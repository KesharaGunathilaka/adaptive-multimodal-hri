"""
Emotion recognition on a single video file or all videos under a folder.

Robustness features for in-the-wild video (vs. controlled RAF-DB):
  - downscale large frames for faster MediaPipe detection
  - eye-keypoint face alignment (correct head tilt)
  - 40% crop padding to match RAF-DB framing
  - CLAHE contrast normalization
  - test-time augmentation (average logits of original + horizontal flip)

Run (from the emotion folder):
    python inference/video.py --video ../../videos/test/C1_D2_T2.mp4
    python inference/video.py                          # all videos under <project>/videos/
    python inference/video.py --videos-dir ../../videos/Classroom
    python inference/video.py --skip-existing          # resume a previous batch run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CHECKPOINT_DIR, DEFAULT_MODEL, EMOTION_LABELS, ROOT
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.postprocess import build_logit_bias, parse_class_bias
from src.transforms import get_test_transforms

MAX_FRAME_WIDTH = 640
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

_PROJECT_ROOT = os.path.normpath(os.path.join(ROOT, "..", ".."))
DEFAULT_VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "videos")


def align_face(image, left_eye, right_eye):
    """Rotate so the eyes are horizontal (skips extreme/negligible tilt)."""
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return image
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def collect_videos(videos_dir):
    """Recursively find all video files under videos_dir, sorted by path."""
    found = []
    for dirpath, _, filenames in os.walk(videos_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def build_output_path(video_path, videos_dir, out_root):
    """Mirror the folder structure of videos_dir under out_root."""
    rel = os.path.relpath(video_path, videos_dir)
    stem = os.path.splitext(rel)[0]
    out_path = os.path.join(out_root, stem + "_emotion.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


_HINT = "[SPACE] Pause   [N] Next   [P] Prev   [Q] Quit"


def _draw_overlay(frame, paused):
    """Draw key-hint bar and optional PAUSED indicator on frame (in-place)."""
    h, w = frame.shape[:2]
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, _HINT, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    if paused:
        text, font, scale, thick = "PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (cx, cy), font, scale, (0, 220, 255), thick, cv2.LINE_AA)


def process_video(video_path, out_path, model, transform, face_detection, device, show,
                  logit_bias=0.0, padding=0.4, use_clahe=True, use_tta=True, use_align=True):
    """Process one video.

    Returns one of: "done" | "next" | "prev" | "quit"
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    action = "done"
    paused = False
    display_frame = None   # last annotated frame, shown while paused

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            small = (cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
                     if w > MAX_FRAME_WIDTH else frame)

            results = face_detection.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if results.detections:
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    x, y = int(box.xmin * w), int(box.ymin * h)
                    bw, bh = int(box.width * w), int(box.height * h)
                    kp = det.location_data.relative_keypoints
                    right_eye = (int(kp[0].x * w), int(kp[0].y * h))
                    left_eye  = (int(kp[1].x * w), int(kp[1].y * h))
                    aligned = align_face(frame, left_eye, right_eye) if use_align else frame

                    pad_w, pad_h = int(bw * padding), int(bh * padding)
                    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                    x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)
                    face = aligned[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    if use_clahe:
                        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
                        l_ch, a_ch, b_ch = cv2.split(lab)
                        face = cv2.cvtColor(cv2.merge((_clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
                    pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    t = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if use_tta:
                            t_flip = transform(pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
                            logits = (model(t) + model(t_flip)) / 2.0
                        else:
                            logits = model(t)
                        top = torch.max(F.softmax(logits + logit_bias, dim=1), dim=1)
                        conf, pred = top.values, top.indices
                    lbl = f"{EMOTION_LABELS[pred.item()]}: {conf.item()*100:.1f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, lbl, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(frame)
            display_frame = frame.copy()

        if show and display_frame is not None:
            view = display_frame.copy()
            _draw_overlay(view, paused)
            cv2.imshow("Emotion Recognition", view)
            key = cv2.waitKey(100 if paused else 30) & 0xFF
            if key == ord("q"):
                action = "quit"
                break
            elif key == ord("n"):
                action = "next"
                break
            elif key == ord("p"):
                action = "prev"
                break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    writer.release()
    return action


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", default=None, help="Path to a single input video file.")
    src.add_argument("--videos-dir", default=None,
                     help=f"Folder to scan recursively (default: {DEFAULT_VIDEOS_DIR}).")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None, help="Output path (single-file mode only).")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open a preview window (always true in batch mode).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip videos whose output file already exists.")
    # Surprise-bias mitigation + preprocessing A/B knobs:
    ap.add_argument("--prior-correction", type=float, default=0.0,
                    help="Re-inject natural class prior (T*log prior). Try 0.5-1.0 "
                         "to cut Surprise over-prediction.")
    ap.add_argument("--class-bias", default=None,
                    help="Manual per-class logit offsets, e.g. '-1,0,0,0,0,0,0.5' "
                         f"(order: {', '.join(EMOTION_LABELS)}).")
    ap.add_argument("--padding", type=float, default=0.4,
                    help="Crop padding fraction per side (A/B: try 0.0-0.4).")
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (A/B test).")
    ap.add_argument("--no-tta", action="store_true", help="Disable flip test-time augmentation.")
    ap.add_argument("--no-align", action="store_true", help="Disable eye-based face alignment.")
    args = ap.parse_args()

    device = get_device()
    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{safe_name(args.model)}.pth")
    model = ALL_MODELS[args.model]()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = get_test_transforms()
    logit_bias = build_logit_bias(parse_class_bias(args.class_bias),
                                  args.prior_correction, device)
    proc_opts = dict(logit_bias=logit_bias, padding=args.padding,
                     use_clahe=not args.no_clahe, use_tta=not args.no_tta,
                     use_align=not args.no_align)
    print(f"Loaded {args.model} from {ckpt}")
    if args.prior_correction or args.class_bias:
        print(f"Logit bias applied: {logit_bias.cpu().numpy().round(2).tolist()}")

    face_detection = mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
        model_selection=1, min_detection_confidence=0.3)

    out_root = os.path.join(ROOT, "outputs")

    # ── Single-file mode ─────────────────────────────────────────────────
    if args.video is not None:
        os.makedirs(out_root, exist_ok=True)
        out_path = args.output or os.path.join(
            out_root, os.path.splitext(os.path.basename(args.video))[0] + "_emotion.mp4")
        print(f"Writing to {out_path}")
        process_video(args.video, out_path, model, transform, face_detection, device,
                      show=not args.no_show, **proc_opts)  # prev/next ignored in single-file mode
        cv2.destroyAllWindows()
        print(f"Saved: {out_path}")
        return

    # ── Batch mode ───────────────────────────────────────────────────────
    videos_dir = os.path.abspath(args.videos_dir or DEFAULT_VIDEOS_DIR)
    videos = collect_videos(videos_dir)
    if not videos:
        print(f"No video files found under: {videos_dir}")
        return

    print(f"Found {len(videos)} video(s) under {videos_dir}")
    done, skipped, idx = 0, 0, 0
    while idx < len(videos):
        video_path = videos[idx]
        out_path = build_output_path(video_path, videos_dir, out_root)
        label = os.path.relpath(video_path, videos_dir)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{idx+1}/{len(videos)}] skip (exists): {label}")
            skipped += 1
            idx += 1
            continue
        print(f"[{idx+1}/{len(videos)}] {label}")
        print(f"         -> {os.path.relpath(out_path, ROOT)}")
        action = process_video(video_path, out_path, model, transform, face_detection, device,
                               show=not args.no_show, **proc_opts)
        cv2.destroyAllWindows()
        if action == "quit":
            print("  Quit by user.")
            break
        elif action == "prev":
            idx = max(0, idx - 1)
            print(f"  Going back to [{idx+1}/{len(videos)}]")
        else:  # "done" or "next"
            done += 1
            idx += 1

    cv2.destroyAllWindows()
    print(f"\nDone. {done} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
