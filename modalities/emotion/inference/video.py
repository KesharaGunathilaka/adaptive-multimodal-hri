"""
Emotion recognition on a single video file or all videos under a folder.

Pipeline (the plain emotion-branch method that works with the deployed
MobileNetV2): frame -> MediaPipe close-range face detection -> tight crop ->
224x224 + ImageNet normalize -> model -> emotion label + confidence overlay.
Large frames are downscaled before detection for speed (the crop is still taken
from the full-resolution frame).

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
import torch
import torch.nn.functional as F
from PIL import Image

from config import CHECKPOINT_DIR, DEFAULT_MODEL, EMOTION_LABELS, ROOT
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms

MAX_FRAME_WIDTH = 640
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

_PROJECT_ROOT = os.path.normpath(os.path.join(ROOT, "..", ".."))
DEFAULT_VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "videos")


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
    cv2.putText(
        frame,
        _HINT,
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if paused:
        text, font, scale, thick = "PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(
            frame,
            text,
            (cx + 2, cy + 2),
            font,
            scale,
            (0, 0, 0),
            thick + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (cx, cy), font, scale, (0, 220, 255), thick, cv2.LINE_AA
        )


def process_video(video_path, out_path, model, transform, face_detection, device, show):
    """Process one video. Returns one of: "done" | "next" | "prev" | "quit"."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    action = "done"
    paused = False
    display_frame = None  # last annotated frame, shown while paused

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            small = (
                cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
                if w > MAX_FRAME_WIDTH
                else frame
            )

            results = face_detection.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if results.detections:
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
                    bw, bh = int(box.width * w), int(box.height * h)
                    face = frame[y : y + bh, x : x + bw]
                    if face.size == 0:
                        continue
                    pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    tensor = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        conf, pred = torch.max(F.softmax(model(tensor), dim=1), 1)
                    lbl = f"{EMOTION_LABELS[pred.item()]}: {conf.item() * 100:.1f}%"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        lbl,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

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
    src.add_argument(
        "--videos-dir",
        default=None,
        help=f"Folder to scan recursively (default: {DEFAULT_VIDEOS_DIR}).",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--output", default=None, help="Output path (single-file mode only)."
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a preview window (always true in batch mode).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose output file already exists.",
    )
    args = ap.parse_args()

    device = get_device()
    ckpt = args.checkpoint or os.path.join(
        CHECKPOINT_DIR, f"best_{safe_name(args.model)}.pth"
    )
    model = ALL_MODELS[args.model]()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = get_test_transforms()
    print(f"Loaded {args.model} from {ckpt}")

    # Close-range model + tight crop, matching the deployed MobileNetV2.
    face_detection = mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
        model_selection=1, min_detection_confidence=0.5
    )

    out_root = os.path.join(ROOT, "outputs")

    # ── Single-file mode ─────────────────────────────────────────────────
    if args.video is not None:
        os.makedirs(out_root, exist_ok=True)
        out_path = args.output or os.path.join(
            out_root, os.path.splitext(os.path.basename(args.video))[0] + "_emotion.mp4"
        )
        print(f"Writing to {out_path}")
        process_video(
            args.video,
            out_path,
            model,
            transform,
            face_detection,
            device,
            show=not args.no_show,
        )
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
            print(f"[{idx + 1}/{len(videos)}] skip (exists): {label}")
            skipped += 1
            idx += 1
            continue
        print(f"[{idx + 1}/{len(videos)}] {label}")
        print(f"         -> {os.path.relpath(out_path, ROOT)}")
        action = process_video(
            video_path,
            out_path,
            model,
            transform,
            face_detection,
            device,
            show=not args.no_show,
        )
        cv2.destroyAllWindows()
        if action == "quit":
            print("  Quit by user.")
            break
        elif action == "prev":
            idx = max(0, idx - 1)
            print(f"  Going back to [{idx + 1}/{len(videos)}]")
        else:  # "done" or "next"
            done += 1
            idx += 1

    cv2.destroyAllWindows()
    print(f"\nDone. {done} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
