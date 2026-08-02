"""Offline per-cue + fused-intent inference on recorded video(s) -> annotated
output video.

Same model loading / windowing as pipeline.py and cues_live.py, but:
  * reads frames from a file instead of a live camera,
  * timestamps frames by FRAME INDEX / VIDEO FPS (not wall-clock), so the
    cue windows (gesture 2.13 s, motion 2.0 s, ...) line up with the video's
    own timeline regardless of how fast this machine can decode/infer,
  * burns a dashboard into the output video: a top banner with the fused
    intent/action (via pipe.step() — same fusion head + policy + temporal
    smoothing/hysteresis as the live pipeline), plus a sidebar with each
    modality's own label + confidence,
  * withholds the fused intent/action for the first --intent-delay seconds
    of each video (default 3s) — the cue windows need to fill up first, so
    early intents are just noise. Cue panels still update as soon as ready.

Usage (run from jetson_deploy/):
    python fusion/cues_video.py --input ../data/raw/clips/classroom/S01_F04/S01_F04_c001.mp4
    python fusion/cues_video.py --input path/to/clips_dir --output-dir out/ --backend onnx
    python fusion/cues_video.py --input clip.mp4 --no-display --json cues.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import HRIPipeline, STRIDE_SEC, EMOTION_LABELS, CONTEXT_LABELS

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MAX_SIDE = 960
SIDEBAR_W = 380

# Must match modalities/gesture/config.py::GESTURE_LABELS and
# modalities/motion/src/inference.py::MOTION_LABELS.
GESTURE_LABELS = ["idle", "wave", "point", "thumbs_up", "thumbs_down",
                  "beckoning", "raise_hand", "both_hands_up"]
MOTION_LABELS = ["sitting", "standing", "walking", "stepping_back"]

# BGR colours per cue label, for the dashboard.
CUE_COLORS = {
    "emotion": {
        "Surprise": (0, 210, 255), "Fear": (200, 80, 255), "Disgust": (60, 180, 75),
        "Happy": (0, 220, 0), "Sad": (255, 160, 0), "Anger": (0, 0, 255),
        "Neutral": (180, 180, 180),
    },
    "gesture": {
        "idle": (160, 160, 160), "wave": (0, 200, 255), "point": (255, 210, 0),
        "thumbs_up": (80, 220, 60), "thumbs_down": (60, 60, 230),
        "beckoning": (255, 120, 0), "raise_hand": (200, 100, 255),
        "both_hands_up": (0, 255, 220),
    },
    "motion": {
        "sitting": (0, 200, 255), "standing": (0, 255, 0),
        "walking": (255, 200, 0), "stepping_back": (0, 0, 255),
    },
    "context": {
        "classroom": (50, 150, 220), "kitchen": (60, 180, 255),
        "hospital": (230, 230, 230), "cloth_store": (200, 100, 255),
        "museum": (0, 170, 110),
    },
}
CUE_ORDER = ["emotion", "gesture", "motion", "context"]
BANNER_H = 58


def _reset_stream_state(pipe):
    """HRIPipeline keeps rolling buffers + hysteresis state on self; when
    batch-processing several videos with one loaded pipeline, clear it
    between videos so the previous clip's tail doesn't bleed into the next
    clip's opening frames."""
    pipe.buf.clear()
    pipe.frames.clear()
    pipe.ctx_probs = None
    pipe.ctx_time = -1e9
    pipe.history.clear()
    pipe.active_intent = "F05"
    pipe.candidate, pipe.candidate_count = None, 0


def _fmt(probs, labels):
    if probs is None:
        return None
    i = int(np.argmax(probs))
    return {"label": labels[i], "confidence": float(probs[i])}


def cues_step(pipe, t):
    emo = pipe._emotion(t)
    ges = pipe._gesture(t)
    mot = pipe._motion(t)
    ctx = pipe._context(t)
    return {
        "t": round(t, 3),
        "emotion": _fmt(emo, EMOTION_LABELS),
        "gesture": _fmt(ges, GESTURE_LABELS),
        "motion": _fmt(mot, MOTION_LABELS),
        "context": _fmt(ctx, CONTEXT_LABELS),
    }


def resize_with_aspect_ratio(image, max_dim=MAX_SIDE):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_banner(canvas, fused, w_total):
    if fused is None:
        cv2.rectangle(canvas, (0, 0), (w_total, BANNER_H), (28, 28, 32), -1)
        cv2.putText(canvas, "buffering...", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (150, 150, 150), 1, cv2.LINE_AA)
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    emergency = fused["emergency"]
    bg = (0, 0, 150) if emergency else (35, 90, 40)
    cv2.rectangle(canvas, (0, 0), (w_total, BANNER_H), bg, -1)

    headline = f"{fused['intent']}  ->  {fused['action']}   p={fused['confidence']:.2f}"
    if emergency:
        headline = "EMERGENCY  " + headline
    cv2.putText(canvas, headline, (16, 27), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    missing = [k for k, v in fused["observed"].items() if not v]
    sub = fused["action_text"]
    if missing:
        sub += "   |  missing: " + ", ".join(missing)
    cv2.putText(canvas, sub[:120], (16, 47), font, 0.42, (215, 215, 215), 1, cv2.LINE_AA)


def draw_dashboard(frame, cues, fused, t, frame_idx, n_frames, proc_fps):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas_h = h + BANNER_H
    canvas = np.zeros((canvas_h, w + SIDEBAR_W, 3), dtype=np.uint8)
    canvas[BANNER_H:, :w] = frame
    cv2.rectangle(canvas, (w, BANNER_H), (w + SIDEBAR_W, canvas_h), (18, 20, 26), -1)
    cv2.line(canvas, (w, BANNER_H), (w, canvas_h), (45, 55, 72), 2)
    draw_banner(canvas, fused, w + SIDEBAR_W)

    top = BANNER_H
    cv2.putText(canvas, "HRI MULTIMODAL CUE ANALYZER", (w + 18, top + 32), font, 0.58,
                (230, 235, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (w + 18, top + 44), (w + SIDEBAR_W - 18, top + 44), (60, 75, 100), 1)
    progress = f"frame {frame_idx}/{n_frames}" if n_frames else f"frame {frame_idx}"
    cv2.putText(canvas, f"t={t:6.2f}s   {progress}", (w + 18, top + 64), font, 0.45,
                (140, 150, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"proc {proc_fps:.1f} fps", (w + 18, top + 82), font, 0.45,
                (140, 150, 170), 1, cv2.LINE_AA)

    row_h = (h - 100) // 4
    y0 = top + 100
    for name in CUE_ORDER:
        cue = cues.get(name) if cues else None
        panel_top = y0
        panel_bot = y0 + row_h - 14

        cv2.putText(canvas, name.upper(), (w + 18, panel_top + 16), font, 0.5,
                    (110, 130, 160), 1, cv2.LINE_AA)

        box_left, box_right = w + 18, w + SIDEBAR_W - 18
        box_top = panel_top + 26
        box_bot = panel_bot - 12
        if cue is None:
            cv2.rectangle(canvas, (box_left, box_top), (box_right, box_bot),
                         (35, 38, 46), -1)
            cv2.putText(canvas, "buffering...", (box_left + 12, (box_top + box_bot) // 2 + 5),
                        font, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        else:
            colour = CUE_COLORS[name].get(cue["label"], (255, 255, 255))
            cv2.rectangle(canvas, (box_left, box_top), (box_right, box_bot),
                         colour, -1)
            text_colour = (255, 255, 255) if np.mean(colour) < 120 else (20, 20, 20)
            label_txt = cue["label"].replace("_", " ")
            (tw, _), _ = cv2.getTextSize(label_txt, font, 0.6, 2)
            cv2.putText(canvas, label_txt, (box_left + 12, box_top + 28), font, 0.6,
                        text_colour, 2, cv2.LINE_AA)
            pct_txt = f"{cue['confidence']*100:.0f}%"
            (ptw, _), _ = cv2.getTextSize(pct_txt, font, 0.5, 1)
            cv2.putText(canvas, pct_txt, (box_right - ptw - 10, box_top + 28), font, 0.5,
                        text_colour, 1, cv2.LINE_AA)
            # confidence bar, strictly inside the sidebar
            bar_y = box_bot - 10
            cv2.rectangle(canvas, (box_left + 12, bar_y), (box_right - 12, bar_y + 6),
                         (0, 0, 0), -1)
            filled = box_left + 12 + int((box_right - box_left - 24) * cue["confidence"])
            cv2.rectangle(canvas, (box_left + 12, bar_y), (filled, bar_y + 6),
                         text_colour, -1)
        y0 += row_h

    return canvas


def process_video(pipe, in_path: Path, out_path: Path, args, jf):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[skip] cannot open {in_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {in_path.name}  fps={fps:.1f}  frames={n_frames}")

    _reset_stream_state(pipe)
    writer = None
    next_step = STRIDE_SEC
    last, last_fused = None, None
    frame_idx = 0
    proc_t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_with_aspect_ratio(frame)
        t = frame_idx / fps
        pipe.push_frame(frame, t)

        if t >= next_step:
            last = cues_step(pipe, t)
            if t >= args.intent_delay:
                last_fused = pipe.step(t)
            next_step += STRIDE_SEC

            msg = f"[{in_path.name} {last['t']:6.2f}s]"
            if last_fused:
                msg += (f" {last_fused['intent']} -> {last_fused['action']} "
                        f"p={last_fused['confidence']:.2f}"
                        f"{'  EMERGENCY' if last_fused['emergency'] else ''}")
            else:
                msg += f" (intent warms up at {args.intent_delay:.1f}s)"
            print(msg, flush=True)

            if jf:
                record = {"video": in_path.name, **last}
                if last_fused:
                    record["fused"] = last_fused
                jf.write(json.dumps(record) + "\n")

        canvas = draw_dashboard(frame, last, last_fused, t, frame_idx, n_frames,
                                proc_fps=frame_idx / max(time.time() - proc_t0, 1e-9))

        if writer is None and args.save:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer is not None:
            writer.write(canvas)

        if args.display:
            cv2.imshow("HRI cues (recorded video)", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[video] saved -> {out_path}")


def collect_inputs(input_path: Path):
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    return [input_path]


def main():
    ap = argparse.ArgumentParser(
        description="Offline per-cue + fused-intent inference on recorded video(s)")
    ap.add_argument("--input", required=True, help="video file or a directory of videos")
    ap.add_argument("--output-dir", default=None,
                    help="where annotated videos go (default: next to the input)")
    ap.add_argument("--backend", choices=("torch", "onnx", "tensorrt"), default="torch")
    ap.add_argument("--display", dest="display", action="store_true", default=True)
    ap.add_argument("--no-display", dest="display", action="store_false")
    ap.add_argument("--no-save", dest="save", action="store_false", default=True,
                    help="skip writing annotated output video (still can log --json)")
    ap.add_argument("--json", default=None, help="append per-step cue results as JSONL")
    ap.add_argument("--intent-delay", type=float, default=3.0,
                    help="withhold fused intent/action until this many seconds into "
                         "each video (cue panels still update immediately); default 3.0")
    args = ap.parse_args()

    input_path = Path(args.input)
    videos = collect_inputs(input_path)
    if not videos:
        raise SystemExit(f"no video files found at {input_path}")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    pipe = HRIPipeline(backend=args.backend)
    jf = open(args.json, "a", encoding="utf-8") if args.json else None

    try:
        for vid in videos:
            out_path = (out_dir or vid.parent) / f"{vid.stem}_cues.mp4"
            process_video(pipe, vid, out_path, args, jf)
    finally:
        if jf:
            jf.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
