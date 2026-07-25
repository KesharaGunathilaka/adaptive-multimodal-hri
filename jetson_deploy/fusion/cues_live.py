"""Live per-cue inference — RealSense/webcam -> 4 perception models.

Same capture/windowing/model loading as pipeline.py, but prints each
modality's OWN predicted label independently (emotion, gesture, motion,
context). No fusion, no intent, no policy.

Usage (run from jetson_deploy/):
    python fusion/cues_live.py --source realsense
    python fusion/cues_live.py --source 0 --backend onnx --no-display
"""
import argparse
import json
import time

import cv2
import numpy as np

from pipeline import HRIPipeline, open_source, STRIDE_SEC, EMOTION_LABELS, CONTEXT_LABELS

# Must match modalities/gesture/config.py::GESTURE_LABELS and
# modalities/motion/src/inference.py::MOTION_LABELS.
GESTURE_LABELS = ["idle", "wave", "point", "thumbs_up", "thumbs_down",
                  "beckoning", "raise_hand", "both_hands_up"]
MOTION_LABELS = ["sitting", "standing", "walking", "stepping_back"]


def _fmt(probs, labels):
    if probs is None:
        return None
    i = int(np.argmax(probs))
    return {"label": labels[i], "confidence": round(float(probs[i]), 3)}


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


def _line(cue):
    return f"{cue['label']} ({cue['confidence']:.2f})" if cue else "..."


def _draw(frame, r):
    if not r:
        return
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 100), (frame.shape[1], h), (0, 0, 0), -1)
    rows = [("emotion", r["emotion"]), ("gesture", r["gesture"]),
            ("motion", r["motion"]), ("context", r["context"])]
    for i, (name, cue) in enumerate(rows):
        colour = (0, 200, 0) if cue else (0, 0, 200)
        cv2.putText(frame, f"{name:8s} {_line(cue)}", (10, h - 80 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="Live per-cue inference (emotion/gesture/motion/context) — no fusion")
    ap.add_argument("--source", default="realsense",
                    help="'realsense', a camera index ('0'), or a video path")
    ap.add_argument("--camera", type=int, default=0,
                    help="Webcam index used if --source realsense finds no device.")
    ap.add_argument("--backend", choices=("torch", "onnx", "tensorrt"), default="torch")
    ap.add_argument("--display", dest="display", action="store_true", default=True)
    ap.add_argument("--no-display", dest="display", action="store_false")
    ap.add_argument("--json", default=None, help="append results as JSONL")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    pipe = HRIPipeline(backend=args.backend)
    read, close = open_source(args.source, camera_fallback=args.camera)
    jf = open(args.json, "a", encoding="utf-8") if args.json else None

    t_start = time.time()
    next_step = STRIDE_SEC
    last = None
    n_frames = 0
    try:
        while True:
            frame = read()
            if frame is None:
                break
            t = time.time() - t_start
            pipe.push_frame(frame, t)
            n_frames += 1
            if t >= next_step:
                last = cues_step(pipe, t)
                next_step += STRIDE_SEC
                print(f"[{last['t']:6.2f}s] emotion={_line(last['emotion'])}  "
                      f"gesture={_line(last['gesture'])}  "
                      f"motion={_line(last['motion'])}  "
                      f"context={_line(last['context'])}", flush=True)
                if jf:
                    jf.write(json.dumps(last) + "\n")
            if args.display:
                _draw(frame, last)
                cv2.imshow("HRI cues (no fusion)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_seconds and t > args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        close()
        if jf:
            jf.close()
        cv2.destroyAllWindows()

    dur = time.time() - t_start
    print(f"\n--- {n_frames} frames in {dur:.1f}s ({n_frames/max(dur,1e-9):.1f} fps)")


if __name__ == "__main__":
    main()
