"""Per-component latency profile — run this ON THE JETSON to check the
end-to-end budget (handover §9: capture->intent <= 300 ms per stride step).

Reports, for both backends if available:
    per-frame  : MediaPipe Holistic (the usual bottleneck)
    per-step   : emotion (face detect + CNN), gesture TCN, motion LSTM,
                 context CLIP, fusion head
    derived    : sustainable stride, whether S=8 @30fps (267 ms) is feasible

Usage:  python fusion/profile_latency.py --source clip.mp4 --backend torch
        python fusion/profile_latency.py --source 0 --backend onnx --steps 30
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import STRIDE_SEC, HRIPipeline, open_source  # noqa: E402


def timed(fn, *a):
    t0 = time.time()
    out = fn(*a)
    return out, (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    pipe = HRIPipeline(backend=args.backend)
    read, close = open_source(args.source)
    marks = {k: [] for k in ("holistic", "emotion", "gesture", "motion",
                             "context", "fusion_head", "step_total")}

    t_start = time.time()
    next_step, steps = STRIDE_SEC, 0
    try:
        while steps < args.steps:
            frame = read()
            if frame is None:
                break
            t = time.time() - t_start
            _, ms = timed(pipe.push_frame, frame, t)
            marks["holistic"].append(ms)
            if t >= next_step:
                t0 = time.time()
                _, e = timed(pipe._emotion, t); marks["emotion"].append(e)
                _, g = timed(pipe._gesture, t); marks["gesture"].append(g)
                _, m = timed(pipe._motion, t);  marks["motion"].append(m)
                _, c = timed(pipe._context, t); marks["context"].append(c)
                marks["step_total"].append((time.time() - t0) * 1000)
                marks["fusion_head"].append(
                    marks["step_total"][-1] - (e + g + m + c))
                next_step += STRIDE_SEC
                steps += 1
    finally:
        close()

    print(f"\nbackend={args.backend}  steps={steps}\n")
    print(f"{'component':14}{'mean ms':>10}{'p95 ms':>10}{'max ms':>10}")
    for k, v in marks.items():
        if not v:
            continue
        p95 = statistics.quantiles(v, n=20)[-1] if len(v) > 1 else v[0]
        print(f"{k:14}{statistics.mean(v):10.1f}{p95:10.1f}{max(v):10.1f}")

    # Frames are processed as they arrive (pipelined), so capture->intent
    # latency for the frame that triggers a step is ONE holistic pass plus the
    # step work — NOT stride_frames x holistic (that would double-count work
    # already done on earlier frames).
    frame_cost = statistics.mean(marks["holistic"])
    step_cost = statistics.mean(marks["step_total"])
    latency = frame_cost + step_cost
    p95_latency = (statistics.quantiles(marks["holistic"], n=20)[-1]
                   + statistics.quantiles(marks["step_total"], n=20)[-1]
                   if len(marks["step_total"]) > 1 else latency)
    sustainable_fps = 1000.0 / frame_cost

    print(f"\ncapture->intent latency = holistic {frame_cost:.0f} ms "
          f"+ step {step_cost:.0f} ms = {latency:.0f} ms "
          f"(p95 ~{p95_latency:.0f} ms)")
    print(f"budget 300 ms -> {'OK' if latency <= 300 else 'OVER BUDGET'}"
          f"{'  (p95 over)' if p95_latency > 300 >= latency else ''}")
    print(f"\nthroughput: Holistic caps input at ~{sustainable_fps:.0f} fps; "
          f"a {STRIDE_SEC*1000:.0f} ms stride then contains "
          f"~{sustainable_fps*STRIDE_SEC:.1f} new frames.")
    print("  Windowing is TIME-based, so a lower frame rate only coarsens the "
          "resampling — it does not break the window semantics (training clips "
          "were 15 fps).")
    if sustainable_fps < 10:
        print("  WARNING: <10 fps starves the 2 s gesture/motion windows "
              "(min 8/15 valid frames). Use Holistic model_complexity=0.")
    if latency > 300:
        print(f"  To cut latency: model_complexity=0 for Holistic, or reduce "
              f"emotion frames/step (now {marks['emotion'] and 'see above'}).")


if __name__ == "__main__":
    main()
