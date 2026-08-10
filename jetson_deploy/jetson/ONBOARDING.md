# Onboarding — if you're new to this project, start here

Written for whoever picks this project up next (including future-you after
a few months away). Five minutes here should save you from re-deriving
things the hard way.

## What this is

A live human-robot-interaction intent detector running entirely on a
Jetson Orin Nano: a RealSense (or webcam) feed goes in, four independent
models each read one "cue" — facial emotion, hand gesture, body motion,
scene context — and a fusion model combines all four into one intent label
(F01–F10) shown on screen in real time. No cloud calls, no internet
dependency once set up.

## Repository layout — what's where, and why

```
hri-jeston-V2/
├── Cue models/              <- ORIGINAL per-cue training repos (PyTorch).
│   ├── Emotion Repo/           Each is a separate, independently-developed
│   ├── Gesture Repo/           project with its own README, its own
│   ├── Motion Repo/            config.py/src/ layout, its own training
│   └── Context Repo/           pipeline. You will not run these on the
│                                Jetson -- they're the source of the .onnx
│                                exports in jetson_deploy/, and the
│                                reference for "what should this
│                                preprocessing/feature logic actually do."
│
├── latest models/           <- Newer fine-tuned checkpoints
│   (best_TCN_finetuned_merged.pth, finetuned_MobileNetV2_merged.pth)
│   that superseded two of the original Cue models/ checkpoints partway
│   through this project. If checkpoints and exported .onnx ever look
│   out of sync, this folder + jetson_deploy/export/*.py --checkpoint
│   is where to look.
│
├── portable_fusion_models/  <- The trained fusion model(s), already
│   portable_fusion/            exported/packaged, NOT part of the four
│     gbt/                      cue repos above. `gbt/` = trained LightGBM
│     rule_based/                (default); `rule_based/` = hand-written
│                                fallback rules needing no ML library.
│                                jetson_deploy/runtime/fusion.py imports
│                                these directly, unchanged.
│
└── jetson_deploy/           <- THE thing that actually runs on-device.
    Everything below is inside here.
```

### Inside `jetson_deploy/`

```
jetson_deploy/
├── README.md                     <- Setup & installation: exporting ONNX,
│                                     building the venv, connecting the
│                                     camera, running the demo, troubleshooting.
│                                     Read this to get the system running
│                                     the first time.
├── ONBOARDING.md                 <- This file.
│
├── export/                       <- One-time (or checkpoint-change-time)
│   ├── export_*.py                  step: PyTorch checkpoint -> ONNX.
│   ├── build_trt_engines.sh         export_all.sh runs all four;
│   ├── export_all.sh                build_trt_engines.sh then turns those
│   └── verify_onnx.py               ONNX files into TensorRT .engine files.
│                                     verify_onnx.py numerically checks the
│                                     ONNX output matches PyTorch's.
│
├── onnx_models/*.onnx            <- Exported model weights, portable.
├── trt_engines/*.engine          <- TensorRT-compiled versions of the
│                                     same models, FP16, hardware-locked
│                                     (not portable across machines).
│
├── runtime/                      <- The actual pipeline code. Read in
│   ├── camera.py                    this order if you want to understand
│   ├── landmarks.py                 it top-to-bottom -- each file's own
│   ├── cue_emotion.py                docstring explains its piece in
│   ├── cue_gesture.py                detail. Or skip straight to
│   ├── cue_motion.py                 reports/jetson_deployment_report
│   ├── cue_context.py                .methodology.md for the guided,
│   ├── ort_session.py                already-synthesized version of this
│   ├── trt_session.py                exact walk-through.
│   ├── feature_aggregator.py
│   └── fusion.py
│
├── live_demo.py                  <- Entry point. Wires all of the above
│                                     together into the real-time loop.
│                                     Run this to see the system work.
├── test_*_live.py                <- Single-cue verification scripts (one
│                                     per cue), for checking one model in
│                                     isolation without the full pipeline.
│
├── benchmark_report.py           <- Runs live_demo.py for a bounded window
│                                     and turns the metrics into a report +
│                                     CSVs (see reports/).
├── reports/
│   ├── jetson_deployment_report.md              <- Latest benchmark results
│   ├── jetson_deployment_report.methodology.md  <- Full pipeline walk-through
│   ├── jetson_deployment_report.raw.csv         <- Per-frame raw data
│   ├── jetson_deployment_report.intent_summary.csv
│   └── jetson_deployment_report.tegrastats.raw.txt
│
├── requirements-jetson.txt       <- What gets installed ON the Jetson
│                                     (no torch -- runtime never needs it).
├── requirements-export.txt       <- What's needed on whatever machine
│                                     does the PyTorch->ONNX export step
│                                     (does need torch).
└── .venv-runtime/                <- The on-device Python venv. Fully
                                       isolated (NOT --system-site-packages
                                       -- see README's "Set up the Jetson
                                       runtime environment" for why that
                                       matters on this JetPack image).
```

## Recommended reading order for a new person

1. **`README.md`** — get it running first. Nothing else here matters if
   you can't reproduce the live demo.
2. **`reports/jetson_deployment_report.methodology.md`** — the full A-to-Z
   pipeline explanation (camera → landmarks → four cues → feature
   aggregation → fusion → intent), with a diagram and file references.
3. **`live_demo.py`** — once you have the conceptual model from #2, the
   actual orchestration code will read quickly; it's ~290 lines and
   mirrors the methodology doc's structure almost line for line.
4. **The `runtime/cue_*.py` file for whichever cue you're actually
   changing** — each has its own detailed docstring on the exact
   preprocessing/model/output contract, and states which original
   `Cue models/*/` file it was ported from.
5. **`reports/jetson_deployment_report.md`** — current measured
   performance, so you know the baseline before you change anything.

## Key vocabulary

| Term | Meaning |
|---|---|
| **Cue** | One of the four independent per-modality classifiers (emotion, gesture, motion, context). Each is its own ONNX model with its own preprocessing. |
| **Intent** | The final fused output, one of ten labels `F01`–`F10` (see `portable_fusion_models/portable_fusion/gbt/classes.json`). |
| **Fusion** | The model that combines all four cues' rolling-window statistics into one intent. Two interchangeable backends: `gbt` (trained LightGBM) and `rule_based`. |
| **Holistic** | MediaPipe's combined pose+hands landmark detector. The single biggest latency cost in the whole pipeline — most of the FPS/latency tuning flags exist because of this one call. |
| **Stale (pose)** | When `--holistic-every N > 1`, landmarks on skipped frames are reused from the last real detection rather than recomputed — marked `pose_stale=True` / shown as `(stale)` in the UI. |
| **Engine (TensorRT)** | A compiled, hardware-locked, FP16 version of an ONNX model, in `trt_engines/`. Auto-selected over plain ONNX by `runtime/ort_session.py` if present. |

## Things that will bite you if you don't know them up front

- **MediaPipe Holistic is the bottleneck, not the four ONNX models.**
  TensorRT-accelerating the models barely moves end-to-end FPS, because
  Holistic (CPU-only, no GPU delegate available for this platform) costs
  60–85% of every frame regardless. See the methodology doc §8 for the
  actual flags that help (`--holistic-complexity`, `--holistic-every`) and
  measured numbers.
- **The 33-column fusion feature schema is a reconstruction, not an
  extraction.** The original clip-level feature-extraction code that built
  the fusion models' training data isn't part of this bundle — only the
  trained models and their fixed input schema are.
  `runtime/feature_aggregator.py`'s docstring has the full caveat; if
  fusion output looks off relative to a cue's own offline accuracy
  numbers, that reconstruction (and `--window-sec`) is the first thing to
  check, not the models themselves.
- **INTENT refreshing slowly is not a bug.** It's gated by `--fusion-hz`
  (default: twice a second) independent of how fast the rest of the
  pipeline runs — a fast compute pass still waits for the next scheduled
  fusion tick. Don't go chasing this in the cue code.
- **The venv must NOT be created with `--system-site-packages`** on this
  JetPack image — it also flips on `site.ENABLE_USER_SITE`, which causes
  pip installs to silently no-op if a matching version already exists in
  `~/.local`. Full explanation in `README.md`.
- **Camera drop-outs are almost always hardware, not software.** If
  `live_demo.py` suddenly can't find the RealSense/webcam, check `lsusb`
  and `/dev/video*` before touching any code — this has happened multiple
  times in this project's history purely from a loose USB connection or a
  webcam's auto-exposure throttling its own frame rate in low light (see
  README's Troubleshooting section for the exposure-vs-FPS story).
