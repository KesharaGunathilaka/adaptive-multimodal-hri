# Jetson live-inference package

Self-contained copy of the 4 deployed perception models — code + the single
deployed checkpoint per modality + a pre-downloaded HuggingFace cache — for
testing live inference on the Jetson Orin Nano. Built 2026-07-16 from the
main repo; each model's full training pipeline, alternate checkpoints, and
evaluation reports stay in the main repo (not copied here — this folder is
inference-only).

> **Working on the Jetson itself with a Claude Code / VS Code agent?** Rename
> `HANDOVER_CLAUDE.md` (in this folder) to `CLAUDE.md` so it auto-loads —
> it's the onboarding doc for a fresh agent session on that machine, and
> points to `JETSON_SETUP_GUIDE.md` (full step-by-step setup) and
> `JETSON_TEST_LOG.md` (where to record findings) also in this folder.

## What's in here

```
jetson_deploy/
├── modalities/
│   ├── emotion/      config.py, src/, inference/, checkpoints/finetuned_MobileNetV2.pth
│   ├── gesture/      config.py, src/, inference/, checkpoints/best_TCN.pth + model_config.json
│   ├── motion/       src/, inference/, checkpoints/best_model_finetuned.pt
│   └── context/      config.py, src/, inference/, scene_classification/ (CLIP backend — no local
│                     checkpoint needed, uses hf_cache/ below)
├── hf_cache/         pre-downloaded HF weights: CLIP ViT-B-32 (~600MB) + SmolVLM2-500M (~2.5GB)
└── requirements.txt   the main repo's shared requirements
```

Every checkpoint here is the one each modality's README currently names as
**deployed**, verified this session against held-out real video:

| Modality | Checkpoint | Held-out test accuracy / macro-F1 |
|---|---|---|
| Emotion | `finetuned_MobileNetV2.pth` | 92.5% / 90.1% |
| Gesture | `best_TCN.pth` | 84.1% / 82.9% |
| Motion | `best_model_finetuned.pt` | 76.8% / 73.2% |
| Context/Scene | CLIP zero-shot (no local weights) | 98.8% / 99.3% |

Known failure modes per model (kitchen `stepping_back` for motion, static
`point` and weak `wave` for gesture, thin Fear/Disgust samples for emotion)
are documented in each modality's own `README.md` — read those before
drawing conclusions from a live demo that looks wrong.

## Setup on the Jetson

1. **PyTorch/torchvision must be Jetson-specific builds** — the generic pip
   wheels in `requirements.txt` (`torch`, `torchvision` with no version/index)
   will not install correctly on Jetson's ARM64 + CUDA setup. Install NVIDIA's
   JetPack-matched wheels first (per your L4T/JetPack version), then install
   the rest:
   ```bash
   pip install -r requirements.txt   # skip/adjust torch+torchvision if already installed above
   ```
2. **Two packages are missing from `requirements.txt`** (a pre-existing gap
   in the main repo, not fixed here) but are required for the context
   modality: `open_clip_torch` and `transformers`. Install them explicitly:
   ```bash
   pip install open_clip_torch transformers
   ```
3. **Point HuggingFace at the bundled cache so context runs fully offline**
   (no internet needed on the Jetson):
   ```bash
   export HF_HOME=/path/to/jetson_deploy/hf_cache
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```
   Verified working offline from this exact cache before copying (both CLIP
   and SmolVLM2 load with zero network calls under these env vars).
4. `pyrealsense2` is in `requirements.txt` for the RealSense entry points;
   if you're testing with a plain USB webcam instead, the emotion/gesture
   realtime scripts fall back to `--camera 0` and motion/context's
   webcam demos use `cv2.VideoCapture(0)` directly — no RealSense required.

## Running each model live

All commands assume you `cd` into `jetson_deploy/` first (or adjust paths).

```bash
# Emotion — RealSense (webcam fallback) / video file
python modalities/emotion/inference/realtime_realsense.py
python modalities/emotion/inference/video.py --video path/to/clip.mp4

# Gesture — RealSense (webcam fallback) / video file
python modalities/gesture/inference/realtime_realsense.py
python modalities/gesture/inference/video.py --input path/to/clip_or_folder --save

# Motion — webcam / video file
python modalities/motion/inference/realtime.py
python modalities/motion/inference/video.py --video path/to/clip.mp4 --save

# Context — full pipeline (CLIP scene + SmolVLM2), needs the HF_HOME env vars above
python modalities/context/inference/realtime.py
python modalities/context/inference/video.py --input path/to/videos --save

# Context — scene only, standalone (lighter weight, skips the VLM)
python modalities/context/scene_classification/inference/realtime.py
```

Each script prints the resolved checkpoint/model path on startup — **check
that line** before trusting results. (This project already got bitten once
by a silently-wrong default checkpoint after a folder copy; it's cheap
insurance to glance at that line every time.)

## Verified before packaging

Every checkpoint above was loaded and every model instantiated from this
exact folder layout this session (not just "files exist" — actually ran):
emotion's `build_model()` + checkpoint load, gesture's `GestureEngine`,
motion's `MotionInference`, and context's `create_scene_classifier()` (CLIP
backend) under `HF_HUB_OFFLINE=1` pointed at the bundled cache. All loaded
without errors or network calls.
