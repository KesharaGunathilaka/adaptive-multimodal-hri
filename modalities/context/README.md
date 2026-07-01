# Context Model

Situational-context perception for the adaptive HRI system. It fuses three
lightweight sub-models into one structured **`ContextState`** per frame that the
downstream policy / fusion module consumes:

1. **Scene classification** — which environment (`classroom` / `kitchen`), from a
   Places365-trained CNN.
2. **Object detection** — HRI-relevant objects (COCO subset) via YOLO11-Nano with
   tracking.
3. **Gaze estimation** — head-pose + iris gaze (MediaPipe), giving *what the user
   is looking at* and *whether they are engaging the robot*.

The fusion layer combines these into `ContextState { scene, objects,
attention_object, activity, engaged, gaze }`. Targeted at **Jetson Orin Nano**.

## Folder structure

```
context/
├── config.py                  # fusion-layer params (scene throttle, gaze gating, attention dist)
├── README.md
├── src/                       # context-level library
│   ├── context_state.py       # the ContextState contract + fusion rules
│   └── pipeline.py            # ContextPipeline: fuses scene + objects + gaze
├── inference/
│   ├── realtime.py            # webcam — full fused context
│   └── video.py               # video files — --mode context|scene|object|gaze (+ scene accuracy)
│
├── scene_classification/      # sub-model (trained here, mirrors the emotion layout)
│   ├── config.py
│   ├── src/                   # models.py, transforms.py, data.py, engine.py, classifier.py
│   ├── scripts/               # compare_models.py, train.py, tune.py, evaluate.py
│   ├── inference/             # realtime.py, video.py — SELF-CONTAINED (mirrors emotion/inference/)
│   ├── checkpoints/           # weights (git-ignored) + classes.json
│   └── reports/               # generated comparison / training / evaluation reports
│
├── object_detection/          # sub-model (pretrained COCO YOLO)
│   ├── config.py · detector.py
│   ├── scripts/train.py       # future custom-dataset fine-tuning
│   ├── inference/             # realtime.py, video.py (objects only)
│   └── checkpoints/yolo11n.pt
│
└── gaze_estimation/           # sub-model (MediaPipe, no training)
    ├── config.py · gaze_estimator.py
    └── inference/             # realtime.py, video.py (gaze only)
```

## Setup

Use the repo's `.venv` (CUDA-enabled PyTorch + mediapipe + ultralytics).

```bash
# from repo root
.venv\Scripts\activate          # PowerShell: .\.venv\Scripts\Activate.ps1
```

Weights (`*.pth`, `yolo11n.pt`) are git-ignored. Scene data lives at
`data/scene/{train,val}/<class>/` (git-ignored Places365 subset).

## Run the full context model

```bash
# Live webcam (scene + objects + gaze fused)
python modalities/context/inference/realtime.py

# Captured videos — full context, with scene accuracy vs the folder name
python modalities/context/inference/video.py --input videos/Classroom

# A single sub-model on captured videos
python modalities/context/inference/video.py --mode scene  --input videos/Kitchen
python modalities/context/inference/video.py --mode object --input videos/Kitchen
python modalities/context/inference/video.py --mode gaze   --input videos/Classroom
```

`inference/video.py` flags: `--save` (annotated mp4 + per-frame JSON log),
`--no-show` (headless), `--stride N` (process every Nth frame).

## Scene-classification pipeline (the only trained sub-model)

Same staged workflow as the emotion model — run from `scene_classification/`:

```bash
cd modalities/context/scene_classification
python scripts/compare_models.py     # Stage 1 — pick best backbone  -> reports/comparison/
python scripts/train.py              # Stage 2 — full training        -> checkpoints/ + reports/training/
python scripts/tune.py               # Stage 3 — hyper-parameter tuning -> reports/tuning/
python scripts/evaluate.py           # Stage 4 — full evaluation       -> reports/evaluation/
```

Deployed model: **EfficientNet-B0** (`checkpoints/best_EfficientNet_B0.pth`),
2-class (classroom, kitchen). Every script supports `--help` and `--model`.

## Per-sub-model inference (run independently)

Each sub-model also ships its own webcam/video scripts for isolated testing, in
the same style as `modalities/emotion/inference/`. `video.py` defaults to
scanning the **repo-root `videos/` folder** when run with no arguments, and
supports single-file mode, batch mode, `--skip-existing`, and pause/next/prev/
quit playback controls (`[SPACE] [N] [P] [Q]`).

```bash
# Scene classification — SELF-CONTAINED: needs only its .pth + pip install
# torch torchvision opencv-python pillow numpy (no project imports at all)
cd modalities/context/scene_classification/inference
python video.py                              # batch: repo videos/ folder
python video.py --video ../../../../myclip.mp4
python realtime.py                            # webcam / RealSense

# Object detection — needs the project's detector.py + a YOLO checkpoint
cd modalities/context/object_detection/inference
python video.py                              # batch: repo videos/ folder
python realtime.py

# Gaze estimation — needs the project's gaze_estimator.py
cd modalities/context/gaze_estimation/inference
python video.py                              # batch: repo videos/ folder
python realtime.py
```

Common `video.py` flags: `--video <file>` / `--videos-dir <folder>` (mutually
exclusive, default is the repo `videos/` folder), `--output` / `--out-dir`
(default `outputs/`), `--no-show`, `--skip-existing`.

## Notes

- **Kitchen scene accuracy** on captured clips is lower than classroom — many
  kitchen clips are person close-ups with little kitchen visible. The fix is
  domain adaptation (fine-tune on captured frames); see `scripts/train.py`.
- **Gaze constants** (signs / gains / engagement thresholds) in
  `gaze_estimation/config.py` are approximate and may need a quick on-device
  calibration.
- Object detection uses pretrained COCO classes; extend to non-COCO objects via
  `object_detection/scripts/train.py`.
