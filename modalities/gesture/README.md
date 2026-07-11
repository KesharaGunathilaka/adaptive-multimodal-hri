# Gesture Recognition (v2)

Real-time hand/body gesture recognition for the adaptive HRI system.
MediaPipe Holistic landmarks (pose + both hands) are turned into 185-dim
per-frame features, a 32-frame window feeds a small temporal network
(BiGRU / TCN / TinyTransformer), and the engine emits one of **8 classes**:

`idle · wave · point · thumbs_up · thumbs_down · beckoning · raise_hand · both_hands_up`

Unlike the removed v1 (static hand-pose MLP + hand-written motion rules),
static and dynamic gestures are learned **uniformly** from keypoint sequences
— no rules, real metrics for every class. Targeted at **Jetson Orin Nano**.

**Design spec & HPC runbook (source of truth):
[`docs/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md`](../../docs/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md)**

## Data

Trained on public datasets + a small custom set (guide §4):

| Source | Classes contributed |
|---|---|
| 20BN-Jester | thumbs_up/down, wave, beckoning (proxy), idle negatives |
| NTU RGB+D 120 | wave, point, thumbs_up/down, both_hands_up, idle negatives |
| Custom recordings | beckoning, raise_hand (+ live test set for ALL classes) |

Raw datasets live outside the repo — set `GESTURE_DATA_ROOT` (defaults to
`data/raw/`), laid out per guide §4.

## Folder structure

```
gesture/
├── config.py               # classes, paths, feature/window spec, engine params
├── src/                    # importable library
│   ├── features.py         # landmark -> 185-dim features, normalization, augmentation
│   ├── data.py             # dataset mapping tables, sequence dataset, splits I/O
│   ├── models.py           # BiGRU / TCN / TinyTransformer (< 1M params each)
│   ├── training.py         # shared fit/eval recipe (early stop on macro-F1)
│   └── engine.py           # GestureEngine: rolling window + EMA + debounce
├── scripts/                # runnable pipeline stages
│   ├── extract_landmarks.py  # Stage 0  MediaPipe over datasets -> .npz (CPU, shardable)
│   ├── prepare_data.py       # Stage 0.5 split index CSVs + data report
│   ├── compare_models.py     # Stage 1  compare architectures -> pick winner
│   ├── train.py              # Stage 2  full training (+ model_config.json)
│   ├── tune.py               # Stage 3  optuna hyper-parameter tuning
│   └── evaluate.py           # Stage 4  test + live-test + latency report
├── inference/
│   ├── realtime_realsense.py # live RealSense / webcam dashboard
│   └── video.py              # video files -> annotated mp4
├── data/index/             # split CSVs (committed); data/raw & landmarks git-ignored
├── checkpoints/            # weights (git-ignored; published as GitHub Releases)
└── reports/                # generated stage reports
```

## Pipeline (run from this folder)

```bash
python scripts/extract_landmarks.py --dataset all      # shardable: --shard i/N
python scripts/prepare_data.py
python scripts/compare_models.py
python scripts/train.py --model <winner>
python scripts/tune.py --model <winner> --trials 40
python scripts/train.py --model <winner> --use-tuned
python scripts/evaluate.py
```

Heavy stages run on the HPC — SLURM templates and setup in the guide (§9).

## Real-time inference

```bash
python inference/realtime_realsense.py                 # RealSense, webcam fallback
python inference/video.py --input ../../videos --save
```

## Main script integration (how to call)

```python
import cv2 as cv
import mediapipe as mp
from modalities.gesture.src.engine import GestureEngine

holistic = mp.solutions.holistic.Holistic(model_complexity=1)
gesture_detector = GestureEngine()

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    res = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # (stable_label_str, confidence_float) — same call style as MotionEngine
    gesture, confidence = gesture_detector.process_holistic(res)
    print(f"Gesture: {gesture} ({confidence:.2f})")
```

`GestureEngine` buffers ~2 s of frames, resamples to the model window,
smooths the softmax with EMA (α = 0.25) and only emits a non-idle label once
it has won for 300 ms at ≥ 0.60 confidence — so the fusion layer sees stable
intents, not per-frame flicker. Call `engine.reset()` when the tracked person
changes.

## Setup

Use the repo `.venv` (PyTorch + mediapipe): `pip install -r requirements.txt`
from the repo root. Trained weights ship as GitHub Releases (`gesture-v2.x`)
— drop `best_*.pth` + `model_config.json` into `checkpoints/`.
