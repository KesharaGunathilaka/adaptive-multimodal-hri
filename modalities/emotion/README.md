# Emotion Recognition (RAF-DB)

Facial emotion recognition for the adaptive HRI system. A lightweight
ImageNet-pretrained CNN classifies a face into 7 emotions
(**Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral**), trained on **RAF-DB**
and targeted at **Jetson Orin Nano** deployment.

The dataset is imbalanced (Happy: 4772 train images vs. Fear: 281), so the whole
pipeline optimizes for **balanced macro-F1**, not raw accuracy: inverse-frequency
weighted loss, label smoothing, mixup, rich augmentation, and model selection by
macro-F1 / balanced accuracy.

## Folder structure

```
emotion/
├── config.py               # paths, classes, default hyper-parameters
├── README.md
├── src/                    # importable library
│   ├── models.py           # model zoo (4 backbones) + helpers
│   ├── transforms.py       # train/eval image transforms (incl. CLAHE)
│   ├── data.py             # datasets, loaders, class weights
│   └── engine.py           # two-stage training recipe, metrics, mixup, AMP
├── scripts/                # runnable pipeline stages
│   ├── eda.py              # Stage 0  EDA
│   ├── compare_models.py   # Stage 1  compare backbones -> pick winner
│   ├── train.py            # Stage 2  full training of chosen model
│   ├── tune.py             # Stage 3  hyper-parameter tuning
│   └── evaluate.py         # Stage 4  full evaluation
├── inference/
│   ├── realtime_realsense.py   # live RealSense camera
│   └── video.py                # video file -> annotated mp4
├── data/                   # RAF-DB (git-ignored): train/1..7, test/1..7
├── checkpoints/            # saved weights (git-ignored)
├── reports/                # generated reports, CSVs, plots
└── outputs/                # inference video outputs (git-ignored)
```

## Setup

Use the repo's `.venv` (CUDA-enabled PyTorch). All commands run **from this
`emotion/` folder**.

```bash
# from repo root
.venv\Scripts\activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
cd modalities/emotion
```

Dataset layout (RAF-DB `_aligned` crops, folders named by label 1–7):
`data/train/<1-7>/*.jpg` and `data/test/<1-7>/*.jpg`.

## Pipeline (run in order)

```bash
# Stage 0 — EDA: class distribution, samples, imbalance  -> reports/eda/
python scripts/eda.py

# Stage 1 — compare all 4 backbones, pick best by macro-F1  -> reports/comparison/
python scripts/compare_models.py --stage1_epochs 3 --stage2_epochs 12

# Stage 2 — full training of the chosen model  -> checkpoints/ + reports/training/
python scripts/train.py --model EfficientNet-B0

# Stage 3 — hyper-parameter tuning  -> reports/tuning/
python scripts/tune.py --model EfficientNet-B0

# Stage 2 (final) — retrain with the best config printed by Stage 3
python scripts/train.py --model EfficientNet-B0 --base_lr 5e-5 --batch_size 64

# Stage 4 — full evaluation of the final checkpoint  -> reports/evaluation/
python scripts/evaluate.py --model EfficientNet-B0
```

Every script supports `--help`. Useful flags:
- `--max_per_class N` (compare) — subsample for a fast smoke test.
- `--no_amp` — disable mixed precision.
- `--batch_size`, `--num_workers` — tune to your GPU.

## Models

| Name | Registry key | Notes |
|---|---|---|
| MobileNetV2 | `MobileNetV2` | smallest, fast |
| MobileNetV3-Large | `MobileNetV3-Large` | strong accuracy/size balance |
| EfficientNet-B0 | `EfficientNet-B0` | default; best accuracy in budget |
| MNASNet 1.0 | `MNASNet1_0` | mobile-optimized |

## Inference

```bash
# Live Intel RealSense camera
python inference/realtime_realsense.py --model EfficientNet-B0

# Annotate a video file (writes to outputs/)
python inference/video.py --video ../../videos/test/C1_D2_T2.mp4
```

Both use MediaPipe for face detection, pad the crop to match RAF-DB framing, and
apply the same normalization as training. `video.py` adds eye-based alignment,
CLAHE, and test-time augmentation for robustness on in-the-wild footage.
