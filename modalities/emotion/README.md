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
│   ├── evaluate.py         # Stage 4  full evaluation (RAF-DB test)
│   ├── extract_realworld_faces.py  # Stage 5  face crops from the HRI dataset
│   ├── finetune_realworld.py       # Stage 6  fine-tune on RAF-DB + real crops
│   └── evaluate_realworld.py       # Stage 7  clip-level real-world evaluation
├── inference/
│   ├── realtime_realsense.py   # live RealSense camera
│   └── video.py                # video file -> annotated mp4
├── data/                   # git-ignored: RAF-DB train/1..7, test/1..7
│                           # + realworld/{train,val}/1..7 (from Stage 5)
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

## Pretrained model (use without retraining)

Trained weights are published as versioned
[**GitHub Releases**](https://github.com/KesharaGunathilaka/adaptive-multimodal-hri/releases)
(kept out of git so the repo stays lean and each model version is its own download).
Fetch the latest into `checkpoints/` with one command — **no GPU or training required**:

```bash
python scripts/download_model.py                 # newest release
python scripts/download_model.py --tag emotion-v1.0   # a specific version
```

This saves `best_MobileNetV2.pth` (the default path the scripts load). Then:

```bash
python scripts/evaluate.py                                   # reproduce the report
python inference/video.py --video ../../videos/classroom/1/20260511_160401.mp4
```

**Deployed model: MobileNetV2 fine-tuned on real-world footage**
(`checkpoints/finetuned_MobileNetV2.pth`, 8.8 MB). The RAF-DB-only models
collapse on far-field deployment footage (subjects 2-5m from the camera), so
the deployed weights are fine-tuned on RAF-DB + face crops from the project's
HRI intent dataset (Stages 5-7 below). Clip-level results on the dataset's
**held-out test subjects** (141 clips, subject-disjoint split):

| Model | Real acc. | Real balanced acc. | Real macro-F1 | RAF-DB acc. / macro-F1 | Size |
|---|---|---|---|---|---|
| **MobileNetV2 fine-tuned (deployed)** | **58.9%** | **50.8%** | **53.2%** | 81.9% / 74.3% | 8.8 MB |
| EfficientNet-B0 fine-tuned (alt) | 57.4% | 47.1% | 49.5% | 84.7% / 78.0% | 16 MB |
| MobileNetV2 RAF-DB only (`best_`) | 44.7% | 28.3% | 24.0% | 84.3% / 76.7% | 8.8 MB |
| EfficientNet-B0 RAF-DB only (`best_`) | 50.4% | 38.7% | 37.3% | 83.9% / 76.4% | 16 MB |

> Inference loads `finetuned_<model>.pth` if present, else falls back to
> `best_<model>.pth`. EfficientNet-B0 is available via `--model EfficientNet-B0`.
> The pipeline below is only needed to **retrain** from scratch.

### Publishing a new model version (maintainers)

1. A `git tag emotion-vX.Y` is created as part of the GitHub Release.
2. On GitHub: **Releases → Draft a new release →** choose/create tag `emotion-vX.Y`,
   add notes (metrics), and **attach `best_MobileNetV2.pth`** as an asset → Publish.
3. `download_model.py` then serves it automatically (it points at the latest release).

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

# Stage 5 — extract face crops from the HRI intent dataset  -> data/realworld/
#           (train/val subjects only; test subjects stay unseen)
python scripts/extract_realworld_faces.py

# Stage 6 — fine-tune on RAF-DB + real crops  -> checkpoints/finetuned_<model>.pth
python scripts/finetune_realworld.py --model MobileNetV2

# Stage 7 — clip-level real-world evaluation  -> reports/evaluation_realworld/
python scripts/evaluate_realworld.py --split test --checkpoint checkpoints/finetuned_MobileNetV2.pth
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
python inference/video.py --video ../../videos/classroom/1/20260511_160401.mp4
```

Both scripts use MediaPipe face detection — **full-range first**
(`model_selection=1`, works to ~5m) **with a close-range fallback**. The old
close-range-only detector found a face in only ~20% of far-field dataset clips;
full-range reaches ~97%. Both accept `--model {MobileNetV2,EfficientNet-B0}`
and `--checkpoint`, and default to the fine-tuned weights when present.

### Real-world fine-tuning (why the deployed weights differ from `best_`)

RAF-DB models see close-up faces; deployment footage has ~40-90px faces that
lose expression detail when upscaled, collapsing predictions to Neutral/Happy
(RAF-DB-only MobileNetV2: 24% macro-F1 on the real-world test subjects, Fear
recall 0%). Two fixes, both in this repo:

1. `src/transforms.py` adds `RandomDownscale` (train-time downscale/upscale) so
   training simulates far-field resolution.
2. Stages 5-6 fine-tune on RAF-DB + real crops from the HRI dataset's
   train/val subjects, with **softened** inverse-frequency class weights
   (`w ~ 1/count^0.5`) — full inverse-frequency weighting over-fires on rare
   classes (the old live "Surprise" bias).

Model selection uses macro-F1 on the held-out val subject; the final comparison
(table above) uses the test subjects, which are never trained on.
