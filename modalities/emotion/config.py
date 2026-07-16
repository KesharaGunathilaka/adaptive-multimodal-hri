"""
Central configuration for the emotion-recognition modality.

All paths are absolute (derived from this file's location) so every script
works regardless of the current working directory.

Dataset:  RAF-DB (7 basic emotions), ImageFolder layout under data/.
Target:   lightweight model for Jetson Orin Nano, optimized for balanced
          (macro-F1) performance on an imbalanced dataset.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
REPORT_DIR = os.path.join(ROOT, "reports")

# ── Dataset / model ──────────────────────────────────────────────────────
NUM_CLASSES = 7
IMAGE_SIZE = 224

# RAF-DB folders 1..7 map (after ImageFolder's alphabetical sort) to:
EMOTION_LABELS = [
    "Surprise",  # folder 1
    "Fear",      # folder 2
    "Disgust",   # folder 3
    "Happy",     # folder 4
    "Sad",       # folder 5
    "Anger",     # folder 6
    "Neutral",   # folder 7
]

# RAF-DB train-set counts per class (same order as EMOTION_LABELS). Used at
# inference to re-inject the natural class prior that inverse-frequency
# class-weighted training removes (see src/postprocess.py).
TRAIN_CLASS_COUNTS = [1290, 281, 717, 4772, 1982, 705, 2524]

# ── Default hyper-parameters ─────────────────────────────────────────────
# Tuned for a high-VRAM GPU; lower BATCH_SIZE / NUM_WORKERS for small cards.
BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-4                 # stage-2 (full fine-tune) base LR
HEAD_LR = 1e-3            # stage-1 (head-only) LR
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
WARMUP_EPOCHS = 2
STAGE1_EPOCHS = 5        # head-only warm-up
STAGE2_EPOCHS = 25       # full fine-tune
SEED = 42

# ImageNet normalization (pretrained backbones expect this)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Deployment budget: models above this size are flagged in the comparison.
SIZE_BUDGET_MB = 20.0

# ── Deployed model (what inference ships with) ───────────────────────────
# These two lines are the single place that decides which model runs by
# default. To ship a different model: drop its .pth in checkpoints/ and update
# both lines. Inference can still be overridden per-run with --model/--checkpoint.
#
# MobileNetV2 fine-tuned on RAF-DB + real-world far-field crops
# (scripts/finetune_realworld.py) is the deployed model: best clip-level
# macro-F1 on the held-out real-world test subjects (53.2% vs 24.0% before
# fine-tuning; RAF-DB stays at 81.9% acc), and it's the smallest (8.8 MB).
DEFAULT_MODEL = "MobileNetV2"
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "finetuned_MobileNetV2.pth")
