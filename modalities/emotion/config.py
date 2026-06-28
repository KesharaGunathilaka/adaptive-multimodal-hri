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

# Default architecture used by train / evaluate / inference when unspecified.
DEFAULT_MODEL = "EfficientNet-B0"
