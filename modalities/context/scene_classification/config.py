"""
Central configuration for the scene-classification sub-model of the context
modality.

All paths are absolute (derived from this file's location) so every script
works regardless of the current working directory.

Dataset:  Places365 subset, ImageFolder layout under ../data/scene/.
Classes:  the two environments we deploy in — classroom, kitchen.
Target:   lightweight CNN for Jetson Orin Nano scene understanding.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(ROOT, "..", "data", "scene"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
REPORT_DIR = os.path.join(ROOT, "reports")
CLASSES_FILE = os.path.join(CHECKPOINT_DIR, "classes.json")

# ── Dataset / model ──────────────────────────────────────────────────────
NUM_CLASSES = 2
IMAGE_SIZE = 224

# Class order MUST match torchvision ImageFolder (alphabetical folder sort).
# Office was dropped (captured "classroom" clips were confidently misread as
# office); we model only the two environments we actually deploy in.
SCENE_LABELS = ["classroom", "kitchen"]

# ── Default hyper-parameters ─────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4                 # stage-2 (full fine-tune) base LR
HEAD_LR = 1e-3            # stage-1 (head-only) LR
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
WARMUP_EPOCHS = 1
STAGE1_EPOCHS = 5         # head-only warm-up
STAGE2_EPOCHS = 15        # full fine-tune
SEED = 42

# ImageNet normalization (pretrained backbones expect this)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Deployment budget: models above this size are flagged in the comparison.
SIZE_BUDGET_MB = 20.0

# ── Deployed model (what inference ships with) ───────────────────────────
# EfficientNet-B0 won the comparison (best accuracy within the size budget).
# To ship a different model: drop its .pth in checkpoints/ and update both lines.
DEFAULT_MODEL = "EfficientNet-B0"
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_EfficientNet_B0.pth")
