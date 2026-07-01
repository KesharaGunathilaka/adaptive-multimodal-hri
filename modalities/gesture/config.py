"""
Central configuration for the gesture-recognition modality.

All paths are absolute (derived from this file's location) so every script
works regardless of the current working directory.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
REPORT_DIR = os.path.join(ROOT, "reports")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset & Model ──────────────────────────────────────────────────────
NUM_CLASSES = 6

# Mapped gesture labels in index order
GESTURE_LABELS = [
    "Open Palm",      # Class 0
    "Close (Fist)",   # Class 1
    "Pointer",        # Class 2
    "Thumbs Up",      # Class 3
    "Thumbs Down",    # Class 4
    "Beckoning",      # Class 5
]

# Deployed PyTorch model paths
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "keypoint_classifier.pth")
DEFAULT_LABEL_CSV = os.path.join(CHECKPOINT_DIR, "keypoint_classifier_label.csv")

# Scenario colors for UI overlay (RGB)
GESTURE_COLORS = {
    "One Hand Raised": (0, 220, 100),
    "Brief Wave": (255, 200, 0),
    "Pointing": (0, 180, 255),
    "None": (160, 160, 160),
    "Arms Waving": (255, 80, 200),
    "Wave": (255, 140, 0),
    "Beckoning": (100, 255, 220),
    "Arms Up": (0, 100, 255),
    "No hands": (80, 80, 80),
    "Thumbs up": (0, 255, 0),
    "Thumbs down": (0, 0, 255),
}
