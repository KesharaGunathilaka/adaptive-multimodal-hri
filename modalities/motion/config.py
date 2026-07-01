"""
Central configuration for the motion-recognition modality.
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

# ── Model & Labels ───────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "motion_lstm_v2_best.pth")
DEFAULT_CONFIG = os.path.join(CHECKPOINT_DIR, "model_config_v2.json")

# HRI Motion Labels
MOTION_LABELS = [
    "Sitting Still",       # Index 0
    "Standing Still",      # Index 1
    "Walking",             # Index 2
    "Walk Across",         # Index 3
    "Run Backward",        # Index 4
    "Run (Fast Movement)", # Index 5
    "Leaning Forward",     # Index 6
    "Frozen/Rigid Stand",  # Index 7
]

# Display colors (BGR) for the UI overlay dashboard
MOTION_COLORS = {
    "Sitting Still":       (160, 160, 160),  # Grey
    "Standing Still":      (200, 200, 200),  # Light Grey
    "Walking":             (255, 210, 0),    # Yellow-cyan
    "Walk Across":         (100, 255, 0),    # Green
    "Run Backward":        (0, 80, 255),     # Orange-red
    "Run (Fast Movement)": (200, 255, 0),    # Turquoise
    "Leaning Forward":     (0, 255, 220),    # Lime-green
    "Frozen/Rigid Stand":  (150, 50, 50),    # Dark Red
}
