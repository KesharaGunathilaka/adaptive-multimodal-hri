"""
Configuration for Base YOLO11-Nano HRI Object Detection.
"""

# ─────────────────────────────────────────────
# Model Parameters
# ─────────────────────────────────────────────
# This will automatically download the standard COCO weights on first run.
# Once exported on the Jetson, change this to "yolo11n.engine"
MODEL_PATH = "yolo11n.pt"

# ─────────────────────────────────────────────
# Target Classes & Categorization
# ─────────────────────────────────────────────
# The specific COCO names we want to detect (ignoring cars, animals, etc.)
TARGET_CLASSES = [
    "person",
    "laptop",
    "mouse",
    "keyboard",
    "cell phone",
    "tv",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "chair",
    "couch",
    "dining table",
    "bed",
    "cup",
    "bottle",
    "bowl",
    "wine glass",
    "fork",
    "knife",
    "spoon",
]

# Map specific items to higher-level context groups for your HRI policy
OBJECT_CATEGORIES = {
    "laptop": "computer",
    "mouse": "computer",
    "keyboard": "computer",
    "tv": "computer",
    "cell phone": "phone",
    "microwave": "appliance",
    "oven": "appliance",
    "toaster": "appliance",
    "refrigerator": "appliance",
    "book": "book",
    "person": "person",
    "chair": "furniture",
    "couch": "furniture",
    "dining table": "furniture",
    "bed": "furniture",
    "cup": "container",
    "bottle": "container",
    "bowl": "container",
    "wine glass": "container",
}

# ─────────────────────────────────────────────
# Inference Settings
# ─────────────────────────────────────────────
INFERENCE_IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.40  # 40% confidence requirement

# ─────────────────────────────────────────────
# Tracking & Temporal Smoothing
# ─────────────────────────────────────────────
# Tracking assigns a persistent ID to each object across frames. This gives
# stable object identities (needed later to answer "which object is the user
# attending to") and steadier boxes than frame-by-frame detection.
USE_TRACKING = True
TRACKER_CONFIG = "bytetrack.yaml"

# A raw detection can flicker on/off between frames. We keep a short rolling
# window and only report a category as "present" if it shows up in at least
# PRESENCE_MIN_RATIO of the last SMOOTH_WINDOW frames. This is what downstream
# fusion/policy should rely on, rather than the noisy per-frame counts.
SMOOTH_WINDOW = 10
PRESENCE_MIN_RATIO = 0.3
