"""
Configuration for the Object Detection module.

Uses YOLO-World — an open-vocabulary detector that can recognise ANY
object you name, not just the 80 COCO classes.  Add your custom objects
to CUSTOM_CLASSES below and they will be detected automatically.
"""

# ─────────────────────────────────────────────
# YOLO-World Model  (open-vocabulary)
# ─────────────────────────────────────────────
# YOLO-World can detect objects by NAME — no retraining needed.
# Available sizes (auto-downloaded on first run):
#   yolov8s-worldv2.pt  — small  (good balance of speed & accuracy)
#   yolov8m-worldv2.pt  — medium (better accuracy, slower)
#   yolov8l-worldv2.pt  — large  (best accuracy, slowest)
MODEL_PATH = "yolov8s-worldv2.pt"

# ─────────────────────────────────────────────
# Custom Classes to Detect
# ─────────────────────────────────────────────
# List EVERY object you want detected — YOLO-World looks for exactly
# these.  Use natural English names.  Be specific (e.g. "electric
# kettle" instead of just "kettle") for better accuracy.
CUSTOM_CLASSES = [
    # People
    "person",

    # Electronics / computing
    "laptop",
    "computer monitor",
    "keyboard",
    "computer mouse",
    "mobile phone",
    "tablet",
    "remote control",
    "television",

    # Kitchen appliances
    "rice cooker",
    "electric kettle",
    "microwave",
    "toaster",
    "blender",
    "coffee maker",

    # Study / reading
    "book",
    "notebook",
    "pen",
    "pencil",

    # Containers / drinkware
    "cup",
    "mug",
    "bottle",
    "bowl",
    "glass",
    "plate",

    # Furniture
    "chair",
    "desk",
    "table",
    "couch",
    "shelf",
    "bed",

    # Bags
    "backpack",
    "handbag",

    # Other context objects
    "clock",
    "whiteboard",
    "fan",
    "lamp",
]

# ─────────────────────────────────────────────
# Class → Category Mapping
# ─────────────────────────────────────────────
# Groups detected class names into high-level HRI context categories.
# Any class not listed here is kept as-is (label = category).
OBJECT_CATEGORIES = {
    # Electronics → "computer"
    "laptop": "computer",
    "computer monitor": "computer",
    "keyboard": "computer",
    "computer mouse": "computer",
    "television": "computer",
    "tablet": "computer",

    # Phone
    "mobile phone": "phone",
    "remote control": "phone",

    # Kitchen appliances → "appliance"
    "rice cooker": "appliance",
    "electric kettle": "appliance",
    "microwave": "appliance",
    "toaster": "appliance",
    "blender": "appliance",
    "coffee maker": "appliance",

    # Study
    "book": "book",
    "notebook": "book",
    "pen": "stationery",
    "pencil": "stationery",

    # People
    "person": "person",

    # Furniture
    "chair": "furniture",
    "desk": "furniture",
    "table": "furniture",
    "couch": "furniture",
    "shelf": "furniture",
    "bed": "furniture",

    # Containers
    "cup": "container",
    "mug": "container",
    "bottle": "container",
    "bowl": "container",
    "glass": "container",
    "plate": "container",

    # Bags
    "backpack": "bag",
    "handbag": "bag",

    # Other
    "clock": "other",
    "whiteboard": "other",
    "fan": "other",
    "lamp": "other",
}

# ─────────────────────────────────────────────
# Inference Parameters
# ─────────────────────────────────────────────
# Input image size. Larger = better for small objects but slower.
INFERENCE_IMG_SIZE = 640

# Test-time augmentation — merges results from multiple scales/flips.
INFERENCE_AUGMENT = False

# NMS IoU threshold — controls duplicate suppression.
INFERENCE_IOU_THRESHOLD = 0.45

# ─────────────────────────────────────────────
# Confidence Thresholds
# ─────────────────────────────────────────────
# Detections below this score are discarded.
CONFIDENCE_THRESHOLD = 0.15

# Per-category overrides (optional).
CATEGORY_THRESHOLDS = {
    "person": 0.15,
    "computer": 0.15,
    "phone": 0.15,
    "appliance": 0.12,
    "book": 0.15,
    "stationery": 0.12,
    "furniture": 0.12,
    "container": 0.15,
    "bag": 0.15,
    "other": 0.15,
}

# ─────────────────────────────────────────────
# Zone Definitions (normalised coordinates)
# ─────────────────────────────────────────────
ZONES_X = {
    "left": (0.00, 0.33),
    "center": (0.33, 0.66),
    "right": (0.66, 1.00),
}

ZONES_Y = {
    "top": (0.00, 0.33),
    "middle": (0.33, 0.66),
    "bottom": (0.66, 1.00),
}

# ─────────────────────────────────────────────
# Visualisation Colours  (BGR for OpenCV)
# ─────────────────────────────────────────────
CATEGORY_COLORS = {
    "person": (255, 140, 50),      # warm orange
    "computer": (255, 200, 60),    # golden yellow
    "phone": (80, 220, 255),       # sky blue
    "appliance": (100, 180, 255),  # coral blue
    "book": (100, 230, 130),       # soft green
    "stationery": (150, 255, 150), # light green
    "furniture": (200, 160, 255),  # lavender
    "container": (130, 210, 230),  # pale teal
    "bag": (180, 130, 255),        # purple
    "other": (200, 200, 200),      # grey
}
DEFAULT_COLOR = (180, 180, 180)

# Box drawing
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.50
LABEL_THICKNESS = 1
LABEL_BG_ALPHA = 0.70

# ─────────────────────────────────────────────
# Temporal Smoothing
# ─────────────────────────────────────────────
PERSISTENCE_FRAMES = 12
MIN_HITS_TO_SHOW = 3
IOU_MATCH_THRESHOLD = 0.25

# ─────────────────────────────────────────────
# Output Settings
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
JSON_OUTPUT_FILE = "detection_results.json"
TIMELINE_CHART_FILE = "detection_timeline.png"
FRAME_SAMPLE_INTERVAL = 1

# ─────────────────────────────────────────────
# Video Detection
# ─────────────────────────────────────────────
DEFAULT_VIDEO_PATH = "../../../videos/c1/20260511_160444.mp4"
SHOW_PREVIEW = True
