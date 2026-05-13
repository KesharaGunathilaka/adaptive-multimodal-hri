from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ultralytics import YOLO
import config as cfg

# Load pretrained YOLOv8 model
model = YOLO(str(_THIS_DIR / cfg.MODEL_PATH))

# Image path from CLI or default
image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

# Run detection with configured confidence threshold
results = model(image_path, show=True, conf=cfg.CONFIDENCE_THRESHOLD)

# Print detected objects
print(f"\n{'Class':<20} {'Confidence':>10}")
print("-" * 32)
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls_id]
        print(f"{name:<20} {conf:>10.1%}")
