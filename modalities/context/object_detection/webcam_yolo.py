from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import cv2
from ultralytics import YOLO
import config as cfg

_model_path = _THIS_DIR / cfg.MODEL_PATH
if _model_path.exists():
    model = YOLO(str(_model_path))
else:
    model = YOLO(cfg.MODEL_PATH)

# Set custom classes for YOLO-World
if hasattr(cfg, "CUSTOM_CLASSES"):
    model.set_classes(cfg.CUSTOM_CLASSES)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    sys.exit(1)

print("YOLO webcam detection (all classes). Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=cfg.CONFIDENCE_THRESHOLD, verbose=False)
    annotated = results[0].plot(line_width=cfg.BOX_THICKNESS)

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
