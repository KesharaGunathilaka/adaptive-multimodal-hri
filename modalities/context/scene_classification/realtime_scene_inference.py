from pathlib import Path
import sys
import time

import cv2

# Ensure repo-root imports work no matter where this script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_classifier import SceneClassifier

classifier = SceneClassifier()
print(f"Scene classifier ready on {classifier.device}. Classes: {classifier.classes}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    sys.exit(1)

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = classifier.predict(frame)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    label = result["label"]
    confidence = result["confidence"] * 100
    color = (0, 255, 0) if label != "uncertain" else (0, 165, 255)

    cv2.putText(
        frame,
        f"Scene: {label} ({confidence:.1f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (frame.shape[1] - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Scene Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
