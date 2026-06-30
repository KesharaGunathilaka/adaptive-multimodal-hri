"""Live webcam scene classification (single sub-model).

Run from the scene_classification/ folder:
    python inference/realtime.py
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import SceneClassifier


def main():
    classifier = SceneClassifier()
    print(f"Scene classifier ready ({classifier.classes}). Press ESC to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        sys.exit(1)

    prev, fps = time.time(), 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = classifier.predict(frame)
        now = time.time(); dt = now - prev; prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt
        color = (0, 255, 0) if r["label"] != "uncertain" else (0, 165, 255)
        cv2.putText(frame, f"{r['label']} ({r['confidence']:.2f})  {fps:.1f} FPS",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Scene Classification", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
