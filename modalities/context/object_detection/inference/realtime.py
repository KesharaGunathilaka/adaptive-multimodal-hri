"""Live webcam object detection (single sub-model).

Run from the object_detection/ folder:
    python inference/realtime.py
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import ContextDetector


def main():
    print("Initializing YOLO11-Nano HRI detector...")
    detector = ContextDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        sys.exit(1)
    print("Ready. Press 'q' or ESC to quit.")

    prev, fps = time.time(), 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, detections, counts, stable = detector.process_frame(frame)

        now = time.time(); dt = now - prev; prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt

        if counts:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (30, 30, 30), -1)
            cv2.putText(annotated, " | ".join(f"{k}: {v}" for k, v in counts.items()),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, "stable: " + (", ".join(sorted(stable)) if stable else "-"),
                    (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"{fps:.1f} FPS", (annotated.shape[1] - 110, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("HRI Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
