"""Live webcam gaze estimation (single sub-model).

Run from repo root or this folder:
    python inference/realtime.py
"""
from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.gaze_estimation.gaze_estimator import GazeEstimator


def main():
    estimator = GazeEstimator()
    print("Gaze estimator ready. Press ESC to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        sys.exit(1)

    prev, fps = time.time(), 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gaze = estimator.estimate(frame)

        now = time.time(); dt = now - prev; prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt

        if gaze.has_face:
            if gaze.face_bbox:
                x1, y1, x2, y2 = gaze.face_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
            if gaze.gaze_point and gaze.face_bbox:
                cx = (gaze.face_bbox[0] + gaze.face_bbox[2]) // 2
                cy = (gaze.face_bbox[1] + gaze.face_bbox[3]) // 2
                gp = (int(gaze.gaze_point[0]), int(gaze.gaze_point[1]))
                cv2.arrowedLine(frame, (cx, cy), gp, (0, 255, 0), 2, tipLength=0.2)
            status = "ROBOT" if gaze.looking_at_robot else "AWAY"
            color = (0, 255, 0) if gaze.looking_at_robot else (0, 165, 255)
            cv2.putText(frame, f"yaw {gaze.yaw:+.0f}  pitch {gaze.pitch:+.0f}  -> {status}"
                        f"  eng {gaze.engagement:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Gaze Estimation", frame)
        if cv2.waitKey(1) == 27:
            break

    estimator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
