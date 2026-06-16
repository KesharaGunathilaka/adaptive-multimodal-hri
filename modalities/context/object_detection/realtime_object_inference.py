import time
import cv2
import sys
from detector import ContextDetector


def main():
    print("Initializing YOLO11-Nano HRI Pipeline...")
    detector = ContextDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        sys.exit(1)

    print("Pipeline ready. Press 'q' or 'ESC' to quit.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        annotated_frame, detections, counts, stable_categories = detector.process_frame(
            frame
        )

        # Smooth FPS estimate
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Top summary bar: raw per-frame counts
        active_counts = [f"{k}: {v}" for k, v in counts.items()]
        if active_counts:
            summary_text = " | ".join(active_counts)
            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (annotated_frame.shape[1], 30),
                (30, 30, 30),
                -1,
            )
            cv2.putText(
                annotated_frame,
                summary_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Bottom-left: temporally stable categories (what downstream should trust)
        stable_text = "stable: " + (
            ", ".join(sorted(stable_categories)) if stable_categories else "-"
        )
        cv2.putText(
            annotated_frame,
            stable_text,
            (10, annotated_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Top-right: FPS
        cv2.putText(
            annotated_frame,
            f"{fps:.1f} FPS",
            (annotated_frame.shape[1] - 110, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("HRI Object Detection (Base COCO)", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
