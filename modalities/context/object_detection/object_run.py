from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import cv2
import numpy as np
from custom_detect import detect_context_objects, DetectionSmoother
import config as cfg


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def draw_detections(frame, result):
    """
    Draw bounding boxes with category-coloured borders and
    semi-transparent label backgrounds on *frame* (in-place).
    """
    overlay = frame.copy()

    for det in result["detections"]:
        cat = det["category"]
        label_text = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        zone = det["zone"]

        color = cfg.CATEGORY_COLORS.get(cat, cfg.DEFAULT_COLOR)

        # --- bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)

        # --- label string ---
        text = f"{label_text} {conf:.0%} [{zone}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(
            text, font, cfg.LABEL_FONT_SCALE, cfg.LABEL_THICKNESS
        )

        # label background (semi-transparent)
        label_y1 = max(y1 - th - baseline - 6, 0)
        label_y2 = y1
        cv2.rectangle(overlay, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)

        # blend overlay
        cv2.addWeighted(
            overlay, cfg.LABEL_BG_ALPHA, frame, 1 - cfg.LABEL_BG_ALPHA, 0, frame
        )
        overlay = frame.copy()  # reset overlay for next box

        # text on top
        cv2.putText(
            frame,
            text,
            (x1 + 3, y1 - baseline - 2),
            font,
            cfg.LABEL_FONT_SCALE,
            (255, 255, 255),
            cfg.LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    # --- summary bar at top ---
    counts = result["counts"]
    active = {k: v for k, v in counts.items() if v > 0}
    if active:
        summary = "  |  ".join(f"{k}: {v}" for k, v in sorted(active.items()))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(
            frame,
            summary,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return frame


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Webcam object detection started. Press ESC to quit.")

    smoother = DetectionSmoother()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_result = detect_context_objects(frame)
        result = smoother.update(raw_result)
        frame = draw_detections(frame, result)

        # Print counts to terminal
        active = {k: v for k, v in result["counts"].items() if v > 0}
        if active:
            print(active)

        cv2.imshow("Object Detection - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
