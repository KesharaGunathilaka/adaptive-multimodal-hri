"""
Real-time emotion recognition from an Intel RealSense camera.
Falls back to the default laptop webcam if no RealSense device is connected.

Pipeline: camera frame -> MediaPipe face detection -> padded crop ->
224x224 + ImageNet normalize -> model -> emotion label + confidence overlay.

Run (from the emotion folder):
    python inference/realtime_realsense.py
    python inference/realtime_realsense.py --model EfficientNet-B0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

from config import CHECKPOINT_DIR, DEFAULT_MODEL, EMOTION_LABELS
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms


def _try_start_realsense():
    """Try to connect to a RealSense device. Returns (pipeline, align) or (None, None)."""
    if not _RS_AVAILABLE:
        return None, None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        return pipeline, rs.align(rs.stream.color)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    device = get_device()
    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{safe_name(args.model)}.pth")
    model = ALL_MODELS[args.model]()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = get_test_transforms()
    print(f"Loaded {args.model} from {ckpt}")

    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    pipeline, align = _try_start_realsense()
    if pipeline is not None:
        print("RealSense camera connected.")
        window_title = "Emotion Recognition (RealSense)"
        cap = None
    else:
        print("RealSense not available — falling back to default webcam (index 0).")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: no camera found.")
            return
        window_title = "Emotion Recognition (Webcam)"

    try:
        while True:
            if pipeline is not None:
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam.")
                    break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            if not results.detections:
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            h, w, _ = frame.shape
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)
                # Pad 40% per side so the crop matches RAF-DB framing.
                pad_w, pad_h = int(bw * 0.4), int(bh * 0.4)
                x, y = max(0, x - pad_w), max(0, y - pad_h)
                bw, bh = min(bw + 2 * pad_w, w - x), min(bh + 2 * pad_h, h - y)
                face = frame[y:y + bh, x:x + bw]
                if face.size == 0:
                    continue
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                tensor = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    conf, pred = torch.max(F.softmax(model(tensor), dim=1), 1)
                label = f"{EMOTION_LABELS[pred.item()]}: {conf.item()*100:.1f}%"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
