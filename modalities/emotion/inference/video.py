"""
Emotion recognition on a video file, written to outputs/.

Robustness features for in-the-wild video (vs. controlled RAF-DB):
  - downscale large frames for faster MediaPipe detection
  - eye-keypoint face alignment (correct head tilt)
  - 40% crop padding to match RAF-DB framing
  - CLAHE contrast normalization
  - test-time augmentation (average logits of original + horizontal flip)

Run (from the emotion folder):
    python inference/video.py --video ../../videos/test/C1_D2_T2.mp4
    python inference/video.py --video path/to.mp4 --model EfficientNet-B0 --no-show
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

from config import CHECKPOINT_DIR, DEFAULT_MODEL, EMOTION_LABELS, ROOT
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms

MAX_FRAME_WIDTH = 640
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def align_face(image, left_eye, right_eye):
    """Rotate so the eyes are horizontal (skips extreme/negligible tilt)."""
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return image
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video file.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None, help="Defaults to outputs/<video>_emotion.mp4")
    ap.add_argument("--no-show", action="store_true", help="Do not open a preview window.")
    args = ap.parse_args()

    device = get_device()
    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{safe_name(args.model)}.pth")
    model = ALL_MODELS[args.model]()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = get_test_transforms()
    print(f"Loaded {args.model} from {ckpt}")

    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(
        out_dir, os.path.splitext(os.path.basename(args.video))[0] + "_emotion.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"Writing to {out_path}  ({width}x{height} @ {fps:.0f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        if w > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / w
            small = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)))
        else:
            small = frame
        results = face_detection.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)
                kp = det.location_data.relative_keypoints
                right_eye = (int(kp[0].x * w), int(kp[0].y * h))
                left_eye = (int(kp[1].x * w), int(kp[1].y * h))
                aligned = align_face(frame, left_eye, right_eye)

                pad_w, pad_h = int(bw * 0.4), int(bh * 0.4)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)
                face = aligned[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                face = cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                t = transform(pil).unsqueeze(0).to(device)
                t_flip = transform(pil.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = (model(t) + model(t_flip)) / 2.0
                    conf, pred = torch.max(F.softmax(logits, dim=1), 1)
                label = f"{EMOTION_LABELS[pred.item()]}: {conf.item()*100:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(frame)
        if not args.no_show:
            cv2.imshow("Emotion Recognition", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
