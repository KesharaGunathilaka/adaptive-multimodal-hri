import cv2
import torch
import torch.nn.functional as F
from models.mobilenet_emotion import get_model
from utils.transforms import get_test_transforms
from config import EMOTION_LABELS
import mediapipe as mp
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = get_model()

model.load_state_dict(torch.load("checkpoints/model_v1.pth", map_location=device))

model.to(device)
model.eval()

transform = get_test_transforms()

# -----------------------------
# MEDIAPIPE FACE DETECTION
# -----------------------------
mp_face = mp.solutions.face_detection

# model_selection=1 → full-range model (detects faces up to ~5 meters)
# model_selection=0 only works for close-up faces within ~2 meters (e.g. webcam)
# min_detection_confidence=0.3 → more sensitive detection for varied video conditions
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)

# Maximum frame width for face detection processing.
# Frames wider than this are downscaled before detection to improve
# MediaPipe performance on high-resolution videos.
MAX_FRAME_WIDTH = 640

# CLAHE for adaptive contrast normalization.
# Video files often have uneven lighting (shadows, backlighting) that
# differ from the controlled conditions in the RAF-DB training set.
# CLAHE equalizes local contrast so the model sees consistent input.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def align_face(image, left_eye, right_eye):
    """
    Align a face crop so the eyes are horizontal.

    RAF-DB training images have roughly upright, aligned faces.
    Video frames often have tilted heads, causing a distribution
    mismatch. This function rotates the image to correct head tilt
    using the eye positions detected by MediaPipe.

    Args:
        image: BGR face crop (numpy array)
        left_eye: (x, y) of left eye in pixel coordinates
        right_eye: (x, y) of right eye in pixel coordinates

    Returns:
        Aligned face crop (same size as input)
    """
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Only align if tilt is noticeable but not extreme
    # (extreme angles likely mean a profile/odd detection)
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return image

    # Rotate around the midpoint between the eyes
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


# -----------------------------
# LOAD VIDEO
# -----------------------------

VIDEO_PATH = "../../videos/c1/0042.mp4"
# VIDEO_PATH = "../../videos/k1/20260509_190437.mp4"
# VIDEO_PATH = "../../videos/k2/20260509_202721.mp4"
# VIDEO_PATH = "../../videos/k3/20260510_003503.mp4"
# VIDEO_PATH = "../../videos/k4/20260509_225918.mp4"
# VIDEO_PATH = "../../videos/k5/20260510_001118.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Set up video writer for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output directory if it doesn't exist
import os

output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "emotion_inference_output.mp4")

# Initialize VideoWriter with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Saving output to: {output_path}")
print(f"Video properties: {width}x{height} @ {fps} fps")

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:
    ret, frame = cap.read()

    # End of video
    if not ret:
        print("Video finished")
        break

    h, w, _ = frame.shape

    # Scale down large frames for better face detection performance
    if w > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / w
        small_frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)))
    else:
        scale = 1.0
        small_frame = frame

    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Map relative coordinates back to original frame dimensions
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)

            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # --- Face alignment using eye keypoints ---
            # MediaPipe FaceDetection provides 6 keypoints:
            #   0: right_eye, 1: left_eye, 2: nose_tip,
            #   3: mouth_center, 4: right_ear, 5: left_ear
            # We use the eyes to correct head tilt before cropping.
            keypoints = detection.location_data.relative_keypoints
            right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
            left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))

            # Align the full frame so the face is upright, then crop
            aligned_frame = align_face(frame, left_eye, right_eye)

            # Add padding around the face (40% on each side).
            # MediaPipe's bbox is tight around the face oval, but the
            # RAF-DB training images include forehead, chin, ears, and
            # some background. Without this padding the model receives
            # an out-of-distribution crop and predicts wrong emotions.
            pad_w = int(bw * 0.4)
            pad_h = int(bh * 0.4)
            x1 = x - pad_w
            y1 = y - pad_h
            x2 = x + bw + pad_w
            y2 = y + bh + pad_h

            # Clamp all coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = aligned_frame[y1:y2, x1:x2]

            if face.size != 0:
                # Apply CLAHE to normalize lighting/contrast.
                # Convert to LAB color space, equalize the L channel,
                # then convert back. This handles shadows, backlighting,
                # and color temperature differences between video and
                # the RAF-DB training set.
                face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(face_lab)
                l_ch = clahe.apply(l_ch)
                face_lab = cv2.merge((l_ch, a_ch, b_ch))
                face = cv2.cvtColor(face_lab, cv2.COLOR_LAB2BGR)

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)

                # --- Test-Time Augmentation (TTA) ---
                # Average predictions over original + horizontally flipped
                # face. This makes the model more robust to asymmetric
                # lighting, slight pose variations, and left/right bias
                # in the training data.
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                face_flipped = face_pil.transpose(Image.FLIP_LEFT_RIGHT)
                face_flipped_tensor = transform(face_flipped).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_orig = model(face_tensor)
                    logits_flip = model(face_flipped_tensor)

                    # Average logits before softmax for better calibration
                    logits_avg = (logits_orig + logits_flip) / 2.0

                    probs = F.softmax(logits_avg, dim=1)
                    conf, pred = torch.max(probs, 1)

                label = EMOTION_LABELS[pred.item()]
                confidence = conf.item() * 100

                text = f"{label}: {confidence:.2f}%"

                # Draw on the original (unaligned) frame for display
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Emotion Recognition", frame)

    # Adjust playback speed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✓ Video saved successfully to: {output_path}")
