import cv2
import torch
import torch.nn.functional as F
from models.mobilenet_emotion import get_model
from utils.transforms import get_test_transforms
from config import EMOTION_LABELS
import mediapipe as mp
from PIL import Image
import pyrealsense2 as rs
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("checkpoints/model_v1.pth", map_location=device))
model.to(device)
model.eval()

transform = get_test_transforms()

# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize Intel RealSense Camera
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device_product_line = str(
    pipeline_profile.get_device().get_info(rs.camera_info.product_line)
)

# Configure streams: RGB and Depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Create alignment object to align depth frames to color frames
align = rs.align(rs.stream.color)

while True:
    # Get frames from RealSense camera
    frames = pipeline.wait_for_frames()

    # Align depth frame to color frame
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert images to numpy arrays
    frame = np.asarray(color_frame.get_data())
    depth_data = np.asarray(depth_frame.get_data())

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Add padding around the face (40% on each side).
            # MediaPipe's bbox is tight around the face oval, but the
            # RAF-DB training images include forehead, chin, ears, and
            # some background. Without this padding the model receives
            # an out-of-distribution crop and predicts wrong emotions.
            pad_w = int(bw * 0.4)
            pad_h = int(bh * 0.4)
            x = x - pad_w
            y = y - pad_h
            bw = bw + 2 * pad_w
            bh = bh + 2 * pad_h

            # Clamp coordinates to frame boundaries
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            face = frame[y : y + bh, x : x + bw]

            if face.size != 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = Image.fromarray(face)
                face = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(face)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                label = EMOTION_LABELS[pred.item()]
                confidence = conf.item() * 100

                text = f"{label}: {confidence:.2f}%"
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition (RealSense)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()
