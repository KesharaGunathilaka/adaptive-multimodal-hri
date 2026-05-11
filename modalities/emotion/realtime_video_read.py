import cv2
import torch
import torch.nn.functional as F
from models.mobilenet_emotion import get_model
from utils.transforms import get_test_transforms
from config import EMOTION_LABELS
import mediapipe as mp
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = get_model()

model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))

model.to(device)
model.eval()

transform = get_test_transforms()

# -----------------------------
# MEDIAPIPE FACE DETECTION
# -----------------------------
mp_face = mp.solutions.face_detection

face_detection = mp_face.FaceDetection(model_selection=0)

# -----------------------------
# LOAD VIDEO
# -----------------------------
VIDEO_PATH = "videos/C1_D1_I1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:
    ret, frame = cap.read()

    # End of video
    if not ret:
        print("Video finished")
        break

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

            # Prevent negative coordinates
            x = max(0, x)
            y = max(0, y)

            face = frame[y : y + bh, x : x + bw]

            if face.size != 0:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                face_pil = Image.fromarray(face_rgb)

                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(face_tensor)

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

    cv2.imshow("Emotion Recognition", frame)

    # Adjust playback speed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
