import cv2
import torch
import torch.nn.functional as F
from models.mobilenet_emotion import get_model
from utils.transforms import get_test_transforms
from config import EMOTION_LABELS
import mediapipe as mp
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load(
    "checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = get_test_transforms()

# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
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

            face = frame[y:y+bh, x:x+bw]

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
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
