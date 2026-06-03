from pathlib import Path
import sys

import cv2
import torch
from torchvision import transforms

# Ensure repo-root imports work no matter where this script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_model import SceneModel

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["classroom", "office", "kitchen"]

# Load model
weights_path = Path(__file__).resolve().parent / "scene_model" / "scene.pth"

model = SceneModel(num_classes=3).to(DEVICE)
if not weights_path.exists():
    raise FileNotFoundError(f"Model weights not found: {weights_path}")

model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
model.eval()

# Transform - must match training
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for proper color channel ordering
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = transform(frame_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, 1).item()

    label = classes[pred]
    confidence = probs[0, pred].item() * 100

    # Display
    cv2.putText(
        frame,
        f"Scene: {label} ({confidence:.1f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Scene Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
