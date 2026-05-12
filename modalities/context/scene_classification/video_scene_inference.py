from pathlib import Path
import sys
import cv2
import torch
from torchvision import transforms

# Ensure repo-root imports work
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_model import SceneModel

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Select video - Options:
# 1. Use webcam (uncomment below)
# VIDEO_PATH = 0  # 0 = default webcam
# 2. Use video file (update path to existing video)
VIDEO_PATH = "../../../videos/c1/20260511_160401.mp4"

# IMPORTANT:
# Must match dataset.class_to_idx from training
classes = [
    "classroom",
    "kitchen",
    "office",
]

# =========================
# LOAD MODEL
# =========================
weights_path = Path(__file__).resolve().parents[1] / "weights" / "scene.pth"

model = SceneModel(num_classes=3).to(DEVICE)

if not weights_path.exists():
    raise FileNotFoundError(f"Model weights not found: {weights_path}")

model.load_state_dict(
    torch.load(
        str(weights_path),
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

print(f"Model loaded successfully on {DEVICE}")

# =========================
# TRANSFORMS
# =========================
# Must match training transforms exactly
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# =========================
# VIDEO CAPTURE
# =========================
print(f"\nAttempting to open: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"✗ Error: Could not open video file at: {VIDEO_PATH}")
    print("  Please check if the file path is correct and the file exists.")
    sys.exit(1)  # Terminate script immediately
else:
    print("✓ Video file opened successfully!")

print("Press 'q' or ESC to quit.")

# =========================
# INFERENCE LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Reached end of video or failed to read frame.")
        break

    # Convert OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    img = transform(frame_rgb).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(img)

        probs = torch.softmax(output, dim=1)

        pred = torch.argmax(probs, dim=1).item()

    # Get label
    label = classes[pred]

    # Print debug info
    print("--------------------------------")
    print("Raw Output:", output.cpu().numpy())
    print("Probabilities:", probs.cpu().numpy())
    print(f"Prediction: {pred} -> {label}")

    # Draw prediction
    cv2.putText(
        frame,
        f"Scene: {label}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show window
    cv2.imshow("Scene Detection - Video", frame)

    # Exit on q or ESC
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
