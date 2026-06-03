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

# VIDEO_PATH = "../../../videos/c1/20260511_160401.mp4"

# VIDEO_PATH = "../../../videos/c1/0042.mp4"
# VIDEO_PATH = "../../../videos/k1/20260509_190437.mp4"
VIDEO_PATH = "../../../videos/k5/20260510_001118.mp4"

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

# Set up video writer for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output directory if it doesn't exist
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "scene_inference_output.mp4"

# Initialize VideoWriter with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

print(f"Saving output to: {output_path}")
print(f"Video properties: {width}x{height} @ {fps} fps")
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

    # Write frame to output video
    out.write(frame)

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
out.release()
cv2.destroyAllWindows()
print(f"\n✓ Video saved successfully to: {output_path}")
