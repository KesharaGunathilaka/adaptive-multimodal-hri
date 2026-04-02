import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACES_DIR = os.path.join(BASE_DIR, "places365")

# Load category labels
categories_file = os.path.join(PLACES_DIR, "categories_places365.txt")
classes = []
with open(categories_file, "r") as f:
    for line in f:
        classes.append(line.strip().split(' ')[0][3:])

# Load Places365 model
print("Loading Places365 ResNet50 model...")
model = models.resnet50(num_classes=365)

checkpoint_path = os.path.join(PLACES_DIR, "resnet50_places365.pth")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

state_dict = checkpoint["state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



def predict_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.topk(1)

    return classes[top_idx.item()], top_prob.item()


def predict_video_scene(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    predictions = []
    confidences = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every N frames
        if frame_count % frame_interval == 0:
            scene, conf = predict_frame(frame)
            predictions.append(scene)
            confidences.append(conf)
            print(f"Frame {frame_count}: Scene - {scene}, Confidence - {conf:.2f}")

        frame_count += 1

    cap.release()

    # Majority voting
    scene_counter = Counter(predictions)
    final_scene = scene_counter.most_common(1)[0][0]

    # Average confidence for that scene
    avg_conf = np.mean([
        conf for scene, conf in zip(predictions, confidences)
        if scene == final_scene
    ])

    return final_scene, avg_conf


# TEST WITH VIDEO
VIDEO_DIR = os.path.join(BASE_DIR, "data")
video_path = os.path.join(
    VIDEO_DIR, "input_video.mp4")

scene, confidence = predict_video_scene(video_path)
print(f"Predicted Scene: {scene}, Confidence: {confidence:.2f}")
