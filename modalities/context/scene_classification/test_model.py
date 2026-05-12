from pathlib import Path
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Ensure repo-root imports work
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_model import SceneModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transform (no augmentation for testing)
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

base_dir = Path(__file__).resolve().parent
dataset_base = base_dir.parent / "data" / "scene"
val_dir = dataset_base / "val"
weights_path = base_dir.parent / "weights" / "scene.pth"

print("=" * 60)
print("SCENE CLASSIFICATION MODEL DIAGNOSTIC TEST")
print("=" * 60)

# Load validation dataset
print("\n[1] Loading validation dataset...")
test_dataset = datasets.ImageFolder(str(val_dir), transform=test_transform)
print(f"✓ Validation set loaded: {len(test_dataset)} samples")
print(f"  Classes: {test_dataset.classes}")
print(f"  Class mapping: {test_dataset.class_to_idx}")

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"✓ Test loader ready: {len(test_dataset)} samples")

# Load model
print("\n[2] Loading model...")
model = SceneModel(num_classes=3).to(DEVICE)

if not weights_path.exists():
    print(f"✗ ERROR: Model weights not found at {weights_path}")
    print("  Please run train_scene.py first!")
    sys.exit(1)

model.load_state_dict(
    torch.load(str(weights_path), map_location=DEVICE, weights_only=True)
)
model.eval()
print(f"✓ Model loaded from {weights_path}")

# Test model
print("\n[3] Evaluating model on test set...")
all_preds = []
all_labels = []
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = 100 * total_correct / total_samples
print(f"✓ Test Accuracy: {accuracy:.2f}%")

# Print classification report
print("\n[4] Detailed Classification Report:")
print("-" * 60)
class_names = test_dataset.classes
report = classification_report(
    all_labels, all_preds, target_names=class_names, digits=4
)
print(report)

# Print confusion matrix
print("\n[5] Confusion Matrix:")
print("-" * 60)
cm = confusion_matrix(all_labels, all_preds)
print("Predicted →")
print("Actual ↓    ", "  ".join([f"{i:>8}" for i in class_names]))
for i, class_name in enumerate(class_names):
    print(f"{class_name:>10}  {cm[i]}")

# Analysis
print("\n[6] Analysis:")
print("-" * 60)
if accuracy >= 95:
    print("✓ Model is performing EXCELLENTLY! Model is well-trained.")
elif accuracy >= 80:
    print("✓ Model is performing WELL. Results should be good.")
elif accuracy >= 60:
    print("⚠ Model is performing MODERATELY. Results may vary.")
else:
    print("✗ Model is performing POORLY. Model needs retraining!")
    print("  Possible issues:")
    print("  1. Dataset might be too small or imbalanced")
    print("  2. Training needs more epochs")
    print("  3. Images might not represent the classes well")

# Per-class analysis
print("\n[7] Per-Class Performance:")
print("-" * 60)
for i, class_name in enumerate(class_names):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_acc = (
            (np.array(all_preds)[class_mask] == i).sum() / class_mask.sum() * 100
        )
        print(
            f"{class_name:>10}: {class_acc:>6.2f}% accuracy ({class_mask.sum()} samples)"
        )

print("\n" + "=" * 60)
