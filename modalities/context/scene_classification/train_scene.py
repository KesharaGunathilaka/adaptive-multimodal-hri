"""
Scene Classification Base Training Script
For tune Parameters and train the model run tune_and_train.py"""

from pathlib import Path
import sys

# Windows consoles default to cp1252, which can't encode the checkmarks printed
# below; force UTF-8 so training doesn't crash mid-run on a print statement.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_model import SceneModel

# Training transforms (with augmentation)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Validation transforms (no augmentation)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

base_dir = Path(__file__).resolve().parent
dataset_base = base_dir.parent / "data" / "scene"
train_dir = dataset_base / "train"
val_dir = dataset_base / "val"
weights_path = base_dir / "scene_model" / "scene.pth"

# Load train and validation datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

# Print dataset info
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")
print(f"Class to index mapping: {train_dataset.class_to_idx}")

# Persist the exact class order alongside the weights so inference never has to
# guess it. SceneClassifier reads this classes.json if present.
import json

weights_path.parent.mkdir(parents=True, exist_ok=True)
(weights_path.parent / "classes.json").write_text(
    json.dumps(train_dataset.classes), encoding="utf-8"
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup (number of classes derived from the dataset folders)
model = SceneModel(num_classes=len(train_dataset.classes)).cuda()
loss_fn = nn.CrossEntropyLoss()

# STAGE 1: WARM-UP (Train final layer only)
print("\n" + "=" * 50)
print("STAGE 1: Warm-up (Training final layer only)")
print("=" * 50)

# 1. Freeze all parameters in the model
for param in model.parameters():
    param.requires_grad = False

# 2. Unfreeze ONLY the final classification layer.
for param in model.fc.parameters():
    param.requires_grad = True

# 3. Setup Stage 1 Optimizer (Higher LR, passing only unfrozen params)
opt_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

for epoch in range(5):
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    progress_bar = tqdm(
        train_loader, desc=f"Stage 1 - Epoch {epoch + 1}/5 [TRAIN]", unit="batch"
    )

    for x, y in progress_bar:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = loss_fn(out, y)

        opt_stage1.zero_grad()
        loss.backward()
        opt_stage1.step()

        running_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    print(
        f"Stage 1 - Epoch {epoch + 1}/5 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
    )

    # Quick Validation phase (No early stopping logic in Stage 1)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item()

            _, predicted = torch.max(out.data, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    print(
        f"Stage 1 - Epoch {epoch + 1}/5 - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
    )


# STAGE 2: FINE-TUNING (Train entire network)
print("\n" + "=" * 50)
print("STAGE 2: Fine-tuning (Training entire network)")
print("=" * 50)

# 1. Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# 2. Setup Stage 2 Optimizer (Very small learning rate)
opt_stage2 = optim.Adam(model.parameters(), lr=1e-5)

best_val_acc = 0
patience = 3
patience_counter = 0
stage2_epochs = 15

for epoch in range(stage2_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    progress_bar = tqdm(
        train_loader,
        desc=f"Stage 2 - Epoch {epoch + 1}/{stage2_epochs} [TRAIN]",
        unit="batch",
    )

    for x, y in progress_bar:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = loss_fn(out, y)

        opt_stage2.zero_grad()
        loss.backward()
        opt_stage2.step()

        running_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    print(
        f"Stage 2 - Epoch {epoch + 1}/{stage2_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
    )

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item()

            _, predicted = torch.max(out.data, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    print(
        f"Stage 2 - Epoch {epoch + 1}/{stage2_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(weights_path))
        print(f"✓ Best model saved with Val Acc: {val_acc:.2f}%\n")
    else:
        patience_counter += 1
        print(f"Patience counter: {patience_counter}/{patience}\n")
        if patience_counter >= patience:
            print(f"Early stopping triggered. Best Val Acc: {best_val_acc:.2f}%")
            break

print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
