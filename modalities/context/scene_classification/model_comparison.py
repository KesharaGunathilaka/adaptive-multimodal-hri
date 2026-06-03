"""
Scene Classification Model Comparison Script
Compares MobileNetV3Small, ResNet18, and EfficientNet-B0 on Places365 custom dataset
For scene understanding in adaptive human-robot interaction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
import os
from tqdm import tqdm
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from pathlib import Path
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CLASSES = 3  # classroom, kitchen, office
IMAGE_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 15
SCENE_LABELS = ["Classroom", "Kitchen", "Office"]

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


def get_mobilenet_v3_small(num_classes):
    """MobileNetV3Small: Current model - Lightweight"""
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model


def get_resnet18(num_classes):
    """ResNet18: Balanced baseline"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes):
    """EfficientNet-B0: Better accuracy potential"""
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ============================================================================
# MODEL METADATA & PARAMETERS
# ============================================================================


def count_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model):
    """Estimate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_model(model, train_loader, val_loader, device, config, model_name):
    """Train model with given configuration"""

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc = 0
    best_epoch = 0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    start_time = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"[{model_name}] Epoch {epoch + 1}/{config['epochs']} [Train]",
            leave=False,
        )

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        history["train_loss"].append(running_loss / len(train_loader))

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        val_pbar = tqdm(
            val_loader,
            desc=f"[{model_name}] Epoch {epoch + 1}/{config['epochs']} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

    elapsed_time = time.time() - start_time

    return best_acc, best_epoch, elapsed_time, history


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================


def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set with detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ============================================================================
# INFERENCE TIME MEASUREMENT
# ============================================================================


def measure_inference_time(model, device, num_samples=100, input_size=(1, 3, 224, 224)):
    """Measure inference time per sample"""
    model.eval()

    times = []
    with torch.no_grad():
        for _ in range(num_samples):
            dummy_input = torch.randn(input_size).to(device)

            start = time.time()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start

            times.append(elapsed * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create comparison directory
    os.makedirs("comparison_results", exist_ok=True)

    # Define data paths
    base_dir = Path(__file__).resolve().parent
    dataset_base = base_dir.parent / "data" / "scene"
    train_dir = dataset_base / "train"
    val_dir = dataset_base / "val"

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(val_dir), transform=val_transform)

    print(f"Dataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Classes: {train_dataset.classes}\n")

    # Configuration for baseline comparison (same for all models)
    baseline_config = {
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "optimizer": "adam",
    }

    # Models to compare
    models_to_compare = {
        "MobileNetV3Small": get_mobilenet_v3_small,
        "ResNet18": get_resnet18,
        "EfficientNet-B0": get_efficientnet_b0,
    }

    results = {}

    print("=" * 100)
    print("BASELINE COMPARISON (Same hyperparameters for all models)")
    print("=" * 100)
    print(
        f"Configuration: LR={baseline_config['lr']}, Batch Size={baseline_config['batch_size']}, "
        f"Optimizer={baseline_config['optimizer']}, Epochs={baseline_config['epochs']}\n"
    )

    for model_name, model_fn in models_to_compare.items():
        print(f"\n{'=' * 100}")
        print(f"Training {model_name}")
        print(f"{'=' * 100}")

        # Create model
        model = model_fn(NUM_CLASSES).to(device)

        # Calculate model statistics
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        model_size_mb = estimate_model_size(model)

        print(f"\n{model_name} Model Statistics:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: {model_size_mb:.2f} MB")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=baseline_config["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=baseline_config["batch_size"], num_workers=0
        )

        # Train model
        best_acc, best_epoch, training_time, history = train_model(
            model, train_loader, val_loader, device, baseline_config, model_name
        )

        # Evaluate model
        metrics = evaluate_model(model, val_loader, device)

        # Measure inference time
        inference_times = measure_inference_time(model, device)

        # Store results
        results[model_name] = {
            "model_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": model_size_mb,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "training_time_sec": training_time,
            "metrics": metrics,
            "inference_time": inference_times,
            "history": history,
        }

        print(f"\n{model_name} Training Results:")
        print(f"  Best Accuracy: {best_acc * 100:.2f}% (Epoch {best_epoch})")
        print(f"  Precision: {metrics['precision'] * 100:.2f}%")
        print(f"  Recall: {metrics['recall'] * 100:.2f}%")
        print(f"  F1-Score: {metrics['f1'] * 100:.2f}%")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Inference Time (avg): {inference_times['mean_ms']:.2f} ms")

    # Save results
    results_json = {}
    for model_name, res in results.items():
        results_json[model_name] = {
            "model_params": res["model_params"],
            "trainable_params": res["trainable_params"],
            "model_size_mb": f"{res['model_size_mb']:.2f}",
            "best_acc": f"{res['best_acc'] * 100:.2f}%",
            "best_epoch": res["best_epoch"],
            "training_time_sec": f"{res['training_time_sec']:.2f}",
            "accuracy": f"{res['metrics']['accuracy'] * 100:.2f}%",
            "precision": f"{res['metrics']['precision'] * 100:.2f}%",
            "recall": f"{res['metrics']['recall'] * 100:.2f}%",
            "f1_score": f"{res['metrics']['f1'] * 100:.2f}%",
            "inference_mean_ms": f"{res['inference_time']['mean_ms']:.2f}",
            "inference_std_ms": f"{res['inference_time']['std_ms']:.2f}",
        }

    with open("comparison_results/baseline_comparison.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    # Create summary table
    print(
        f"\n{'Model':<20} {'Params':<15} {'Size (MB)':<12} {'Accuracy':<12} {'F1-Score':<12} {'Inf.Time (ms)':<15}"
    )
    print("-" * 86)
    for model_name, res in results.items():
        print(
            f"{model_name:<20} {res['model_params']:<15,} {res['model_size_mb']:<12.2f} "
            f"{res['metrics']['accuracy'] * 100:<12.2f} {res['metrics']['f1'] * 100:<12.2f} "
            f"{res['inference_time']['mean_ms']:<15.2f}"
        )

    print(
        "\n✓ Comparison complete! Results saved to comparison_results/baseline_comparison.json"
    )
