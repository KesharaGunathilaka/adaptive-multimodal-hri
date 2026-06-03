"""
Scene Classification Hyperparameter Tuning Script
Tests different learning rates, batch sizes, and optimizers
Focuses on the best performing model from baseline comparison
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from pathlib import Path
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CLASSES = 3
IMAGE_SIZE = 224
EPOCHS = 15
SCENE_LABELS = ["Classroom", "Kitchen", "Office"]

# ============================================================================
# MODEL DEFINITION
# ============================================================================


def get_mobilenet_v3_small(num_classes):
    """MobileNetV3Small: Current model - Lightweight"""
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_model_tuning(model, train_loader, val_loader, device, config, trial_name):
    """Train model with given configuration and return best metrics"""

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc = 0
    best_epoch = 0
    best_metrics = {}

    start_time = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"[{trial_name}] Epoch {epoch + 1}/{config['epochs']}",
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

        scheduler.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_metrics = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    elapsed_time = time.time() - start_time

    return best_acc, best_epoch, elapsed_time, best_metrics


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create results directory
    os.makedirs("hyperparameter_tuning", exist_ok=True)

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
    print(f"  Val samples: {len(val_dataset)}\n")

    # Define hyperparameter grid
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [32, 64]
    optimizers = ["adam"]
    epochs = 5

    total_combinations = len(learning_rates) * len(batch_sizes) * len(optimizers)

    print("=" * 100)
    print("HYPERPARAMETER TUNING GRID SEARCH")
    print("=" * 100)
    print(f"Learning Rates: {learning_rates}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Optimizers: {optimizers}")
    print(f"Epochs: {epochs}")
    print(f"Total Combinations: {total_combinations}\n")

    results = []
    best_overall_acc = 0
    best_config = {}

    trial = 0
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt in optimizers:
                trial += 1
                trial_name = f"Trial {trial}/{total_combinations}"

                print(f"\n{trial_name}: LR={lr}, BS={batch_size}, Opt={opt}")

                # Create model
                model = get_mobilenet_v3_small(NUM_CLASSES).to(device)

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, num_workers=0
                )

                # Configure
                config = {
                    "lr": lr,
                    "batch_size": batch_size,
                    "optimizer": opt,
                    "epochs": epochs,
                }

                # Train
                best_acc, best_epoch, training_time, metrics = train_model_tuning(
                    model, train_loader, val_loader, device, config, trial_name
                )

                result = {
                    "trial": trial,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "optimizer": opt,
                    "best_epoch": best_epoch,
                    "accuracy": best_acc,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1"],
                    "training_time_sec": training_time,
                }

                results.append(result)

                print(f"  Best Accuracy: {best_acc * 100:.2f}% (Epoch {best_epoch})")
                print(f"  F1-Score: {metrics['f1'] * 100:.2f}%")
                print(f"  Training Time: {training_time:.2f}s")

                # Track best
                if best_acc > best_overall_acc:
                    best_overall_acc = best_acc
                    best_config = config.copy()
                    best_config["best_accuracy"] = best_acc
                    best_config["metrics"] = metrics
                    best_config["best_epoch"] = best_epoch

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparameter_tuning/tuning_results.csv", index=False)

    with open("hyperparameter_tuning/best_config.json", "w") as f:
        json.dump(
            {
                "learning_rate": best_config["lr"],
                "batch_size": best_config["batch_size"],
                "optimizer": best_config["optimizer"],
                "accuracy": f"{best_config['best_accuracy'] * 100:.2f}%",
                "precision": f"{best_config['metrics']['precision'] * 100:.2f}%",
                "recall": f"{best_config['metrics']['recall'] * 100:.2f}%",
                "f1_score": f"{best_config['metrics']['f1'] * 100:.2f}%",
                "best_epoch": best_config["best_epoch"],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("TUNING SUMMARY")
    print("=" * 100)

    # Best configurations by metric
    print("\nTop 5 Configurations by Accuracy:")
    print(
        results_df.nlargest(5, "accuracy")[
            [
                "learning_rate",
                "batch_size",
                "optimizer",
                "accuracy",
                "f1_score",
                "training_time_sec",
            ]
        ]
    )

    print("\n" + "-" * 100)
    print(f"\nBest Configuration Overall:")
    print(f"  Learning Rate: {best_config['lr']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Optimizer: {best_config['optimizer']}")
    print(f"  Best Epoch: {best_config['best_epoch']}")
    print(f"  Accuracy: {best_config['best_accuracy'] * 100:.2f}%")
    print(f"  Precision: {best_config['metrics']['precision'] * 100:.2f}%")
    print(f"  Recall: {best_config['metrics']['recall'] * 100:.2f}%")
    print(f"  F1-Score: {best_config['metrics']['f1'] * 100:.2f}%")

    print(f"\n✓ Tuning complete! Results saved to hyperparameter_tuning/")
    print(f"  - tuning_results.csv (all {total_combinations} combinations)")
    print(f"  - best_config.json (best configuration)")
