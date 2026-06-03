"""
Scene Classification Hyperparameter Tuning Script
Use Optuna to perform Bayesian optimization over learning rates, batch sizes, and optimizers
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import optuna

# ============================================================================
# CONFIGURATION & PATH SETUP
# ============================================================================
IMAGE_SIZE = 224
TUNING_TRIALS = 15  # Number of Bayesian optimization combinations to test
TUNING_EPOCHS = 5  # Short epochs per trial to find parameters quickly
FINAL_EPOCHS = 15  # Full training epochs for the final optimized model

# Handle repository paths safely
BASE_DIR = Path(__file__).resolve().parent
DATASET_BASE = BASE_DIR.parent / "data" / "scene"
TRAIN_DIR = DATASET_BASE / "train"
VAL_DIR = DATASET_BASE / "val"

OUTPUT_DIR = Path("hyperparameter_tuning")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# OPTIMAL EDGE MODEL ARCHITECTURE
# ============================================================================
class SceneModel(nn.Module):
    def __init__(self, num_classes):
        super(SceneModel, self).__init__()
        # Swapped to EfficientNet-B0 for superior extraction capacity on the Jetson Orin Nano
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        self.fc = self.model.classifier[1]  # Alias for training script compatibility

    def forward(self, x):
        return self.model(x)


# ============================================================================
# DATA PREPARATION (Loaded once globally to conserve VRAM)
# ============================================================================
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

train_dataset = ImageFolder(str(TRAIN_DIR), transform=train_transform)
val_dataset = ImageFolder(str(VAL_DIR), transform=val_transform)
NUM_CLASSES = len(
    train_dataset.classes
)  # Dynamically handles variations in class count


# ============================================================================
# OPTUNA TRIAL OBJECTIVE
# ============================================================================
def objective(trial, device):
    """Optuna objective function that evaluates a single hyperparameter combination."""

    # Define the Bayesian search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    # Initialize dynamic DataLoaders for this trial's batch size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = SceneModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TUNING_EPOCHS)

    # Execute a lightweight training run
    best_trial_acc = 0.0
    for epoch in range(TUNING_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Fast Validation Phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_trial_acc:
            best_trial_acc = acc

        # Handle early pruning if the trial is performing significantly below baseline
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_trial_acc


# ============================================================================
# FULL STANDARD TRAINING LOOP (For final production model)
# ============================================================================
def train_final_model(best_params, device):
    """Trains the final model thoroughly using the best parameters found by Optuna."""
    print("\n" + "=" * 80)
    print("STARTING PRODUCTION TRAINING WITH OPTIMAL HYPERPARAMETERS")
    print("=" * 80)

    train_loader = DataLoader(
        train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=0
    )

    model = SceneModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    if best_params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    best_acc = 0.0
    final_metrics = {}

    for epoch in range(FINAL_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader, desc=f"Final Model - Epoch {epoch + 1}/{FINAL_EPOCHS}"
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Complete Validation Assessment
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        print(
            f" -> Epoch {epoch + 1} Evaluation: Accuracy = {acc * 100:.2f}%, F1-Score = {f1 * 100:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            final_metrics = {"accuracy": acc, "f1_score": f1, "epoch": epoch + 1}
            # Save weights separately to clear deployment target location
            torch.save(model.state_dict(), OUTPUT_DIR / "best_scene_model.pth")

    print(
        f"\n✓ Optimization Complete. Production model saved to: {OUTPUT_DIR / 'best_scene_model.pth'}"
    )
    return final_metrics


# ============================================================================
# MAIN ROUTINE EXECUTION
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device: {device}")
    print(f"Classes Found: {train_dataset.classes} (Total: {NUM_CLASSES})\n")

    # 1. Initialize Optuna Study using the TPE (Tree-structured Parzen Estimator) Bayesian sampler
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )

    print("Starting Bayesian Optimization via Optuna...")
    study.optimize(lambda trial: objective(trial, device), n_trials=TUNING_TRIALS)

    # 2. Export Tuning Logs and Diagnostics
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Best Configuration Found: {study.best_params}")
    print(f"Top Tuning Accuracy Achieved: {study.best_value * 100:.2f}%")

    # Save the log history to CSV for visualization records
    study.trials_dataframe().to_csv(
        OUTPUT_DIR / "optuna_tuning_history.csv", index=False
    )

    # 3. Train the final production model using the best configurations discovered
    production_metrics = train_final_model(study.best_params, device)

    # 4. Save Final Metrical Summary Configuration File
    summary_report = {
        "optimal_hyperparameters": study.best_params,
        "production_metrics": {
            "accuracy": f"{production_metrics['accuracy'] * 100:.2f}%",
            "f1_score": f"{production_metrics['f1_score'] * 100:.2f}%",
            "best_epoch": production_metrics["epoch"],
        },
    }

    with open(OUTPUT_DIR / "production_config_summary.json", "w") as f:
        json.dump(summary_report, f, indent=4)

    print(
        "✓ Diagnostic configurations and history logs compiled inside hyperparameter_tuning/"
    )
