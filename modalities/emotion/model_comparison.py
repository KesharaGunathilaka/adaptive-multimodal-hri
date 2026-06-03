"""
Comprehensive Model Comparison Script
Compares MobileNetV2, EfficientNet-B0, and ResNet18 on RAF-DB emotion dataset
For Jetson Orin Nano deployment optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from utils.transforms import get_train_transforms, get_test_transforms
from config import *
import os
from tqdm import tqdm
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


def get_mobilenet_v2(num_classes):
    """MobileNetV2: Current model - Lightweight, efficient"""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_efficientnet_b0(num_classes):
    """EfficientNet-B0: Lightweight, better accuracy potential"""
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_resnet18(num_classes):
    """ResNet18: Baseline comparison - Balanced"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """Estimate FLOPs for model"""
    try:
        from thop import profile

        dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # Return in GFLOPs
    except:
        return None


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_model(model, train_loader, test_loader, device, config, model_name):
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
            test_loader,
            desc=f"[{model_name}] Epoch {epoch + 1}/{config['epochs']} [Val]",
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix(
                    {"acc": f"{100 * accuracy_score(all_labels, all_preds):.2f}%"}
                )

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, Val Acc: {acc * 100:.2f}%, Val F1: {f1 * 100:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

    elapsed_time = time.time() - start_time

    return best_acc, best_epoch, elapsed_time, history


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


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
            torch.cuda.synchronize() if torch.cuda.is_available() else None
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

    # Load data
    train_dataset = ImageFolder("data/train", transform=get_train_transforms())
    test_dataset = ImageFolder("data/test", transform=get_test_transforms())

    print(f"Dataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {NUM_CLASSES}\n")

    # Configuration for baseline comparison (same for all models)
    baseline_config = {
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "optimizer": "adam",
    }

    # Models to compare
    models_to_compare = {
        "MobileNetV2": get_mobilenet_v2,
        "EfficientNet-B0": get_efficientnet_b0,
        "ResNet18": get_resnet18,
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
        test_loader = DataLoader(
            test_dataset, batch_size=baseline_config["batch_size"], num_workers=0
        )

        # Train model
        best_acc, best_epoch, training_time, history = train_model(
            model, train_loader, test_loader, device, baseline_config, model_name
        )

        # Evaluate model
        metrics = evaluate_model(model, test_loader, device)

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
        print(f"  Inference Time (std): {inference_times['std_ms']:.2f} ms")

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
