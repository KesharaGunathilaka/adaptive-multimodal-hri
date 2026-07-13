"""
finetune.py

Fine-tunes the NTU-only MotionLSTM checkpoint (checkpoints/best_model.pt)
on real-world data from hri-multimodal-intent-v1.0.0, to close the domain
gap between Kinect-derived training data and MediaPipe-derived inference
data (measured baseline: ~24% clip accuracy on this dataset vs 96.7% on
NTU val — see evaluation results).

Warm-starts all weights from the NTU checkpoint (not a resized head — the
class count already matches at 4), trains at a lower learning rate than
the from-scratch run, and early-stops on the held-out hri val split. The
hri test split is never touched here — reserved for a final, honest
before/after comparison.

Requires data/processed/hri_finetune/{X,y}_{train,val}.npy — produced by
build_hri_dataset.py --split train / --split val.

Usage:
    python finetune.py
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import MotionLSTM

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data", "processed", "hri_finetune")
CKPT_DIR = os.path.join(SRC_DIR, "..", "checkpoints")
LOG_DIR = os.path.join(SRC_DIR, "..", "logs")
BASE_CKPT = os.path.join(CKPT_DIR, "best_model.pt")

MOTION_LABELS = {0: "sitting", 1: "standing", 2: "walking", 3: "stepping_back"}

CONFIG = {
    "batch_size":   64,     # smaller batches — fine-tune dataset is much smaller than NTU
    "epochs":       80,
    "lr":           0.0002, # ~10x lower than the from-scratch run (0.0022) — warm start, don't overwrite the pretrained backbone too fast
    "weight_decay": 0.006,  # slightly higher — more regularization given the smaller dataset
    "patience":     15,
}


class MotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    print(f"  Train : X={X_train.shape}  y={y_train.shape}")
    print(f"  Val   : X={X_val.shape}    y={y_val.shape}")
    counts = Counter(y_train.tolist())
    print("\n  Class distribution (fine-tune train):")
    for idx, name in MOTION_LABELS.items():
        print(f"    {idx} {name:<15}: {counts.get(idx, 0):>6,}")
    return X_train, y_train, X_val, y_val


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())
    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_preds, all_labels


def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nFine-tuning MotionLSTM on hri-multimodal-intent-v1.0.0\nDevice: {device}\n{'='*60}\n")

    print("Loading fine-tune data...")
    X_train, y_train, X_val, y_val = load_data()
    train_ds = MotionDataset(X_train, y_train)
    val_ds = MotionDataset(X_val, y_val)

    label_counts = Counter(y_train.tolist())
    sample_weights = [1.0 / label_counts[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    classes = np.arange(len(MOTION_LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\n  Class weights: {dict(zip(MOTION_LABELS.values(), np.round(weights, 3)))}")

    print(f"\nLoading base checkpoint: {BASE_CKPT}")
    base_ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=True)
    base_cfg = base_ckpt.get("config", {})
    model = MotionLSTM(
        hidden_size=base_cfg.get("hidden_size", 256),
        num_layers=base_cfg.get("num_layers", 3),
        dropout=base_cfg.get("dropout", 0.35),
        num_classes=4,
    ).to(device)
    model.load_state_dict(base_ckpt["model_state_dict"])
    print(f"Warm-started from checkpoint with NTU val_acc={base_ckpt.get('val_acc', 0):.3%}")
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Note'}")
    print(f"{'─'*60}")

    best_val_acc = 0.0
    patience_ctr = 0
    log_history = []
    epoch = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]

        note = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": {**base_cfg, **CONFIG},
                "warm_started_from": BASE_CKPT,
            }, os.path.join(CKPT_DIR, "best_model_finetuned.pt"))
            note = "← best"
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs)")
                break

        log_history.append({
            "epoch": epoch, "train_loss": round(avg_train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "lr": round(current_lr, 6),
        })
        print(f"{epoch:>6}  {avg_train_loss:>10.4f}  {train_acc:>8.3%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.3%}  {current_lr:>8.6f}  {note}")

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
               os.path.join(CKPT_DIR, "last_model_finetuned.pt"))

    print(f"\n{'='*60}\nBest fine-tuned val accuracy (hri val split): {best_val_acc:.3%}")
    checkpoint = torch.load(os.path.join(CKPT_DIR, "best_model_finetuned.pt"), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

    label_names = [MOTION_LABELS[i] for i in range(len(MOTION_LABELS))]
    print(f"\nPer-class report (hri val set):")
    print(classification_report(val_labels, val_preds, target_names=label_names, digits=3))
    print(f"Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(val_labels, val_preds)
    print(f"{'':>15}", end="")
    for name in label_names: print(f"{name[:10]:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{label_names[i]:>15}", end="")
        for val in row: print(f"{val:>12}", end="")
        print()

    with open(os.path.join(LOG_DIR, "training_log_finetuned.json"), "w") as f:
        json.dump({"config": CONFIG, "history": log_history}, f, indent=2)
    print(f"\nLogs saved to: {LOG_DIR}/training_log_finetuned.json")
    print(f"Best model  : {CKPT_DIR}/best_model_finetuned.pt")


if __name__ == "__main__":
    finetune()
