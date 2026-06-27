"""
Improved emotion recognition training.

Key improvements over train.py:
  - Weighted CrossEntropyLoss to fix class imbalance (Fear: 281 vs Happy: 4,772)
  - Two-stage fine-tuning: head only → full backbone
  - Richer augmentation: CLAHE, ColorJitter, RandomErasing, RandomAffine (see transforms.py)
  - Mixup data augmentation (alpha=0.2, applied to 50% of batches in stage 2)
  - Label smoothing (0.1) in stage 2
  - Cosine LR schedule with 2-epoch linear warmup in stage 2
  - Gradient clipping (max norm=1.0)

Usage:
    python train_v2.py --model EfficientNet-B0 --stage1_epochs 5 --stage2_epochs 25
    python train_v2.py --model MobileNetV3-Large
    python train_v2.py --model MobileNetV2
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from config import BATCH_SIZE, NUM_CLASSES
from models.model_zoo import ALL_MODELS, model_size_mb
from utils.transforms import get_test_transforms, get_train_transforms


def compute_class_weights(dataset):
    counts = np.bincount([label for _, label in dataset.samples])
    w = 1.0 / counts.astype(float)
    w = w / w.sum() * len(counts)
    return torch.FloatTensor(w)


def _mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _mixup_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


def _cosine_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, use_mixup, alpha=0.2):
    model.train()
    total_loss, preds, true_labels = 0.0, [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixup and np.random.rand() < 0.5:
            mixed, ya, yb, lam = _mixup(imgs, labels, alpha, device)
            out = model(mixed)
            loss = _mixup_loss(criterion, out, ya, yb, lam)
            preds.extend(out.argmax(1).cpu().tolist())
            true_labels.extend(ya.cpu().tolist())
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            preds.extend(out.argmax(1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader), accuracy_score(true_labels, preds)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, true_labels = [], []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        preds.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
        true_labels.extend(labels.tolist())
    return accuracy_score(true_labels, preds), f1_score(true_labels, preds, average="weighted")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    train_ds = ImageFolder("data/train", transform=get_train_transforms())
    val_ds = ImageFolder("data/test", transform=get_test_transforms())

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Training: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ALL_MODELS[args.model]().to(device)
    size = model_size_mb(model)
    print(f"\nModel: {args.model} | Size: {size:.1f} MB")

    safe_name = args.model.replace(" ", "_").replace("-", "_")
    ckpt_path = f"checkpoints/best_emotion_v2_{safe_name}.pth"
    best_acc = 0.0
    history = []

    # ── Stage 1: head only ────────────────────────────────────────────
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name

    head_params = [p for p in model.parameters() if p.requires_grad]
    criterion_s1 = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_s1 = optim.Adam(head_params, lr=1e-3)

    print(f"\n── Stage 1: {args.stage1_epochs} epochs (head only) ──")
    for ep in range(1, args.stage1_epochs + 1):
        loss, tr_acc = train_one_epoch(
            model, train_loader, criterion_s1, optimizer_s1,
            None, device, use_mixup=False
        )
        val_acc, val_f1 = validate(model, val_loader, device)
        print(
            f"  [{ep:2d}/{args.stage1_epochs}]  loss={loss:.4f}  "
            f"train={tr_acc*100:.2f}%  val={val_acc*100:.2f}%  f1={val_f1*100:.2f}%"
        )
        history.append(dict(stage=1, epoch=ep, loss=loss, train_acc=tr_acc, val_acc=val_acc, val_f1=val_f1))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    # ── Stage 2: full fine-tune ───────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = True

    criterion_s2 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer_s2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    total_steps = args.stage2_epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer_s2,
        lambda s: _cosine_warmup(s, warmup_steps, total_steps)
    )

    print(f"\n── Stage 2: {args.stage2_epochs} epochs (full fine-tune + mixup + label smoothing) ──")
    for ep in range(1, args.stage2_epochs + 1):
        loss, tr_acc = train_one_epoch(
            model, train_loader, criterion_s2, optimizer_s2,
            scheduler, device, use_mixup=True, alpha=0.2
        )
        val_acc, val_f1 = validate(model, val_loader, device)
        print(
            f"  [{ep:2d}/{args.stage2_epochs}]  loss={loss:.4f}  "
            f"train={tr_acc*100:.2f}%  val={val_acc*100:.2f}%  f1={val_f1*100:.2f}%"
        )
        history.append(dict(stage=2, epoch=ep, loss=loss, train_acc=tr_acc, val_acc=val_acc, val_f1=val_f1))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest val accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint saved: {ckpt_path}")

    hist_path = f"checkpoints/history_{safe_name}.json"
    with open(hist_path, "w") as f:
        json.dump(dict(model=args.model, best_acc=best_acc, checkpoint=ckpt_path, history=history), f, indent=2)
    print(f"History saved: {hist_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EfficientNet-B0", choices=list(ALL_MODELS.keys()))
    ap.add_argument("--stage1_epochs", type=int, default=5)
    ap.add_argument("--stage2_epochs", type=int, default=25)
    main(ap.parse_args())
