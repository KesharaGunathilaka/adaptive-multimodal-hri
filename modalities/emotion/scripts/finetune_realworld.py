"""
Fine-tune a trained RAF-DB checkpoint on RAF-DB + real-world face crops.

Adapts the model to far-field deployment footage while keeping general
expression knowledge:
  - Train data: RAF-DB train + data/realworld/train (from
    scripts/extract_realworld_faces.py), both with the rich train transforms
    (now including RandomDownscale).
  - Loss: CE with SOFTENED inverse-frequency class weights (w ~ 1/count^0.5)
    + label smoothing. Full inverse-frequency weighting over-fires on rare
    classes (the live "Surprise" bias); the square root is a middle ground.
  - Selection: macro-F1 on the real-world val crops (held-out subject P04).
    RAF-DB test metrics are logged each epoch to watch for regression.

Run (from modalities/emotion/):
    python scripts/finetune_realworld.py --model MobileNetV2
    python scripts/finetune_realworld.py --model EfficientNet-B0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEFAULT_MODEL,
    LABEL_SMOOTHING,
    NUM_WORKERS,
    REPORT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from src.engine import evaluate, get_device, set_seed, train_one_epoch, _cosine_warmup
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms, get_train_transforms

REALWORLD_TRAIN = os.path.join(DATA_DIR, "realworld", "train")
REALWORLD_VAL = os.path.join(DATA_DIR, "realworld", "val")


def softened_class_weights(datasets, num_classes=7, power=0.5):
    """w ~ (1/count)^power over the combined datasets, normalized to sum to C."""
    counts = np.zeros(num_classes)
    for ds in datasets:
        counts += np.bincount([label for _, label in ds.samples], minlength=num_classes)
    w = (1.0 / counts.clip(min=1)) ** power
    w = w / w.sum() * num_classes
    return torch.FloatTensor(w), counts.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--init_checkpoint", default=None,
                    help="Starting weights (default: checkpoints/best_<Model>.pth)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    sname = safe_name(args.model)
    init_ckpt = args.init_checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")
    out_ckpt = os.path.join(CHECKPOINT_DIR, f"finetuned_{sname}.pth")

    if not os.path.isdir(REALWORLD_TRAIN):
        raise SystemExit("data/realworld/ not found - run "
                         "scripts/extract_realworld_faces.py first.")

    # ── Data ─────────────────────────────────────────────────────────────
    rafdb_train = ImageFolder(TRAIN_DIR, transform=get_train_transforms())
    real_train = ImageFolder(REALWORLD_TRAIN, transform=get_train_transforms())
    train_ds = ConcatDataset([rafdb_train, real_train])
    real_val = ImageFolder(REALWORLD_VAL, transform=get_test_transforms())
    rafdb_test = ImageFolder(TEST_DIR, transform=get_test_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    real_val_loader = DataLoader(real_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    rafdb_test_loader = DataLoader(rafdb_test, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    weights, counts = softened_class_weights([rafdb_train, real_train])
    print(f"Train: {len(rafdb_train)} RAF-DB + {len(real_train)} real-world "
          f"= {len(train_ds)} images  |  val: {len(real_val)} real, "
          f"{len(rafdb_test)} RAF-DB test")
    print(f"Combined class counts: {counts.tolist()}")
    print(f"Softened class weights: {[round(float(w), 3) for w in weights]}")

    # ── Model ────────────────────────────────────────────────────────────
    model = ALL_MODELS[args.model](pretrained=False)
    model.load_state_dict(torch.load(init_ckpt, map_location=device, weights_only=True))
    model.to(device)
    print(f"Initialized from {init_ckpt}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                    label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_warmup(s, len(train_loader), total_steps))
    scaler = (torch.amp.GradScaler("cuda")
              if (not args.no_amp and device.type == "cuda") else None)

    # ── Fine-tune ────────────────────────────────────────────────────────
    best_f1, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                       device, scheduler=scheduler, use_mixup=False,
                                       scaler=scaler, desc=f"FT {ep}/{args.epochs}")
        real_m = evaluate(model, real_val_loader, device)
        raf_m = evaluate(model, rafdb_test_loader, device)
        history.append({"epoch": ep, "loss": loss, "train_acc": tr_acc,
                        "real_val": real_m, "rafdb_test": raf_m})
        print(f"  [{ep:2d}/{args.epochs}] loss={loss:.4f} train_acc={tr_acc*100:.2f}%  "
              f"REAL val: acc={real_m['accuracy']*100:.2f}% "
              f"macroF1={real_m['f1_macro']*100:.2f}%  |  "
              f"RAF-DB test: acc={raf_m['accuracy']*100:.2f}% "
              f"macroF1={raf_m['f1_macro']*100:.2f}%")
        if real_m["f1_macro"] > best_f1:
            best_f1 = real_m["f1_macro"]
            torch.save(model.state_dict(), out_ckpt)
            print(f"      saved (best real val macro-F1 {best_f1*100:.2f}%)")

    out_dir = os.path.join(REPORT_DIR, "finetune_realworld")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"history_{sname}.json"), "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nBest real-world val macro-F1: {best_f1*100:.2f}%  ->  {out_ckpt}")


if __name__ == "__main__":
    main()
