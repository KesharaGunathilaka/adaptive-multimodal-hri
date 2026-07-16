"""
Fine-tune the RAF-DB-trained CNN-LSTM on real-world face-crop SEQUENCES.

Counterpart of scripts/finetune_realworld.py for the temporal model: instead of
independent crops, each training sample is an ordered frame sequence from one
clip (data/realworld/{train,val}, grouped by clip id), so the LSTM finally sees
real temporal dynamics rather than RAF-DB pseudo-sequences.

  - Train data: real-world train clips (each seen ``--repeats`` times per
    epoch with a fresh random frame subset + augmentations) + an equal-sized
    random draw of RAF-DB pseudo-sequences to keep general expression
    knowledge without drowning out the ~900 clips.
  - LRs: backbone at --lr (gentle, it's already domain-trained), LSTM +
    classifier at --head_lr (they must adapt to real temporal statistics).
  - Loss: CE with SOFTENED inverse-frequency class weights (w ~ 1/count^0.5)
    + label smoothing, as in finetune_realworld.py.
  - Selection: macro-F1 on the real-world val clips (held-out subjects).
    RAF-DB test metrics are logged each epoch to watch for regression.

Run (from modalities/emotion/):
    python scripts/finetune_lstm_realworld.py
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

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    LABEL_SMOOTHING,
    NUM_WORKERS,
    REPORT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from src.data import MultiViewImageFolder, RandomSubsetDataset, RealWorldClipDataset
from src.engine import _cosine_warmup, evaluate, get_device, set_seed, train_one_epoch
from src.models import EmotionCNNLSTM
from src.transforms import get_test_transforms, get_train_transforms

REALWORLD_TRAIN = os.path.join(DATA_DIR, "realworld", "train")
REALWORLD_VAL = os.path.join(DATA_DIR, "realworld", "val")
INIT_CKPT = os.path.join(CHECKPOINT_DIR, "best_MobileNetV2_LSTM.pth")
OUT_CKPT = os.path.join(CHECKPOINT_DIR, "finetuned_MobileNetV2_LSTM.pth")


def softened_class_weights(datasets, num_classes=7, power=0.5):
    """w ~ (1/count)^power over per-sequence labels, normalized to sum to C."""
    counts = np.zeros(num_classes)
    for ds in datasets:
        counts += np.bincount([label for _, label in ds.samples], minlength=num_classes)
    w = (1.0 / counts.clip(min=1)) ** power
    w = w / w.sum() * num_classes
    return torch.FloatTensor(w), counts.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4,
                    help="LR for LSTM + classifier (backbone uses --lr)")
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16, help="Sequences per batch")
    ap.add_argument("--repeats", type=int, default=4,
                    help="Times each real clip appears per epoch")
    ap.add_argument("--rafdb_ratio", type=float, default=1.0,
                    help="RAF-DB pseudo-sequences per real sequence, per epoch")
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    if not os.path.isdir(REALWORLD_TRAIN):
        raise SystemExit("data/realworld/ not found - run "
                         "scripts/extract_realworld_faces.py first.")

    # ── Data ─────────────────────────────────────────────────────────────
    real_train = RealWorldClipDataset(REALWORLD_TRAIN, get_train_transforms(),
                                      seq_len=args.seq_len, train=True,
                                      repeats=args.repeats)
    rafdb_train = MultiViewImageFolder(TRAIN_DIR, get_train_transforms(),
                                       views=args.seq_len)
    n_raf = int(len(real_train) * args.rafdb_ratio)
    train_ds = ConcatDataset([real_train, RandomSubsetDataset(rafdb_train, n_raf)])
    real_val = RealWorldClipDataset(REALWORLD_VAL, get_test_transforms(),
                                    seq_len=args.seq_len)
    rafdb_test = MultiViewImageFolder(TEST_DIR, get_test_transforms(), views=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    real_val_loader = DataLoader(real_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    rafdb_test_loader = DataLoader(rafdb_test, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    weights, counts = softened_class_weights([real_train, rafdb_train])
    print(f"Train: {len(real_train)} real seqs ({len(real_train.clips)} clips x "
          f"{args.repeats}) + {n_raf}/{len(rafdb_train)} RAF-DB pseudo-seqs/epoch "
          f"(seq_len={args.seq_len})  |  val: {len(real_val)} real clips, "
          f"{len(rafdb_test)} RAF-DB test images")
    print(f"Combined class counts: {counts.tolist()}")
    print(f"Softened class weights: {[round(float(w), 3) for w in weights]}")

    # ── Model ────────────────────────────────────────────────────────────
    model = EmotionCNNLSTM(pretrained=False)
    model.load_state_dict(torch.load(INIT_CKPT, map_location=device, weights_only=True))
    model.to(device)
    print(f"Initialized from {INIT_CKPT}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                    label_smoothing=LABEL_SMOOTHING)
    backbone = [p for n, p in model.named_parameters() if n.startswith("features.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("features.")]
    optimizer = optim.Adam([{"params": backbone, "lr": args.lr},
                            {"params": head, "lr": args.head_lr}],
                           weight_decay=1e-5)
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
            torch.save(model.state_dict(), OUT_CKPT)
            print(f"      saved (best real val macro-F1 {best_f1*100:.2f}%)")

    out_dir = os.path.join(REPORT_DIR, "finetune_realworld")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history_MobileNetV2_LSTM.json"), "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nBest real-world val macro-F1: {best_f1*100:.2f}%  ->  {OUT_CKPT}")


if __name__ == "__main__":
    main()
