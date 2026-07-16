"""
Train the CNN-LSTM temporal emotion model on RAF-DB.

RAF-DB is static images, so each image becomes a pseudo-sequence of ``--views``
independently augmented copies: the LSTM learns to aggregate noisy views of one
expression, which is what it does at deployment over the frames of a clip.

The MobileNetV2 backbone starts from the RAF-DB-trained ``best_MobileNetV2.pth``
(falling back to ImageNet weights if missing), then the whole model is trained
on RAF-DB in the usual two stages:
  Stage 1 - LSTM + classifier only (frozen backbone), weighted CE.
  Stage 2 - full fine-tune, weighted CE + label smoothing, cosine LR w/ warmup.
Selection: macro-F1 on the RAF-DB test set (same protocol as train.py).

Run (from modalities/emotion/):
    python scripts/train_lstm_rafdb.py
    python scripts/train_lstm_rafdb.py --stage1_epochs 1 --stage2_epochs 1 --limit 200
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    LABEL_SMOOTHING,
    NUM_WORKERS,
    REPORT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from src.data import MultiViewImageFolder, compute_class_weights, subset_per_class
from src.engine import _cosine_warmup, evaluate, get_device, set_seed, train_one_epoch
from src.models import EmotionCNNLSTM, model_size_mb, count_params
from src.transforms import get_test_transforms, get_train_transforms

CKPT = os.path.join(CHECKPOINT_DIR, "best_MobileNetV2_LSTM.pth")
BACKBONE_INIT = os.path.join(CHECKPOINT_DIR, "best_MobileNetV2.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=4,
                    help="Augmented views per image = pseudo-sequence length")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Sequences per batch (x views images through the CNN)")
    ap.add_argument("--stage1_epochs", type=int, default=3)
    ap.add_argument("--stage2_epochs", type=int, default=12)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max images per class (smoke test)")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()

    train_ds = MultiViewImageFolder(TRAIN_DIR, get_train_transforms(), views=args.views)
    test_ds = MultiViewImageFolder(TEST_DIR, get_test_transforms(), views=args.views)
    weights = compute_class_weights(train_ds).to(device)
    if args.limit:
        full = train_ds
        train_ds = subset_per_class(train_ds, args.limit)
        test_ds = subset_per_class(test_ds, args.limit // 4 or 1)
        train_ds.samples = [full.samples[i] for i in train_ds.indices]  # for logs

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = EmotionCNNLSTM(pretrained=True)
    if os.path.exists(BACKBONE_INIT):
        model.load_backbone(torch.load(BACKBONE_INIT, map_location="cpu",
                                       weights_only=True))
        print(f"Backbone initialized from {BACKBONE_INIT}")
    model.to(device)
    print(f"MobileNetV2-LSTM: {count_params(model)/1e6:.2f}M params, "
          f"{model_size_mb(model):.1f} MB  |  {len(train_ds)} train images x "
          f"{args.views} views, {len(test_ds)} test  |  device: {device}")

    scaler = (torch.amp.GradScaler("cuda")
              if (not args.no_amp and device.type == "cuda") else None)
    best_f1, history = -1.0, []

    def run_epochs(stage, epochs, criterion, optimizer, scheduler=None):
        nonlocal best_f1
        for ep in range(1, epochs + 1):
            loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=scheduler, use_mixup=False, scaler=scaler,
                desc=f"S{stage} {ep}/{epochs}")
            m = evaluate(model, test_loader, device)
            m.update(stage=stage, epoch=ep, loss=loss, train_acc=tr_acc)
            history.append(m)
            print(f"  [S{stage}] {ep:2d}/{epochs}  loss={loss:.4f}  "
                  f"train_acc={tr_acc*100:.2f}%  test_acc={m['accuracy']*100:.2f}%  "
                  f"macroF1={m['f1_macro']*100:.2f}%")
            if m["f1_macro"] > best_f1:
                best_f1 = m["f1_macro"]
                torch.save(model.state_dict(), CKPT)
                print(f"      saved (best macro-F1 {best_f1*100:.2f}%)")

    # Stage 1: LSTM + classifier only
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith("features.")
    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n-- Stage 1: {args.stage1_epochs} epochs (LSTM head only) --")
    run_epochs(1, args.stage1_epochs,
               nn.CrossEntropyLoss(weight=weights),
               optim.Adam(head_params, lr=args.head_lr, weight_decay=1e-5))

    # Stage 2: full fine-tune
    for p in model.parameters():
        p.requires_grad = True
    opt2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.stage2_epochs * len(train_loader)
    sched = optim.lr_scheduler.LambdaLR(
        opt2, lambda s: _cosine_warmup(s, len(train_loader), total))
    print(f"\n-- Stage 2: {args.stage2_epochs} epochs (full fine-tune) --")
    run_epochs(2, args.stage2_epochs,
               nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING),
               opt2, scheduler=sched)

    out_dir = os.path.join(REPORT_DIR, "training")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history_MobileNetV2_LSTM.json"), "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nBest RAF-DB test macro-F1: {best_f1*100:.2f}%  ->  {CKPT}")


if __name__ == "__main__":
    main()
