"""Fine-tune MotionLSTM on the COMPLETE `data/final_merged` dataset.

Warm-starts from `best_model.pt` (the NTU-only base checkpoint) rather than
from the currently-deployed `best_model_finetuned.pt` — this replays the
ORIGINAL documented recipe (modalities/motion/scripts/finetune.py: "Warm-starts
all weights from the NTU checkpoint") on the now-3x-larger real dataset, instead
of stacking a second fine-tune on top of the first. Same hyperparameters as the
original script (lr 2e-4, weight-decay 0.006, patience 15) for comparability.

Training windows are the RAW 84-dim pos+vel features from
`WindowFeaturizer.raw_training_windows` — the exact same window-selection logic
used at inference (fusion/extraction/windows.py), so there is no train/serve
skew between what this script trains on and what the deployed pipeline feeds
the model. Source: `data/final_merged/features/perframe/*.npz` (already
cached — no video decoding needed).

Train/val = the `split` column from `23_build_splits.py` (actor-disjoint,
P04+P03 in val). Test is never touched here.

After training, the new checkpoint is saved SEPARATELY
(`best_model_finetuned_merged.pt`) -- it does not overwrite the deployed
checkpoint. Run `scripts/32_refresh_and_compare.py --modality motion` next to
regenerate the feature cache with it and get a before/after comparison; only
promote it to deployed once that comparison is satisfactory.

    .venv/Scripts/python scripts/31_finetune_motion.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.windows import WindowFeaturizer  # noqa: E402
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval.merged_unimodal import PERFRAME_DIR, load_clips  # noqa: E402

MOT_DIR = ROOT / "modalities" / "motion"
NTU_BASE_CKPT = MOT_DIR / "checkpoints" / "best_model.pt"              # NTU-only
DEPLOYED_CKPT = MOT_DIR / "checkpoints" / "best_model_finetuned.pt"    # deployed

MOTION_LABELS = {0: "sitting", 1: "standing", 2: "walking", 3: "stepping_back"}
GT_TO_IDX = {"sitting": 0, "standing": 1, "walking": 2, "stepping_back": 3}

# lr is 4x lower for the "deployed" warm start: continuing an ALREADY
# domain-adapted checkpoint needs a gentler step than the original NTU->real
# adaptation did, or it risks overwriting exactly the generalisation the
# 2026-08-04 experiment showed the from-scratch-on-NTU run destroyed.
CONFIGS = {
    "ntu": {"base": NTU_BASE_CKPT, "batch_size": 64, "epochs": 80, "lr": 0.0002,
           "weight_decay": 0.006, "patience": 15},
    "deployed": {"base": DEPLOYED_CKPT, "batch_size": 64, "epochs": 60, "lr": 0.00005,
                "weight_decay": 0.006, "patience": 15},
}


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_windows(clips, fz) -> tuple[list, list]:
    X, y = [], []
    t0 = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        if r.motion_masked or r.gt_motion not in GT_TO_IDX:
            continue
        p = PERFRAME_DIR / f"{r.clip_id}.npz"
        if not p.exists():
            continue
        npz = np.load(p)
        label = GT_TO_IDX[r.gt_motion]
        for feat in fz.raw_training_windows(npz, "motion"):
            X.append(feat)
            y.append(label)
        if n % 300 == 0:
            print(f"  [{n}/{len(clips)}] {len(X)} windows ({n/(time.time()-t0):.1f} clips/s)",
                  flush=True)
    return X, y


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(yb.cpu().tolist())
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, macro_f1, preds, labels


CACHE_DIR = ROOT / "data" / "final_merged" / "features" / "_motion_finetune_windows"


def get_windows(device):
    """Raw train/val windows -- identical regardless of warm-start choice, so
    cache them once instead of rebuilding (~3 min) per experiment."""
    if CACHE_DIR.exists():
        print(f"reusing cached windows from {CACHE_DIR}")
        Xtr = np.load(CACHE_DIR / "Xtr.npy"); ytr = np.load(CACHE_DIR / "ytr.npy")
        Xva = np.load(CACHE_DIR / "Xva.npy"); yva = np.load(CACHE_DIR / "yva.npy")
        return list(Xtr), list(ytr), list(Xva), list(yva)
    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    fz = WindowFeaturizer(device=device)
    print("Building TRAIN windows...")
    Xtr, ytr = build_windows(clips[clips.split == "train"], fz)
    print(f"  {len(Xtr)} train windows")
    print("Building VAL windows...")
    Xva, yva = build_windows(clips[clips.split == "val"], fz)
    print(f"  {len(Xva)} val windows\n")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "Xtr.npy", np.stack(Xtr)); np.save(CACHE_DIR / "ytr.npy", np.array(ytr))
    np.save(CACHE_DIR / "Xva.npy", np.stack(Xva)); np.save(CACHE_DIR / "yva.npy", np.array(yva))
    return Xtr, ytr, Xva, yva


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", choices=("ntu", "deployed"), default="deployed")
    ap.add_argument("--no-sampler", action="store_true",
                    help="class-weighted LOSS only, no WeightedRandomSampler -- "
                         "2026-08-04 finding: the sampler (exact-duplicate "
                         "oversampling of stepping_back) coincided with a "
                         "val-up/test-down collapse; emotion+gesture use "
                         "loss-weighting only and both generalised. Default "
                         "off for backward compatibility with the original run.")
    args = ap.parse_args()
    CONFIG = CONFIGS[args.warm_start]
    BASE_CKPT = CONFIG["base"]
    tag = args.warm_start + ("_noresample" if args.no_sampler else "")
    OUT_CKPT = MOT_DIR / "checkpoints" / f"best_model_finetuned_merged_{tag}.pt"
    LOG_PATH = MOT_DIR / "logs" / f"training_log_finetuned_merged_{tag}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  warm_start={args.warm_start}  base={BASE_CKPT.name}\n")

    Xtr, ytr, Xva, yva = get_windows(device)
    counts = Counter(ytr)
    print("class distribution (train):")
    for i, name in MOTION_LABELS.items():
        print(f"  {name:<15}: {counts.get(i, 0):>6,}")

    train_ds, val_ds = WindowDataset(Xtr, ytr), WindowDataset(Xva, yva)
    if args.no_sampler:
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        print("\nsampler: NONE (class-weighted loss only)")
    else:
        sample_w = [1.0 / counts[l] for l in ytr]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler)
        print("\nsampler: WeightedRandomSampler (+ class-weighted loss, the "
              "original/suspect combination)")
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    weights = compute_class_weight("balanced", classes=np.arange(4), y=np.array(ytr))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(f"\nWarm-starting from {BASE_CKPT}")
    base = torch.load(BASE_CKPT, map_location=device, weights_only=True)
    base_cfg = base.get("config", {})
    sys.path.insert(0, str(MOT_DIR / "src"))
    from model import MotionLSTM  # noqa: E402
    model = MotionLSTM(hidden_size=base_cfg.get("hidden_size", 256),
                       num_layers=base_cfg.get("num_layers", 3),
                       dropout=base_cfg.get("dropout", 0.35), num_classes=4).to(device)
    model.load_state_dict(base["model_state_dict"])
    print(f"base val_acc={base.get('val_acc', 0):.3%}, "
          f"{model.count_parameters():,} params")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

    best_score, best_val_acc, patience_ctr, history = -1.0, 0.0, 0, []
    best_state = None
    print(f"\n{'epoch':>6}{'train_loss':>12}{'train_acc':>11}{'val_loss':>10}{'val_acc':>9}{'val_f1':>8}")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += len(yb)
        sched.step()
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        note = ""
        # select on macro-F1, matching emotion/gesture -- val_acc alone lets a
        # model that ignores rare classes look best (exactly motion's failure mode)
        if val_f1 > best_score:
            best_score, best_val_acc, patience_ctr = val_f1, val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            note = "<- best"
        else:
            patience_ctr += 1
        history.append({"epoch": epoch, "train_loss": round(tr_loss / len(train_loader), 4),
                        "train_acc": round(tr_correct / tr_total, 4),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                        "val_macro_f1": round(val_f1, 4)})
        print(f"{epoch:>6}{tr_loss/len(train_loader):>12.4f}"
              f"{tr_correct/tr_total:>10.2%}{val_loss:>10.4f}{val_acc:>9.2%}{val_f1:>8.3f}  {note}")
        if patience_ctr >= CONFIG["patience"]:
            print(f"early stop at epoch {epoch}")
            break

    log_params = {k: v for k, v in CONFIG.items() if k != "base"}
    model.load_state_dict(best_state)
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": len(history), "model_state_dict": model.state_dict(),
               "val_acc": best_val_acc, "val_macro_f1": best_score,
               "config": {**base_cfg, **log_params},
               "warm_started_from": str(BASE_CKPT),
               "trained_on": "data/final_merged (complete)"}, OUT_CKPT)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps({"config": log_params, "history": history}, indent=2))

    _, _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    names = [MOTION_LABELS[i] for i in range(4)]
    print(f"\nbest val_acc={best_val_acc:.3%}  val_macro_f1={best_score:.3f}")
    print(classification_report(val_labels, val_preds, labels=list(range(4)),
                                target_names=names, digits=3, zero_division=0))
    print(confusion_matrix(val_labels, val_preds))
    print(f"\n-> {OUT_CKPT}")

    with start_run("02_fusion", f"motion__finetune_merged_{tag}",
                   dataset="final_merged", split_kind="scenarios", cues="real",
                   params={"model": "motion_lstm", **log_params,
                           "warm_start": args.warm_start,
                           "sampler": "none" if args.no_sampler else "weighted_random",
                           "n_train_windows": len(Xtr), "n_val_windows": len(Xva)},
                   notes="fine-tune on complete dataset; see script 32 for "
                        "before/after headline comparison") as run:
        run.log_metrics({"val_acc": best_val_acc, "val_macro_f1": best_score,
                         "best_epoch": history[-1]["epoch"]})
        run.log_checkpoint(OUT_CKPT)


if __name__ == "__main__":
    main()
