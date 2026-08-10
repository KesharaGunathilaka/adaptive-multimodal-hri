"""2-fold held-out retraining of gesture (TCN) and emotion (MobileNetV2), for
out-of-fold recombination pools (2026-08-08).

`POOL_PURITY.md` found gesture/emotion's TRAIN-split embedding pools are
contaminated: both were fine-tuned on train, so pool purity is unrealistically
high there (gesture 1.000, emotion 0.959) vs test (0.864, 0.762). Motion/
context were never fine-tuned and are NOT contaminated (motion's train/test
gap is actually negative) -- only gesture and emotion need this fix.

Splits TRAIN scenarios (V3 rows, stratified by intent so each fold keeps
reasonable class balance) into 2 folds. For fold f, trains EXCLUDING fold f's
clips (only fold `1-f`'s clips are used as training data), validates on the
REAL val split unchanged (same early-stopping signal as the original
recipes). Reuses `scripts/34_finetune_emotion.py` / `36_finetune_gesture.py`'s
exact hyperparameters and warm-start-from-deployed recipe -- the only change
is which train clips are visible.

Output checkpoints are NEW files (`*_fold0.pth`/`*_fold1.pth`), never
overwriting the deployed or `_merged`-promoted checkpoints:
    modalities/gesture/checkpoints/best_TCN_fold{0,1}.pth
    modalities/emotion/checkpoints/finetuned_MobileNetV2_fold{0,1}.pth

Each fold's own held-out clips (i.e. the ones EXCLUDED from ITS training) are
exactly the clips `scripts/60_outfold_pools.py` will later score with that
fold's checkpoint -- true out-of-fold predictions, at the SCENARIO grain
(the axis `EMBEDDING_PROBE.md` and `POOL_PURITY.md` implicated, not just
actor identity).

    .venv/Scripts/python scripts/59_outfold_finetune.py
    .venv/Scripts/python scripts/59_outfold_finetune.py --only gesture
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from fusion.extraction.windows import WindowFeaturizer  # noqa: E402
from scripts.realworld_eval.merged_unimodal import (PERFRAME_DIR,  # noqa: E402
                                                     load_clips)

GES_DIR = ROOT / "modalities" / "gesture"
EMO_DIR = ROOT / "modalities" / "emotion"
CROPS_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"
GES_DEPLOYED = GES_DIR / "checkpoints" / "best_TCN.pth"
EMO_DEPLOYED = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2.pth"
FOLD_SPLIT_PATH = ROOT / "data" / "final_merged" / "features" / "outfold_split.json"

GESTURE_LABELS = ["idle", "wave", "point", "thumbs_up", "thumbs_down",
                  "beckoning", "raise_hand", "both_hands_up"]
EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GES_CFG = {"batch_size": 256, "epochs": 60, "lr": 0.0001, "weight_decay": 0.0001,
          "patience": 12, "label_smoothing": 0.1}
EMO_CFG = {"batch_size": 64, "epochs": 25, "lr": 0.00003, "weight_decay": 1e-5,
          "patience": 8, "label_smoothing": 0.1}


def make_folds(clips, seed=0) -> dict[int, set]:
    """-> {fold: set(v3_row)} -- 2 folds over TRAIN v3_rows, stratified by
    intent (round-robin within each intent group after shuffling) so both
    folds keep reasonable class balance. Persisted to disk so downstream
    scripts (60) use the IDENTICAL split, not a re-derived one."""
    if FOLD_SPLIT_PATH.exists():
        d = json.loads(FOLD_SPLIT_PATH.read_text())
        return {int(k): set(v) for k, v in d.items()}
    rng = np.random.default_rng(seed)
    train_rows = clips[clips.split == "train"][["v3_row", "intent"]].drop_duplicates()
    folds = {0: set(), 1: set()}
    for intent, grp in train_rows.groupby("intent"):
        rows = grp.v3_row.tolist()
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            folds[i % 2].add(int(r))
    FOLD_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FOLD_SPLIT_PATH.write_text(json.dumps({str(k): sorted(v) for k, v in folds.items()}, indent=2))
    return folds


# ── gesture ──────────────────────────────────────────────────────────────
class WindowArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_gesture_windows(clips, fz):
    gt_to_idx = {n: i for i, n in enumerate(GESTURE_LABELS)}
    X, y = [], []
    for r in clips.itertuples():
        if r.gesture_masked or r.gt_gesture not in gt_to_idx:
            continue
        p = PERFRAME_DIR / f"{r.clip_id}.npz"
        if not p.exists():
            continue
        npz = np.load(p)
        label = gt_to_idx[r.gt_gesture]
        for feat in fz.raw_training_windows(npz, "gesture"):
            X.append(feat)
            y.append(label)
    return X, y


def finetune_gesture_fold(clips, exclude_rows: set, fold: int, device):
    out_ckpt = GES_DIR / "checkpoints" / f"best_TCN_fold{fold}.pth"
    if out_ckpt.exists():
        print(f"  [gesture fold{fold}] already exists, skip: {out_ckpt.name}", flush=True)
        return out_ckpt

    import json as _json
    cfg = _json.loads((GES_DIR / "checkpoints" / "model_config.json").read_text())
    # Collision-safe import (see fusion/extraction/modloader.py's docstring --
    # raw sys.path mutation + `from src.models import ...` would leave
    # sys.modules['src'] pointing at gesture's package even after this
    # function returns, corrupting emotion's OWN `src`-named module when
    # `finetune_emotion_fold` runs later in the SAME process).
    gm = load_module("hri_gesture_models_ft", GES_DIR / "src" / "models.py", [GES_DIR])
    build_model = gm.build_model

    fz = WindowFeaturizer(device=device)
    tr_clips = clips[(clips.split == "train") & ~clips.v3_row.isin(exclude_rows)]
    va_clips = clips[clips.split == "val"]
    print(f"  [gesture fold{fold}] train scenarios={tr_clips.v3_row.nunique()} "
          f"(excluding {len(exclude_rows)}), building windows...", flush=True)
    Xtr, ytr = build_gesture_windows(tr_clips, fz)
    Xva, yva = build_gesture_windows(va_clips, fz)
    print(f"    {len(ytr)} train windows, {len(yva)} val windows", flush=True)

    counts = Counter(ytr)
    weights = np.array([1.0 / counts.get(i, 1) for i in range(8)])
    weights = weights / weights.sum() * 8
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = build_model(cfg["model"], **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(GES_DEPLOYED, map_location=device, weights_only=True))

    train_loader = DataLoader(WindowArrayDataset(Xtr, ytr),
                              batch_size=GES_CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(WindowArrayDataset(Xva, yva),
                            batch_size=GES_CFG["batch_size"], shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=GES_CFG["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=GES_CFG["lr"],
                            weight_decay=GES_CFG["weight_decay"])

    best_score, bad, best_state = -1.0, 0, None
    t0 = time.time()
    for epoch in range(1, GES_CFG["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                labels.extend(yb.tolist())
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_score:
            best_score, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= GES_CFG["patience"]:
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch}: val_macro_f1={f1:.4f} best={best_score:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    model.load_state_dict(best_state)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print(f"  [gesture fold{fold}] done, best val macro-F1={best_score:.4f} -> {out_ckpt.name} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return out_ckpt


# ── emotion ──────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.labels[i]


def load_emotion_split(clips):
    label_to_idx = {n: i for i, n in enumerate(EMOTION_LABELS)}
    images, labels = [], []
    for r in clips.itertuples():
        if r.emotion_masked or r.gt_emotion not in label_to_idx:
            continue
        p = CROPS_DIR / f"{r.clip_id}.npz"
        if not p.exists():
            continue
        crops = np.load(p)["crops"]
        if len(crops) == 0:
            continue
        images.append(crops)
        labels.extend([label_to_idx[r.gt_emotion]] * len(crops))
    images = np.concatenate(images, axis=0) if images else np.zeros((0, 240, 240, 3), np.uint8)
    return images, np.array(labels, np.int64)


def finetune_emotion_fold(clips, exclude_rows: set, fold: int, device):
    out_ckpt = EMO_DIR / "checkpoints" / f"finetuned_MobileNetV2_fold{fold}.pth"
    if out_ckpt.exists():
        print(f"  [emotion fold{fold}] already exists, skip: {out_ckpt.name}", flush=True)
        return out_ckpt

    emo = load_module("hri_emo_ft", EMO_DIR / "inference" / "video.py")
    tfm = load_module("hri_emo_transforms", EMO_DIR / "src" / "transforms.py", [EMO_DIR])

    tr_clips = clips[(clips.split == "train") & ~clips.v3_row.isin(exclude_rows)]
    va_clips = clips[clips.split == "val"]
    print(f"  [emotion fold{fold}] train scenarios={tr_clips.v3_row.nunique()} "
          f"(excluding {len(exclude_rows)}), loading crops...", flush=True)
    Xtr, ytr = load_emotion_split(tr_clips)
    Xva, yva = load_emotion_split(va_clips)
    print(f"    {len(ytr)} train images, {len(yva)} val images", flush=True)

    counts = Counter(ytr.tolist())
    weights = np.array([1.0 / counts.get(i, 1) for i in range(7)])
    weights = weights / weights.sum() * 7
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    train_loader = DataLoader(CropDataset(Xtr, ytr, tfm.get_train_transforms()),
                              batch_size=EMO_CFG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(Xva, yva, tfm.get_test_transforms()),
                            batch_size=EMO_CFG["batch_size"], shuffle=False, num_workers=0)

    model = emo.build_model().to(device)
    model.load_state_dict(torch.load(EMO_DEPLOYED, map_location=device, weights_only=True))
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=EMO_CFG["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=EMO_CFG["lr"],
                            weight_decay=EMO_CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EMO_CFG["epochs"])

    best_score, bad, best_state = -1.0, 0, None
    t0 = time.time()
    for epoch in range(1, EMO_CFG["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                labels.extend(yb.tolist())
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_score:
            best_score, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= EMO_CFG["patience"]:
                break
        print(f"    epoch {epoch}: val_macro_f1={f1:.4f} best={best_score:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    model.load_state_dict(best_state)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print(f"  [emotion fold{fold}] done, best val macro-F1={best_score:.4f} -> {out_ckpt.name} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return out_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["gesture", "emotion"], default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    folds = make_folds(clips)
    print(f"folds: fold0={len(folds[0])} scenarios, fold1={len(folds[1])} scenarios "
          f"-> {FOLD_SPLIT_PATH.relative_to(ROOT)}", flush=True)

    for fold in (0, 1):
        exclude = folds[fold]           # this fold's clips are HELD OUT of training
        if args.only in (None, "gesture"):
            finetune_gesture_fold(clips, exclude, fold, device)
        if args.only in (None, "emotion"):
            finetune_emotion_fold(clips, exclude, fold, device)

    print("\nAll fold checkpoints ready. Next: scripts/60_outfold_pools.py")


if __name__ == "__main__":
    main()
