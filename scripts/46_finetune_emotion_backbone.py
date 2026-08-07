"""Fine-tune an arbitrary emotion backbone on real face crops, for a fair
4-way backbone ablation (methodology panel proof: does the Stage-1 RAF-DB
winner also win after everyone gets the same real-data adaptation step?).

Unlike `scripts/34_finetune_emotion.py` (which warm-starts MobileNetV2 from
the already-deployed, already-once-fine-tuned checkpoint), this warm-starts
every backbone from its OWN plain RAF-DB checkpoint (`best_<Arch>.pth`), so
all four models get the identical single fine-tune pass on the identical
`data/final_merged` train split with the identical recipe. That symmetry is
the point: it isolates backbone choice as the only variable.

    .venv/Scripts/python scripts/46_finetune_emotion_backbone.py --model "MobileNetV3-Large"
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from scripts.realworld_eval.merged_unimodal import load_clips  # noqa: E402

EMO_DIR = ROOT / "modalities" / "emotion"
CROPS_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
LABEL_TO_IDX = {n: i for i, n in enumerate(EMOTION_LABELS)}

CONFIG = {"batch_size": 64, "epochs": 25, "lr": 0.00003, "weight_decay": 1e-5,
         "patience": 8, "label_smoothing": 0.1}


def safe_name(name):
    return name.replace(" ", "_").replace("-", "_")


class LazyCropDataset(Dataset):
    """Indexes (clip_id, crop_idx, label) triples; loads pixels from the npz
    cache on access instead of holding every crop in RAM at once. The full
    real-world crop cache is ~4 GB uncompressed -- concatenating it eagerly
    (the original approach) spiked a single process to ~5 GB and risked OOM
    on this machine's 15.5 GB. Caches the last-opened file since DataLoader
    batches often touch the same clip's crops consecutively."""

    def __init__(self, index, transform):
        self.index = index  # list of (clip_id, crop_idx, label)
        self.transform = transform
        self._cache_id = None
        self._cache_arr = None

    def __len__(self):
        return len(self.index)

    def _crops_for(self, clip_id):
        if self._cache_id != clip_id:
            self._cache_arr = np.load(CROPS_DIR / f"{clip_id}.npz")["crops"]
            self._cache_id = clip_id
        return self._cache_arr

    def __getitem__(self, i):
        clip_id, crop_idx, label = self.index[i]
        pil = Image.fromarray(self._crops_for(clip_id)[crop_idx])
        return self.transform(pil), label


def build_index(clips, split_name):
    sub = clips[(clips.split == split_name) & ~clips.emotion_masked
               & clips.gt_emotion.notna()]
    index, missing = [], 0
    for r in sub.itertuples():
        p = CROPS_DIR / f"{r.clip_id}.npz"
        if not p.exists():
            missing += 1
            continue
        n_crops = np.load(p)["crops"].shape[0]
        if n_crops == 0:
            continue
        label = LABEL_TO_IDX[r.gt_emotion]
        index.extend((r.clip_id, k, label) for k in range(n_crops))
    if missing:
        print(f"  ({missing} {split_name} clips missing a crop cache)")
    return index


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(yb.tolist())
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, macro_f1, preds, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["MobileNetV2", "MobileNetV3-Large", "EfficientNet-B0", "MNASNet1_0"])
    args = ap.parse_args()
    arch = args.model
    sname = safe_name(arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={arch}\n")

    sys.path.insert(0, str(EMO_DIR))
    from src.models import build_model as build_backbone  # noqa: E402
    sys.path.remove(str(EMO_DIR))

    tfm = load_module("hri_emo_transforms_ft", EMO_DIR / "src" / "transforms.py",
                      [EMO_DIR])

    warm_ckpt = EMO_DIR / "checkpoints" / f"best_{sname}.pth"
    out_ckpt = EMO_DIR / "checkpoints" / f"backbonecmp_{sname}.pth"
    if not warm_ckpt.exists():
        raise SystemExit(f"missing {warm_ckpt} -- run scripts/train.py --model \"{arch}\" first")

    clips = load_clips()
    clips = clips[clips.v3_row.notna()]

    print("Indexing TRAIN crops...")
    train_index = build_index(clips, "train")
    print(f"  {len(train_index)} train images")
    print("Indexing VAL crops...")
    val_index = build_index(clips, "val")
    print(f"  {len(val_index)} val images\n")

    counts = Counter(label for _, _, label in train_index)
    print("class distribution (train):")
    for i, name in enumerate(EMOTION_LABELS):
        print(f"  {name:<10}: {counts.get(i, 0):>6,}")

    train_ds = LazyCropDataset(train_index, tfm.get_train_transforms())
    val_ds = LazyCropDataset(val_index, tfm.get_test_transforms())
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)

    weights = np.array([1.0 / counts.get(i, 1) for i in range(7)])
    weights = weights / weights.sum() * 7
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(f"\nWarm-starting from {warm_ckpt.name} (RAF-DB checkpoint, same for every arch)")
    model = build_backbone(arch).to(device)
    model.load_state_dict(torch.load(warm_ckpt, map_location=device, weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} params")

    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=CONFIG["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])

    best_score, patience_ctr, history, best_state = -1.0, 0, [], None
    print(f"\n{'epoch':>6}{'train_loss':>12}{'train_acc':>11}{'val_acc':>9}{'val_f1':>9}")
    t0 = time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += len(yb)
        sched.step()
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        score = val_f1
        note = ""
        if score > best_score:
            best_score, patience_ctr = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            note = "<- best"
        else:
            patience_ctr += 1
        history.append({"epoch": epoch, "train_loss": round(tr_loss / len(train_loader), 4),
                        "train_acc": round(tr_correct / tr_total, 4),
                        "val_acc": round(val_acc, 4), "val_macro_f1": round(val_f1, 4)})
        print(f"{epoch:>6}{tr_loss/len(train_loader):>12.4f}"
              f"{tr_correct/tr_total:>10.2%}{val_acc:>9.2%}{val_f1:>9.3f}  {note}"
              f"  ({time.time()-t0:.0f}s)")
        if patience_ctr >= CONFIG["patience"]:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    log_path = EMO_DIR / "logs" / f"training_log_backbonecmp_{sname}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"model": arch, "config": CONFIG, "history": history}, indent=2))

    val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
    print(f"\nbest val macro-F1={best_score:.3f} (acc={val_acc:.3%})")
    print(classification_report(val_labels, val_preds, target_names=EMOTION_LABELS,
                                digits=3, zero_division=0))
    print(confusion_matrix(val_labels, val_preds))
    print(f"\n-> {out_ckpt}")


if __name__ == "__main__":
    main()
