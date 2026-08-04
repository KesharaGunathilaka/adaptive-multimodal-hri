"""Fine-tune the emotion MobileNetV2 on the COMPLETE `data/final_merged` dataset.

Warm-starts from the DEPLOYED checkpoint (`finetuned_MobileNetV2.pth`), not
from the RAF-DB-only base -- unlike motion's first (NTU-base) attempt, this
mirrors what worked comparably for motion, and re-doing the RAF-DB->real
domain adaptation from scratch would throw away already-useful features for
no evidenced benefit.

**Loss weighting, not resampling** (deliberate difference from the motion
recipe): emotion's ORIGINAL recipe class-weights the loss
(`src/data.py::compute_class_weights`), it does not resample/duplicate
minority-class examples. 2026-08-04 finding: motion's `WeightedRandomSampler`
(exact-duplicate oversampling of the rarest class) coincided with a stepping_back
generalisation collapse (val 66%, test 0%) -- consistent with overfitting to a
few duplicated hard examples. Loss-weighting nudges gradients without
duplicating any image, a gentler intervention. If emotion shows the same
val-up/test-down pattern anyway, that would isolate the narrow actor-disjoint
val set (not the sampler) as the real cause -- see script 36's diagnosis.

Trains on face crops cached by `scripts/33_extract_emotion_crops.py` (already
decoded once; no video I/O here). Train/val = the `split` column
(actor-disjoint, P04+P03 in val). Uses every cached crop as an independent
sample (~15/clip) with the SAME train-time augmentation as the original recipe
(`modalities/emotion/src/transforms.py`).

    .venv/Scripts/python scripts/34_finetune_emotion.py
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
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval.merged_unimodal import load_clips  # noqa: E402

EMO_DIR = ROOT / "modalities" / "emotion"
CROPS_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"
DEPLOYED_CKPT = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2.pth"
OUT_CKPT = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2_merged.pth"
LOG_PATH = EMO_DIR / "logs" / "training_log_finetuned_merged.json"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
LABEL_TO_IDX = {n: i for i, n in enumerate(EMOTION_LABELS)}

CONFIG = {"batch_size": 64, "epochs": 25, "lr": 0.00003, "weight_decay": 1e-5,
         "patience": 8, "label_smoothing": 0.1}


class CropDataset(Dataset):
    """(image[240,240,3] uint8, label_idx) pairs, transform applied lazily."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pil = Image.fromarray(self.images[i])
        return self.transform(pil), self.labels[i]


def load_split(clips, split_name):
    sub = clips[(clips.split == split_name) & ~clips.emotion_masked
               & clips.gt_emotion.notna()]
    images, labels = [], []
    missing = 0
    for r in sub.itertuples():
        p = CROPS_DIR / f"{r.clip_id}.npz"
        if not p.exists():
            missing += 1
            continue
        z = np.load(p)
        crops = z["crops"]
        if len(crops) == 0:
            continue
        label = LABEL_TO_IDX[r.gt_emotion]
        images.append(crops)
        labels.extend([label] * len(crops))
    if missing:
        print(f"  ({missing} {split_name} clips missing a crop cache)")
    images = np.concatenate(images, axis=0) if images else np.zeros((0, 240, 240, 3), np.uint8)
    return images, np.array(labels, dtype=np.int64)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\n")

    emo = load_module("hri_emo_ft", EMO_DIR / "inference" / "video.py")
    tfm = load_module("hri_emo_transforms", EMO_DIR / "src" / "transforms.py",
                      [EMO_DIR])

    clips = load_clips()
    clips = clips[clips.v3_row.notna()]

    print("Loading TRAIN crops...")
    Xtr, ytr = load_split(clips, "train")
    print(f"  {len(ytr)} train images from {len(set(ytr))} classes")
    print("Loading VAL crops...")
    Xva, yva = load_split(clips, "val")
    print(f"  {len(yva)} val images\n")

    counts = Counter(ytr.tolist())
    print("class distribution (train):")
    for i, name in enumerate(EMOTION_LABELS):
        print(f"  {name:<10}: {counts.get(i, 0):>6,}")

    train_ds = CropDataset(Xtr, ytr, tfm.get_train_transforms())
    val_ds = CropDataset(Xva, yva, tfm.get_test_transforms())
    # num_workers=0: the train/test transforms come from a dynamically loaded
    # module (fusion/extraction/modloader.py's load_module), whose class is
    # not importable in a spawned worker process on Windows -- num_workers>0
    # fails with "pickle data was truncated". Dataset is fully in-memory
    # already, so single-process loading is not a real bottleneck here.
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)

    weights = np.array([1.0 / counts.get(i, 1) for i in range(7)])
    weights = weights / weights.sum() * 7
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"\nclass weights: {dict(zip(EMOTION_LABELS, weights.round(2)))}")

    print(f"\nWarm-starting from {DEPLOYED_CKPT.name}")
    model = emo.build_model().to(device)
    model.load_state_dict(torch.load(DEPLOYED_CKPT, map_location=device, weights_only=True))
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
        score = val_f1               # select on macro-F1, matching the original recipe
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
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_CKPT)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps({"config": CONFIG, "history": history}, indent=2))

    val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
    print(f"\nbest val macro-F1={best_score:.3f} (acc={val_acc:.3%})")
    print(classification_report(val_labels, val_preds, target_names=EMOTION_LABELS,
                                digits=3, zero_division=0))
    print(confusion_matrix(val_labels, val_preds))
    print(f"\n-> {OUT_CKPT}")

    with start_run("02_fusion", "emotion__finetune_merged", dataset="final_merged",
                   split_kind="scenarios", cues="real",
                   params={"model": "emotion_mobilenetv2", **CONFIG,
                           "warm_start": "deployed", "loss_weighting": "class_weighted_ce",
                           "n_train_images": len(ytr), "n_val_images": len(yva)},
                   notes="fine-tune on complete dataset face crops; "
                        "see script 35 for before/after test comparison") as run:
        run.log_metrics({"val_acc": val_acc, "val_macro_f1": best_score,
                         "best_epoch": history[-1]["epoch"]})
        run.log_checkpoint(OUT_CKPT)


if __name__ == "__main__":
    main()
