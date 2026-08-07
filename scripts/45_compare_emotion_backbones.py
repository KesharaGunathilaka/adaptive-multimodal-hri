"""Head-to-head backbone comparison for the methodology write-up: does
EfficientNet-B0 (winner of the Stage-1 RAF-DB-only search) actually beat the
deployed MobileNetV2 on OUR real-world footage, once both are fine-tuned on
real face crops?

Same protocol as `scripts/35_compare_emotion.py` (clip-level mean-softmax over
cached crops, scored on the held-out `test` split, `headline_eval` rows) so
numbers are directly comparable to the deployed MobileNetV2 figures already
reported. Generalized to load an arbitrary torchvision backbone via
`modalities/emotion/src/models.py::build_model(name)` instead of the
MobileNetV2-only `inference/video.py::build_model()`.

    .venv/Scripts/python scripts/45_compare_emotion_backbones.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from scripts.realworld_eval.merged_unimodal import load_clips  # noqa: E402

EMO_DIR = ROOT / "modalities" / "emotion"
CROPS_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"
OUT_DIR = ROOT / "results" / "realworld_eval_merged"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]

# (tag, architecture name in src/models.py, checkpoint filename)
CANDIDATES = [
    ("MobileNetV2_RAFDB_only", "MobileNetV2", "best_MobileNetV2.pth"),
    ("MobileNetV2_finetuned_DEPLOYED", "MobileNetV2", "finetuned_MobileNetV2.pth"),
    ("EfficientNetB0_RAFDB_only", "EfficientNet-B0", "best_EfficientNet_B0.pth"),
    ("EfficientNetB0_finetuned", "EfficientNet-B0", "finetuned_EfficientNet_B0.pth"),
    ("EfficientNetB0_finetuned_v2", "EfficientNet-B0", "finetuned_v2_EfficientNet_B0.pth"),
    ("MobileNetV3Large_RAFDB_only", "MobileNetV3-Large", "best_MobileNetV3_Large.pth"),
    ("MobileNetV3Large_finetuned", "MobileNetV3-Large", "backbonecmp_MobileNetV3_Large.pth"),
]


@torch.no_grad()
def predict_clip_means(clips, model, tfm, device):
    from PIL import Image
    rows = []
    for r in clips.itertuples():
        p = CROPS_DIR / f"{r.clip_id}.npz"
        if not p.exists():
            continue
        crops = np.load(p)["crops"]
        if len(crops) == 0:
            continue
        imgs = torch.stack([tfm(Image.fromarray(c)) for c in crops]).to(device)
        probs = torch.softmax(model(imgs), dim=1).mean(0).cpu().numpy()
        rows.append({"clip_id": r.clip_id, "pred": EMOTION_LABELS[int(probs.argmax())],
                    "n_crops": len(crops)})
    return pd.DataFrame(rows)


def score(y_true, y_pred):
    if len(y_true) == 0:
        return {"n": 0, "acc": None, "macro_f1": None}
    return {"n": int(len(y_true)), "acc": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\n")

    # Import the backbone registry directly (needs modalities/emotion on
    # sys.path for its own `from config import NUM_CLASSES`); grab the
    # function reference before modloader purges the generic "src"/"config"
    # module names for the transforms load below.
    sys.path.insert(0, str(EMO_DIR))
    from src.models import build_model as build_backbone  # noqa: E402
    sys.path.remove(str(EMO_DIR))

    tfm = load_module("hri_emo_transforms3", EMO_DIR / "src" / "transforms.py",
                      [EMO_DIR])
    transform = tfm.get_test_transforms()

    clips = load_clips()
    clips = clips[clips.v3_row.notna() & ~clips.emotion_masked & clips.gt_emotion.notna()]
    te = clips[clips.split == "test"]
    print(f"{len(te)} test clips with an emotion target\n")

    results = {}
    for tag, arch, ckpt_name in CANDIDATES:
        ckpt = EMO_DIR / "checkpoints" / ckpt_name
        if not ckpt.exists():
            print(f"[skip] {tag}: {ckpt_name} not found")
            continue
        model = build_backbone(arch).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        pred = predict_clip_means(te, model, transform, device)
        m = pred.merge(te[["clip_id", "gt_emotion", "context", "headline_eval",
                           "source"]], on="clip_id")
        hl = m[m.headline_eval]
        results[tag] = {
            "arch": arch, "checkpoint": ckpt_name,
            "headline": score(hl.gt_emotion, hl.pred),
            "by_class": {c: score(g.gt_emotion, g.pred) for c, g in hl.groupby("gt_emotion")},
            "curated_clip": score(m[m.source == "curated_clip"].gt_emotion,
                                  m[m.source == "curated_clip"].pred),
            "raw_take": score(m[m.source.astype(str).str.startswith("raw_take")].gt_emotion,
                              m[m.source.astype(str).str.startswith("raw_take")].pred),
        }
        h = results[tag]["headline"]
        print(f"{tag:34} n={h['n']:>4}  acc={h['acc']:.4f}  macroF1={h['macro_f1']:.4f}")

    print(f"\n{'model':34}{'headline acc':>14}{'headline macroF1':>18}")
    for tag, r in results.items():
        h = r["headline"]
        print(f"{tag:34}{h['acc']:>14.4f}{h['macro_f1']:>18.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "emotion_backbone_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT_DIR / 'emotion_backbone_comparison.json'}")


if __name__ == "__main__":
    main()
