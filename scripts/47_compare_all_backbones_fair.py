"""Final 4-way backbone proof table for the methodology panel: RAF-DB-only vs
real-data-fine-tuned accuracy, same protocol for every backbone, evaluated on
the same held-out real-world test clips used everywhere else in this project.

Each backbone follows the identical path:
  ImageNet-pretrained -> best_<Arch>.pth (RAF-DB Stage 2, scripts/train.py)
  -> backbonecmp_<Arch>.pth (real-data fine-tune, scripts/46_finetune_emotion_backbone.py)
  -> scored here on data/final_merged's `split=="test"`, `headline_eval` clips.

    .venv/Scripts/python scripts/47_compare_all_backbones_fair.py
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

ARCHES = ["MobileNetV2", "MobileNetV3-Large", "EfficientNet-B0", "MNASNet1_0"]


def safe_name(name):
    return name.replace(" ", "_").replace("-", "_")


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

    sys.path.insert(0, str(EMO_DIR))
    from src.models import build_model as build_backbone, count_params, model_size_mb  # noqa: E402
    sys.path.remove(str(EMO_DIR))

    tfm = load_module("hri_emo_transforms_fair", EMO_DIR / "src" / "transforms.py",
                      [EMO_DIR])
    transform = tfm.get_test_transforms()

    clips = load_clips()
    clips = clips[clips.v3_row.notna() & ~clips.emotion_masked & clips.gt_emotion.notna()]
    te = clips[clips.split == "test"]
    print(f"{len(te)} test clips with an emotion target\n")

    results = {}
    for arch in ARCHES:
        sname = safe_name(arch)
        row = {"arch": arch}
        model = build_backbone(arch)
        row["params_m"] = round(count_params(model) / 1e6, 2)
        row["size_mb"] = round(model_size_mb(model), 2)

        for tag, ckpt_name in [("rafdb_only", f"best_{sname}.pth"),
                               ("finetuned_real", f"backbonecmp_{sname}.pth")]:
            ckpt = EMO_DIR / "checkpoints" / ckpt_name
            if not ckpt.exists():
                print(f"[skip] {arch}/{tag}: {ckpt_name} not found")
                continue
            model = build_backbone(arch).to(device).eval()
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            pred = predict_clip_means(te, model, transform, device)
            m = pred.merge(te[["clip_id", "gt_emotion", "context", "headline_eval",
                               "source"]], on="clip_id")
            hl = m[m.headline_eval]
            s = score(hl.gt_emotion, hl.pred)
            row[f"{tag}_acc"] = s["acc"]
            row[f"{tag}_macro_f1"] = s["macro_f1"]
            row[f"{tag}_n"] = s["n"]
            print(f"{arch:20}{tag:16}  acc={s['acc']:.4f}  macroF1={s['macro_f1']:.4f}")
        results[arch] = row

    df = pd.DataFrame(results.values())
    print(f"\n{df.to_string(index=False)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "emotion_backbone_fair_comparison.csv", index=False)
    (OUT_DIR / "emotion_backbone_fair_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT_DIR / 'emotion_backbone_fair_comparison.csv'}")


if __name__ == "__main__":
    main()
