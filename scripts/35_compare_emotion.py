"""Compare the DEPLOYED vs FINE-TUNED emotion checkpoint on the SAME cached
test-split face crops -- isolates the checkpoint as the only variable (no
windowing/aggregation differences to confound the comparison, unlike motion's
first attempt which compared through two different pipelines).

Reports headline (test, headline_eval), by context, by emotion class, and the
`curated_clip` vs `raw_take` split (fine-tune-overlap vs genuinely unseen) --
same structure as the motion diagnosis so the two are directly comparable.

    .venv/Scripts/python scripts/35_compare_emotion.py
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
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval.merged_unimodal import load_clips  # noqa: E402

EMO_DIR = ROOT / "modalities" / "emotion"
CROPS_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"
DEPLOYED_CKPT = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2.pth"
NEW_CKPT = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2_merged.pth"
OUT_DIR = ROOT / "results" / "realworld_eval_merged"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]


@torch.no_grad()
def predict_clip_means(clips, model, tfm, device, batch=128):
    """One mean-softmax prediction per clip from its cached crops."""
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
    emo = load_module("hri_emo_compare", EMO_DIR / "inference" / "video.py")
    tfm = load_module("hri_emo_transforms2", EMO_DIR / "src" / "transforms.py",
                      [EMO_DIR])
    transform = tfm.get_test_transforms()

    clips = load_clips()
    clips = clips[clips.v3_row.notna() & ~clips.emotion_masked & clips.gt_emotion.notna()]
    te = clips[clips.split == "test"]
    print(f"{len(te)} test clips with an emotion target", flush=True)

    results = {}
    for tag, ckpt in [("before_deployed", DEPLOYED_CKPT), ("after_finetuned", NEW_CKPT)]:
        model = emo.build_model().to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        pred = predict_clip_means(te, model, transform, device)
        m = pred.merge(te[["clip_id", "gt_emotion", "context", "headline_eval",
                           "source"]], on="clip_id")
        m["hit"] = m.pred == m.gt_emotion
        hl = m[m.headline_eval]
        results[tag] = {
            "headline": score(hl.gt_emotion, hl.pred),
            "by_context": {c: score(g.gt_emotion, g.pred) for c, g in hl.groupby("context")},
            "by_class": {c: score(g.gt_emotion, g.pred) for c, g in hl.groupby("gt_emotion")},
            "curated_clip": score(m[m.source == "curated_clip"].gt_emotion,
                                  m[m.source == "curated_clip"].pred),
            "raw_take": score(m[m.source.astype(str).str.startswith("raw_take")].gt_emotion,
                              m[m.source.astype(str).str.startswith("raw_take")].pred),
        }
        results[tag]["_df"] = m
        print(f"\n{tag}: headline acc={results[tag]['headline']['acc']} "
              f"macroF1={results[tag]['headline']['macro_f1']}")

    print(f"\n{'metric':24}{'BEFORE':>10}{'AFTER':>10}{'DELTA':>9}")
    for label, key in [("headline acc", ("headline", "acc")),
                       ("headline macro-F1", ("headline", "macro_f1")),
                       ("test curated_clip acc", ("curated_clip", "acc")),
                       ("test raw_take acc", ("raw_take", "acc"))]:
        b = results["before_deployed"][key[0]][key[1]]
        a = results["after_finetuned"][key[0]][key[1]]
        d = round(a - b, 4) if b is not None and a is not None else None
        arrow = "  ^" if (d or 0) > 0.001 else ("  v" if (d or 0) < -0.001 else "")
        print(f"{label:24}{b:>10}{a:>10}{d:>9}{arrow}")

    print(f"\n{'class':10}{'BEFORE acc':>12}{'AFTER acc':>11}{'DELTA':>9}")
    for c in EMOTION_LABELS:
        b = results["before_deployed"]["by_class"].get(c, {}).get("acc")
        a = results["after_finetuned"]["by_class"].get(c, {}).get("acc")
        d = round(a - b, 4) if b is not None and a is not None else None
        print(f"{c:10}{b if b is not None else '-':>12}{a if a is not None else '-':>11}"
             f"{d if d is not None else '-':>9}")

    print(f"\n{'context':10}{'BEFORE acc':>12}{'AFTER acc':>11}")
    for c in ("classroom", "kitchen"):
        b = results["before_deployed"]["by_context"].get(c, {}).get("acc")
        a = results["after_finetuned"]["by_context"].get(c, {}).get("acc")
        print(f"{c:10}{b if b is not None else '-':>12}{a if a is not None else '-':>11}")

    out = {k: {kk: vv for kk, vv in v.items() if kk != "_df"} for k, v in results.items()}
    (OUT_DIR / "emotion_compare.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT_DIR / 'emotion_compare.json'}")

    h = results["after_finetuned"]["headline"]
    with start_run("03_diagnostics", "final_merged__unimodal_emotion_finetuned",
                   dataset="final_merged", split_kind="scenarios", cues="real",
                   params={"model": "emotion", "checkpoint": "finetuned_merged"},
                   notes="after fine-tune on complete dataset; compare to "
                        "final_merged__unimodal_emotion") as run:
        run.log_metrics({"headline_clip_acc": h["acc"], "headline_clip_macro_f1": h["macro_f1"],
                         "test_raw_take_acc": results["after_finetuned"]["raw_take"]["acc"]})


if __name__ == "__main__":
    main()
