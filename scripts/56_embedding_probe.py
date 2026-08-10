"""Sanity gate + purity check for the pooled penultimate embeddings
(2026-08-08). Two questions, one script:

1. **Pipeline correctness.** A simple linear probe fit on TRAIN embeddings,
   scored on train/val/test, should roughly recover the known per-modality
   accuracy numbers (`POOL_PURITY.md`'s argmax figures: gesture 0.864 test,
   emotion 0.762 test, motion 0.671 test). If it can't get anywhere close,
   that is a red flag for the extraction pipeline (`scripts/54`/`55`), not a
   finding about embeddings -- catch it here before building a fusion model
   on top.

2. **Does the pool-purity problem get WORSE in embedding space?**
   `POOL_PURITY.md` found argmax purity collapses train->test for the two
   fine-tuned-on-train cues (gesture 1.000->0.864, emotion 0.959->0.762) but
   not for motion (0.655->0.671, never fine-tuned). A linear probe trained on
   TRAIN embeddings and scored on TEST measures the same generalisation gap
   one level upstream of argmax -- if raw embeddings encode train-specific
   "fingerprints" beyond what the classifier head uses, the probe's
   train-vs-test gap should be even wider than the argmax gap.

    .venv/Scripts/python scripts/56_embedding_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.merged_unimodal import (  # noqa: E402
    GT_MAPS, LABELS, MASK_COL, load_clips)

EMBED_PATH = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled.npz"
OUT_DIR = ROOT / "results" / "realworld_eval_merged"
MOD_KEY = {"emo": "emotion", "ges": "gesture", "mot": "motion", "ctx": "context"}
FINETUNED = {"emo": True, "ges": True, "mot": False, "ctx": False}

# argmax-purity reference numbers from POOL_PURITY.md (scripts/53), for the
# side-by-side sanity comparison -- NOT recomputed here, just quoted.
ARGMAX_REF = {
    "emo": {"train": 0.9589, "val": 0.7524, "test": 0.7617},
    "ges": {"train": 1.0000, "val": 0.9966, "test": 0.8643},
    "mot": {"train": 0.6550, "val": 0.6708, "test": 0.6710},
}


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    z = np.load(EMBED_PATH, allow_pickle=True)
    clip_id = z["clip_id"].astype(str)
    clips = load_clips().set_index("clip_id")
    clips = clips.reindex(clip_id)                    # align row order to z

    results = {}
    for pref in ("emo", "ges", "mot"):
        modality = MOD_KEY[pref]
        emb = z[f"{pref}_embed"]
        obs = z[f"{pref}_obs"].astype(bool)
        gt_col = f"gt_{modality}"
        masked = clips[MASK_COL[modality]].to_numpy()
        gt = clips[gt_col].to_numpy(dtype=object)
        classes = LABELS[modality]

        usable = obs & ~masked & pd.notna(gt) & clips.split.notna().to_numpy()
        split = clips.split.to_numpy()

        Xtr = emb[usable & (split == "train")]
        ytr = gt[usable & (split == "train")]
        sets = {s: (emb[usable & (split == s)], gt[usable & (split == s)])
               for s in ("train", "val", "test")}

        print(f"[{modality}] n_train={len(Xtr)} n_val={len(sets['val'][0])} "
              f"n_test={len(sets['test'][0])}", flush=True)

        t0 = time.time()
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        probe = {}
        for s, (X, y) in sets.items():
            if len(X) == 0:
                probe[s] = None
                continue
            pred = clf.predict(X)
            probe[s] = {
                "acc": round(float(accuracy_score(y, pred)), 4),
                "macro_f1": round(float(f1_score(y, pred, average="macro",
                                                 labels=classes, zero_division=0)), 4),
                "n": int(len(X)),
            }
        print(f"  probe: train={probe['train']['acc']} val={probe['val']['acc']} "
              f"test={probe['test']['acc']}  ({time.time()-t0:.1f}s)", flush=True)

        gap = round(probe["train"]["acc"] - probe["test"]["acc"], 4)
        argmax_gap = round(ARGMAX_REF[pref]["train"] - ARGMAX_REF[pref]["test"], 4)
        results[modality] = {
            "finetuned_on_train": FINETUNED[pref],
            "probe": probe,
            "probe_train_minus_test_gap": gap,
            "argmax_train_minus_test_gap_reference": argmax_gap,
            "gap_worse_than_argmax": bool(gap > argmax_gap),
        }

    (OUT_DIR / "embedding_probe_results.json").write_text(json.dumps(results, indent=2))
    write_report(results)
    print(f"\n-> {OUT_DIR / 'EMBEDDING_PROBE.md'}")


def write_report(results: dict) -> None:
    lines = [
        "# Embedding probe — pipeline sanity check + purity-gap comparison",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · linear probe "
        "(`LogisticRegression`) fit on TRAIN pooled embeddings, scored on "
        "train/val/test.",
        "",
        "## 1. Pipeline sanity — does the probe recover known accuracy?", "",
        "| Modality | Probe train | Probe val | Probe test | Argmax test (reference) |",
        "|---|---|---|---|---|",
    ]
    for m, r in results.items():
        p = r["probe"]
        ref = ARGMAX_REF[{"emotion": "emo", "gesture": "ges", "motion": "mot"}[m]]["test"]
        lines.append(f"| {m} | {p['train']['acc']:.4f} | {p['val']['acc']:.4f} | "
                     f"{p['test']['acc']:.4f} | {ref:.4f} |")
    lines += [
        "",
        "Close agreement between probe-test and argmax-test confirms the "
        "extraction pipeline (`scripts/54`/`55`) produced embeddings that "
        "carry the same classification signal as the deployed classifier "
        "heads -- a large mismatch here would point at a pipeline bug, not a "
        "property of embeddings.",
        "",
        "## 2. Does the train/test gap widen in embedding space?", "",
        "| Modality | Fine-tuned on train? | Probe train−test gap | Argmax train−test gap (ref) | Worse? |",
        "|---|---|---|---|---|",
    ]
    for m, r in results.items():
        lines.append(f"| {m} | {'**yes**' if r['finetuned_on_train'] else 'no'} | "
                     f"{r['probe_train_minus_test_gap']:+.4f} | "
                     f"{r['argmax_train_minus_test_gap_reference']:+.4f} | "
                     f"{'**YES**' if r['gap_worse_than_argmax'] else 'no'} |")
    lines += [
        "",
        "If gesture/emotion (fine-tuned on train) show a probe gap wider than "
        "their argmax gap, embedding-level recombination pools inherit "
        "`POOL_PURITY.md`'s contamination at least as badly as probability "
        "pools, likely worse given the extra capacity a 1280/128-d vector "
        "has to encode train-specific detail a 7/8-way softmax cannot.",
    ]
    (OUT_DIR / "EMBEDDING_PROBE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
