"""Per-intent breakdown of the deployed fusion recipe on `data/final_merged`.

`scripts/30_merged_recombination.py` reports headline accuracy / macro-F1 for
the four-way ablation but keeps no per-class record, so the thesis had no way
to say *which* intents the final model gets wrong. This script re-runs the
winning `full` config through the identical code path (same loader, same
recombination pools, same `train_fusion` call, same seeds, same clip-level
evaluation) and additionally writes:

  * a pooled 3-seed confusion matrix over the headline test clips,
  * per-intent precision / recall / F1 / support (mean over seeds),
  * the same headline acc / macro-F1 the ablation reports, as a check that this
    run reproduces `RECOMBINATION.md` rather than measuring something else.

It adds no new modelling decision. Outputs go to
`results/realworld_eval_merged/fusion_confusion_full.{json,md}`.

    .venv/Scripts/python scripts/41_fusion_confusion.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR

# the winning row of the four-way ablation in scripts/30_merged_recombination.py
FULL = dict(dropout_p=0.3, jitter_sigma=0.15, recombine=True, select_masked=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--seed-pool", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    print(f"{clips.clip_id.nunique()} clips · headline test clips: {len(hl)}",
          flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    X, obs, y, rep = generate(pools, n_per_combo=args.n_per_combo,
                             seed=args.seed_pool)
    print(f"recombination: {rep.n_generated} synthetic samples", flush=True)

    y_true = hl.y.to_numpy()
    preds, per_seed = [], []
    for seed in SEEDS:
        t0 = time.time()
        model, val_acc = T.train_fusion(
            splits, seed=seed, dropout_p=FULL["dropout_p"],
            jitter_sigma=FULL["jitter_sigma"], extra=(X, obs, y), device=device,
            missing_mode="exclude", select_masked=FULL["select_masked"])
        pred = T._eval_arrays(model, *T.frame_arrays(hl), device)
        preds.append(pred)
        per_seed.append(G.eval_clip(y_true, pred))
        print(f"  full seed{seed}: val={val_acc:.4f} "
              f"headline={per_seed[-1]} ({time.time()-t0:.0f}s)", flush=True)

    headline = G.agg(per_seed)
    print("\nheadline (should reproduce RECOMBINATION.md `full`):", headline)

    # ── per-intent metrics, mean over seeds ────────────────────────────────
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    present = sorted(set(y_true.tolist()))
    names = [common.INTENTS[i] for i in present]

    cm = np.zeros((len(present), len(present)), dtype=int)
    for pred in preds:
        cm += confusion_matrix(y_true, pred, labels=present)

    rows = []
    for pred in preds:
        p, r, f, s = precision_recall_fscore_support(
            y_true, pred, labels=present, zero_division=0)
        rows.append(np.stack([p, r, f, s]))
    stack = np.stack(rows)                       # [seed, 4, class]
    mean, std = stack.mean(0), stack.std(0)

    per_class = {
        n: {"precision": round(float(mean[0, i]), 4),
            "precision_std": round(float(std[0, i]), 4),
            "recall": round(float(mean[1, i]), 4),
            "recall_std": round(float(std[1, i]), 4),
            "f1": round(float(mean[2, i]), 4),
            "f1_std": round(float(std[2, i]), 4),
            "support": int(mean[3, i])}
        for i, n in enumerate(names)
    }

    out = {
        "config": "full (dropout 0.3 + jitter 0.15 + recombination)",
        "model": "attention_fusion (self-attention, 70,090 params)",
        "seeds": list(SEEDS),
        "n_per_combo": args.n_per_combo,
        "n_synthetic": rep.n_generated,
        "eval": "headline test clips, clip-level mean-pooled cues",
        "headline": headline,
        "per_seed": per_seed,
        "intents_present": names,
        "confusion_pooled_3seeds": cm.tolist(),
        "per_class": per_class,
        "note": ("macro-F1 elsewhere in this project is computed over all 10 "
                 "intent labels while final_merged contains only 9 (F09 was "
                 "removed 2026-07-31), so those macro-F1 values are capped at "
                 "0.900. The per-class table here is over the 9 present "
                 "classes only."),
    }
    (OUT_DIR / "fusion_confusion_full.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # ── markdown companion ────────────────────────────────────────────────
    cmdf = pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=names)
    pcdf = pd.DataFrame(per_class).T[["precision", "recall", "f1", "support"]]
    lines = [
        "# Fusion per-intent breakdown — `full` recipe on `data/final_merged`",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(SEEDS)} seeds · "
        f"headline test clips (n={headline['n']}) · same code path as "
        "`scripts/30_merged_recombination.py`'s winning `full` config.",
        "",
        f"Headline clip acc **{headline['acc_mean']} ± {headline['acc_std']}**, "
        f"macro-F1 **{headline['macro_f1_mean']} ± {headline['macro_f1_std']}** "
        "— reproduces `RECOMBINATION.md`.",
        "",
        "## Per intent (mean over 3 seeds, 9 present classes)", "",
        pcdf.to_markdown(),
        "",
        "## Confusion matrix (rows = truth, pooled over 3 seeds)", "",
        cmdf.to_markdown(),
    ]
    (OUT_DIR / "FUSION_CONFUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'fusion_confusion_full.json'}")
    print(f"-> {OUT_DIR / 'FUSION_CONFUSION.md'}")


if __name__ == "__main__":
    main()
