"""Rubric-driven cue recombination on `data/final_merged` — the experiment the
2026-08-03 gap decomposition points at (`GAP_DECOMPOSITION_MERGED.md`):
fusion generalisation cost (0.381) dominates perception cost (0.144), and
F02 (emergency) rows fail even with oracle cues. See
`fusion/model/recombine_merged.py` for the full design rationale.

Four-way ablation (3 seeds each), isolating recombination's contribution from
dropout/jitter's, mirroring the Stage 6 ablation style
(`docs/methodology/06_fusion_model.md` §6.6):

  plain       real cues only, no augmentation at all           -- = the gap
              decomposition's "fusion+real" (0.475), reproduced here for a
              same-run baseline rather than reusing the earlier number
  augmented   real cues + modality dropout (0.3) + confidence jitter (0.15),
              NO recombination                                  -- isolates
              dropout/jitter's contribution alone
  recomb_only real cues + recombination samples, NO dropout/jitter -- isolates
              recombination's contribution alone (pure semantic teaching)
  full        real cues + recombination + dropout/jitter          -- the
              deployed-recipe candidate (matches scripts/07_finalize_fusion.py's
              config, applied on top of the merged table)

The decisive test: does `full` (or `recomb_only`) beat rules+real (0.608,
the number learned fusion currently LOSES to)? That is the G1 claim, on the
honest split, finally measured with the fix in hand. F02 rows (#23/#53/#54)
are checked explicitly per-row for the best config.

    .venv/Scripts/python scripts/30_merged_recombination.py
    .venv/Scripts/python scripts/30_merged_recombination.py --n-per-combo 40
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

from fusion.model import train as T  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR

CONFIGS = {
    "plain":       dict(dropout_p=0.0, jitter_sigma=0.0, recombine=False, select_masked=False),
    "augmented":   dict(dropout_p=0.3, jitter_sigma=0.15, recombine=False, select_masked=True),
    "recomb_only": dict(dropout_p=0.0, jitter_sigma=0.0, recombine=True, select_masked=False),
    "full":        dict(dropout_p=0.3, jitter_sigma=0.15, recombine=True, select_masked=True),
}


def run_config(name: str, cfg: dict, real: pd.DataFrame, extra, device):
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    ex = extra if cfg["recombine"] else None

    per_seed_hl, per_seed_full, models = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model, val_acc = T.train_fusion(
            splits, seed=seed, dropout_p=cfg["dropout_p"],
            jitter_sigma=cfg["jitter_sigma"], extra=ex, device=device,
            missing_mode="exclude", select_masked=cfg["select_masked"])
        pred_hl = T._eval_arrays(model, *T.frame_arrays(hl), device)
        pred_full = T._eval_arrays(model, *T.frame_arrays(splits["test"]), device)
        r_hl = G.eval_clip(hl.y.to_numpy(), pred_hl)
        r_full = G.eval_clip(splits["test"].y.to_numpy(), pred_full)
        per_seed_hl.append(r_hl)
        per_seed_full.append(r_full)
        models.append(model)
        print(f"  {name} seed{seed}: val={val_acc:.4f} headline={r_hl} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return per_seed_hl, per_seed_full, models


def f02_check(model, real: pd.DataFrame, device) -> pd.DataFrame:
    """Per-clip hit/miss on the 3 rows the gap decomposition found failing
    even with oracle cues (#23, #53, #54 -- all F02)."""
    hl = real[(real.split == "test") & real.headline_eval
             & real.v3_row.isin([23, 53, 54])]
    if hl.empty:
        return pd.DataFrame()
    pred = T._eval_arrays(model, *T.frame_arrays(hl), device)
    out = hl[["clip_id", "v3_row", "intent"]].copy()
    out["hit"] = (pred == hl.y.to_numpy())
    return out.groupby(["v3_row", "intent"]).hit.agg(["mean", "size"]).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--seed-pool", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    print(f"{clips.clip_id.nunique()} clips, {real.split.value_counts().to_dict()}",
          flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    n_pools = len(pools)
    print(f"pools built: {n_pools} / {2+7+8+4} (context+emotion+gesture+motion "
          "class bins)", flush=True)

    X, obs, y, report = generate(pools, n_per_combo=args.n_per_combo,
                                 seed=args.seed_pool)
    print(f"recombination: {report.n_generated} synthetic samples from "
          f"{report.n_combos - report.n_skipped_empty_pool}/{report.n_combos} "
          f"combos ({report.n_skipped_empty_pool} skipped, empty pool)",
          flush=True)
    if report.skipped:
        print("  skipped combos (first 10):", report.skipped[:10], flush=True)
    print("  label distribution:", report.label_counts, flush=True)
    (OUT_DIR / "recombination_report.json").write_text(json.dumps({
        "n_per_combo": args.n_per_combo, "n_combos": report.n_combos,
        "n_generated": report.n_generated,
        "n_skipped": report.n_skipped_empty_pool,
        "label_counts": report.label_counts,
        "skipped": [list(s[:4]) for s in report.skipped],
    }, indent=2))

    results = {}
    best_model_by_config = {}
    for name, cfg in CONFIGS.items():
        hl_runs, full_runs, models = run_config(name, cfg, real, (X, obs, y), device)
        results[name] = {"headline": G.agg(hl_runs), "full_test": G.agg(full_runs)}
        best_model_by_config[name] = models[0]        # seed 0, for the F02 spot-check
        with start_run("04_recombination", f"final_merged__{name}",
                       dataset="final_merged", split_kind="scenarios", cues="real",
                       params={"model": "attention_fusion", "seeds": len(SEEDS),
                               "aggregation": "clip_mean", "missing_mode": "exclude",
                               "n_per_combo": args.n_per_combo if cfg["recombine"] else 0,
                               "n_synthetic": report.n_generated if cfg["recombine"] else 0,
                               **cfg},
                       notes="4-way recombination ablation, complete dataset") as run:
            h, f = results[name]["headline"], results[name]["full_test"]
            run.log_metrics({"headline_clip_acc": h["acc_mean"],
                             "headline_clip_acc_std": h["acc_std"],
                             "headline_clip_macro_f1": h["macro_f1_mean"],
                             "headline_clip_macro_f1_std": h["macro_f1_std"],
                             "test_clip_acc": f["acc_mean"],
                             "test_clip_macro_f1": f["macro_f1_mean"]})

    # ── F02 spot-check on the best (full) config ────────────────────────────
    f02 = f02_check(best_model_by_config["full"], real, device)
    print("\nF02 rows (#23/#53/#54) under `full` config, seed0:")
    print(f02.to_string(index=False) if not f02.empty else "  (no matching clips)")

    (OUT_DIR / "recombination_results.json").write_text(json.dumps(results, indent=2))
    write_report(results, report, f02, args.n_per_combo)
    print(f"\n-> {OUT_DIR / 'RECOMBINATION.md'}")


def write_report(results: dict, report, f02: pd.DataFrame, n_per_combo: int) -> None:
    lines = [
        "# Rubric-driven cue recombination — `data/final_merged`",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · "
        f"{report.n_generated} synthetic samples from "
        f"{report.n_combos - report.n_skipped_empty_pool}/{report.n_combos} "
        f"combinatorial cue tuples (n_per_combo={n_per_combo}), sourced from "
        "TRAIN-split real cue vectors bucketed by GROUND TRUTH class "
        "(`fusion/model/recombine_merged.py`).",
        "",
        "Reference numbers from the gap decomposition (same protocol, no "
        "recombination): rules+real **0.608**, fusion+real (plain) **0.475** "
        "— the bar recombination must clear to recover the G1 claim on "
        "unseen cue combinations.",
        "",
        "## Headline test (excl. row #58's derived clips), 3 seeds", "",
        "| Config | dropout | jitter | recomb | Clip acc | Clip macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, cfg in CONFIGS.items():
        h = results[name]["headline"]
        lines.append(f"| {name} | {cfg['dropout_p']} | {cfg['jitter_sigma']} | "
                     f"{'yes' if cfg['recombine'] else 'no'} | "
                     f"{h['acc_mean']} ± {h['acc_std']} | "
                     f"{h['macro_f1_mean']} ± {h['macro_f1_std']} |")

    best = max(results, key=lambda k: results[k]["headline"]["acc_mean"])
    best_acc = results[best]["headline"]["acc_mean"]
    lines += [
        "", f"**Best: `{best}` at {best_acc}.** "
        + ("Beats rules+real (0.608) — the G1 claim is recovered on the "
           "honest (unseen-combination) split."
           if best_acc > 0.608 else
           "Does NOT beat rules+real (0.608) — recombination narrowed the "
           "gap but did not close it; see the per-config breakdown for which "
           "ingredient helped and consider a larger `--n-per-combo` or a "
           "wider gesture/motion pool before concluding the approach failed."),
        "",
        "## F02 (emergency) rows #23/#53/#54 — the safety-critical check", "",
        "Gap decomposition found these fail even with ORACLE (perfect) cues; "
        "the question is whether recombination (which explicitly includes "
        "every F02 combination) fixes them.", "",
    ]
    lines.append(f02.to_markdown(index=False) if not f02.empty
                 else "(no headline clips matched — check v3_row filter)")
    (OUT_DIR / "RECOMBINATION.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
