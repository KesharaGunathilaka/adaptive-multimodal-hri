"""Study 3 — window-SIZE sweep: how long should each cue model's own lookback
SPAN be (gesture/motion input span), as opposed to Study 2's window-SEQUENCE
question (how many windows / how much order matters). These are explicitly
different axes (`docs/methodology/07_evaluation.md` §7.9's own words: "a
different question from Stage 7.2's *output aggregation* window").

This RE-RUNS an already-answered sweep, not a new question: §7.9 measured
x0.5/x1/x2 lookback spans on the OLDER, smaller classroom-only table (no
recombination, single seed, window-level train+eval) and found **shorter is
better**: x0.5=0.976, x1.0=0.951, x2.0=0.756 test clip-acc, with x2 "exceeding
typical clip length and collapsing." The deployed spans stayed at x1 anyway,
because they match the unimodal engines' own *validated* training buffers, not
because x1 scored best.

Re-running on `final_merged` (promoted emotion/gesture checkpoints, both
contexts, actor-disjoint scenario split) with the CURRENT protocol (clip-pool
+ recombination `full` config, matching Study 1/2's headline metric) checks
whether that "shorter is better" finding still holds now that the dataset and
models have changed substantially, and gives an updated, directly comparable
number for the deployment recommendation.

Rebuilds window features straight from the cached Pass-1 per-frame arrays
(`WindowFeaturizer(scale=...)`, `fusion/extraction/windows.py`) — cheap, no
video re-decode — into a SCRATCH parquet per scale (the canonical
`unimodal_windows.parquet` at scale=1.0 is never touched).

    .venv/Scripts/python scripts/40_window_size_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.windows import WindowFeaturizer  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.merged_unimodal import (  # noqa: E402
    LABELS, PERFRAME_DIR, PROB_PREFIX)

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
SCALES = (0.5, 1.0, 1.5, 2.0)
SCRATCH_DIR = ROOT / "data" / "final_merged" / "features" / "_window_sweep_scratch"
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True)


def build_windows_at_scale(clips: pd.DataFrame, scale: float, device) -> pd.DataFrame:
    fz = WindowFeaturizer(device=device, scale=scale)
    rows = []
    t0 = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        npz_path = PERFRAME_DIR / f"{r.clip_id}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        for w in fz.featurize_clip(npz):
            row = {"clip_id": r.clip_id, "window_idx": w["window_idx"], "t_end": w["t_end"]}
            for m, key in [("emotion", "emo_probs"), ("gesture", "ges_probs"),
                           ("motion", "mot_probs"), ("context", "ctx_probs")]:
                p, pref = w[key], PROB_PREFIX[m]
                row[f"{pref}_obs"] = p is not None
                for i, c in enumerate(LABELS[m]):
                    row[f"{pref}_{c}"] = float(p[i]) if p is not None else np.nan
            rows.append(row)
        if n % 500 == 0 or n == len(clips):
            rate = n / (time.time() - t0)
            print(f"  [{n}/{len(clips)}] ({rate:.1f} clips/s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    clips = G.load_clips_and_windows()[0]   # clips only; windows rebuilt per scale
    print(f"{clips.clip_id.nunique()} clips", flush=True)

    results = {}
    for scale in SCALES:
        print(f"\n=== scale x{scale} ===", flush=True)
        scratch = SCRATCH_DIR / f"windows_x{scale}.parquet"
        if scratch.exists():
            windows = pd.read_parquet(scratch)
            print(f"  reusing cached {scratch.name} ({len(windows)} windows)", flush=True)
        else:
            windows = build_windows_at_scale(clips, scale, device)
            windows.to_parquet(scratch)
            print(f"  wrote {scratch.name} ({len(windows)} windows)", flush=True)

        real = G.to_common_schema(G.build_real(clips, windows))
        splits = {s: real[real.split == s] for s in ("train", "val", "test")}
        hl = real[(real.split == "test") & real.headline_eval]

        pools = build_pools(real[real.split == "train"], clips)
        X, obs, y, report = generate(pools, n_per_combo=100, seed=0)
        extra = (X, obs, y)

        hl_runs = []
        for seed in SEEDS:
            t0 = time.time()
            model, val_acc = T.train_fusion(
                splits, seed=seed, dropout_p=FULL_CFG["dropout_p"],
                jitter_sigma=FULL_CFG["jitter_sigma"], extra=extra, device=device,
                select_masked=FULL_CFG["select_masked"], missing_mode="exclude",
                model_factory=lambda: AttentionFusion(missing_mode="exclude"))
            pred = T._eval_arrays(model, *T.frame_arrays(hl), device)
            hl_runs.append(G.eval_clip(hl.y.to_numpy(), pred))
            print(f"    seed{seed}: val={val_acc:.4f} headline={hl_runs[-1]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        agg = G.agg(hl_runs)
        results[f"x{scale}"] = {
            "ges_span_s": round(64 / 30 * scale, 3), "mot_span_s": round(2.0 * scale, 3),
            "n_windows": len(windows), **agg}
        with start_run("06_temporal_representation", f"window_size_x{scale}",
                       dataset="final_merged", split_kind="scenarios", cues="real",
                       params={"scale": scale, "seeds": len(SEEDS), **FULL_CFG},
                       notes="Study 3: re-run of 07_evaluation.md §7.9's span sweep "
                             "on final_merged + recombination") as run:
            run.log_metrics({"headline_clip_acc": agg["acc_mean"],
                             "headline_clip_acc_std": agg["acc_std"],
                             "headline_clip_macro_f1": agg["macro_f1_mean"]})

    (OUT_DIR / "window_size_sweep_results.json").write_text(json.dumps(results, indent=2))
    write_report(results)
    print(f"\n-> {OUT_DIR / 'WINDOW_SIZE_SWEEP.md'}")


def write_report(results: dict) -> None:
    lines = [
        "# Study 3 — window-size (lookback span) sweep", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · 3 seeds · headline test · "
        "`full` recipe (recombination + dropout 0.3 + jitter 0.15), clip-pooled "
        "(R1) — matches Study 1/2's protocol, unlike the original single-seed "
        "window-level §7.9 sweep.", "",
        "Prior result (`docs/methodology/07_evaluation.md` §7.9, old classroom-only "
        "table, no recombination): x0.5=0.976, x1.0=0.951, x2.0=0.756 — "
        "**shorter was better**, deployed spans kept at x1 anyway (matches the "
        "unimodal engines' own validated buffers, not chosen for this number).",
        "", "| Scale | Gesture span | Motion span | Clip acc | Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for k, r in results.items():
        lines.append(f"| {k} | {r['ges_span_s']}s | {r['mot_span_s']}s | "
                     f"{r['acc_mean']} ± {r['acc_std']} | {r['macro_f1_mean']} |")
    best = max(results, key=lambda k: results[k]["acc_mean"])
    lines += ["", f"**Best on final_merged: `{best}`.** "
             "Compare against the deployed x1.0 span and §7.9's old-table trend "
             "before changing the deployed lookback."]
    (OUT_DIR / "WINDOW_SIZE_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
