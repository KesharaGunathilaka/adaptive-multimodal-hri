"""Phase 3 — trajectory recombination for the window-SEQUENCE model (R3).

Study 2 (2026-08-05, `TEMPORAL_REPRESENTATION.md`) found R3 (order-aware
sequence transformer) scoring far below R1 (clip-pool, ~0.533 vs ~0.7191) but
flagged the comparison as CONFOUNDED: R3 never got a recombination analogue,
so it trained on ~30x fewer augmented samples than R1/R2. This script builds
that analogue (`fusion/model/recombine_sequences.py`, trajectory-level, not
just pooled-vector-level) and re-runs R3 with it, to find out whether the gap
was really about temporal modelling being unhelpful, or just about coverage.

This is also the ONE experiment in the whole "why does fusion only tie rules"
investigation (Phase 1/2, `ROBUSTNESS_BATTERY.md`/`VIDEO_DEGRADATION.md`) that
can genuinely produce HIGHER accuracy than rules on real cues: rules see one
pooled snapshot and structurally cannot use temporal dynamics at all -- if
sequence order carries information the rubric's four static labels don't, only
a temporal model can use it.

    .venv/Scripts/python scripts/45_sequence_recombination.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.recombine_sequences import build_sequence_pools, generate_sequences  # noqa: E402
from fusion.model.sequences import build_sequences  # noqa: E402
from fusion.model.temporal import _eval_arrays, train_temporal  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
MAX_LEN = 40                            # matches scripts/39_temporal_representation.py
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15)

# Study 2's baseline numbers (2026-08-05), reproduced here for reference only
# -- NOT retrained, to keep this script's cost bounded to the new experiment.
BASELINE = {
    "R1_clip_pool": 0.7191,
    "rules": 0.712,
    "R3_bidi_no_recomb": 0.5328,
    "R3_causal_no_recomb": 0.5329,
}


def run(causal: bool, extra, seq_splits, device, tag: str) -> list[dict]:
    runs = []
    for seed in SEEDS:
        t0 = time.time()
        model, va = train_temporal(seq_splits, seed=seed, causal=causal,
                                   dropout_p=FULL_CFG["dropout_p"],
                                   jitter_sigma=FULL_CFG["jitter_sigma"],
                                   device=device, extra=extra)
        Xte, obste, valte, yte = seq_splits["test"]
        pred = _eval_arrays(model, Xte, obste, valte, device)
        r = G.eval_clip(yte, pred)
        runs.append(r)
        print(f"  {tag} seed{seed}: val={va:.4f} headline={r} "
             f"({time.time()-t0:.0f}s)", flush=True)
    return runs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    # same headline-eval fix as Study 2 (WORKLOG 2026-08-05): only non-test
    # rows pass through unfiltered, test rows are restricted to headline_eval
    real_hl = real[(real.split != "test") | real.headline_eval]

    print("building sequence pools (train split, ground-truth indexed)...", flush=True)
    pools = build_sequence_pools(windows, clips, real_hl[real_hl.split == "train"])
    print(f"  {len(pools)} / {2+7+8+4} (context+emotion+gesture+motion class bins) "
         "have a real trajectory pool", flush=True)

    train_ids = set(real_hl[real_hl.split == "train"].clip_id)
    target_lens = windows[windows.clip_id.isin(train_ids)].groupby("clip_id").size().to_numpy()
    print(f"  target length distribution: n={len(target_lens)} clips, "
         f"median={np.median(target_lens):.0f}, max={target_lens.max()}", flush=True)

    Xs, obss, valids, ys, rep = generate_sequences(
        pools, target_lens, T=MAX_LEN, n_per_combo=100, seed=0)
    print(f"generated {rep.n_generated} synthetic trajectories from "
         f"{rep.n_combos - rep.n_skipped_empty_pool}/{rep.n_combos} combos "
         f"({rep.n_skipped_empty_pool} skipped, empty pool)", flush=True)

    print("\nbuilding real sequences (train/val/test)...", flush=True)
    seq_splits = {s: build_sequences(windows, real_hl, s, max_len=MAX_LEN)[:4]
                 for s in ("train", "val", "test")}
    for s, (X, obs, valid, y) in seq_splits.items():
        print(f"  {s}: n={len(y)} T={X.shape[1]}", flush=True)

    extra = (Xs, obss, valids, ys)

    print("\n=== causal, WITHOUT recombination (reproduces Study 2's baseline) ===",
         flush=True)
    causal_base = run(True, None, seq_splits, device, "causal_base")

    print("\n=== causal, WITH trajectory recombination ===", flush=True)
    causal_recomb = run(True, extra, seq_splits, device, "causal_recomb")

    print("\n=== bidirectional, WITH trajectory recombination (offline ceiling) ===",
         flush=True)
    bidi_recomb = run(False, extra, seq_splits, device, "bidi_recomb")

    results = {
        "causal_base": G.agg(causal_base),
        "causal_recomb": G.agg(causal_recomb),
        "bidi_recomb": G.agg(bidi_recomb),
        "n_synthetic": rep.n_generated,
        "n_combos_covered": rep.n_combos - rep.n_skipped_empty_pool,
    }
    (OUT_DIR / "sequence_recombination_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(f"\n({time.time()-t0:.0f}s total) -> {OUT_DIR / 'SEQUENCE_RECOMBINATION.md'}")


def write_report(res: dict) -> None:
    cb, cr, br = res["causal_base"], res["causal_recomb"], res["bidi_recomb"]
    L = [
        "# Phase 3 — trajectory recombination for the window-sequence model (R3)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test (n=979) · {res['n_synthetic']} synthetic trajectories "
        f"covering {res['n_combos_covered']}/448 combos · 3 seeds.",
        "",
        "## Does closing the augmentation gap change R3's verdict?",
        "",
        "| Config | Clip acc | Macro-F1 |",
        "|---|---|---|",
        f"| R1 clip-pool (reference, Study 1/2) | {BASELINE['R1_clip_pool']} | — |",
        f"| Rules (reference) | {BASELINE['rules']} | — |",
        f"| R3 causal, no recombination (Study 2 baseline, reproduced here) "
        f"| {cb['acc_mean']} ± {cb['acc_std']} | {cb['macro_f1_mean']} |",
        f"| **R3 causal, WITH trajectory recombination** "
        f"| **{cr['acc_mean']} ± {cr['acc_std']}** | **{cr['macro_f1_mean']}** |",
        f"| R3 bidirectional, WITH recombination (offline ceiling) "
        f"| {br['acc_mean']} ± {br['acc_std']} | {br['macro_f1_mean']} |",
        "",
    ]
    delta = cr["acc_mean"] - cb["acc_mean"]
    vs_r1 = cr["acc_mean"] - BASELINE["R1_clip_pool"]
    vs_rules = cr["acc_mean"] - BASELINE["rules"]
    L += [
        f"Recombination moves causal R3 by **{delta:+.4f}** "
        f"({cb['acc_mean']} → {cr['acc_mean']}).",
        "",
    ]
    if vs_r1 > 0.01:
        L.append(f"**R3 now beats R1 by {vs_r1:+.4f}** — order-aware temporal "
                 "modelling has a genuine edge once given the same combinatorial "
                 "coverage. This is the one result in the whole Phase 1-3 "
                 "investigation that can exceed rules on real cues by more than "
                 "seed noise, because rules structurally cannot use dynamics at "
                 "all — this is a capability gap, not a tie.")
    elif abs(vs_r1) <= 0.01:
        L.append(f"R3 now **ties** R1 (delta {vs_r1:+.4f}, within seed noise) — "
                 "the augmentation gap was indeed most of the story. Temporal "
                 "modelling does not clearly help OR hurt once coverage is equal; "
                 "no strong claim either way from this result alone.")
    else:
        L.append(f"R3 still trails R1 by {vs_r1:+.4f} even with matched "
                 "augmentation. The remaining gap is now attributable to "
                 "something other than coverage -- e.g. harder optimisation "
                 "over sequences, or genuinely less useful information in this "
                 "dataset's window-to-window dynamics than in the pooled mean.")
    L += ["", f"vs rules: {vs_rules:+.4f}."]
    (OUT_DIR / "SEQUENCE_RECOMBINATION.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
