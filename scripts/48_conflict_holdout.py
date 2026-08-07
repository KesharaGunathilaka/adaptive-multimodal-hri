"""Conflict-holdout generalization test (2026-08-07, discussed and refined
twice with the user after reviewing all 164 conflicting rows).

Attempt 1 (see WORKLOG) restricted BOTH recombination and real training to
the 284 "aligned" combos, excluding all 164 conflicting combos -- including
753 real training clips whose true tuple happened to be conflicting. Result:
0.0% accuracy on every conflicting test, exactly. Diagnosed as a design flaw,
not a generalization finding: F02/F07/F08/F10 (4 of 9 intent classes) turn
out to be produced EXCLUSIVELY by an emotion overriding a gesture's default
reading -- they have no "calm default" pathway anywhere in the rubric. So the
attempt-1 model had literally zero training exposure to these four classes;
0% measures "can you name a color you've never been shown," not
generalization.

Attempt 2 (this version), per the user's explicit instruction: recombination
and real training use EVERY real training scenario as-is (no exclusion) plus
the 284 aligned combos; recombination is additionally ALLOWED to reinforce
(synthesise more examples of) any conflicting combo that is already present
among real TRAINING scenarios -- this is not new information, just more
synthetic repeats of a pattern the model would see anyway. Only conflicting
combos with ZERO presence in real training -- synthetic or real -- are
excluded from training entirely ("truly novel", 143 of the 164, covering all
9 intent classes, verified before running). Testing uses ONLY real
data-table TEST scenarios (headline test set), split by whether each clip's
true combo was truly-novel or already-seen during training.

    .venv/Scripts/python scripts/48_conflict_holdout.py
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

from fusion.baselines import common  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.recombine_merged import (  # noqa: E402
    build_pools, classify_combos, generate)
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.merged_gap import rule_predict_fair  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True)


def tuple_of(row) -> tuple:
    return (row.context, row.gt_emotion, row.gt_gesture, row.gt_motion)


def is_determinate(row) -> bool:
    return not (row.emotion_masked or row.gesture_masked or row.motion_masked)


def combos_present_in(clips: pd.DataFrame, real_split: pd.DataFrame) -> set:
    """Distinct (ctx,emo,ges,mot) tuples actually recorded among this split's
    determinate (no masked cue) clips."""
    c2 = clips.set_index("clip_id")
    out = set()
    for cid in real_split.clip_id:
        row = c2.loc[cid]
        if is_determinate(row):
            out.add(tuple_of(row))
    return out


def classify_clips(clips: pd.DataFrame, real_split: pd.DataFrame,
                   novel_combos: set) -> tuple[set, set]:
    """-> (novel_clip_ids, seen_clip_ids) within `real_split`."""
    c2 = clips.set_index("clip_id")
    novel_ids, seen_ids = set(), set()
    for cid in real_split.clip_id:
        row = c2.loc[cid]
        if not is_determinate(row):
            continue
        (novel_ids if tuple_of(row) in novel_combos else seen_ids).add(cid)
    return novel_ids, seen_ids


def train_variant(tag: str, splits: dict, extra, device) -> list:
    models = []
    for seed in SEEDS:
        t0 = time.time()
        m, va = T.train_fusion(
            splits, seed=seed, dropout_p=FULL_CFG["dropout_p"],
            jitter_sigma=FULL_CFG["jitter_sigma"], extra=extra, device=device,
            missing_mode="exclude", select_masked=FULL_CFG["select_masked"])
        models.append(m)
        print(f"  {tag} seed{seed}: val={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return models


def ensemble_eval(models, X, obs, device) -> np.ndarray:
    preds = np.stack([T._eval_arrays(m, X, obs, device) for m in models])
    return np.array([np.bincount(preds[:, i], minlength=10).argmax()
                     for i in range(preds.shape[1])])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits_full = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval].copy()
    print(f"{clips.clip_id.nunique()} clips · headline test {len(hl)}", flush=True)

    aligned_combos, conflicting_combos = classify_combos()
    train_real_combos = combos_present_in(clips, splits_full["train"])
    always_trained = aligned_combos | train_real_combos
    novel_combos = conflicting_combos - always_trained
    print(f"aligned: {len(aligned_combos)}  distinct real-train combos: "
         f"{len(train_real_combos)}  reinforced-by-recombination (conflicting "
         f"AND in real train): {len(conflicting_combos & train_real_combos)}",
         flush=True)
    print(f"always_trained: {len(always_trained)}  truly novel (excluded from "
         f"everything): {len(novel_combos)}", flush=True)

    # ── pools + synthetic data (BOTH variants use the FULL, unfiltered real
    # train split for pools and for direct training -- only the SYNTHETIC
    # combo coverage differs) ────────────────────────────────────────────────
    pools = build_pools(splits_full["train"], clips)

    X_r, obs_r, y_r, rep_r = generate(pools, n_per_combo=100, seed=0,
                                      combo_filter=always_trained)
    print(f"restricted synthetic: {rep_r.n_generated} samples from "
         f"{rep_r.n_combos - rep_r.n_skipped_empty_pool}/{len(always_trained)} "
         "always_trained combos", flush=True)

    X_full, obs_full, y_full, rep_full = generate(pools, n_per_combo=100, seed=0)
    print(f"full-reference synthetic: {rep_full.n_generated} samples from "
         f"{rep_full.n_combos - rep_full.n_skipped_empty_pool}/448 combos", flush=True)

    print("\ntraining restricted (3 seeds)...", flush=True)
    models_restricted = train_variant("restricted", splits_full,
                                      (X_r, obs_r, y_r), device)

    print("\ntraining full_reference (3 seeds)...", flush=True)
    models_full = train_variant("full_reference", splits_full,
                                (X_full, obs_full, y_full), device)

    # ── real headline test clips, split by novel vs already-seen combo ─────
    novel_ids, seen_ids = classify_clips(clips, hl, novel_combos)
    hl_novel = hl[hl.clip_id.isin(novel_ids)]
    hl_seen = hl[hl.clip_id.isin(seen_ids)]
    print(f"\nheadline test: {len(hl_novel)} novel-combo, "
         f"{len(hl_seen)} seen-combo clips", flush=True)

    results = {}
    for name, sub in (("real_novel", hl_novel), ("real_seen", hl_seen)):
        Xs, obss = T.frame_arrays(sub)
        y_true = sub.y.to_numpy()
        pred_r = ensemble_eval(models_restricted, Xs, obss, device)
        pred_full = ensemble_eval(models_full, Xs, obss, device)
        pred_rules = rule_predict_fair(sub, use_predicted_context=False)
        results[name] = {
            "n": len(sub),
            "restricted": G.eval_clip(y_true, pred_r),
            "full_reference": G.eval_clip(y_true, pred_full),
            "rules": G.eval_clip(y_true, pred_rules),
        }
        print(f"  {name} (n={len(sub)}): restricted={results[name]['restricted']} "
             f"full_reference={results[name]['full_reference']} "
             f"rules={results[name]['rules']}", flush=True)

    (OUT_DIR / "conflict_holdout_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, len(aligned_combos), len(conflicting_combos),
                len(always_trained), len(novel_combos))
    print(f"\n({time.time()-t0:.0f}s) -> {OUT_DIR / 'CONFLICT_HOLDOUT.md'}")


def write_report(res, n_aligned, n_conflicting, n_always_trained, n_novel):
    L = [
        "# Conflict-holdout generalization test (v2)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"3-seed ensembles · {n_aligned} aligned + {n_conflicting} conflicting "
        f"combos overall; {n_always_trained} \"always trained\" (aligned + "
        f"whatever's in real training) vs {n_novel} truly novel (zero "
        "presence in training, real or synthetic).",
        "",
        "v1 (WORKLOG 2026-08-07) excluded ALL conflicting combos, including "
        "753 real training clips -- this made 4 of 9 intent classes "
        "(F02/F07/F08/F10) entirely absent from training, producing an "
        "uninformative 0.0% ('never seen this class at all', not a "
        "generalization measurement). v2 fixes this: real training is used "
        "unfiltered, and recombination is allowed to reinforce any "
        "conflicting combo already present in real training data. Only "
        "combos with ZERO training presence at all are excluded.",
        "",
        "**`restricted`**: trained on real data (unfiltered) + synthetic "
        "recombination limited to `always_trained` combos.",
        "**`full_reference`**: standard recipe, all 448 combos, retrained "
        "fresh for a clean paired comparison.",
        "",
        "| Test set | n | restricted | full_reference | rules |",
        "|---|---|---|---|---|",
    ]
    for name, r in res.items():
        L.append(f"| {name} | {r['n']} | {r['restricted']['acc']} / "
                 f"{r['restricted']['macro_f1']} | {r['full_reference']['acc']} / "
                 f"{r['full_reference']['macro_f1']} | {r['rules']['acc']} / "
                 f"{r['rules']['macro_f1']} |")
    L.append("")
    L.append("(cells are accuracy / macro-F1)")

    rn, rs = res["real_novel"], res["real_seen"]
    gap = rn["full_reference"]["acc"] - rn["restricted"]["acc"]
    L += [
        "",
        "## Reading this",
        "",
        f"On real NOVEL-combo test clips (n={rn['n']}, true combo never seen "
        f"in training, real or synthetic) -- `restricted` scores "
        f"{rn['restricted']['acc']}, `full_reference` scores "
        f"{rn['full_reference']['acc']} (it saw these combos directly), "
        f"rules score {rn['rules']['acc']}. The `full_reference`-minus-"
        f"`restricted` gap ({gap:+.4f}) is the genuine generalization cost "
        "of not having a labelled example of this specific combo -- with "
        "every class still represented in `restricted`'s training, so this "
        "measures compositional generalization, not label-set coverage.",
        "",
        f"On real SEEN-combo test clips (n={rs['n']}) -- both fusion models "
        f"had training exposure -- `restricted` scores {rs['restricted']['acc']} "
        f"vs `full_reference`'s {rs['full_reference']['acc']}, expected to be "
        "much closer.",
    ]
    (OUT_DIR / "CONFLICT_HOLDOUT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
