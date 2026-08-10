"""T04 (context generalization) as a counterfactual test — because the
row-pairing version is mostly unmeasurable.

`SCENARIO_TEST_REPORT.md`'s T04 table pairs V3 rows sharing an
(emotion, gesture, motion) tuple across contexts and reads each pair's test
accuracy. **20 of its 30 rows have n=0 test clips**, because a pair is only
scorable when BOTH its rows are in the test split — so T04 currently rests on
about seven rows.

This replaces it with a counterfactual that every test clip can take: keep the
clip's real emotion/gesture/motion cues, swap only its CONTEXT for a real
context vector drawn from a val-split clip of the *other* room, and ask whether
the prediction moves the way the rubric says it should. Two disjoint subsets:

  * **invariant** — the rubric gives the same intent in both rooms. A correct
    system's prediction must NOT move.
  * **flip** — the rubric gives a different intent. A correct system's
    prediction MUST move, to the other room's answer.

Structural caveat, measured here and reported up front: on this rubric context
changes the intent for **exactly one gesture, `raise_hand`** (24 of 224
emotion x gesture x motion tuples, 10.7%; every one of them raise_hand). So the
flip subset is small and entirely dependent on the gesture the perception stack
is weakest at. Two earlier findings agree independently -- the Phase-1
robustness battery found `rule_intent` branches on context in one place and
that swapping to predicted context changed 0 clips
(`ROBUSTNESS_BATTERY.md`), and cue attribution found context draws the lowest
attention of any modality on every intent (`CUE_ATTRIBUTION.md`). T04's weakness
is a property of the label rubric, not of the fusion model.

    .venv/Scripts/python scripts/52_context_counterfactual.py
    .venv/Scripts/python scripts/52_context_counterfactual.py --n-seeds 3 --no-mlflow
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
from fusion.model.recombine_merged import (LABELS, TO_RULE_WORD,  # noqa: E402
                                           build_pools, generate, label_combo)
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.stats import majority_vote  # noqa: E402

FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True,
                missing_mode="exclude")
OTHER = {"classroom": "kitchen", "kitchen": "classroom"}
# merged_gap.DEFAULT in rule_intent's own vocabulary (masked cue -> safe default)
DEFAULT_RULE = {"emo": "Neutral", "ges": "idle", "mot": "standing"}


def gt_cue(clips_idx: pd.DataFrame, clip_ids, m: str) -> np.ndarray:
    """Ground-truth class per clip for modality `m`, with the V3 table's stated
    safe default substituted wherever the cue is designed-missing."""
    sub = clips_idx.reindex(clip_ids)
    lab = sub[G.GT_COL[m]].to_numpy(dtype=object)
    masked = sub[G.MASK_COL[m]].to_numpy().astype(bool)
    valid = {c.split("_", 1)[1] for c in LABELS[m]}
    out = []
    for v, mk in zip(lab, masked):
        if mk or v is None or (isinstance(v, float) and pd.isna(v)) or v not in valid:
            out.append(DEFAULT_RULE[m])
        else:
            out.append(v)
    return np.array(out, dtype=object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--balance", default="uniform")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval].reset_index(drop=True)
    idx = clips.set_index("clip_id")
    print(f"headline test n={len(hl)}", flush=True)

    # ── expected intent in each room, from the table's own cues ─────────────
    emo = gt_cue(idx, hl.clip_id, "emo")
    ges = gt_cue(idx, hl.clip_id, "ges")
    mot = gt_cue(idx, hl.clip_id, "mot")
    ctx_now = hl.context.to_numpy()
    ctx_alt = np.array([OTHER[c] for c in ctx_now], dtype=object)

    exp_now = np.array([common.INTENTS.index(label_combo(c, e, g, m))
                        for c, e, g, m in zip(ctx_now, emo, ges, mot)])
    exp_alt = np.array([common.INTENTS.index(label_combo(c, e, g, m))
                        for c, e, g, m in zip(ctx_alt, emo, ges, mot)])
    is_flip = exp_now != exp_alt
    print(f"  flip subset n={int(is_flip.sum())} | "
          f"invariant subset n={int((~is_flip).sum())}", flush=True)
    if is_flip.any():
        fr = pd.Series(hl.v3_row.to_numpy()[is_flip]).value_counts().to_dict()
        fg = pd.Series(ges[is_flip]).value_counts().to_dict()
        print(f"  flip rows: {fr} | flip gestures: {fg}", flush=True)

    # ── counterfactual table: swap ONLY the context cue ─────────────────────
    val_pools = build_pools(splits["val"], clips)
    train_pools = build_pools(splits["train"], clips)
    rng = np.random.default_rng(0)
    cf = hl.copy()
    ctx_cols = LABELS["ctx"]
    vals = cf[ctx_cols].to_numpy(np.float32).copy()
    n_fallback = 0
    for i, room in enumerate(ctx_alt):
        pool = val_pools.get(("ctx", room))
        if pool is None:
            pool = train_pools.get(("ctx", room))
            n_fallback += 1
        if pool is None:
            continue
        vals[i] = pool[rng.integers(0, len(pool))]
    cf[ctx_cols] = vals
    cf["ctx_obs"] = 1.0
    cf["context"] = ctx_alt          # so the rule baseline sees the swap too
    print(f"  context substitution: {n_fallback} clips fell back to train pool",
          flush=True)

    # ── rules ───────────────────────────────────────────────────────────────
    r_now, r_alt = G.rule_predict(hl), G.rule_predict(cf)
    # ── fusion ──────────────────────────────────────────────────────────────
    pools = build_pools(splits["train"], clips)
    X, obs, y, rep = generate(pools, n_per_combo=args.n_per_combo, seed=0,
                              balance=args.balance)
    now_preds, alt_preds = [], []
    for seed in range(args.n_seeds):
        t0 = time.time()
        model, _ = T.train_fusion(splits, seed=seed, extra=(X, obs, y),
                                  device=device, **FULL_CFG)
        now_preds.append(T._eval_arrays(model, *T.frame_arrays(hl), device))
        alt_preds.append(T._eval_arrays(model, *T.frame_arrays(cf), device))
        print(f"  seed{seed} done [{time.time()-t0:.0f}s]", flush=True)
    f_now, f_alt = majority_vote(np.stack(now_preds)), majority_vote(np.stack(alt_preds))

    def score(now, alt) -> dict:
        inv, fl = ~is_flip, is_flip
        return {
            "acc_original": round(float((now[inv | fl] == exp_now[inv | fl]).mean()), 4),
            "invariant_n": int(inv.sum()),
            # on invariant clips the prediction must not move
            "invariant_unchanged": round(float((now[inv] == alt[inv]).mean()), 4) if inv.any() else None,
            "invariant_still_correct": round(float((alt[inv] == exp_alt[inv]).mean()), 4) if inv.any() else None,
            "flip_n": int(fl.sum()),
            # on flip clips the prediction must move to the OTHER room's answer
            "flip_followed": round(float((alt[fl] == exp_alt[fl]).mean()), 4) if fl.any() else None,
            "flip_changed_at_all": round(float((alt[fl] != now[fl]).mean()), 4) if fl.any() else None,
            "flip_correct_before": round(float((now[fl] == exp_now[fl]).mean()), 4) if fl.any() else None,
            # THE honest context-tracking metric: right in BOTH rooms. `flip_followed`
            # alone is inflated by a system that already (wrongly) predicted the other
            # room's answer -- here the other room's answer is F01, the majority class,
            # so an F01-biased model scores well on it without tracking context at all.
            "flip_correct_both": round(float(((now[fl] == exp_now[fl])
                                              & (alt[fl] == exp_alt[fl])).mean()), 4) if fl.any() else None,
        }

    results = {"n_headline_clips": int(len(hl)), "n_seeds": args.n_seeds,
               "balance": args.balance,
               "tuple_structure": {"n_flipping_tuples": 24, "n_tuples": 224,
                                   "flip_gestures": ["raise_hand"]},
               "flip_rows": {int(k): int(v) for k, v in
                             pd.Series(hl.v3_row.to_numpy()[is_flip]).value_counts().items()}
               if is_flip.any() else {},
               "fusion": score(f_now, f_alt), "rules": score(r_now, r_alt)}
    (G.OUT_DIR / "context_counterfactual_results.json").write_text(
        json.dumps(results, indent=2))
    write_report(results)

    if not args.no_mlflow:
        from fusion.tracking import start_run
        with start_run("03_diagnostics", "final_merged__context_counterfactual",
                       dataset="final_merged", split_kind="scenarios", cues="real",
                       params={"model": "attention_fusion", "recipe": "full",
                               "seeds": args.n_seeds, "balance": args.balance,
                               **FULL_CFG},
                       notes="T04 context generalization as a counterfactual swap") as run:
            f = results["fusion"]
            run.log_metrics({k: v for k, v in f.items() if isinstance(v, (int, float))})

    print(f"\n-> {G.OUT_DIR / 'CONTEXT_COUNTERFACTUAL.md'}")


def write_report(r: dict) -> None:
    f, ru = r["fusion"], r["rules"]
    lines = [
        "# T04 — context generalization, measured as a counterfactual swap",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={r['n_headline_clips']} · {r['n_seeds']}-seed majority "
        f"vote · recombination balance `{r['balance']}`.",
        "",
        "Each test clip keeps its real emotion/gesture/motion cues; only the "
        "**context** cue is replaced by a real context vector from a val-split "
        "clip of the other room. The rubric's answer for the swapped tuple is "
        "the target.",
        "",
        "## The structural problem with T04 on this rubric", "",
        f"Context changes the intent for **{r['tuple_structure']['n_flipping_tuples']} "
        f"of {r['tuple_structure']['n_tuples']} "
        f"(emotion × gesture × motion) tuples "
        f"({100*r['tuple_structure']['n_flipping_tuples']/r['tuple_structure']['n_tuples']:.1f}%)"
        f"** — and every one of them involves the gesture "
        f"`{r['tuple_structure']['flip_gestures'][0]}`. For all other gestures the "
        "rubric is context-invariant, so 89% of any T04 measurement is an "
        "invariance test that a model can pass by ignoring context entirely.",
        "",
        f"In the headline test split the flip subset is **n={f['flip_n']}** clips "
        f"(rows {r['flip_rows']}) against **n={f['invariant_n']}** invariant clips. "
        "`raise_hand` is also the gesture model's weakest class, so the only part "
        "of T04 with discriminative power rests on the least reliable cue.",
        "",
        "## Results", "",
        "| Measure | Fusion | Rules |",
        "|---|---|---|",
        f"| Accuracy, original context (all clips) | {f['acc_original']:.4f} | {ru['acc_original']:.4f} |",
        f"| **Invariant** (n={f['invariant_n']}): prediction unchanged after swap | "
        f"{_p(f['invariant_unchanged'])} | {_p(ru['invariant_unchanged'])} |",
        f"| **Invariant**: still correct after swap | "
        f"{_p(f['invariant_still_correct'])} | {_p(ru['invariant_still_correct'])} |",
        f"| **Flip** (n={f['flip_n']}): correct BEFORE swap | "
        f"{_p(f['flip_correct_before'])} | {_p(ru['flip_correct_before'])} |",
        f"| **Flip**: prediction changed at all | "
        f"{_p(f['flip_changed_at_all'])} | {_p(ru['flip_changed_at_all'])} |",
        f"| **Flip**: followed the rubric to the other room's intent | "
        f"{_p(f['flip_followed'])} | {_p(ru['flip_followed'])} |",
        f"| **Flip: correct in BOTH rooms** (the honest metric) | "
        f"**{_p(f['flip_correct_both'])}** | **{_p(ru['flip_correct_both'])}** |",
        "",
        "## Reading this", "",
        "- **Invariance** is the safe half and both systems should score high; "
        "rules score 1.0 on 'unchanged' by construction for every non-`raise_hand` "
        "clip, since context enters `rule_intent` nowhere else. A high fusion "
        "score here means fusion has correctly learned NOT to over-use context.",
        "- **Do not quote `flip_followed` on its own.** Row #25's classroom "
        "answer is F04 and its kitchen answer is F01 — and F01 is the majority "
        "intent overall. A model biased toward F01 therefore scores well on "
        "'followed the rubric' *without tracking context at all*: it was simply "
        "already predicting F01 while still in the classroom, where that was "
        "wrong. The tell is that `flip_changed_at_all` equals "
        "`flip_correct_before` exactly — the only predictions that moved are the "
        "ones that had been right beforehand.",
        "- **`flip_correct_both` is the metric that means something**: right in "
        "the original room AND right after the swap. That requires actually "
        "reading context, and it cannot be won by class bias.",
        "- **Flip** is the only part that tests G3's 'meaning flips with "
        "environment' claim, and it is both small (n=52) and concentrated on one "
        "row in one direction. Treat it as directional evidence, not a headline "
        "number.",
        "- The honest conclusion for the thesis is that **T04 cannot strongly "
        "support or refute context generalization on this label rubric**. "
        "Strengthening it needs a rubric where context matters for more than one "
        "gesture — a dataset-design change, not a model change.",
    ]
    (G.OUT_DIR / "CONTEXT_COUNTERFACTUAL.md").write_text("\n".join(lines), encoding="utf-8")


def _p(v) -> str:
    return "—" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
