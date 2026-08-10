"""How much of the DEPLOYED model's remaining error is perception? (2026-08-08)

`GAP_DECOMPOSITION_MERGED.md` decomposed the error of a *plain* fusion model
(no augmentation): generalisation cost 0.381, perception cost 0.072. That model
is obsolete -- recombination has since closed most of the generalisation gap
(0.475 -> 0.7191). The decomposition was never re-run on the `full` recipe, and
`docs/WORKLOG.md` 2026-08-03 flagged exactly that as the outstanding follow-up.
Until it exists, we do not know where the deployed model's remaining ~0.28 of
error actually lives, and therefore cannot say whether better perception (or
richer cue representations) would help at all.

Three test-time cue conditions, all scored with the SAME trained models (the
deployed `full` recipe: real train cues + recombination + dropout/jitter +
masked-val selection), so the only thing varying is what the model is fed:

  real              what deployment actually gets.
  oracle_onehot     the V3 table's true cue labels as one-hot vectors. Matches
                    `scripts/29`'s oracle condition, BUT note that script both
                    TRAINED and tested on one-hot; here the model was trained on
                    soft real vectors, so one-hot input is out-of-distribution
                    and this number is a LOWER bound on "perfect perception".
  oracle_realistic  perfect perception WITHOUT the distribution shift: each
                    clip's cue vector is replaced by a real, soft probability
                    vector drawn from a VAL-split clip whose GROUND TRUTH is
                    this clip's true class. Val is actor-disjoint from train and
                    is NOT used to build the recombination pools, so these
                    vectors are not ones the model was trained on -- which
                    `train`-drawn vectors would have been, inflating the result.

`oracle_realistic - real` is the honest perception band: the accuracy the
deployed model would gain if the four cue models were perfect but still emitted
realistically-shaped distributions.

    .venv/Scripts/python scripts/50_perception_band.py
    .venv/Scripts/python scripts/50_perception_band.py --n-seeds 3 --no-mlflow
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
from fusion.model.recombine_merged import (COL_OFFSET, LABELS,  # noqa: E402
                                           build_pools, generate)
from scripts.realworld_eval import merged_gap as G  # noqa: E402

FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True,
                missing_mode="exclude")
MODS = ("emo", "ges", "mot", "ctx")


def build_realistic_oracle(tbl: pd.DataFrame, clips: pd.DataFrame,
                           pools: dict, fallback: dict, seed: int = 0):
    """Replace every observed cue vector with a real vector of the clip's TRUE
    class, drawn from `pools`. Masked cues keep obs=0 and an all-zero vector,
    exactly as `build_oracle` does. -> (table, stats)."""
    rng = np.random.default_rng(seed)
    out = tbl.copy()
    gt = clips.set_index("clip_id")
    n_fallback, n_missing_bin, n_filled = 0, 0, 0

    for m in MODS:
        cols = LABELS[m]
        vals = out[cols].to_numpy(np.float32).copy()
        labels = gt.reindex(out.clip_id)[G.GT_COL[m]].to_numpy()
        masked = gt.reindex(out.clip_id)[G.MASK_COL[m]].to_numpy().astype(bool)
        obs = np.ones(len(out), np.float32)

        for i, (cls, is_masked) in enumerate(zip(labels, masked)):
            if is_masked or cls is None or (isinstance(cls, float) and pd.isna(cls)):
                vals[i] = 0.0
                obs[i] = 0.0
                continue
            pool = pools.get((m, cls))
            if pool is None:
                pool = fallback.get((m, cls))
                if pool is not None:
                    n_fallback += 1
            if pool is None:                      # no example of this class anywhere
                n_missing_bin += 1
                vals[i] = 0.0
                obs[i] = 0.0
                continue
            vals[i] = pool[rng.integers(0, len(pool))]
            n_filled += 1

        out[cols] = vals
        out[f"{m}_obs"] = obs

    return out, {"n_filled": n_filled, "n_from_train_fallback": n_fallback,
                 "n_no_pool_anywhere": n_missing_bin}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    oracle = G.to_common_schema(G.build_oracle(clips))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}

    hl_mask = (real.split == "test") & real.headline_eval
    hl_real = real[hl_mask]
    hl_oracle = oracle[(oracle.split == "test") & oracle.headline_eval]
    y_true = hl_real.y.to_numpy()
    assert (hl_oracle.clip_id.to_numpy() == hl_real.clip_id.to_numpy()).all(), \
        "real/oracle headline tables must be row-aligned for a paired comparison"
    print(f"headline test n={len(hl_real)}", flush=True)

    # Recombination pools come from TRAIN (as in the deployed recipe); the
    # realistic-oracle substitution draws from VAL so the model is never fed a
    # vector it was trained on.
    train_pools = build_pools(splits["train"], clips)
    val_pools = build_pools(splits["val"], clips)
    print(f"pools: train={len(train_pools)} bins, val={len(val_pools)} bins", flush=True)

    X, obs, y, rep = generate(train_pools, n_per_combo=args.n_per_combo, seed=0)
    print(f"recombination: {rep.n_generated} synthetic samples", flush=True)

    hl_realistic, sub_stats = build_realistic_oracle(
        hl_real, clips, val_pools, train_pools, seed=0)
    print(f"realistic-oracle substitution: {sub_stats}", flush=True)

    conditions = {"real": hl_real, "oracle_onehot": hl_oracle,
                  "oracle_realistic": hl_realistic}

    # ── rules under each condition (deterministic) ──────────────────────────
    rules = {name: G.eval_clip(y_true, G.rule_predict(tbl))
             for name, tbl in conditions.items()}
    for k, v in rules.items():
        print(f"rules[{k}]: {v}", flush=True)

    # ── fusion: one set of models, three evaluations each ───────────────────
    per_seed = {name: [] for name in conditions}
    for seed in range(args.n_seeds):
        t0 = time.time()
        model, val_acc = T.train_fusion(
            splits, seed=seed, extra=(X, obs, y), device=device, **FULL_CFG)
        line = []
        for name, tbl in conditions.items():
            pred = T._eval_arrays(model, *T.frame_arrays(tbl), device)
            res = G.eval_clip(y_true, pred)
            per_seed[name].append(res)
            line.append(f"{name}={res['acc']:.4f}")
        print(f"  seed{seed}: val={val_acc:.4f} " + " ".join(line)
              + f" [{time.time()-t0:.0f}s]", flush=True)

    fusion = {name: G.agg(runs) for name, runs in per_seed.items()}

    real_acc = fusion["real"]["acc_mean"]
    band_realistic = round(fusion["oracle_realistic"]["acc_mean"] - real_acc, 4)
    band_onehot = round(fusion["oracle_onehot"]["acc_mean"] - real_acc, 4)
    gen_cost = round(1.0 - fusion["oracle_realistic"]["acc_mean"], 4)

    results = {
        "n_seeds": args.n_seeds, "n_headline_clips": int(len(hl_real)),
        "substitution_stats": sub_stats,
        "rules": rules, "fusion": fusion,
        "perception_band_realistic": band_realistic,
        "perception_band_onehot": band_onehot,
        "generalisation_cost_realistic": gen_cost,
    }
    (G.OUT_DIR / "perception_band_results.json").write_text(json.dumps(results, indent=2))
    write_report(results)

    if not args.no_mlflow:
        from fusion.tracking import start_run
        for cues, tag in [("real", "real"), ("oracle", "oracle_onehot"),
                          ("oracle", "oracle_realistic")]:
            f = fusion[tag]
            with start_run("03_diagnostics", f"final_merged__fullrecipe_{tag}",
                           dataset="final_merged", split_kind="scenarios", cues=cues,
                           params={"model": "attention_fusion", "recipe": "full",
                                   "seeds": args.n_seeds, "cue_condition": tag,
                                   "n_per_combo": args.n_per_combo, **FULL_CFG},
                           notes="perception-band decomposition on the full recipe") as run:
                run.log_metrics({"headline_clip_acc": f["acc_mean"],
                                 "headline_clip_acc_std": f["acc_std"],
                                 "headline_clip_macro_f1": f["macro_f1_mean"],
                                 "rules_same_condition": rules[tag]["acc"]})

    print(f"\n-> {G.OUT_DIR / 'PERCEPTION_BAND.md'}")


def write_report(r: dict) -> None:
    fu, ru = r["fusion"], r["rules"]
    lines = [
        "# Where does the deployed model's remaining error live?",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={r['n_headline_clips']} · **{r['n_seeds']} seeds** · "
        "deployed `full` recipe (real train cues + recombination + dropout 0.3 "
        "+ jitter 0.15 + masked-val selection).",
        "",
        "`GAP_DECOMPOSITION_MERGED.md`'s decomposition was measured on a *plain* "
        "fusion model and has been stale since recombination landed. This re-runs "
        "it on the recipe actually deployed, and adds a distribution-shift-free "
        "oracle condition.",
        "",
        "## Results", "",
        "| Test-time cues | Fusion clip acc | Fusion macro-F1 | Rules clip acc |",
        "|---|---|---|---|",
    ]
    for tag, label in [("real", "`real` (what deployment gets)"),
                       ("oracle_onehot", "`oracle_onehot` (one-hot true labels)"),
                       ("oracle_realistic", "`oracle_realistic` (true class, real soft vectors)")]:
        f = fu[tag]
        lines.append(f"| {label} | {f['acc_mean']:.4f} ± {f['acc_std']:.4f} | "
                     f"{f['macro_f1_mean']:.4f} | {ru[tag]['acc']:.4f} |")

    onehot = fu["oracle_onehot"]["acc_mean"]
    realistic = fu["oracle_realistic"]["acc_mean"]
    real_acc = fu["real"]["acc_mean"]
    gen_onehot = round(1.0 - onehot, 4)
    band_onehot = round(onehot - real_acc, 4)
    lines += [
        "",
        "## What each oracle condition actually measures", "",
        "**`oracle_realistic` is NOT perfect perception.** The substituted "
        "vector is guaranteed to come from a clip of the right class, but that "
        "clip's cue model output is still noisy — so class-conditional noise "
        "survives the substitution. The proof is in the rules column: rules "
        f"score **{ru['oracle_realistic']['acc']:.4f}** under `oracle_realistic` "
        f"but exactly **{ru['oracle_onehot']['acc']:.4f}** under `oracle_onehot`. "
        "A rule system given a genuinely perfect cue is right by construction; "
        "the shortfall is the noise the substitution keeps. So "
        "`oracle_realistic` measures *decorrelating cue noise from a clip's own "
        "difficulty*, not eliminating it.",
        "",
        "Read the two conditions as bracketing the truth:",
        "",
        f"- `oracle_onehot` — **noise-free but out-of-distribution** for a model "
        f"trained on soft vectors. Fusion {onehot:.4f}.",
        f"- `oracle_realistic` — **in-distribution but still noisy**. "
        f"Fusion {realistic:.4f}.",
        "",
        "## Decomposition", "",
        "| Using | Generalisation cost (1.0 − oracle) | Perception cost (oracle − real) |",
        "|---|---|---|",
        f"| `oracle_onehot` | {gen_onehot:.4f} | {band_onehot:.4f} |",
        f"| `oracle_realistic` | {r['generalisation_cost_realistic']:.4f} | "
        f"{r['perception_band_realistic']:.4f} |",
        "",
        "**The headline change: fusion+oracle has moved from 0.619 to "
        f"{onehot:.4f}.** `GAP_DECOMPOSITION_MERGED.md` measured 0.619 on the "
        "*plain* recipe and concluded generalisation cost 0.381 dominated "
        "perception cost 0.072. On the deployed `full` recipe that is now "
        f"reversed: given correct cues the model is right {onehot:.1%} of the "
        "time, so **recombination has largely closed the generalisation gap it "
        "was built to close**, and what remains is dominated by perception.",
        "",
        "**Only the `oracle_onehot` row is a clean split.** In the "
        "`oracle_realistic` row the \"generalisation\" column is not pure "
        "generalisation — it also contains the class-conditional noise the "
        "substitution keeps, so it overstates the model's rubric error. Rules "
        "calibrate exactly how much: rules have zero generalisation error by "
        f"construction, so their entire {1.0 - ru['oracle_realistic']['acc']:.4f} "
        "shortfall under `oracle_realistic` IS that residual noise. Fusion's "
        f"shortfall is {1.0 - realistic:.4f}, i.e. only "
        f"**{ru['oracle_realistic']['acc'] - realistic:+.4f}** worse than a "
        "system that cannot generalise wrongly at all.",
        "",
        f"So both conditions agree that fusion's generalisation cost is now "
        f"small — {gen_onehot:.4f} by the one-hot route, "
        f"{ru['oracle_realistic']['acc'] - realistic:.4f} by the "
        "rules-as-reference route — even though their raw \"1.0 − oracle\" "
        "columns look contradictory. Perception is the shared bottleneck of "
        "both systems (rules lose "
        f"{ru['oracle_onehot']['acc'] - ru['real']['acc']:.4f} to it, fusion "
        f"{band_onehot:.4f}), which is also why the two are statistically "
        "indistinguishable on headline accuracy (`SIGNIFICANCE.md`).",
        "",
        "## Caveats", "",
        "- The 0.619 comparison is indicative, not exact: that number came from "
        "a model both TRAINED and tested on one-hot cues, and from the plain "
        "recipe. Here the model is trained on soft real vectors, so "
        "`oracle_onehot` is if anything a **lower** bound on its rubric "
        "competence.",
        "- `oracle_realistic` draws substitutes from the **val** split, which is "
        "actor-disjoint from train and is not used to build the recombination "
        "pools, so the model is not fed vectors it memorised. Substitution "
        f"stats: `{r['substitution_stats']}`.",
        "- Both oracle conditions keep scenario-designed `[missing]` cues "
        "masked, so they model perfect perception, not perfect information.",
    ]
    (G.OUT_DIR / "PERCEPTION_BAND.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
