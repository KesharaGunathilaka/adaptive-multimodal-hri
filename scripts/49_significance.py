"""Significance testing for the headline fusion-vs-rules claim (2026-08-08).

`results/realworld_eval_merged/SCENARIO_TEST_REPORT.md` currently states
"Fusion beats rules by 0.0071 acc -- the G1/T05 claim holds" on the strength of
fusion 0.7191 +- 0.0164 vs rules 0.712. The seed standard deviation is more
than twice the margin, so that sentence is not supported by its own numbers.
This script replaces the assertion with a measured one.

Two independent sources of variability are reported separately, because they
answer different questions and conflating them is what produced the overclaim:

  * **Seed variability** -- would a different random init change the verdict?
    Measured by training `n_seeds` models and reporting mean, std and a
    t-based 95% CI over per-seed accuracies.
  * **Test-set variability** -- would a different sample of clips change the
    verdict? Measured by bootstrapping CLIPS (not seeds) and taking the
    percentile CI of the fusion-minus-rules accuracy difference.

The verdict itself uses **McNemar's exact test** on the paired per-clip
correct/incorrect outcomes. McNemar is the right test here (rather than a
two-sample t-test on the two accuracies) because both systems are evaluated on
the SAME clips, so the comparison is paired and only the discordant clips --
those where exactly one system is right -- carry information. Rules are
deterministic, so they contribute no seed variance; McNemar is run per seed and
also for the majority-vote ensemble.

    .venv/Scripts/python scripts/49_significance.py
    .venv/Scripts/python scripts/49_significance.py --n-seeds 5 --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model import train as T  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.stats import (N_BOOT, bootstrap_diff,  # noqa: E402
                                          ci95_over_seeds, majority_vote,
                                          mcnemar)

# The deployed `full` recipe (scripts/30_merged_recombination.py's winner).
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True,
                missing_mode="exclude")

# stats.mcnemar reports the generic "a"/"b"; name them for this comparison.
_SIDE = {"a": "fusion", "b": "rules", "tie": "tie"}


def compare(fusion_correct: np.ndarray, rules_correct: np.ndarray) -> dict:
    mc = mcnemar(fusion_correct, rules_correct)
    return {**mc, "favours": _SIDE[mc["favours"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    y_true = hl.y.to_numpy()
    print(f"{clips.clip_id.nunique()} clips | headline test n={len(hl)}", flush=True)

    pools = build_pools(splits["train"], clips)
    X, obs, y, rep = generate(pools, n_per_combo=args.n_per_combo, seed=0)
    print(f"recombination: {rep.n_generated} synthetic samples from "
          f"{rep.n_combos - rep.n_skipped_empty_pool}/{rep.n_combos} combos",
          flush=True)

    # ── rules (deterministic; no seed loop) ─────────────────────────────────
    rules_pred = G.rule_predict(hl)
    rules_correct = rules_pred == y_true
    rules_res = G.eval_clip(y_true, rules_pred)
    print(f"rules: {rules_res}", flush=True)

    # ── fusion, n_seeds independent trainings ───────────────────────────────
    seed_preds, seed_accs, seed_f1s, per_seed = [], [], [], []
    for seed in range(args.n_seeds):
        t0 = time.time()
        model, val_acc = T.train_fusion(
            splits, seed=seed, extra=(X, obs, y), device=device, **FULL_CFG)
        pred = T._eval_arrays(model, *T.frame_arrays(hl), device)
        res = G.eval_clip(y_true, pred)
        mc = compare(pred == y_true, rules_correct)
        seed_preds.append(pred)
        seed_accs.append(res["acc"])
        seed_f1s.append(res["macro_f1"])
        per_seed.append({"seed": seed, "val_acc": round(float(val_acc), 4),
                         **res, "mcnemar": mc})
        print(f"  seed{seed}: acc={res['acc']:.4f} f1={res['macro_f1']:.4f} "
              f"McNemar p={mc['p_value']:.4f} ({mc['favours']}, "
              f"n10={mc['n10']} n01={mc['n01']}) [{time.time()-t0:.0f}s]",
              flush=True)

    seed_preds = np.stack(seed_preds)
    ens_pred = majority_vote(seed_preds)
    ens_correct = ens_pred == y_true
    ens_res = G.eval_clip(y_true, ens_pred)
    ens_mc = compare(ens_correct, rules_correct)
    ens_boot = bootstrap_diff(ens_correct, rules_correct)

    acc_ci = ci95_over_seeds(seed_accs)
    f1_ci = ci95_over_seeds(seed_f1s)
    # Does the per-seed CI for fusion accuracy exclude the (fixed) rules number?
    beats_rules_ci = acc_ci["ci95_low"] > rules_res["acc"]
    n_sig = sum(1 for s in per_seed
                if s["mcnemar"]["p_value"] < 0.05 and s["mcnemar"]["favours"] == "fusion")

    results = {
        "n_seeds": args.n_seeds, "n_headline_clips": int(len(hl)),
        "n_per_combo": args.n_per_combo, "n_synthetic": int(rep.n_generated),
        "rules": rules_res,
        "fusion_per_seed": per_seed,
        "fusion_acc_over_seeds": acc_ci,
        "fusion_macro_f1_over_seeds": f1_ci,
        "fusion_ensemble": ens_res,
        "ensemble_vs_rules_mcnemar": ens_mc,
        "ensemble_vs_rules_bootstrap": ens_boot,
        "seed_ci_excludes_rules": bool(beats_rules_ci),
        "n_seeds_significantly_beating_rules": n_sig,
    }
    (G.OUT_DIR / "significance_results.json").write_text(json.dumps(results, indent=2))
    write_report(results)

    if not args.no_mlflow:
        from fusion.tracking import start_run
        with start_run("03_diagnostics", "final_merged__significance",
                       dataset="final_merged", split_kind="scenarios", cues="real",
                       params={"model": "attention_fusion", "recipe": "full",
                               "seeds": args.n_seeds,
                               "n_per_combo": args.n_per_combo, **FULL_CFG},
                       notes="McNemar + bootstrap on the fusion-vs-rules headline") as run:
            run.log_metrics({
                "fusion_acc_mean": acc_ci["mean"], "fusion_acc_std": acc_ci["std"],
                "fusion_acc_ci95_low": acc_ci["ci95_low"],
                "fusion_acc_ci95_high": acc_ci["ci95_high"],
                "rules_acc": rules_res["acc"],
                "ensemble_acc": ens_res["acc"],
                "mcnemar_p": ens_mc["p_value"],
                "boot_diff_low": ens_boot["ci95_low"],
                "boot_diff_high": ens_boot["ci95_high"]})

    print(f"\n-> {G.OUT_DIR / 'SIGNIFICANCE.md'}")


def write_report(r: dict) -> None:
    a, f1, ru = r["fusion_acc_over_seeds"], r["fusion_macro_f1_over_seeds"], r["rules"]
    mc, bt, ens = r["ensemble_vs_rules_mcnemar"], r["ensemble_vs_rules_bootstrap"], r["fusion_ensemble"]
    sig = mc["p_value"] < 0.05

    lines = [
        "# Is the fusion-vs-rules headline statistically significant?",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={r['n_headline_clips']} · `full` recipe "
        f"(recombination n_per_combo={r['n_per_combo']} → {r['n_synthetic']} "
        f"synthetic samples, dropout 0.3, jitter 0.15, masked-val selection) · "
        f"**{r['n_seeds']} seeds**.",
        "",
        "Replaces the unqualified claim in `SCENARIO_TEST_REPORT.md` "
        "(\"Fusion beats rules by 0.0071 acc — the G1/T05 claim holds\"), which "
        "rested on a margin less than half its own seed standard deviation.",
        "",
        "## Headline", "",
        "| | Clip acc | Macro-F1 |",
        "|---|---|---|",
        f"| Rules (deterministic) | {ru['acc']:.4f} | {ru['macro_f1']:.4f} |",
        f"| Fusion, mean ± std over {r['n_seeds']} seeds | "
        f"{a['mean']:.4f} ± {a['std']:.4f} | {f1['mean']:.4f} ± {f1['std']:.4f} |",
        f"| Fusion, {r['n_seeds']}-seed majority vote | {ens['acc']:.4f} | "
        f"{ens['macro_f1']:.4f} |",
        "",
        f"**Fusion accuracy 95% CI over seeds: "
        f"[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]** — "
        + (f"excludes rules' {ru['acc']:.4f}, so the seed-level advantage is real."
           if r["seed_ci_excludes_rules"] else
           f"**contains rules' {ru['acc']:.4f}**, so a differently-seeded run "
           "could plausibly land at or below the rule baseline."),
        "",
        "## McNemar's exact test (paired, per clip)", "",
        "Only clips where exactly one system is correct carry information. "
        "`n10` = fusion right / rules wrong; `n01` = rules right / fusion wrong.",
        "",
        "| Comparison | n10 | n01 | discordant | p | favours |",
        "|---|---|---|---|---|---|",
        f"| {r['n_seeds']}-seed ensemble vs rules | {mc['n10']} | {mc['n01']} | "
        f"{mc['n_discordant']} | {mc['p_value']:.4f} | {mc['favours']} |",
    ]
    for s in r["fusion_per_seed"]:
        m = s["mcnemar"]
        lines.append(f"| seed {s['seed']} vs rules | {m['n10']} | {m['n01']} | "
                     f"{m['n_discordant']} | {m['p_value']:.4f} | {m['favours']} |")

    lines += [
        "",
        f"**{r['n_seeds_significantly_beating_rules']} of {r['n_seeds']} seeds "
        f"beat rules significantly (p<0.05).**",
        "",
        "## Bootstrap over clips (ensemble − rules accuracy)", "",
        f"{N_BOOT:,} resamples of the {r['n_headline_clips']} headline clips, "
        "resampled as pairs so the within-clip pairing is preserved.",
        "",
        f"- mean difference **{bt['mean_diff']:+.4f}**",
        f"- 95% CI **[{bt['ci95_low']:+.4f}, {bt['ci95_high']:+.4f}]**"
        + ("  (excludes 0)" if bt["ci95_low"] > 0 or bt["ci95_high"] < 0
           else "  (**contains 0**)"),
        f"- {bt['frac_boot_favouring_a']:.1%} of resamples favour fusion",
        "",
        "## Verdict", "",
        (f"On this test set the ensemble's advantage over rules IS statistically "
         f"significant (McNemar p={mc['p_value']:.4f})."
         if sig and mc["favours"] == "fusion" else
         f"On this test set the difference between fusion and rules is **not "
         f"statistically significant** (McNemar p={mc['p_value']:.4f}). "
         "The two systems are statistically indistinguishable on overall "
         "headline accuracy; any claim of superiority must rest on the axes "
         "where they genuinely diverge (missing-cue robustness T03, F02 "
         "recall), not on this number."),
        "",
        "Note both systems are scored against V3 intent labels that are "
        "themselves `rule_intent()`'s output, so rules+oracle = 1.000 by "
        "construction (`GAP_DECOMPOSITION_MERGED.md`). Overall accuracy on this "
        "dataset is therefore an arena that structurally favours the rule "
        "baseline; this test quantifies how much of the remaining difference is "
        "signal rather than noise.",
    ]
    (G.OUT_DIR / "SIGNIFICANCE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
