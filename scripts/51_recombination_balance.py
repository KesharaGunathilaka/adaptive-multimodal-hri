"""Does rebalancing recombination's intent prior fix the T02 `point` collapse?

Diagnosis (2026-08-08). `SCENARIO_TEST_REPORT.md` T02 shows fusion scoring
**0.038** on row #27 (classroom / angry / point / walking -> F06) against rules'
0.642, and **0.381** on row #55 (kitchen / happy / point / walking -> F03)
against 0.571. Probing those clips' cue vectors rules perception OUT as the
cause -- they are clean (row #27: anger p=0.80, point p=0.65, walking p=0.81,
and rules read them correctly 64% of the time).

The cause is the recombination sampler's prior. A constant `n_per_combo` gives
every cue COMBO equal weight, so each intent's share of the synthetic set is
proportional to how many combos map to it. That is skewed globally (F01 104/448
combos = 23.2%, F10 8/448 = 1.8%) and, decisively, WITHIN a cue family:

    Anger + point  ->  F06 2/8 (25%)  vs  F07 6/8 (75%)
    Happy + point  ->  F03 2/8 (25%)  vs  F05 6/8 (75%)

Both collapsing rows sit in the 25% minority branch of their own family, so the
model learns the 3:1 majority and answers F07/F05 even when motion clearly says
walking. `fusion/model/recombine_merged.py::allocate` adds three corrections;
this script measures them.

Note the distinction between the GLOBAL and the CONDITIONAL prior, which the
first run of this script established empirically: `intent` equalises the global
marginal but leaves `Anger+point` at 1:2.80 (from 1:3.00), and row #27 stayed
broken; `family` equalises within the (context, emotion, gesture) family and
takes it to 1:1.00, which is the ratio that actually governs that decision.

**Selection protocol:** the balance mode is chosen on VALIDATION macro-F1, and
the test split is read once per mode purely for reporting. Choosing the mode by
its test score would be tuning on test, which this project's own rules forbid
(`HANDOVER_CLAUDE.md` §11) -- so the selected mode is named before its test
number is used for any claim.

    .venv/Scripts/python scripts/51_recombination_balance.py
    .venv/Scripts/python scripts/51_recombination_balance.py --n-seeds 3 --no-mlflow
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
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.stats import (bootstrap_diff, ci95_over_seeds,  # noqa: E402
                                          majority_vote, mcnemar)

FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True,
                missing_mode="exclude")
MODES = ("uniform", "sqrt", "intent", "family")
SPOTLIGHT_ROWS = (27, 55)          # the two collapsing `point` rows
PRESENT = [i for i, c in enumerate(common.INTENTS) if c != "F09"]


def per_class_recall(y_true, y_pred) -> dict:
    r = recall_score(y_true, y_pred, labels=PRESENT, average=None, zero_division=0)
    return {common.INTENTS[c]: round(float(v), 4) for c, v in zip(PRESENT, r)}


def row_accuracy(tbl: pd.DataFrame, pred: np.ndarray) -> dict:
    out = {}
    y = tbl.y.to_numpy()
    rows = tbl.v3_row.to_numpy()
    for r in SPOTLIGHT_ROWS:
        m = rows == r
        out[f"row_{int(r)}"] = round(float((pred[m] == y[m]).mean()), 4) if m.any() else None
    return out


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
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    y_true, y_val = hl.y.to_numpy(), splits["val"].y.to_numpy()
    print(f"headline test n={len(hl)} | val n={len(splits['val'])}", flush=True)

    pools = build_pools(splits["train"], clips)
    rules_pred = G.rule_predict(hl)
    rules_correct = rules_pred == y_true
    rules = {**G.eval_clip(y_true, rules_pred),
             "per_class_recall": per_class_recall(y_true, rules_pred),
             **row_accuracy(hl, rules_pred)}
    print(f"rules: acc={rules['acc']} macro_f1={rules['macro_f1']} "
          f"row27={rules['row_27']} row55={rules['row_55']}", flush=True)

    results = {}
    preds_by_mode = {}
    for mode in MODES:
        X, obs, y, rep = generate(pools, n_per_combo=args.n_per_combo, seed=0,
                                  balance=mode)
        print(f"\n[{mode}] {rep.n_generated} synthetic samples | "
              f"intent totals {rep.label_counts}", flush=True)

        val_f1s, val_accs, te_accs, te_f1s, seed_preds = [], [], [], [], []
        for seed in range(args.n_seeds):
            t0 = time.time()
            model, _ = T.train_fusion(splits, seed=seed, extra=(X, obs, y),
                                      device=device, **FULL_CFG)
            pv = T._eval_arrays(model, *T.frame_arrays(splits["val"]), device)
            pt = T._eval_arrays(model, *T.frame_arrays(hl), device)
            val_accs.append(float((pv == y_val).mean()))
            val_f1s.append(float(f1_score(y_val, pv, labels=PRESENT,
                                          average="macro", zero_division=0)))
            r = G.eval_clip(y_true, pt)
            te_accs.append(r["acc"])
            te_f1s.append(r["macro_f1"])
            seed_preds.append(pt)
            print(f"  seed{seed}: val_f1={val_f1s[-1]:.4f} test_acc={r['acc']:.4f} "
                  f"test_f1={r['macro_f1']:.4f} [{time.time()-t0:.0f}s]", flush=True)

        ens = majority_vote(np.stack(seed_preds))
        preds_by_mode[mode] = ens
        results[mode] = {
            "n_synthetic": int(rep.n_generated),
            "intent_totals": rep.label_counts,
            "val_macro_f1": ci95_over_seeds(val_f1s),
            "val_acc": ci95_over_seeds(val_accs),
            "test_acc": ci95_over_seeds(te_accs),
            "test_macro_f1": ci95_over_seeds(te_f1s),
            "ensemble": {**G.eval_clip(y_true, ens),
                         "per_class_recall": per_class_recall(y_true, ens),
                         **row_accuracy(hl, ens)},
        }

    # ── selection on VALIDATION macro-F1 (never on test) ────────────────────
    selected = max(MODES, key=lambda m: results[m]["val_macro_f1"]["mean"])
    print(f"\nSELECTED on val macro-F1: {selected}", flush=True)

    sel_correct = preds_by_mode[selected] == y_true
    mc = mcnemar(sel_correct, rules_correct)
    bt = bootstrap_diff(sel_correct, rules_correct)
    verdict = {"selected_mode": selected,
               "mcnemar": {**mc, "favours": {"a": "fusion", "b": "rules",
                                             "tie": "tie"}[mc["favours"]]},
               "bootstrap": bt}

    payload = {"n_seeds": args.n_seeds, "n_per_combo": args.n_per_combo,
               "n_headline_clips": int(len(hl)), "rules": rules,
               "modes": results, "verdict": verdict}
    (G.OUT_DIR / "recombination_balance_results.json").write_text(
        json.dumps(payload, indent=2))
    write_report(payload)

    if not args.no_mlflow:
        from fusion.tracking import start_run
        for mode in MODES:
            r = results[mode]
            with start_run("04_recombination", f"final_merged__balance_{mode}",
                           dataset="final_merged", split_kind="scenarios", cues="real",
                           params={"model": "attention_fusion", "recipe": "full",
                                   "balance": mode, "seeds": args.n_seeds,
                                   "n_per_combo": args.n_per_combo,
                                   "n_synthetic": r["n_synthetic"], **FULL_CFG},
                           notes="recombination intent-prior rebalancing (T02 point fix)") as run:
                run.log_metrics({
                    "val_macro_f1": r["val_macro_f1"]["mean"],
                    "headline_clip_acc": r["test_acc"]["mean"],
                    "headline_clip_acc_std": r["test_acc"]["std"],
                    "headline_clip_macro_f1": r["test_macro_f1"]["mean"],
                    "ensemble_acc": r["ensemble"]["acc"],
                    "ensemble_macro_f1": r["ensemble"]["macro_f1"]})

    print(f"\n-> {G.OUT_DIR / 'RECOMBINATION_BALANCE.md'}")


def write_report(p: dict) -> None:
    ru, modes, v = p["rules"], p["modes"], p["verdict"]
    sel = v["selected_mode"]
    lines = [
        "# Rebalancing recombination's intent prior (T02 `point` fix)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={p['n_headline_clips']} · {p['n_seeds']} seeds per mode · "
        f"budget fixed at ~{p['n_per_combo'] * 448:,} synthetic samples for every mode.",
        "",
        "## Why", "",
        "Fusion scored 0.038 on row #27 and 0.381 on row #55 despite **clean "
        "cues** (row #27: anger 0.80, point 0.65, walking 0.81 — rules read the "
        "same vectors correctly 64% of the time). Perception is not the cause. "
        "Uniform `n_per_combo` weights cue COMBOS equally, which makes each "
        "intent's synthetic share proportional to its combo count — and inside "
        "a cue family that is lopsided:",
        "",
        "| Cue family | Majority branch | Minority branch |",
        "|---|---|---|",
        "| `Anger + point` | F07 6/8 (75%) | **F06 2/8 (25%)** ← row #27 |",
        "| `Happy + point` | F05 6/8 (75%) | **F03 2/8 (25%)** ← row #55 |",
        "",
        "## Modes", "",
        "| Mode | Intent share of synthetic set |",
        "|---|---|",
    ]
    for m in MODES:
        tot = sum(modes[m]["intent_totals"].values())
        share = ", ".join(f"{k} {100*vv/tot:.0f}%"
                          for k, vv in sorted(modes[m]["intent_totals"].items()))
        lines.append(f"| `{m}` | {share} |")

    lines += [
        "",
        "## Results", "",
        f"**Mode selected on VALIDATION macro-F1: `{sel}`.** Test columns are "
        "reported for all modes for transparency, but played no part in the "
        "choice.",
        "",
        "| Mode | Val macro-F1 | Test acc | Test macro-F1 | Ensemble acc | Ensemble macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODES:
        r = modes[m]
        star = " **←selected**" if m == sel else ""
        lines.append(
            f"| `{m}`{star} | {r['val_macro_f1']['mean']:.4f} | "
            f"{r['test_acc']['mean']:.4f} ± {r['test_acc']['std']:.4f} | "
            f"{r['test_macro_f1']['mean']:.4f} ± {r['test_macro_f1']['std']:.4f} | "
            f"{r['ensemble']['acc']:.4f} | {r['ensemble']['macro_f1']:.4f} |")
    lines.append(f"| rules (reference) | — | {ru['acc']:.4f} | {ru['macro_f1']:.4f} | "
                 f"{ru['acc']:.4f} | {ru['macro_f1']:.4f} |")

    lines += [
        "",
        "## The two spotlight rows (ensemble)", "",
        "| Mode | row #27 (F06) | row #55 (F03) |",
        "|---|---|---|",
    ]
    for m in MODES:
        e = modes[m]["ensemble"]
        lines.append(f"| `{m}` | {e['row_27']} | {e['row_55']} |")
    lines.append(f"| rules | {ru['row_27']} | {ru['row_55']} |")

    lines += ["", "## Per-intent recall (ensemble)", "",
              "| Mode | " + " | ".join(sorted(ru["per_class_recall"])) + " |",
              "|---|" + "---|" * len(ru["per_class_recall"])]
    for m in MODES:
        rc = modes[m]["ensemble"]["per_class_recall"]
        lines.append(f"| `{m}` | " + " | ".join(f"{rc[k]:.3f}" for k in sorted(rc)) + " |")
    lines.append("| rules | " + " | ".join(f"{ru['per_class_recall'][k]:.3f}"
                                           for k in sorted(ru["per_class_recall"])) + " |")

    mc, bt = v["mcnemar"], v["bootstrap"]
    lines += [
        "",
        f"## Selected mode (`{sel}`) vs rules — paired tests", "",
        f"- McNemar exact: n10={mc['n10']} (fusion right/rules wrong), "
        f"n01={mc['n01']}, p={mc['p_value']:.4f}, favours **{mc['favours']}**",
        f"- Bootstrap over clips: mean diff {bt['mean_diff']:+.4f}, "
        f"95% CI [{bt['ci95_low']:+.4f}, {bt['ci95_high']:+.4f}]"
        + ("  (excludes 0)" if bt["ci95_low"] > 0 or bt["ci95_high"] < 0
           else "  (**contains 0**)"),
        f"- {bt['frac_boot_favouring_a']:.1%} of resamples favour fusion",
        "",
        ("**Significant.**" if mc["p_value"] < 0.05 else
         "**Not significant on overall accuracy** — read the per-intent recall "
         "and spotlight-row tables instead, which is where rebalancing acts."),
    ]
    (G.OUT_DIR / "RECOMBINATION_BALANCE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
