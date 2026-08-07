"""F02 (emergency) recall fix via class-weighted loss.

Phase 1 (`scripts/43_robustness_battery.py` part C, `ROBUSTNESS_BATTERY.md`)
found a genuine, specific fusion weakness: fusion's F02 recall CEILING across
the entire threshold sweep (0.645) never reaches rules' fixed-point recall
(0.727), at any threshold. That is a training-time deficiency (the model
itself under-predicts F02), not a thresholding problem -- so the fix has to
happen in training, not at inference.

Lever: class-weighted `CrossEntropyLoss`, isolated to ONLY F02 (every other
class stays at weight 1.0) -- the same mechanism that fixed emotion/gesture's
per-class recall gaps during unimodal fine-tuning
(`unimodal-finetune-promotion` memory), applied here to fusion instead.
Isolating the intervention to one class keeps the experiment legible: any
accuracy change elsewhere is a side-effect to report, not something the
sweep is trying to fix.

Sweeps F02 weight in {1.0 (control, = the deployed `full` recipe),
2.0, 3.0, 5.0}, 3 seeds each, and reports both headline accuracy (does the
fix cost overall performance?) and the SAME threshold-swept F02
precision/recall curve Phase 1 used (does it actually close the recall gap
against rules' 0.727?).

    .venv/Scripts/python scripts/46_f02_recall_fix.py
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

from fusion.baselines import common  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.merged_gap import rule_predict_fair  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True)
F02 = common.INTENTS.index("F02")
F02_WEIGHTS = [1.0, 2.0, 3.0, 5.0]

# Phase 1 reference point (rules' fixed, non-adjustable operating point)
RULES_F02_PRECISION = 0.481
RULES_F02_RECALL = 0.727


@torch.no_grad()
def eval_probs(model, X, obs, device, bs=2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        ob = torch.from_numpy(obs[i:i + bs]).to(device)
        out.append(torch.softmax(model(xb, ob), dim=1).cpu().numpy())
    return np.concatenate(out)


def ensemble_probs(models, X, obs, device) -> np.ndarray:
    return np.mean([eval_probs(m, X, obs, device) for m in models], axis=0)


def pr_at_threshold(p_f02: np.ndarray, y_true: np.ndarray, tau: float) -> dict:
    pred_pos = p_f02 >= tau
    true_pos = y_true == F02
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"tau": round(float(tau), 3), "precision": prec, "recall": rec}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval].copy()
    y_true = hl.y.to_numpy()
    print(f"{clips.clip_id.nunique()} clips · headline test {len(hl)} · "
         f"true F02 clips: {int((y_true == F02).sum())}", flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    X, obs, y, rep = generate(pools, n_per_combo=100, seed=0)
    print(f"recombination: {rep.n_generated} synthetic samples", flush=True)

    rule_pred = rule_predict_fair(hl, use_predicted_context=False)
    rule_hit = rule_pred == F02
    rule_true = y_true == F02
    rules_p = (rule_hit & rule_true).sum() / max(1, rule_hit.sum())
    rules_r = (rule_hit & rule_true).sum() / max(1, rule_true.sum())
    print(f"rules F02 (reproduced): precision={rules_p:.3f} recall={rules_r:.3f}",
         flush=True)

    results = {}
    for w_f02 in F02_WEIGHTS:
        tag = f"w{w_f02}"
        weights = np.ones(10, np.float32)
        weights[F02] = w_f02
        print(f"\n=== F02 weight = {w_f02} ===", flush=True)
        models, headline_runs = [], []
        for seed in SEEDS:
            t1 = time.time()
            m, va = T.train_fusion(
                splits, seed=seed, dropout_p=FULL_CFG["dropout_p"],
                jitter_sigma=FULL_CFG["jitter_sigma"], extra=(X, obs, y),
                device=device, missing_mode="exclude",
                select_masked=FULL_CFG["select_masked"], class_weights=weights)
            models.append(m)
            pred = T._eval_arrays(m, *T.frame_arrays(hl), device)
            r = G.eval_clip(y_true, pred)
            headline_runs.append(r)
            print(f"  seed{seed}: val={va:.4f} headline={r} ({time.time()-t1:.0f}s)",
                 flush=True)

        Xh, obsh = T.frame_arrays(hl)
        probs = ensemble_probs(models, Xh, obsh, device)
        p_f02 = probs[:, F02]
        curve = [pr_at_threshold(p_f02, y_true, t) for t in np.linspace(0.02, 0.95, 32)]
        best_recall = max((c["recall"] for c in curve if not np.isnan(c["recall"])),
                          default=float("nan"))
        recall_at_rules_precision = max(
            (c["recall"] for c in curve
             if not np.isnan(c["precision"]) and c["precision"] >= RULES_F02_PRECISION),
            default=None)

        results[tag] = {
            "f02_weight": w_f02,
            "headline": G.agg(headline_runs),
            "f02_recall_ceiling": round(float(best_recall), 4),
            "f02_recall_at_rules_precision": recall_at_rules_precision,
            "beats_rules_recall": bool(best_recall >= RULES_F02_RECALL),
            "curve": curve,
        }
        print(f"  headline acc={results[tag]['headline']['acc_mean']} "
             f"F02 recall ceiling={best_recall:.3f} "
             f"(rules={RULES_F02_RECALL}) "
             f"{'BEATS RULES' if best_recall >= RULES_F02_RECALL else 'still below rules'}",
             flush=True)

    (OUT_DIR / "f02_recall_fix_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, rules_p, rules_r)
    print(f"\n({time.time()-t0:.0f}s) -> {OUT_DIR / 'F02_RECALL_FIX.md'}")


def write_report(results, rules_p, rules_r):
    L = [
        "# F02 (emergency) recall fix — class-weighted loss",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        "headline test · isolated intervention: ONLY F02's loss weight "
        "changes, every other class stays at 1.0.",
        "",
        f"**Reference (Phase 1, `ROBUSTNESS_BATTERY.md` part C):** rules are a "
        f"fixed, non-adjustable point at precision={rules_p:.3f}, "
        f"recall={rules_r:.3f}. The unweighted `full` model's F02 recall "
        "ceiling across its entire threshold sweep was 0.645 — never reaching "
        "rules' 0.727, at any threshold. That is what this fix targets.",
        "",
        "| F02 weight | Headline acc | Headline macro-F1 | F02 recall ceiling | "
        "F02 recall @ rules' precision | Beats rules' recall (0.727)? |",
        "|---|---|---|---|---|---|",
    ]
    for tag, r in results.items():
        h = r["headline"]
        beats = "**YES**" if r["beats_rules_recall"] else "no"
        rap = ("n/a" if r["f02_recall_at_rules_precision"] is None
              else f"{r['f02_recall_at_rules_precision']:.3f}")
        L.append(f"| {r['f02_weight']} | {h['acc_mean']} ± {h['acc_std']} | "
                 f"{h['macro_f1_mean']} | {r['f02_recall_ceiling']:.3f} | {rap} | {beats} |")

    baseline = results.get("w1.0")
    best = max(results.values(), key=lambda r: r["f02_recall_ceiling"])
    L += ["", ""]
    if baseline:
        acc_cost = best["headline"]["acc_mean"] - baseline["headline"]["acc_mean"]
        L.append(
            f"Best F02 recall ceiling: **{best['f02_recall_ceiling']:.3f}** at "
            f"weight={best['f02_weight']} "
            f"({'now beats' if best['beats_rules_recall'] else 'still short of'} "
            f"rules' {RULES_F02_RECALL}). "
            f"Headline accuracy cost vs the unweighted baseline: "
            f"{acc_cost:+.4f} ({baseline['headline']['acc_mean']} -> "
            f"{best['headline']['acc_mean']}).")
    (OUT_DIR / "F02_RECALL_FIX.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
