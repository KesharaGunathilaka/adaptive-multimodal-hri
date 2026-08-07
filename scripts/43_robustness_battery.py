"""Phase 1 robustness battery — where learned fusion can beat rules, and why
the headline accuracy comparison was never the right question.

MOTIVATION (2026-08-06). The headline numbers are fusion 0.7191 vs rules 0.712
-- a tie. That is not a failed hypothesis, it is a measurement artifact of how
the labels were made:

  * every clip's intent label comes from its V3 scenario row, and that row's
    intent is a function of its cue tuple -- i.e. `y = rule_intent(true_cues)`;
  * therefore a rule system given the TRUE cues is Bayes-optimal by
    construction, which is exactly what the gap decomposition measured
    (rules+oracle = 1.000, dead on the ceiling);
  * so rules can only lose points where PERCEPTION misreads a cue, never
    where their reasoning is wrong -- their reasoning defines correctness;
  * and recombination trains fusion on `rule_intent()` labels, making it a
    distillation student of that same teacher (see
    `fusion/model/recombine_merged.py`). We engineered fusion to converge on
    rules, then measured that it did.

The room left for fusion to genuinely win is therefore NOT clean-data accuracy.
It is the regimes where a hand-written argmax rubric structurally cannot
compete:

  A. FAIR CONTEXT -- `merged_gap.rule_predict` hands rules the clip's TRUE
     context while fusion must infer it. Measured here honestly. NOTE (checked
     before building): `rule_intent` branches on context in exactly ONE place
     (`raise_hand`, classroom vs kitchen), so this is expected to be nearly a
     no-op -- and that near-no-op is itself the finding: the rubric barely
     encodes context at all, because hand-authoring context-dependence is hard.
  B. DEGRADATION -- rules take `argmax` and discard confidence entirely, so a
     51/49 call becomes a hard commitment; fusion sees the whole distribution.
     Sweeping cue corruption should separate them. This is the G2 experiment.
  C. SAFETY OPERATING POINT -- rules emit one hard label with no confidence,
     so they are a SINGLE POINT in precision/recall space. Fusion emits a
     distribution, so F02 (emergency) recall can be tuned. A capability rules
     cannot have at any accuracy.

Fairness note for B: the deployed `full` recipe trains WITH jitter, so it is
partly adapted to this corruption. That is a legitimate advantage of learned
systems (you can train them to be robust; you cannot train a rubric), but to
keep it honest we also evaluate `recomb_only`, which sees no jitter or dropout
in training, and report both.

    .venv/Scripts/python scripts/43_robustness_battery.py
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
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.merged_gap import rule_predict_fair  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
F02 = common.INTENTS.index("F02")

# reuse the exact configs from scripts/30_merged_recombination.py
CONFIGS = {
    "full":        dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True),
    "recomb_only": dict(dropout_p=0.0, jitter_sigma=0.0, select_masked=False),
}
SIGMAS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


# ── fusion probability output (argmax is not enough for the F02 curve) ──────
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
    """Mean softmax over seeds -- the natural probabilistic ensemble, and what
    a deployed system would actually do with 3 checkpoints."""
    return np.mean([eval_probs(m, X, obs, device) for m in models], axis=0)


# `rule_predict_fair` (rule baseline with switchable true/predicted context)
# now lives in `scripts/realworld_eval/merged_gap.py` -- shared with
# `scripts/44_video_degradation.py` (Phase 2), which can't import a sibling
# script whose module name starts with a digit.


# ── B. corruption model ─────────────────────────────────────────────────────
def corrupt(tbl: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    """Gaussian noise on log-probabilities, renormalised -- the same functional
    form as the training-time confidence jitter (`fusion/model/datasets.py`),
    applied at EVAL time to both systems identically.

    Chosen over temperature-smoothing because a monotonic rescaling leaves
    `argmax` unchanged, so rules would be trivially invariant to it and the
    comparison would be rigged. Log-prob noise genuinely flips argmax at a rate
    that grows with sigma, which is the realistic failure mode: a marginally
    misread cue.

    Only OBSERVED cues are corrupted (an unobserved cue is already all-zero and
    must stay that way so `*_obs` keeps its meaning for both systems).
    """
    if sigma <= 0:
        return tbl
    rng = np.random.default_rng(seed)
    out = tbl.copy()
    for m in ("emo", "ges", "mot", "ctx"):
        cols = [f"{m}_{c}" for c in G.LABELS[m]]
        vals = out[cols].to_numpy(np.float64)
        obs = out[f"{m}_obs"].to_numpy().astype(bool)
        logp = np.log(np.clip(vals, 1e-9, None))
        logp = logp + rng.normal(0.0, sigma, logp.shape)
        e = np.exp(logp - logp.max(axis=1, keepdims=True))
        newv = e / e.sum(axis=1, keepdims=True)
        vals[obs] = newv[obs]
        out[cols] = vals.astype(np.float32)
    return out


# ── C. F02 operating point ──────────────────────────────────────────────────
def pr_at_threshold(p_f02: np.ndarray, y_true: np.ndarray, tau: float) -> dict:
    pred_pos = p_f02 >= tau
    true_pos = y_true == F02
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"tau": round(float(tau), 3), "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval].copy()
    y_true = hl.y.to_numpy()
    print(f"{clips.clip_id.nunique()} clips · headline test {len(hl)}", flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    X, obs, y, rep = generate(pools, n_per_combo=100, seed=0)
    print(f"recombination: {rep.n_generated} synthetic samples", flush=True)

    trained = {}
    for name, cfg in CONFIGS.items():
        print(f"training {name} ({len(SEEDS)} seeds)...", flush=True)
        models = []
        for seed in SEEDS:
            m, va = T.train_fusion(
                splits, seed=seed, dropout_p=cfg["dropout_p"],
                jitter_sigma=cfg["jitter_sigma"], extra=(X, obs, y),
                device=device, missing_mode="exclude",
                select_masked=cfg["select_masked"])
            models.append(m)
            print(f"  seed{seed}: val={va:.4f}", flush=True)
        trained[name] = models

    results = {}

    # ── A. fair context ────────────────────────────────────────────────────
    r_true = rule_predict_fair(hl, use_predicted_context=False)
    r_pred = rule_predict_fair(hl, use_predicted_context=True)
    n_changed = int((r_true != r_pred).sum())
    results["A_fair_context"] = {
        "rules_true_context": G.eval_clip(y_true, r_true),
        "rules_predicted_context": G.eval_clip(y_true, r_pred),
        "n_clips_changed": n_changed,
        "note": ("rule_intent branches on context in exactly one place "
                 "(raise_hand: classroom vs kitchen), so a near-zero delta is "
                 "expected and is itself the finding"),
    }
    print(f"\nA. fair context: true={results['A_fair_context']['rules_true_context']} "
          f"pred={results['A_fair_context']['rules_predicted_context']} "
          f"({n_changed} clips changed)", flush=True)

    # ── B. degradation sweep ───────────────────────────────────────────────
    print("\nB. degradation sweep", flush=True)
    sweep = []
    for sigma in SIGMAS:
        # identical corrupted table for every system at this sigma
        ctbl = corrupt(hl, sigma, seed=1234)
        Xc, obsc = T.frame_arrays(ctbl)
        row = {"sigma": sigma}
        rp = rule_predict_fair(ctbl, use_predicted_context=False)
        row["rules"] = G.eval_clip(y_true, rp)["acc"]
        for name, models in trained.items():
            probs = ensemble_probs(models, Xc, obsc, device)
            row[name] = round(float((probs.argmax(1) == y_true).mean()), 4)
        sweep.append(row)
        print(f"  sigma={sigma:<5} rules={row['rules']:.4f}  "
              + "  ".join(f"{n}={row[n]:.4f}" for n in trained), flush=True)
    results["B_degradation"] = sweep

    # ── C. F02 safety operating point ──────────────────────────────────────
    print("\nC. F02 operating point", flush=True)
    Xh, obsh = T.frame_arrays(hl)
    probs = ensemble_probs(trained["full"], Xh, obsh, device)
    p_f02 = probs[:, F02]

    rule_pred = rule_predict_fair(hl, use_predicted_context=False)
    rt = rule_pred == F02
    tp = int((rt & (y_true == F02)).sum())
    fp = int((rt & (y_true != F02)).sum())
    fn = int((~rt & (y_true == F02)).sum())
    rules_point = {"precision": tp / (tp + fp) if tp + fp else float("nan"),
                   "recall": tp / (tp + fn) if tp + fn else float("nan"),
                   "tp": tp, "fp": fp, "fn": fn}

    curve = [pr_at_threshold(p_f02, y_true, t) for t in np.linspace(0.02, 0.95, 32)]
    # fusion's recall at (or above) the rules' precision, and vice versa
    ok_prec = [c for c in curve if not np.isnan(c["precision"])
               and c["precision"] >= rules_point["precision"]]
    ok_rec = [c for c in curve if not np.isnan(c["recall"])
              and c["recall"] >= rules_point["recall"]]
    results["C_f02"] = {
        "rules_single_point": rules_point,
        "curve": curve,
        "fusion_best_recall_at_rules_precision": (
            max((c["recall"] for c in ok_prec), default=None)),
        "fusion_best_precision_at_rules_recall": (
            max((c["precision"] for c in ok_rec), default=None)),
        "n_true_f02": int((y_true == F02).sum()),
    }
    print(f"  rules point: P={rules_point['precision']:.3f} R={rules_point['recall']:.3f}",
          flush=True)
    print(f"  fusion recall @ rules' precision: "
          f"{results['C_f02']['fusion_best_recall_at_rules_precision']}", flush=True)
    print(f"  fusion precision @ rules' recall: "
          f"{results['C_f02']['fusion_best_precision_at_rules_recall']}", flush=True)

    (OUT_DIR / "robustness_battery.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    write_report(results)
    print(f"\n({time.time()-t0:.0f}s) -> {OUT_DIR / 'ROBUSTNESS_BATTERY.md'}")


def write_report(res: dict) -> None:
    A, B, C = res["A_fair_context"], res["B_degradation"], res["C_f02"]
    L = [
        "# Phase 1 — robustness battery: where fusion can actually beat rules",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test clips · {len(SEEDS)}-seed ensemble.",
        "",
        "## Why this study exists",
        "",
        "Headline accuracy is fusion **0.7191** vs rules **0.712** — a tie. That "
        "is a property of how the labels were built, not a failed hypothesis: "
        "each clip's intent comes from its V3 row, and that row's intent is a "
        "function of its cue tuple, so `y = rule_intent(true_cues)`. A rule "
        "system given true cues is therefore **Bayes-optimal by construction** "
        "— confirmed independently by the gap decomposition (rules+oracle = "
        "1.000, exactly the ceiling). Rules can only lose points to "
        "*perception* error, never to reasoning error, because their reasoning "
        "defines correctness. Recombination then trains fusion on "
        "`rule_intent()` labels, making it a distillation student of that same "
        "teacher. Clean-data accuracy was never a fair arena.",
        "",
        "These three tests probe the regimes a hand-written argmax rubric "
        "structurally cannot handle.",
        "",
        "## A — Does the rule baseline get an unfair context advantage?",
        "",
        "`merged_gap.rule_predict` hands rules the clip's TRUE context while "
        "fusion must infer it from the CLIP classifier.",
        "",
        "| Rule variant | Clip acc | Macro-F1 |",
        "|---|---|---|",
        f"| true context (as previously reported) | {A['rules_true_context']['acc']} "
        f"| {A['rules_true_context']['macro_f1']} |",
        f"| predicted context (fair) | {A['rules_predicted_context']['acc']} "
        f"| {A['rules_predicted_context']['macro_f1']} |",
        "",
        f"Only **{A['n_clips_changed']} clips** change. The advantage is real but "
        "nearly worthless — because `rule_intent` branches on context in "
        "**exactly one place** (`raise_hand`: classroom vs kitchen). "
        "**That is the finding**: the rubric is effectively a *three-cue* "
        "system. Hand-authoring context-dependence across every branch is hard, "
        "so it was never written — a structural limit of rule systems that "
        "fusion does not share (it can condition on context everywhere, for "
        "free). It also means T04-style context claims cannot be strongly "
        "supported by *this* rule baseline in its current form.",
        "",
        "## B — Degradation: the G2 experiment",
        "",
        "Gaussian noise on log-probabilities (same functional form as training "
        "jitter), renormalised, applied to the **identical** corrupted table for "
        "every system at each sigma. Chosen over temperature smoothing because "
        "a monotonic rescale leaves `argmax` untouched — rules would be "
        "trivially invariant and the comparison rigged.",
        "",
        "`full` = deployed recipe (trained with dropout+jitter, so partly "
        "adapted to this noise — a legitimate advantage of learned systems, but "
        "flagged). `recomb_only` = no dropout, no jitter in training, so its "
        "robustness is architectural rather than trained-in.",
        "",
        "| sigma | Rules | Fusion `full` | Fusion `recomb_only` |",
        "|---|---|---|---|",
    ]
    for r in B:
        L.append(f"| {r['sigma']} | {r['rules']} | {r['full']} | {r['recomb_only']} |")
    base = B[0]
    worst = B[-1]
    L += [
        "",
        f"From sigma 0 → {worst['sigma']}: rules "
        f"{base['rules']} → {worst['rules']} "
        f"(**{worst['rules']-base['rules']:+.4f}**), fusion `full` "
        f"{base['full']} → {worst['full']} "
        f"(**{worst['full']-base['full']:+.4f}**), fusion `recomb_only` "
        f"{base['recomb_only']} → {worst['recomb_only']} "
        f"(**{worst['recomb_only']-base['recomb_only']:+.4f}**).",
        "",
        "Read the *slopes*, not the intercepts: the intercept is the tie we "
        "already knew about; the slope is whether discarding confidence "
        "(`argmax`) costs you when cues get unreliable.",
        "",
        "## C — F02 emergency: a capability rules cannot have",
        "",
        "Rules emit one hard label with no confidence, so they occupy a "
        "**single point** in precision/recall space. Fusion emits a "
        "distribution, so the emergency threshold is tunable — the operating "
        "point can be moved toward recall, which is what a safety-critical HRI "
        "system actually needs.",
        "",
        f"True F02 clips in the headline test set: **{C['n_true_f02']}**.",
        "",
        f"- **Rules (single point):** precision "
        f"{C['rules_single_point']['precision']:.3f}, recall "
        f"{C['rules_single_point']['recall']:.3f} "
        f"(tp={C['rules_single_point']['tp']}, fp={C['rules_single_point']['fp']}, "
        f"fn={C['rules_single_point']['fn']}) — **not adjustable**",
        f"- **Fusion recall at rules' precision:** "
        f"{C['fusion_best_recall_at_rules_precision']}",
        f"- **Fusion precision at rules' recall:** "
        f"{C['fusion_best_precision_at_rules_recall']}",
        "",
        "| tau | Precision | Recall | TP | FP | FN |",
        "|---|---|---|---|---|---|",
    ]
    for c in C["curve"]:
        p = "n/a" if np.isnan(c["precision"]) else f"{c['precision']:.3f}"
        r = "n/a" if np.isnan(c["recall"]) else f"{c['recall']:.3f}"
        L.append(f"| {c['tau']} | {p} | {r} | {c['tp']} | {c['fp']} | {c['fn']} |")
    L += [
        "",
        "Even where fusion's *accuracy* ties rules, only fusion can be asked "
        "\"never miss an emergency, and I'll accept more false alarms.\" A "
        "rubric has no dial to turn.",
    ]
    (OUT_DIR / "ROBUSTNESS_BATTERY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
