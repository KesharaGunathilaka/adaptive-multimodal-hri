"""Study 1 — fusion ARCHITECTURE ablation on the clip-pooled cue vector (R1).

Compares a roster of fusion heads against the rule-based baseline on the exact
same clip-level input and evaluation contract as the recombination study
(`scripts/30_merged_recombination.py`), under two training configs:

  plain  real cues only, no recombination / no augmentation
  full   recombination + modality-dropout(0.3) + confidence-jitter(0.15)
         -- the validated deployed recipe

Roster (rule-based = baseline):
  rule                G.rule_predict (explicit rubric, deterministic)
  concat_mlp          the learned floor (baselines.learned.ConcatMLP), lifted
  gbt                 LightGBM on the 28-dim tabular vector
  self_attention      the incumbent AttentionFusion (== "transformer")
  cross_attention     learned-query cross-attention (fusion_zoo)
  channel_attention   SE-style modality-channel recalibration (fusion_zoo)
  gmu                 Gated Multimodal Unit (fusion_zoo)
  lmf                 Low-rank Multimodal Fusion (fusion_zoo)

The question this answers: on the pooled input, does any architecture beat the
incumbent self-attention head and the rule baseline (headline clip-acc ~0.712),
or is fusion capacity NOT the lever (the gap decomposition's implication)?

    .venv/Scripts/python scripts/38_fusion_architectures.py
    .venv/Scripts/python scripts/38_fusion_architectures.py --n-per-combo 40 --configs full
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

from fusion.model import gbt as GBT  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.fusion_zoo import ZOO, ConcatMLPFusion  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
RULE_REF = 0.712  # rules+real headline clip-acc on the promoted features

CONFIGS = {
    "plain": dict(dropout_p=0.0, jitter_sigma=0.0, recombine=False, select_masked=False),
    "full":  dict(dropout_p=0.3, jitter_sigma=0.15, recombine=True, select_masked=True),
}

# name -> no-arg factory returning a model with the forward(x, obs) contract
TORCH_FACTORIES = {
    "concat_mlp":        ConcatMLPFusion,
    "self_attention":    lambda: AttentionFusion(missing_mode="exclude"),
    "cross_attention":   ZOO["cross_attention"],
    "channel_attention": ZOO["channel_attention"],
    "gmu":               ZOO["gmu"],
    "lmf":               ZOO["lmf"],
}


def _params(factory) -> int:
    return sum(p.numel() for p in factory().parameters())


def eval_torch(name, factory, cfg, splits, hl, extra, device):
    ex = extra if cfg["recombine"] else None
    hl_runs, full_runs = [], []
    first_model = None
    for seed in SEEDS:
        t0 = time.time()
        model, val_acc = T.train_fusion(
            splits, seed=seed, dropout_p=cfg["dropout_p"],
            jitter_sigma=cfg["jitter_sigma"], extra=ex, device=device,
            select_masked=cfg["select_masked"], model_factory=factory)
        first_model = first_model or model
        pred_hl = T._eval_arrays(model, *T.frame_arrays(hl), device)
        pred_full = T._eval_arrays(model, *T.frame_arrays(splits["test"]), device)
        hl_runs.append(G.eval_clip(hl.y.to_numpy(), pred_hl))
        full_runs.append(G.eval_clip(splits["test"].y.to_numpy(), pred_full))
        print(f"    {name} seed{seed}: val={val_acc:.4f} headline={hl_runs[-1]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return hl_runs, full_runs, first_model


def eval_gbt(cfg, splits, hl, extra):
    ex = extra if cfg["recombine"] else None
    hl_runs, full_runs = [], []
    for seed in SEEDS:
        t0 = time.time()
        clf = GBT.train_gbt(splits, seed=seed, dropout_p=cfg["dropout_p"],
                            jitter_sigma=cfg["jitter_sigma"], extra=ex)
        hl_runs.append(G.eval_clip(hl.y.to_numpy(), GBT.gbt_predict(clf, hl)))
        full_runs.append(G.eval_clip(splits["test"].y.to_numpy(),
                                     GBT.gbt_predict(clf, splits["test"])))
        print(f"    gbt seed{seed}: headline={hl_runs[-1]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return hl_runs, full_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    print(f"{clips.clip_id.nunique()} clips, {real.split.value_counts().to_dict()}, "
          f"headline n={hl.clip_id.nunique()}", flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    X, obs, y, report = generate(pools, n_per_combo=args.n_per_combo, seed=0)
    extra = (X, obs, y)
    print(f"recombination: {report.n_generated} synthetic samples from "
          f"{report.n_combos - report.n_skipped_empty_pool}/{report.n_combos} "
          f"combos", flush=True)

    # rule baseline (deterministic, config-independent) ------------------------
    rule_hl = G.eval_clip(hl.y.to_numpy(), G.rule_predict(hl))
    rule_full = G.eval_clip(splits["test"].y.to_numpy(), G.rule_predict(splits["test"]))
    print(f"\nrule baseline: headline={rule_hl}  full={rule_full}", flush=True)

    params = {n: _params(f) for n, f in TORCH_FACTORIES.items()}

    results = {}
    for cname in args.configs:
        cfg = CONFIGS[cname]
        print(f"\n=== config: {cname} ({cfg}) ===", flush=True)
        results[cname] = {}
        for mname, factory in TORCH_FACTORIES.items():
            hl_runs, full_runs, model = eval_torch(mname, factory, cfg, splits,
                                                   hl, extra, device)
            results[cname][mname] = {"headline": G.agg(hl_runs),
                                     "full_test": G.agg(full_runs),
                                     "params": params[mname]}
            _log(cname, mname, cfg, results[cname][mname], report, args)
        hl_runs, full_runs = eval_gbt(cfg, splits, hl, extra)
        results[cname]["gbt"] = {"headline": G.agg(hl_runs),
                                 "full_test": G.agg(full_runs), "params": None}
        _log(cname, "gbt", cfg, results[cname]["gbt"], report, args)

    rule = {"headline": {**rule_hl, "acc_mean": rule_hl["acc"], "acc_std": 0.0,
                         "macro_f1_mean": rule_hl["macro_f1"], "macro_f1_std": 0.0},
            "full_test": {**rule_full, "acc_mean": rule_full["acc"], "acc_std": 0.0,
                          "macro_f1_mean": rule_full["macro_f1"], "macro_f1_std": 0.0},
            "params": None}

    (OUT_DIR / "fusion_architectures_results.json").write_text(
        json.dumps({"rule": rule, "configs": results}, indent=2))
    write_report(results, rule, args.n_per_combo)
    print(f"\n-> {OUT_DIR / 'FUSION_ARCHITECTURES.md'}")


def _log(cname, mname, cfg, res, report, args):
    with start_run("05_fusion_architectures", f"{mname}__{cname}",
                   dataset="final_merged", split_kind="scenarios", cues="real",
                   params={"model": mname, "config": cname, "aggregation": "clip_mean",
                           "seeds": len(SEEDS), "params": res["params"],
                           "n_per_combo": args.n_per_combo if cfg["recombine"] else 0,
                           **cfg},
                   notes="Study 1: fusion architecture ablation on R1 pooled") as run:
        h, f = res["headline"], res["full_test"]
        run.log_metrics({"headline_clip_acc": h["acc_mean"],
                         "headline_clip_acc_std": h["acc_std"],
                         "headline_clip_macro_f1": h["macro_f1_mean"],
                         "test_clip_acc": f["acc_mean"]})


ORDER = ["rule", "concat_mlp", "gbt", "self_attention", "cross_attention",
         "channel_attention", "gmu", "lmf"]
PRETTY = {"rule": "Rule-based (baseline)", "concat_mlp": "Concat-MLP (floor)",
          "gbt": "GBT (LightGBM)", "self_attention": "Self-attention / transformer",
          "cross_attention": "Cross-attention", "channel_attention": "Channel attn (CAM)",
          "gmu": "GMU (gated)", "lmf": "LMF (low-rank tensor)"}


def _cell(res):
    h = res["headline"]
    return f"{h['acc_mean']} ± {h['acc_std']} / {h['macro_f1_mean']}"


def write_report(results: dict, rule: dict, n_per_combo: int) -> None:
    configs = list(results)
    lines = [
        "# Study 1 — fusion architecture ablation (clip-pooled input, R1)", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · 3 seeds · headline test "
        "(excl. row #58 derived clips) · clip acc ± std / macro-F1 · "
        "recombination n_per_combo=" + str(n_per_combo) + ".",
        "", f"Reference: **rule-based baseline = {rule['headline']['acc_mean']}** "
        "headline clip-acc. The question is whether any *architecture* beats it "
        "and the incumbent self-attention head on the same pooled input.", "",
        "| Model | Params | " + " | ".join(PRETTY.get(c, c) or c for c in
                                            [f"{c}" for c in configs]) + " |",
        "|---|---|" + "---|" * len(configs),
    ]
    # header row uses config names
    lines[-2] = "| Model | Params | " + " | ".join(configs) + " |"
    for m in ORDER:
        if m == "rule":
            p = "—"
            cells = " | ".join(f"{rule['headline']['acc_mean']} / "
                               f"{rule['headline']['macro_f1_mean']}" for _ in configs)
            lines.append(f"| {PRETTY[m]} | {p} | {cells} |")
            continue
        present = any(m in results[c] for c in configs)
        if not present:
            continue
        p = next((results[c][m]["params"] for c in configs if m in results[c]), None)
        pstr = f"{p:,}" if p else "—"
        cells = " | ".join(_cell(results[c][m]) if m in results[c] else "—"
                           for c in configs)
        lines.append(f"| {PRETTY.get(m, m)} | {pstr} | {cells} |")

    # verdict on the best `full` (or last) config
    ref_cfg = "full" if "full" in results else configs[-1]
    scored = {m: results[ref_cfg][m]["headline"]["acc_mean"]
              for m in results[ref_cfg]}
    best = max(scored, key=scored.get)
    lines += [
        "", f"**Best architecture ({ref_cfg} config): `{best}` at "
        f"{scored[best]}** vs rule baseline {rule['headline']['acc_mean']}. "
        + ("Learned fusion beats the explicit rubric on real cues."
           if scored[best] > rule["headline"]["acc_mean"] else
           "Does not clear the rule baseline."),
        "", "Cells are headline clip-acc±std / macro-F1. `full` = recombination "
        "+ dropout(0.3) + jitter(0.15); `plain` = real cues only.",
    ]
    (OUT_DIR / "FUSION_ARCHITECTURES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
