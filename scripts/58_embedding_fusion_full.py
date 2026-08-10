"""Embedding-level fusion at the DEPLOYED operating point -- recombination +
dropout + jitter + masked-val selection, not the "plain" floor
`scripts/57_embedding_fusion.py` tested (2026-08-08).

`scripts/57` found plain PCA-embeddings significantly WORSE than plain
probabilities (p=0.0014). But "plain" is the wrong arena to judge this in --
the deployed system scores 0.7133 only because recombination closes most of
the generalisation gap (`PERCEPTION_BAND.md`); comparing un-augmented
embeddings against that number would be unfair in the other direction. This
re-tests both representations THROUGH THE SAME full recipe, paired on
identical seeds via one training loop (`fusion/model/generic_train.py`), for
the comparison that actually matters: does richer representation help at the
system's best validated operating point?

Uses the SAME 448-combo recombination rubric as `recombine_merged.py`
(`fusion/model/recombine_embed.py`, in-fold pools -- built from TRAIN
embeddings, inheriting `POOL_PURITY.md`'s known contamination for the two
fine-tuned cues; flagged, not fixed, here). PCA output is z-scored (fit on
train) so a single `jitter_sigma` is meaningful across modalities of very
different raw scale.

    .venv/Scripts/python scripts/58_embedding_fusion_full.py
    .venv/Scripts/python scripts/58_embedding_fusion_full.py --n-seeds 10 --n-components 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model.generic_train import _eval_arrays, train_fusion_generic  # noqa: E402
from fusion.model.model import MODALITIES  # noqa: E402
from fusion.model.recombine_embed import build_embed_pools, generate_embed  # noqa: E402
from fusion.model.recombine_merged import build_pools as build_prob_pools  # noqa: E402
from fusion.model.recombine_merged import generate as generate_probs  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402
from scripts.realworld_eval.stats import (bootstrap_diff, ci95_over_seeds,  # noqa: E402
                                          majority_vote, mcnemar)
from scripts.embedding_fusion_common import (PREF_OF, PROB_DIMS,  # noqa: E402
                                             load_embed_table)

FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True,
                missing_mode="exclude")


def fit_pca_and_scaler(train_embed: dict, n_components: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pcas, scalers, dims = {}, {}, {}
    for m in MODALITIES:
        pref = PREF_OF[m]
        X = train_embed[f"{pref}_embed"]
        obs = train_embed[f"{pref}_obs"].astype(bool)
        Xo = X[obs]
        k = min(n_components, Xo.shape[0] - 1, Xo.shape[1])
        pca = PCA(n_components=k, random_state=0).fit(Xo)
        Xp = pca.transform(Xo)
        scaler = StandardScaler().fit(Xp)
        pcas[m], scalers[m], dims[m] = pca, scaler, k
        print(f"  PCA[{m}]: -> {k}d, explained_var="
              f"{pca.explained_variance_ratio_.sum():.3f}", flush=True)
    return pcas, scalers, dims


def transform(embed: dict, pcas: dict, scalers: dict) -> tuple[np.ndarray, np.ndarray]:
    Xs, obss = [], []
    for m in MODALITIES:
        pref = PREF_OF[m]
        Xp = pcas[m].transform(embed[f"{pref}_embed"])
        Xs.append(scalers[m].transform(Xp).astype(np.float32))
        obss.append(embed[f"{pref}_obs"])
    return np.concatenate(Xs, axis=1), np.stack(obss, axis=1)


def build_pool_ready_embed(embed_table_train: dict, pcas, scalers) -> dict:
    """PCA+scale the raw train embedding table in place (same transform used
    everywhere else), so `build_embed_pools` buckets already-transformed
    vectors -- the pools live in the SAME space the model trains on."""
    out = {}
    for m in MODALITIES:
        pref = PREF_OF[m]
        Xp = pcas[m].transform(embed_table_train[f"{pref}_embed"])
        out[f"{pref}_embed"] = scalers[m].transform(Xp).astype(np.float32)
        out[f"{pref}_obs"] = embed_table_train[f"{pref}_obs"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-components", type=int, default=32)
    ap.add_argument("--n-per-combo", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()

    G.OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    y_hl = hl.y.to_numpy()
    ytr, yva = splits["train"].y.to_numpy(), splits["val"].y.to_numpy()
    print(f"headline test n={len(hl)}", flush=True)

    # ── probability side (baseline, SAME code path for a fair paired run) ──
    def probs_xy(frame):
        X = frame[common.PROB_COLS].fillna(0.0).to_numpy(np.float32)
        obs = frame[common.OBS_COLS].to_numpy(np.float32)
        return X, obs
    Xp_tr, obsp_tr = probs_xy(splits["train"])
    Xp_va, obsp_va = probs_xy(splits["val"])
    Xp_hl, obsp_hl = probs_xy(hl)
    prob_pools = build_prob_pools(splits["train"], clips)
    Xp_synth, obsp_synth, yp_synth, rep = generate_probs(
        prob_pools, n_per_combo=args.n_per_combo, seed=0)
    print(f"prob recombination: {rep.n_generated} synthetic samples", flush=True)

    # ── embedding side ───────────────────────────────────────────────────
    embed_tr = load_embed_table(splits["train"].clip_id.to_numpy())
    embed_va = load_embed_table(splits["val"].clip_id.to_numpy())
    embed_hl = load_embed_table(hl.clip_id.to_numpy())
    print("fitting PCA + scaler on train embeddings...", flush=True)
    pcas, scalers, embed_dims = fit_pca_and_scaler(embed_tr, args.n_components)

    Xe_tr, obse_tr = transform(embed_tr, pcas, scalers)
    Xe_va, obse_va = transform(embed_va, pcas, scalers)
    Xe_hl, obse_hl = transform(embed_hl, pcas, scalers)

    gt_col = {m: f"gt_{m}" for m in ("emotion", "gesture", "motion", "context")}
    embed_tr_transformed = build_pool_ready_embed(embed_tr, pcas, scalers)
    embed_pools = build_embed_pools(
        embed_tr_transformed, clips, splits["train"].clip_id.to_numpy(),
        gt_col, MU.MASK_COL)
    Xe_synth, obse_synth, ye_synth, skipped = generate_embed(
        embed_pools, embed_dims, n_per_combo=args.n_per_combo, seed=0)
    print(f"embed recombination: {len(ye_synth)} synthetic samples "
          f"({len(skipped)} combos skipped, empty pool)", flush=True)

    configs = {
        "probs_full": dict(Xtr=Xp_tr, obstr=obsp_tr, Xva=Xp_va, obsva=obsp_va,
                          Xhl=Xp_hl, obshl=obsp_hl, modality_dims=PROB_DIMS,
                          extra=(Xp_synth, obsp_synth, yp_synth)),
        "embed_full": dict(Xtr=Xe_tr, obstr=obse_tr, Xva=Xe_va, obsva=obse_va,
                          Xhl=Xe_hl, obshl=obse_hl, modality_dims=embed_dims,
                          extra=(Xe_synth, obse_synth, ye_synth)),
    }

    rules_pred = G.rule_predict(hl)
    rules_correct = rules_pred == y_hl
    rules_res = G.eval_clip(y_hl, rules_pred)
    print(f"rules: {rules_res}", flush=True)

    results, preds_by_config = {}, {}
    for name, cfg in configs.items():
        print(f"\n[{name}] modality_dims={cfg['modality_dims']} "
              f"(total {sum(cfg['modality_dims'].values())}d)", flush=True)
        seed_preds, seed_accs, seed_f1s = [], [], []
        for seed in range(args.n_seeds):
            t0 = time.time()
            model, val_acc = train_fusion_generic(
                cfg["Xtr"], cfg["obstr"], ytr, cfg["Xva"], cfg["obsva"], yva,
                cfg["modality_dims"], MODALITIES, seed=seed, extra=cfg["extra"],
                **FULL_CFG)
            pred = _eval_arrays(model, cfg["Xhl"], cfg["obshl"], "cuda"
                                if __import__("torch").cuda.is_available() else "cpu")
            res = G.eval_clip(y_hl, pred)
            seed_preds.append(pred)
            seed_accs.append(res["acc"])
            seed_f1s.append(res["macro_f1"])
            print(f"  seed{seed}: val={val_acc:.4f} test_acc={res['acc']:.4f} "
                  f"test_f1={res['macro_f1']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        seed_preds = np.stack(seed_preds)
        ens = majority_vote(seed_preds)
        preds_by_config[name] = ens
        results[name] = {"modality_dims": cfg["modality_dims"],
                         "acc": ci95_over_seeds(seed_accs),
                         "macro_f1": ci95_over_seeds(seed_f1s),
                         "ensemble": G.eval_clip(y_hl, ens)}

    comparisons = {}
    for name in configs:
        mc = mcnemar(preds_by_config[name] == y_hl, rules_correct)
        bt = bootstrap_diff(preds_by_config[name] == y_hl, rules_correct)
        comparisons[f"{name}_vs_rules"] = {"mcnemar": mc, "bootstrap": bt}
    base_correct = preds_by_config["probs_full"] == y_hl
    mc = mcnemar(preds_by_config["embed_full"] == y_hl, base_correct)
    bt = bootstrap_diff(preds_by_config["embed_full"] == y_hl, base_correct)
    comparisons["embed_full_vs_probs_full"] = {"mcnemar": mc, "bootstrap": bt}

    payload = {"n_seeds": args.n_seeds, "n_components": args.n_components,
              "n_headline_clips": int(len(hl)), "rules": rules_res,
              "results": results, "comparisons": comparisons}
    (G.OUT_DIR / "embedding_fusion_full_results.json").write_text(
        json.dumps(payload, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_report(payload)
    print(f"\n-> {G.OUT_DIR / 'EMBEDDING_FUSION_FULL.md'}")


def write_report(p: dict) -> None:
    ru = p["rules"]
    lines = [
        "# Embedding-level fusion at the deployed (full-recipe) operating point",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={p['n_headline_clips']} · {p['n_seeds']} seeds · "
        f"PCA n_components={p['n_components']} (z-scored) · recombination + "
        "dropout 0.3 + jitter 0.15 + masked-val selection · in-fold pools "
        "(known contamination risk, see `POOL_PURITY.md`/`EMBEDDING_PROBE.md`, "
        "not fixed in this pass).",
        "",
        "Reference: probs-only full recipe already established at "
        "**0.7133 ± 0.0160** (`SIGNIFICANCE.md`, `scripts/49`). This run "
        "reproduces that number fresh (`probs_full`) via the SAME generic "
        "training loop as `embed_full`, for a paired, apples-to-apples "
        "comparison rather than citing the old number across different code.",
        "",
        "## Results", "",
        "| Config | Dims | Acc (mean±std) | Macro-F1 (mean±std) | Ensemble acc | Ensemble F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in p["results"].items():
        dims = sum(r["modality_dims"].values())
        a, f1, e = r["acc"], r["macro_f1"], r["ensemble"]
        lines.append(f"| `{name}` | {dims} | {a['mean']:.4f} ± {a['std']:.4f} | "
                     f"{f1['mean']:.4f} ± {f1['std']:.4f} | {e['acc']:.4f} | "
                     f"{e['macro_f1']:.4f} |")
    lines.append(f"| rules (reference) | — | {ru['acc']:.4f} | {ru['macro_f1']:.4f} | "
                 f"{ru['acc']:.4f} | {ru['macro_f1']:.4f} |")

    lines += ["", "## Significance", "",
             "| Comparison | n10 | n01 | p | favours | boot mean diff | boot 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for name, c in p["comparisons"].items():
        mc, bt = c["mcnemar"], c["bootstrap"]
        fav = {"a": "left", "b": "right", "tie": "tie"}[mc["favours"]]
        lines.append(f"| {name} | {mc['n10']} | {mc['n01']} | {mc['p_value']:.4f} | "
                     f"{fav} | {bt['mean_diff']:+.4f} | "
                     f"[{bt['ci95_low']:+.4f}, {bt['ci95_high']:+.4f}] |")

    lines += [
        "", "## Reading this", "",
        "If `embed_full` beats `probs_full` significantly here despite losing "
        "significantly in the plain test (`EMBEDDING_FUSION.md`), that means "
        "recombination is doing more useful work with the richer "
        "representation than dropout/jitter alone could exploit -- plausible, "
        "since embeddings give the recombination sampler more real structure "
        "to draw from per class. If it still loses or ties, the representation "
        "itself is the bottleneck, not the lack of augmentation, and the "
        "pool-purity contamination (`POOL_PURITY.md`) is the more likely "
        "explanation -- worth testing with out-of-fold pools before concluding "
        "embeddings don't help at all.",
    ]
    (G.OUT_DIR / "EMBEDDING_FUSION_FULL.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
