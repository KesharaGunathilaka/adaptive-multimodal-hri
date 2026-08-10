"""Does fixing recombination-pool contamination change the embedding-fusion
result? (2026-08-08) -- `scripts/58`'s in-fold pools gave a dead tie with
probabilities (McNemar p=1.0000). This is the SAME full-recipe protocol, only
the recombination pool SOURCE changes: gesture/emotion pool vectors now come
from `scripts/60_outfold_pools.py`'s out-of-fold embeddings (scored by a
checkpoint that never trained on that clip) instead of the deployed model's
in-fold embeddings.

Design: PCA/scaler are fit on the DEPLOYED model's TRAIN embeddings, UNCHANGED
from script 58 -- that is still what real train/val/test data looks like at
evaluation time, and must stay consistent across every config for a fair
comparison. Only the pool-building step reads
`embeddings_pooled_outfold.npz` instead of the regular pooled file, and those
out-of-fold vectors are transformed through the SAME (deployed-fit) PCA/
scaler before being drawn into synthetic samples -- a linear projection
applied consistently, not a second, incompatible feature space.

    .venv/Scripts/python scripts/61_embedding_fusion_outfold.py --n-seeds 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model.generic_train import _eval_arrays, train_fusion_generic  # noqa: E402
from fusion.model.model import MODALITIES  # noqa: E402
from fusion.model.recombine_embed import build_embed_pools, generate_embed  # noqa: E402
from fusion.model.recombine_merged import build_pools as build_prob_pools  # noqa: E402
from fusion.model.recombine_merged import generate as generate_probs  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.stats import (bootstrap_diff, ci95_over_seeds,  # noqa: E402
                                          majority_vote, mcnemar)
from scripts.embedding_fusion_common import PREF_OF, PROB_DIMS  # noqa: E402
from scripts.embedding_fusion_common import load_embed_table as load_deployed_embed  # noqa: E402

OUTFOLD_PATH = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled_outfold.npz"
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True,
                missing_mode="exclude")


def load_outfold_embed_table(clip_ids: np.ndarray) -> dict:
    """Same contract as `embedding_fusion_common.load_embed_table`, reading
    `embeddings_pooled_outfold.npz` instead (already has all 4 modalities --
    `scripts/60` copies motion/context through unchanged)."""
    z = np.load(OUTFOLD_PATH, allow_pickle=True)
    idx = {cid: i for i, cid in enumerate(z["clip_id"].astype(str))}
    pos = np.array([idx.get(cid, -1) for cid in clip_ids])
    have = pos >= 0
    out = {}
    for m in MODALITIES:
        pref = PREF_OF[m]
        dim = z[f"{pref}_embed"].shape[1]
        emb = np.zeros((len(clip_ids), dim), np.float32)
        obs = np.zeros(len(clip_ids), np.float32)
        emb[have] = z[f"{pref}_embed"][pos[have]]
        obs[have] = z[f"{pref}_obs"][pos[have]]
        out[f"{pref}_embed"] = emb
        out[f"{pref}_obs"] = obs
    return out


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
        scaler = StandardScaler().fit(pca.transform(Xo))
        pcas[m], scalers[m], dims[m] = pca, scaler, k
    return pcas, scalers, dims


def transform(embed: dict, pcas: dict, scalers: dict):
    Xs, obss = [], []
    for m in MODALITIES:
        pref = PREF_OF[m]
        Xp = pcas[m].transform(embed[f"{pref}_embed"])
        Xs.append(scalers[m].transform(Xp).astype(np.float32))
        obss.append(embed[f"{pref}_obs"])
    return np.concatenate(Xs, axis=1), np.stack(obss, axis=1)


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

    # ── probability side: UNCHANGED reference (not the point of this script,
    # but needed for the significance comparisons) ──────────────────────────
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

    # ── embedding side: DEPLOYED embeddings for real train/val/test + PCA fit
    embed_tr = load_deployed_embed(splits["train"].clip_id.to_numpy())
    embed_va = load_deployed_embed(splits["val"].clip_id.to_numpy())
    embed_hl = load_deployed_embed(hl.clip_id.to_numpy())
    print("fitting PCA + scaler on DEPLOYED train embeddings (unchanged vs script 58)...",
          flush=True)
    pcas, scalers, embed_dims = fit_pca_and_scaler(embed_tr, args.n_components)

    Xe_tr, obse_tr = transform(embed_tr, pcas, scalers)
    Xe_va, obse_va = transform(embed_va, pcas, scalers)
    Xe_hl, obse_hl = transform(embed_hl, pcas, scalers)

    # ── pool source: OUT-OF-FOLD embeddings, transformed through the SAME
    # (deployed-fit) PCA/scaler ──────────────────────────────────────────
    embed_tr_outfold = load_outfold_embed_table(splits["train"].clip_id.to_numpy())
    outfold_transformed = {}
    for m in MODALITIES:
        pref = PREF_OF[m]
        Xp = pcas[m].transform(embed_tr_outfold[f"{pref}_embed"])
        outfold_transformed[f"{pref}_embed"] = scalers[m].transform(Xp).astype(np.float32)
        outfold_transformed[f"{pref}_obs"] = embed_tr_outfold[f"{pref}_obs"]

    gt_col = {m: f"gt_{m}" for m in ("emotion", "gesture", "motion", "context")}
    from scripts.realworld_eval import merged_unimodal as MU
    embed_pools = build_embed_pools(
        outfold_transformed, clips, splits["train"].clip_id.to_numpy(),
        gt_col, MU.MASK_COL)
    Xe_synth, obse_synth, ye_synth, skipped = generate_embed(
        embed_pools, embed_dims, n_per_combo=args.n_per_combo, seed=0)
    print(f"embed recombination (OUT-OF-FOLD pools): {len(ye_synth)} synthetic samples "
          f"({len(skipped)} combos skipped, empty pool)", flush=True)

    configs = {
        "probs_full": dict(Xtr=Xp_tr, obstr=obsp_tr, Xva=Xp_va, obsva=obsp_va,
                          Xhl=Xp_hl, obshl=obsp_hl, modality_dims=PROB_DIMS,
                          extra=(Xp_synth, obsp_synth, yp_synth)),
        "embed_full_outfold": dict(Xtr=Xe_tr, obstr=obse_tr, Xva=Xe_va, obsva=obse_va,
                                  Xhl=Xe_hl, obshl=obse_hl, modality_dims=embed_dims,
                                  extra=(Xe_synth, obse_synth, ye_synth)),
    }

    rules_pred = G.rule_predict(hl)
    rules_correct = rules_pred == y_hl
    rules_res = G.eval_clip(y_hl, rules_pred)
    print(f"rules: {rules_res}", flush=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            pred = _eval_arrays(model, cfg["Xhl"], cfg["obshl"], device)
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
    mc = mcnemar(preds_by_config["embed_full_outfold"] == y_hl, base_correct)
    bt = bootstrap_diff(preds_by_config["embed_full_outfold"] == y_hl, base_correct)
    comparisons["embed_full_outfold_vs_probs_full"] = {"mcnemar": mc, "bootstrap": bt}

    payload = {"n_seeds": args.n_seeds, "n_components": args.n_components,
              "n_headline_clips": int(len(hl)), "rules": rules_res,
              "results": results, "comparisons": comparisons}
    (G.OUT_DIR / "embedding_fusion_outfold_results.json").write_text(
        json.dumps(payload, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_report(payload)
    print(f"\n-> {G.OUT_DIR / 'EMBEDDING_FUSION_OUTFOLD.md'}")


def write_report(p: dict) -> None:
    ru = p["rules"]
    lines = [
        "# Embedding-level fusion with OUT-OF-FOLD recombination pools",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={p['n_headline_clips']} · {p['n_seeds']} seeds · "
        f"PCA n_components={p['n_components']} (fit on DEPLOYED train "
        "embeddings, unchanged vs `EMBEDDING_FUSION_FULL.md`) · gesture/"
        "emotion recombination pools now drawn from `scripts/60`'s "
        "out-of-fold embeddings instead of in-fold; motion/context "
        "unchanged (never contaminated).",
        "",
        "Reference (in-fold pools, `EMBEDDING_FUSION_FULL.md`): "
        "`embed_full` 0.6904±0.0070 vs `probs_full` 0.6993±0.0135, "
        "McNemar p=1.0000 (exact tie).",
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
        "If out-of-fold pools move `embed_full_outfold` meaningfully vs the "
        "in-fold `embed_full` (0.6904), that confirms pool contamination was "
        "suppressing the embedding representation's real value. If the "
        "number barely moves, contamination was not the binding constraint -- "
        "the flat tie is a property of the representation/architecture, not "
        "of dirty pools, and the dimensionality-overfitting risk flagged in "
        "`EMBEDDING_PROBE.md` is the more likely explanation.",
    ]
    (G.OUT_DIR / "EMBEDDING_FUSION_OUTFOLD.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
