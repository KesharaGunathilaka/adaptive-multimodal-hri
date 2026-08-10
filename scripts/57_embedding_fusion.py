"""Embedding-level fusion: does a richer per-modality representation beat the
24-dim probability baseline? (2026-08-08)

Motivated by `PERCEPTION_BAND.md`: on the deployed `full` recipe, perception
error (0.224) now dominates fusion's remaining headroom (generalisation cost
only 0.063). Rules and the current fusion head both consume ARGMAX-COLLAPSED
probability vectors; the penultimate embedding each cue model computes
BEFORE its classifier head retains uncertainty argmax throws away. This
tests whether that retained signal helps the fusion head on cases the
classifier itself got wrong.

**PCA is required, not optional** (`EMBEDDING_PROBE.md`): raw combined width
is 2176 (1280+128+256+512) against ~1547 train clips. A linear probe found
motion's train/test gap LARGER than gesture's or emotion's despite motion
never being fine-tuned on train -- evidence that some of the risk here is
plain dimensionality/overfitting, not only fine-tune contamination. PCA is
fit on TRAIN ONLY, per modality, to a modest component count.

Three configs, same architecture family (`AttentionFusion`, generalised to
arbitrary per-modality width 2026-08-08), same seeds, NO recombination/
augmentation (isolates the representation question; embedding-level
recombination inherits `POOL_PURITY.md`'s contamination and is an explicit
follow-up, not folded in here):
  probs_only   the existing 24-dim contract (baseline, reproduced fresh here
               for an apples-to-apples paired comparison)
  embed_only   PCA-reduced embeddings only (`n_components` per modality)
  embed_probs  PCA-reduced embeddings concatenated with the 24-dim probs

Scored against rules and each other with McNemar + bootstrap
(`scripts/realworld_eval/stats.py`), matching `scripts/49_significance.py`'s
protocol.

    .venv/Scripts/python scripts/57_embedding_fusion.py
    .venv/Scripts/python scripts/57_embedding_fusion.py --n-components 32 --n-seeds 10
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model.model import AttentionFusion, MODALITIES  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.stats import (bootstrap_diff, ci95_over_seeds,  # noqa: E402
                                          majority_vote, mcnemar)
from scripts.embedding_fusion_common import PREF_OF, PROB_DIMS, load_embed_table  # noqa: E402


def fit_pca(train_embed: dict, train_obs_mask: dict, n_components: int) -> dict:
    """Per-modality PCA, fit on TRAIN's OBSERVED rows only. -> {modality: PCA}."""
    from sklearn.decomposition import PCA
    pcas = {}
    for m in MODALITIES:
        pref = PREF_OF[m]
        X = train_embed[f"{pref}_embed"]
        obs = train_obs_mask[f"{pref}_obs"].astype(bool)
        Xo = X[obs]
        k = min(n_components, Xo.shape[0] - 1, Xo.shape[1])
        pca = PCA(n_components=k, random_state=0)
        pca.fit(Xo)
        pcas[m] = pca
        print(f"  PCA[{m}]: {Xo.shape[1]}d -> {k}d, "
              f"explained_var={pca.explained_variance_ratio_.sum():.3f} "
              f"(fit on {Xo.shape[0]} observed train rows)", flush=True)
    return pcas


def transform_embed(embed: dict, pcas: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> (X [N, sum(k)], obs [N, n_modalities]), MODALITIES order."""
    Xs, obss = [], []
    for m in MODALITIES:
        pref = PREF_OF[m]
        Xs.append(pcas[m].transform(embed[f"{pref}_embed"]).astype(np.float32))
        obss.append(embed[f"{pref}_obs"])
    return np.concatenate(Xs, axis=1), np.stack(obss, axis=1)


def build_probs_xy(real: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = real[common.PROB_COLS].fillna(0.0).to_numpy(np.float32)
    obs = real[common.OBS_COLS].to_numpy(np.float32)
    return X, obs


# ── lightweight training loop (plain, no augmentation -- matches the gap-
# decomposition "plain" condition; WindowDataset is 24-dim-probability-
# specific and not reused here) ─────────────────────────────────────────────
def train_plain(Xtr, obstr, ytr, Xva, obsva, yva, modality_dims, seed,
                device, epochs=80, patience=10, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttentionFusion(missing_mode="exclude",
                            modality_dims=modality_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(obstr),
                       torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    xva_t = torch.from_numpy(Xva).to(device)
    obsva_t = torch.from_numpy(obsva).to(device)

    best_acc, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(device), ob.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xva_t, obsva_t).argmax(1).cpu().numpy() == yva).mean())
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def predict(model, X, obs, device, bs=2048):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        ob = torch.from_numpy(obs[i:i + bs]).to(device)
        preds.append(model(xb, ob).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-components", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    y_hl = hl.y.to_numpy()
    print(f"headline test n={len(hl)}", flush=True)

    # ── probability arrays ───────────────────────────────────────────────
    Xp = {s: build_probs_xy(splits[s]) for s in ("train", "val")}
    Xp_hl = build_probs_xy(hl)

    # ── embedding arrays + PCA (fit on train only) ──────────────────────
    embed = {s: load_embed_table(splits[s].clip_id.to_numpy()) for s in ("train", "val")}
    embed_hl = load_embed_table(hl.clip_id.to_numpy())
    print(f"embedding coverage: train has_embed={embed['train']['has_embed'].mean():.4f} "
          f"headline has_embed={embed_hl['has_embed'].mean():.4f}", flush=True)

    print("fitting PCA...", flush=True)
    pcas = fit_pca(embed["train"], embed["train"], args.n_components)
    embed_dims = {m: pcas[m].n_components_ for m in MODALITIES}

    Xe = {s: transform_embed(embed[s], pcas) for s in ("train", "val")}
    Xe_hl = transform_embed(embed_hl, pcas)

    ytr, yva = splits["train"].y.to_numpy(), splits["val"].y.to_numpy()

    def concat_dims(a, b):
        return {m: a[m] + b[m] for m in MODALITIES}

    configs = {
        "probs_only": dict(
            Xtr=Xp["train"][0], obstr=Xp["train"][1],
            Xva=Xp["val"][0], obsva=Xp["val"][1],
            Xhl=Xp_hl[0], obshl=Xp_hl[1],
            modality_dims=PROB_DIMS),
        "embed_only": dict(
            Xtr=Xe["train"][0], obstr=Xe["train"][1],
            Xva=Xe["val"][0], obsva=Xe["val"][1],
            Xhl=Xe_hl[0], obshl=Xe_hl[1],
            modality_dims=embed_dims),
        "embed_probs": dict(
            Xtr=np.concatenate([Xe["train"][0], Xp["train"][0]], axis=1),
            obstr=Xe["train"][1],  # obs flags identical across probs/embed (same detector)
            Xva=np.concatenate([Xe["val"][0], Xp["val"][0]], axis=1),
            obsva=Xe["val"][1],
            Xhl=np.concatenate([Xe_hl[0], Xp_hl[0]], axis=1),
            obshl=Xe_hl[1],
            modality_dims=concat_dims(embed_dims, PROB_DIMS)),
    }

    rules_pred = G.rule_predict(hl)
    rules_correct = rules_pred == y_hl
    rules_res = G.eval_clip(y_hl, rules_pred)
    print(f"rules: {rules_res}", flush=True)

    results = {}
    preds_by_config = {}
    for name, cfg in configs.items():
        print(f"\n[{name}] modality_dims={cfg['modality_dims']} "
              f"(total {sum(cfg['modality_dims'].values())}d)", flush=True)
        seed_preds, seed_accs, seed_f1s = [], [], []
        for seed in range(args.n_seeds):
            t0 = time.time()
            model, val_acc = train_plain(
                cfg["Xtr"], cfg["obstr"], ytr, cfg["Xva"], cfg["obsva"], yva,
                cfg["modality_dims"], seed, device)
            pred = predict(model, cfg["Xhl"], cfg["obshl"], device)
            res = G.eval_clip(y_hl, pred)
            seed_preds.append(pred)
            seed_accs.append(res["acc"])
            seed_f1s.append(res["macro_f1"])
            print(f"  seed{seed}: val={val_acc:.4f} test_acc={res['acc']:.4f} "
                  f"test_f1={res['macro_f1']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        seed_preds = np.stack(seed_preds)
        ens = majority_vote(seed_preds)
        preds_by_config[name] = ens
        results[name] = {
            "modality_dims": cfg["modality_dims"],
            "acc": ci95_over_seeds(seed_accs),
            "macro_f1": ci95_over_seeds(seed_f1s),
            "ensemble": G.eval_clip(y_hl, ens),
        }

    # ── pairwise significance: each config vs rules, and embed configs vs probs_only ──
    comparisons = {}
    for name in configs:
        mc = mcnemar(preds_by_config[name] == y_hl, rules_correct)
        bt = bootstrap_diff(preds_by_config[name] == y_hl, rules_correct)
        comparisons[f"{name}_vs_rules"] = {"mcnemar": mc, "bootstrap": bt}
    base_correct = preds_by_config["probs_only"] == y_hl
    for name in ("embed_only", "embed_probs"):
        mc = mcnemar(preds_by_config[name] == y_hl, base_correct)
        bt = bootstrap_diff(preds_by_config[name] == y_hl, base_correct)
        comparisons[f"{name}_vs_probs_only"] = {"mcnemar": mc, "bootstrap": bt}

    payload = {"n_seeds": args.n_seeds, "n_components": args.n_components,
               "n_headline_clips": int(len(hl)), "rules": rules_res,
               "results": results, "comparisons": comparisons}
    (G.OUT_DIR / "embedding_fusion_results.json").write_text(
        json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_report(payload)
    print(f"\n-> {G.OUT_DIR / 'EMBEDDING_FUSION.md'}")


def write_report(p: dict) -> None:
    ru = p["rules"]
    lines = [
        "# Embedding-level fusion — does richer representation beat probabilities?",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"headline test n={p['n_headline_clips']} · {p['n_seeds']} seeds · "
        f"PCA n_components={p['n_components']} per modality (fit on train, "
        "observed rows only) · plain (no recombination/augmentation) — "
        "isolates the representation question.",
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
        "`_vs_rules` favours=left means the config beat rules; `_vs_probs_only` "
        "favours=left means the embedding config beat the probability baseline. "
        "A CI containing 0 means not significant at this seed count — read "
        "`scripts/49_significance.py`'s standing lesson (σ≈0.016 needs ~10+ "
        "seeds to resolve gaps below ~0.023) before treating any single-digit "
        "point difference here as established.",
    ]
    (G.OUT_DIR / "EMBEDDING_FUSION.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
