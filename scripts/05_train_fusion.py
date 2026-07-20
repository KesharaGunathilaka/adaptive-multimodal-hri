"""Train the attention fusion model — ablation grid x 3 seeds — then run the
T03 masking sweep (attention vs concat-MLP). Writes results/fusion_v1/.

Configs:
  attn_base      no augmentation
  attn_do        + modality dropout p=0.2
  attn_do_jit    + confidence jitter sigma=0.15
  attn_full      + cue recombination (fills F10 + unrecorded V3 train rows)

Run from repo root:  .venv/Scripts/python scripts/05_train_fusion.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common, learned  # noqa: E402
from fusion.model import recombine, train as T  # noqa: E402
from fusion.model.model import MODALITIES  # noqa: E402

OUT = ROOT / "results" / "fusion_v1"
SEEDS = (0, 1, 2)

CONFIGS = {
    "attn_base":   {"dropout_p": 0.0, "jitter_sigma": 0.0,  "recombine": False},
    "attn_do":     {"dropout_p": 0.2, "jitter_sigma": 0.0,  "recombine": False},
    "attn_do_jit": {"dropout_p": 0.2, "jitter_sigma": 0.15, "recombine": False},
    "attn_full":   {"dropout_p": 0.2, "jitter_sigma": 0.15, "recombine": True},
    # robustness-selected: higher dropout + early stopping on masked-val mix
    "attn_robust": {"dropout_p": 0.3, "jitter_sigma": 0.15, "recombine": True,
                    "select_masked": True},
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    df = common.load_windows()
    splits = common.split(df)

    full_df = __import__("pandas").read_parquet(common.FEATURES)
    full_df["y"] = full_df.intent.map(common.INTENTS.index)
    synth = recombine.generate(full_df, n_per_row=400, seed=0)
    (OUT / "recombine_report.txt").write_text("\n".join(synth[3]))
    print("recombination:", len(synth[2]), "synthetic samples")

    results = {}
    best_models = {}
    for name, cfg in CONFIGS.items():
        extra = synth[:3] if cfg["recombine"] else None
        runs = []
        for seed in SEEDS:
            t0 = time.time()
            model, val_acc = T.train_fusion(
                splits, seed=seed, dropout_p=cfg["dropout_p"],
                jitter_sigma=cfg["jitter_sigma"], extra=extra, device=device,
                select_masked=cfg.get("select_masked", False))
            run = {}
            for sp in ("val", "test"):
                pred = T._eval_arrays(model, *T.frame_arrays(splits[sp]), device)
                run[sp] = common.evaluate(splits[sp], pred)
            runs.append(run)
            print(f"{name} s{seed}: val_win={val_acc:.4f} "
                  f"test_clip={run['test']['clip']} ({time.time()-t0:.0f}s)")
            if name not in best_models or val_acc > best_models[name][1]:
                best_models[name] = (model, val_acc, seed)
        agg = {}
        for sp, lvl, m in itertools.product(("val", "test"), ("window", "clip"),
                                            ("acc", "macro_f1")):
            vals = [r[sp][lvl][m] for r in runs]
            agg[f"{sp}.{lvl}.{m}"] = {"mean": round(float(np.mean(vals)), 4),
                                      "std": round(float(np.std(vals)), 4)}
        results[name] = {"runs": runs, "agg": agg}

    # ── T03 masking sweep: best attn_full vs freshly trained concat-MLP ────
    sweep_masks = ([()] + [(m,) for m in MODALITIES]
                   + list(itertools.combinations(MODALITIES, 2)))
    attn_model = best_models["attn_robust"][0]
    sweep = {"attention": {}, "concat_mlp": {}}
    for mask in sweep_masks:
        r = T.evaluate_masked(attn_model, splits["test"], mask, device)
        sweep["attention"]["+".join(mask) or "none"] = r["clip"]

    # concat-MLP (no dropout training) for the degradation contrast
    mlp_preds_fn = _concat_mlp_masked_factory(splits, device)
    for mask in sweep_masks:
        sweep["concat_mlp"]["+".join(mask) or "none"] = mlp_preds_fn(mask)

    results["masking_sweep_test_clip"] = sweep
    results["_meta"] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": SEEDS, "configs": CONFIGS,
        "best_model_seed": {k: v[2] for k, v in best_models.items()},
        "note": "val/test = actor-disjoint subject splits of recorded train scenarios",
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    torch.save(best_models["attn_robust"][0].state_dict(), OUT / "attn_robust_best.pt")
    _write_md(results)
    print(f"\nwrote {OUT}/results.json, RESULTS.md, attn_full_best.pt")


def _concat_mlp_masked_factory(splits, device):
    """Train one concat-MLP (seed 0, no modality dropout) and return
    mask -> test clip metrics."""
    import torch.nn as nn

    from fusion.model.datasets import CUE_SLICES

    Xtr, ytr = common.xy(splits["train"])
    torch.manual_seed(0)
    model = learned.ConcatMLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr.copy()),
                                       torch.from_numpy(ytr.copy())),
        batch_size=512, shuffle=True)
    Xva, yva = common.xy(splits["val"])
    xva = torch.from_numpy(Xva).to(device)
    best, state, bad = 0, None, 0
    for _ in range(60):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xva).argmax(1).cpu().numpy() == yva).mean())
        if acc > best:
            best, bad = acc, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(state)

    def run(mask):
        X, _ = common.xy(splits["test"])
        X = X.copy()
        for m in mask:
            X[:, CUE_SLICES[m]] = 0.0
            X[:, 24 + MODALITIES.index(m)] = 0.0     # obs flag
        with torch.no_grad():
            pred = model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()
        return common.evaluate(splits["test"], pred)["clip"]

    return run


def _write_md(results):
    lines = ["# Attention fusion results (features_v1, actor-disjoint test)", "",
             "## Ablations (3 seeds, test clip-level)", "",
             "| config | clip acc | clip macro-F1 |", "|---|---|---|"]
    for name in CONFIGS:
        a = results[name]["agg"]
        lines.append(f"| {name} | {a['test.clip.acc']['mean']:.3f}±{a['test.clip.acc']['std']:.3f} "
                     f"| {a['test.clip.macro_f1']['mean']:.3f}±{a['test.clip.macro_f1']['std']:.3f} |")
    lines += ["", "## T03 masking sweep (test clip acc, attention=attn_robust best seed)", "",
              "| masked | attention | concat-MLP |", "|---|---|---|"]
    sw = results["masking_sweep_test_clip"]
    for key in sw["attention"]:
        lines.append(f"| {key} | {sw['attention'][key]['acc']:.3f} "
                     f"| {sw['concat_mlp'][key]['acc']:.3f} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
