"""Robustness follow-up (motivated by the first T03 sweep, WORKLOG 2026-07-17):
the token-substitution attention model degraded WORSE under masking than the
plain concat-MLP. Two candidate fixes, evaluated head-to-head:

  A. attn_exclude — attention with missing cues EXCLUDED via key-padding mask
     (architectural marginalization) instead of a learned [MISSING] token.
  B. mlp_do — concat-MLP trained WITH modality dropout (fair comparison: is
     the MLP's grace architectural or would it too improve with dropout?).

Both use dropout 0.3 + jitter 0.15 + recombination + masked-val selection,
3 seeds. Writes results/fusion_v1/robustness.json + updates RESULTS.md picture.

Run from repo root:  .venv/Scripts/python scripts/06_robustness_experiments.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common, learned  # noqa: E402
from fusion.model import recombine, train as T  # noqa: E402
from fusion.model.datasets import CUE_SLICES, WindowDataset  # noqa: E402
from fusion.model.model import MODALITIES  # noqa: E402

OUT = ROOT / "results" / "fusion_v1"
SEEDS = (0, 1, 2)
MASKS = ([()] + [(m,) for m in MODALITIES]
         + list(itertools.combinations(MODALITIES, 2)))


def train_mlp_do(splits, extra, seed, device):
    """Concat-MLP trained on the same dropout/jitter/recombination stream."""
    torch.manual_seed(seed)
    ds = WindowDataset(splits["train"], dropout_p=0.3, jitter_sigma=0.15,
                       seed=seed, extra=extra)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)
    Xva, yva = common.xy(splits["val"])
    xva = torch.from_numpy(Xva).to(device)
    model = learned.ConcatMLP(28).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = 0, None, 0
    for _ in range(80):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            inp = torch.cat([xb, ob], dim=1).to(device)
            nn.functional.cross_entropy(model(inp), yb.to(device)).backward()
            opt.step()
        model.eval()
        accs = []
        with torch.no_grad():
            accs.append(float((model(xva).argmax(1).cpu().numpy() == yva).mean()))
            for m in MODALITIES:                      # masked-val selection too
                X2 = Xva.copy()
                X2[:, CUE_SLICES[m]] = 0.0
                X2[:, 24 + MODALITIES.index(m)] = 0.0
                x2 = torch.from_numpy(X2).to(device)
                accs.append(float((model(x2).argmax(1).cpu().numpy() == yva).mean()))
        score = float(np.mean(accs))
        if score > best:
            best, bad = score, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    return model


def mlp_masked_eval(model, frame, mask, device):
    X, _ = common.xy(frame)
    X = X.copy()
    for m in mask:
        X[:, CUE_SLICES[m]] = 0.0
        X[:, 24 + MODALITIES.index(m)] = 0.0
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()
    return common.evaluate(frame, pred)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = common.load_windows()
    splits = common.split(df)
    full_df = pd.read_parquet(common.FEATURES)
    full_df["y"] = full_df.intent.map(common.INTENTS.index)
    X_s, obs_s, y_s, _ = recombine.generate(full_df, n_per_row=400, seed=0)
    extra = (X_s, obs_s, y_s)

    out = {}
    for name in ("attn_exclude", "mlp_do"):
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            if name == "attn_exclude":
                model, _ = T.train_fusion(
                    splits, seed=seed, dropout_p=0.3, jitter_sigma=0.15,
                    extra=extra, device=device, select_masked=True,
                    missing_mode="exclude")
                evals = {"+".join(m) or "none":
                         T.evaluate_masked(model, splits["test"], m, device)["clip"]
                         for m in MASKS}
            else:
                model = train_mlp_do(splits, extra, seed, device)
                evals = {"+".join(m) or "none":
                         mlp_masked_eval(model, splits["test"], m, device)["clip"]
                         for m in MASKS}
            per_seed.append(evals)
            print(f"{name} s{seed}: none={evals['none']['acc']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        agg = {}
        for key in per_seed[0]:
            accs = [r[key]["acc"] for r in per_seed]
            f1s = [r[key]["macro_f1"] for r in per_seed]
            agg[key] = {"acc_mean": round(float(np.mean(accs)), 4),
                        "acc_std": round(float(np.std(accs)), 4),
                        "f1_mean": round(float(np.mean(f1s)), 4)}
        out[name] = {"per_seed": per_seed, "agg": agg}

    (OUT / "robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'masked':18} {'attn_exclude':>14} {'mlp_do':>10}")
    for key in out["attn_exclude"]["agg"]:
        a = out["attn_exclude"]["agg"][key]
        b = out["mlp_do"]["agg"][key]
        print(f"{key:18} {a['acc_mean']:.3f}±{a['acc_std']:.3f} "
              f"{b['acc_mean']:.3f}±{b['acc_std']:.3f}")


if __name__ == "__main__":
    main()
