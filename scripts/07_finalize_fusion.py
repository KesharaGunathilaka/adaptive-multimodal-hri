"""Train the FINAL deployable fusion model and save it under jetson_deploy/fusion/.

Config (per DECISIONS.md 2026-07-17): attention with missing_mode='exclude',
modality dropout 0.3, confidence jitter 0.15, cue recombination, masked-val
early stopping. 3 seeds; the seed with the best masked-val score is kept.

Run from repo root:  .venv/Scripts/python scripts/07_finalize_fusion.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model import recombine, train as T  # noqa: E402

OUT = ROOT / "jetson_deploy" / "fusion"
CFG = {"missing_mode": "exclude", "dropout_p": 0.3, "jitter_sigma": 0.15,
       "recombine": True, "select_masked": True, "n_per_row": 400}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    df = common.load_windows()
    splits = common.split(df)
    full_df = pd.read_parquet(common.FEATURES)
    full_df["y"] = full_df.intent.map(common.INTENTS.index)
    X_s, obs_s, y_s, _ = recombine.generate(full_df, n_per_row=CFG["n_per_row"], seed=0)

    best = None
    for seed in (0, 1, 2):
        t0 = time.time()
        model, val_acc = T.train_fusion(
            splits, seed=seed, dropout_p=CFG["dropout_p"],
            jitter_sigma=CFG["jitter_sigma"], extra=(X_s, obs_s, y_s),
            device=device, select_masked=True, missing_mode="exclude")
        Xva, obs_va = T.frame_arrays(splits["val"])
        yva = splits["val"]["y"].to_numpy()
        score, _ = T._masked_val_acc(model, Xva, obs_va, yva, device)
        pred = T._eval_arrays(model, *T.frame_arrays(splits["test"]), device)
        test = common.evaluate(splits["test"], pred)
        print(f"seed {seed}: masked_val={score:.4f} val={val_acc:.4f} "
              f"test_clip={test['clip']} ({time.time()-t0:.0f}s)")
        if best is None or score > best["score"]:
            best = {"seed": seed, "score": score, "model": model, "test": test}

    torch.save(best["model"].state_dict(), OUT / "fusion_attn.pt")
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": CFG, "chosen_seed": best["seed"],
        "masked_val_score": round(best["score"], 4),
        "test_clip": best["test"]["clip"],
        "features": "features_v1.parquet",
        "intents": common.INTENTS,
        "input": "x[24]=emo(7,RAF-DB order)|ges(8)|mot(4)|ctx(5), obs[4] float 1/0",
        "arch": "AttentionFusion d=64 heads=4 layers=2 ff=128 missing_mode=exclude",
    }
    (OUT / "fusion_config.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {OUT/'fusion_attn.pt'} (seed {best['seed']}, "
          f"test clip {best['test']['clip']})")


if __name__ == "__main__":
    main()
