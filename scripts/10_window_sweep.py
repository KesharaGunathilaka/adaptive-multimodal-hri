"""Window-size sweep (handover §8.3): rebuild window features with the
gesture/motion lookback spans scaled x0.5 / x1 / x2 (≈ W=16/32/64 @ 30 fps
equivalents) from the per-frame caches (no video re-extraction), train the
fusion model (1 seed, dropout+jitter, no recombination for comparability),
and report val/test clip accuracy vs span.

Writes results/fusion_v1/window_sweep.json.

Run from repo root:  .venv/Scripts/python scripts/10_window_sweep.py
"""
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
from fusion.extraction.windows import WindowFeaturizer  # noqa: E402
from fusion.model import train as T  # noqa: E402

PERFRAME = ROOT / "data" / "features" / "perframe"
OUT = ROOT / "results" / "fusion_v1" / "window_sweep.json"
SCALES = (0.5, 1.0, 2.0)

EMO_L = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
CTX_L = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]


def build_frame(scale, device):
    fz = WindowFeaturizer(device=device, scale=scale)
    clips = pd.read_csv(ROOT / "data" / "annotations" / "clips.csv")
    labels = pd.read_csv(ROOT / "data" / "labels.csv").set_index("scenario_id")
    rows = []
    for r in clips.itertuples():
        lab = labels.loc[r.scenario_id]
        npz = np.load(PERFRAME / f"{r.clip_id}.npz")
        for w in fz.featurize_clip(npz):
            row = {"clip_id": r.clip_id, "scenario_id": r.scenario_id,
                   "split_subject": r.split_subject, "status": lab.status,
                   "intent": lab.intent,
                   "emo_obs": w["emo_probs"] is not None,
                   "ges_obs": w["ges_probs"] is not None,
                   "mot_obs": w["mot_probs"] is not None,
                   "ctx_obs": w["ctx_probs"] is not None}
            for names, key, pref in [(EMO_L, "emo_probs", "emo"),
                                     (fz.gesture_labels, "ges_probs", "ges"),
                                     (fz.motion_labels, "mot_probs", "mot"),
                                     (CTX_L, "ctx_probs", "ctx")]:
                p = w[key]
                for i, c in enumerate(names):
                    row[f"{pref}_{c}"] = float(p[i]) if p is not None else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df.status == "train"].copy()
    df["y"] = df.intent.map(common.INTENTS.index)
    return df


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for scale in SCALES:
        t0 = time.time()
        df = build_frame(scale, device)
        splits = common.split(df)
        model, _ = T.train_fusion(splits, seed=0, dropout_p=0.3,
                                  jitter_sigma=0.15, device=device,
                                  select_masked=True, missing_mode="exclude")
        out = {}
        for sp in ("val", "test"):
            pred = T._eval_arrays(model, *T.frame_arrays(splits[sp]), device)
            out[sp] = common.evaluate(splits[sp], pred)
        results[f"x{scale}"] = {
            "ges_span_s": round(64 / 30 * scale, 3),
            "mot_span_s": round(2.0 * scale, 3),
            "n_windows": len(df),
            "val_clip": out["val"]["clip"], "test_clip": out["test"]["clip"]}
        print(f"x{scale}: {results[f'x{scale}']} ({time.time()-t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
