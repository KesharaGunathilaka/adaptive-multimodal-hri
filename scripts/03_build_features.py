"""Pass 2 runner: per-frame caches -> data/features/features_v1.parquet + manifest.json.

One row per (clip, window). Probability columns are flattened with class-name
suffixes in each model's NATIVE order (see docs/MODEL_AUDIT.md). NaN probs +
*_obs=0 mean the cue was runtime-missing in that window; the scenario-designed
[MISSING] flag comes from data/labels.csv (`missing` column).

Run from repo root:  .venv/Scripts/python scripts/03_build_features.py
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction import windows as W  # noqa: E402
from fusion.extraction.perframe import EMOTION_CKPT  # noqa: E402
from fusion.extraction.windows import WindowFeaturizer  # noqa: E402

PERFRAME = ROOT / "data" / "features" / "perframe"
OUT_PARQUET = ROOT / "data" / "features" / "features_v1.parquet"
OUT_MANIFEST = ROOT / "data" / "features" / "manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    clips = pd.read_csv(ROOT / "data" / "annotations" / "clips.csv")
    labels = pd.read_csv(ROOT / "data" / "labels.csv").set_index("scenario_id")
    fz = WindowFeaturizer()

    emo_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
    ctx_labels = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]

    rows, skipped = [], []
    t0 = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        npz_path = PERFRAME / f"{r.clip_id}.npz"
        if not npz_path.exists():
            skipped.append(r.clip_id)
            continue
        lab = labels.loc[r.scenario_id]
        npz = np.load(npz_path)
        for w in fz.featurize_clip(npz):
            row = {
                "clip_id": r.clip_id, "scenario_id": r.scenario_id,
                "person_id": r.person_id, "split_subject": r.split_subject,
                "v3_row": int(lab.v3_row), "status": lab.status,
                "context_gt": lab.context, "intent": lab.intent,
                "missing_designed": lab.missing if isinstance(lab.missing, str) else "",
                "window_idx": w["window_idx"], "t_end": w["t_end"],
                "emo_obs": w["emo_probs"] is not None,
                "ges_obs": w["ges_probs"] is not None,
                "mot_obs": w["mot_probs"] is not None,
                "ctx_obs": w["ctx_probs"] is not None,
                "emo_cov": round(float(w["emo_cov"]), 3),
                "ges_cov": round(float(w["ges_cov"]), 3),
                "mot_cov": round(float(w["mot_cov"]), 3),
                "ctx_cov": float(w["ctx_cov"]),
            }
            for names, key, pref in [(emo_labels, "emo_probs", "emo"),
                                     (fz.gesture_labels, "ges_probs", "ges"),
                                     (fz.motion_labels, "mot_probs", "mot"),
                                     (ctx_labels, "ctx_probs", "ctx")]:
                p = w[key]
                for i, c in enumerate(names):
                    row[f"{pref}_{c}"] = float(p[i]) if p is not None else np.nan
            rows.append(row)
        if n % 100 == 0:
            print(f"[{n}/{len(clips)}] {len(rows)} windows", flush=True)

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    manifest = {
        "version": "v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "git_commit": commit,
        "windows": {"stride_sec": W.STRIDE_SEC, "ges_span": W.GES_SPAN,
                    "mot_span": W.MOT_SPAN, "emo_span": W.EMO_SPAN,
                    "ctx_span": W.CTX_SPAN},
        "checkpoints": {
            "emotion": {"path": str(EMOTION_CKPT.relative_to(ROOT)),
                        "sha256": sha256(EMOTION_CKPT)},
            "gesture": {"path": str(W.GESTURE_CKPT.relative_to(ROOT)),
                        "sha256": sha256(W.GESTURE_CKPT)},
            "motion": {"path": str(W.MOTION_CKPT.relative_to(ROOT)),
                       "sha256": sha256(W.MOTION_CKPT)},
            "context": "CLIP ViT-B-32 zero-shot (jetson_deploy/hf_cache)",
        },
        "class_orders": {"emotion": emo_labels, "gesture": fz.gesture_labels,
                         "motion": fz.motion_labels, "context": ctx_labels},
        "n_rows": len(df), "n_clips": df.clip_id.nunique(),
        "skipped_clips": skipped,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(df)} windows from {df.clip_id.nunique()} clips "
          f"({len(skipped)} clips had no cache) -> {OUT_PARQUET}")
    print("observed rates:",
          {c: round(float(df[c].mean()), 3)
           for c in ["emo_obs", "ges_obs", "mot_obs", "ctx_obs"]})


if __name__ == "__main__":
    main()
