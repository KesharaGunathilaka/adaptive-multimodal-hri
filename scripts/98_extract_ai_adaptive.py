"""Run the AI new-context clips (Hospital / Museum / Kitchen) through the SAME
adaptive cue pipeline the v4 fusion model was trained on, so T04 measures a
change of ENVIRONMENT and not a change of cue pipeline.

The AI clips already have cue measurements in `hri_fusion-engine`'s feature
table, but that pipeline measures cues differently (its rule baseline scores
0.51 where the adaptive pipeline scores 0.66 on identical clips). Feeding those
to a model trained on adaptive cues would confound the two effects.

Writes `results/fusion_v2/ai_clips_adaptive_cues.parquet`:
    clip_id, scenario_dir, context_folder, intent,
    emo_* (7), ges_* (8), mot_* (4), ctx_* (5), *_obs
i.e. the same clip-level pooled probability layout `merged_gap.build_real`
produces for recorded clips.

Resumable: clips already in the output parquet are skipped.

    .venv/Scripts/python scripts/98_extract_ai_adaptive.py
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AI_ROOT = ROOT / "data" / "Dataset v3"
OUT = ROOT / "results" / "fusion_v2" / "ai_clips_adaptive_cues.parquet"
CTX_DIRS = {"hospita": "hospital", "museum": "museum", "kitchen": "kitchen"}
EMO = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GES = ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
       "raise_hand", "both_hands_up"]
MOT = ["sitting", "standing", "walking", "stepping_back"]
CTX = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]


def find_clips():
    out = []
    for d, ctx in CTX_DIRS.items():
        base = AI_ROOT / d
        if not base.exists():
            continue
        for scen in sorted(base.iterdir()):
            if scen.is_dir():
                for mp4 in sorted(scen.glob("*.mp4")):
                    out.append((mp4.stem, scen.name, ctx, mp4))
    return out


def pool(rows, key, n):
    """mean over windows where the cue fired -> (vector, observed flag)."""
    vecs = [r[key] for r in rows if r.get(key) is not None]
    if not vecs:
        return np.zeros(n, np.float32), 0.0
    v = np.nanmean(np.stack(vecs), axis=0).astype(np.float32)
    if np.isnan(v).any():
        return np.zeros(n, np.float32), 0.0
    return v, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    clips = find_clips()
    done = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT).clip_id)
        print(f"resuming: {len(done)} clips already extracted", flush=True)
    todo = [c for c in clips if c[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(clips)} AI clips total, {len(todo)} to extract", flush=True)
    if not todo:
        return

    from fusion.extraction.perframe import PerFrameExtractor
    from fusion.extraction.windows import WindowFeaturizer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = PerFrameExtractor(device=dev)
    fz = WindowFeaturizer()

    rows_out, t0 = [], time.time()
    for i, (cid, scen, ctx, path) in enumerate(todo, 1):
        try:
            npz = extractor.extract_clip(path)
            wins = fz.featurize_clip(npz)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {cid} FAILED: {e}", flush=True)
            continue
        if not wins:
            print(f"  [{i}/{len(todo)}] {cid} no windows", flush=True)
            continue
        row = {"clip_id": cid, "scenario_dir": scen, "context_folder": ctx,
               "intent": scen.split("_")[1] if "_" in scen else None}
        for key, names, pref in (("emo_probs", EMO, "emo"), ("ges_probs", GES, "ges"),
                                 ("mot_probs", MOT, "mot"), ("ctx_probs", CTX, "ctx")):
            v, obs = pool(wins, key, len(names))
            for j, nm in enumerate(names):
                row[f"{pref}_{nm}"] = float(v[j]) if j < len(v) else 0.0
            row[f"{pref}_obs"] = obs
        rows_out.append(row)
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {cid} ok  ({el/i:.1f}s/clip, "
                  f"eta {(len(todo)-i)*el/i/60:.0f} min)", flush=True)
            df = pd.DataFrame(rows_out)
            if OUT.exists():
                df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUT, index=False)
            rows_out = []

    if rows_out:
        df = pd.DataFrame(rows_out)
        if OUT.exists():
            df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
        df.to_parquet(OUT, index=False)
    final = pd.read_parquet(OUT)
    print(f"\n{len(final)} clips extracted -> {OUT}")
    print(final.groupby(["context_folder", "scenario_dir"]).size().to_string())


if __name__ == "__main__":
    main()
