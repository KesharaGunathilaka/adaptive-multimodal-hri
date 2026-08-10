"""Pass 2 for embeddings: per-clip mean-pooled penultimate-layer vectors for
embedding-level fusion (2026-08-08). Mirrors the EXACT pooling contract
`merged_unimodal.clip_pool` uses for probabilities (mean over OBSERVED
windows only) so the embedding table lines up clip-for-clip with the existing
24-dim probability table (`scripts/realworld_eval/merged_gap.py::build_real`)
-- same `obs` semantics, same clip set, same split/label join.

Reads, per clip:
  * `data/final_merged/features/perframe/<clip_id>.npz` (existing Pass-1 cache
    — gesture_feats, joints25, face_valid, ...)
  * `data/final_merged/features/embeddings/<clip_id>.npz`
    (`scripts/54_extract_embeddings.py`'s output — emotion/context per-frame
    embeddings)

Runs `WindowFeaturizer.featurize_clip(npz, embed_npz)` ONCE per clip (gesture/
motion embeddings come free from the same forward pass already computing
ges_probs/mot_probs; emotion/context embeddings are pooled from the Pass-1.5
cache using the identical span logic `featurize_clip` already applies to
probs), then mean-pools each modality's embedding over its own observed
windows -- a clip where gesture never fired gets ges_obs=0 and an all-zero
ges_embed, exactly like the existing 24-dim table's `ges_obs`/zero-fill
convention (`fusion/baselines/common.py`).

Writes a single compact `data/final_merged/features/embeddings_pooled.npz`:
    clip_id  [N] str
    emo_embed[N,1280]  ges_embed[N,128]  mot_embed[N,256]  ctx_embed[N,512]  float32
    emo_obs  ges_obs  mot_obs  ctx_obs   [N] float32 (1.0/0.0)
    n_windows[N] int64

Deliberately NOT a parquet (2176 float columns is the wrong shape for that
format) and deliberately separate from label/split columns -- join by
clip_id against `scripts/realworld_eval/merged_gap.py::build_real`'s output
at train time, the same way the probability table is already built.

    .venv/Scripts/python scripts/55_pool_embeddings.py
    .venv/Scripts/python scripts/55_pool_embeddings.py --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.merged_unimodal import (  # noqa: E402
    PERFRAME_DIR, load_clips)

EMBED_DIR = ROOT / "data" / "final_merged" / "features" / "embeddings"
OUT_PATH = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled.npz"
EMBED_DIMS = {"emo": 1280, "ges": 128, "mot": 256, "ctx": 512}
KEY_OF = {"emo": "emo_embed", "ges": "ges_embed", "mot": "mot_embed", "ctx": "ctx_embed"}


def pool_clip(rows: list[dict]) -> dict:
    """window rows (from featurize_clip) -> one pooled dict for this clip.
    Mean over windows where that modality's embedding is not None (= observed
    -- same condition merged_unimodal.clip_pool uses via *_obs on probs)."""
    out = {}
    for pref, dim in EMBED_DIMS.items():
        key = KEY_OF[pref]
        vecs = [r[key] for r in rows if r[key] is not None]
        if vecs:
            out[key] = np.mean(np.stack(vecs), axis=0).astype(np.float32)
            out[f"{pref}_obs"] = 1.0
        else:
            out[key] = np.zeros(dim, np.float32)
            out[f"{pref}_obs"] = 0.0
    out["n_windows"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from fusion.extraction.windows import WindowFeaturizer
    fz = WindowFeaturizer()

    clips = load_clips()
    if args.limit:
        clips = clips.head(args.limit)
    print(f"{len(clips)} usable clips", flush=True)

    clip_ids, pooled, missing = [], [], []
    t0 = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        pf_path = PERFRAME_DIR / f"{r.clip_id}.npz"
        eb_path = EMBED_DIR / f"{r.clip_id}.npz"
        if not (pf_path.exists() and eb_path.exists()):
            missing.append(r.clip_id)
            continue
        npz = np.load(pf_path)
        embed_npz = np.load(eb_path)
        rows = fz.featurize_clip(npz, embed_npz=embed_npz)
        clip_ids.append(r.clip_id)
        pooled.append(pool_clip(rows))
        if n % 200 == 0 or n == len(clips):
            print(f"  [{n}/{len(clips)}] {len(clip_ids)} pooled, "
                  f"{len(missing)} missing cache "
                  f"({n / (time.time() - t0):.1f} clips/s)", flush=True)

    if missing:
        print(f"WARNING: {len(missing)} clip(s) missing perframe or embeddings "
              f"cache, skipped. First 5: {missing[:5]}", flush=True)
    if not clip_ids:
        raise SystemExit("nothing pooled -- check that scripts/54 has run")

    out = {"clip_id": np.array(clip_ids, dtype=object)}
    for pref, dim in EMBED_DIMS.items():
        key = KEY_OF[pref]
        out[key] = np.stack([p[key] for p in pooled]).astype(np.float32)
        out[f"{pref}_obs"] = np.array([p[f"{pref}_obs"] for p in pooled], np.float32)
    out["n_windows"] = np.array([p["n_windows"] for p in pooled], np.int64)

    np.savez_compressed(OUT_PATH, **out)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n-> {OUT_PATH.relative_to(ROOT)} ({size_mb:.1f} MB, {len(clip_ids)} clips)")
    for pref in EMBED_DIMS:
        obs_rate = out[f"{pref}_obs"].mean()
        print(f"  {pref}: obs_rate={obs_rate:.4f}")


if __name__ == "__main__":
    main()
