"""Pass 1.5 — penultimate-layer embeddings for embedding-level fusion
(2026-08-08). Motivated by `PERCEPTION_BAND.md`: on the deployed recipe,
perception error (0.224) now dominates fusion's remaining headroom
(generalisation cost 0.063), so a cue representation richer than the 4/5/7/8-
class softmax the current 24-dim fusion input collapses to is the lever with
real room to help. See the corrupted `fusion/fusion-engine-embeddings/`
attempt ([[embeddings-corrupt-on-win3060]] memory) for why this is a fresh,
local re-extraction rather than a re-transfer.

Writes `data/final_merged/features/embeddings/<clip_id>.npz`:
    emotion_probs [T,7]      (duplicated from the existing cache, for a
                              free correctness check against it)
    emotion_embed [T,1280]   NaN where no face -- MobileNetV2 penultimate
    face_valid    [T]        bool
    context_probs [Tc,5]
    context_embed [Tc,512]   CLIP image embedding (was already computed and
                              discarded in the original Pass 1)
    context_frames[Tc]       int64 frame index
    fps, n_frames            scalars

Deliberately does NOT touch `data/final_merged/features/perframe/` (the
production cache every other script reads) and does NOT re-run MediaPipe
Holistic — `PerFrameExtractor.extract_embeddings_only` skips it entirely,
since emotion's face crop comes from the separate FaceDetection model and
Holistic was the dominant cost of the original extraction. Gesture/motion
embeddings need no video decode at all: they're derived in Pass 1.5b
(`scripts/55_pool_embeddings.py`) from `gesture_feats`/`joints25` already
sitting in the existing perframe cache.

Resumable: clips whose output already exists are skipped.

    .venv/Scripts/python scripts/54_extract_embeddings.py            # everything
    .venv/Scripts/python scripts/54_extract_embeddings.py --limit 5  # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MERGED = ROOT / "data" / "final_merged"
CLIPS_DIR = MERGED / "raw" / "clips"
PERFRAME_DIR = MERGED / "features" / "perframe"
OUT_DIR = MERGED / "features" / "embeddings"


def sanity_check(clip_id: str, arrays: dict) -> str | None:
    """Compare this pass's emotion/context PROBS against the existing
    production cache's -- they should be near-identical (same models, same
    preprocessing; only the code path differs). A large mismatch means the
    lean extractor diverged from `extract_clip` and should be investigated
    before trusting the embeddings. Returns an error string, or None if OK."""
    p = PERFRAME_DIR / f"{clip_id}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        old_emo = z["emotion_probs"]
        old_ctx = z["context_probs"]
    new_emo, new_ctx = arrays["emotion_probs"], arrays["context_probs"]
    if old_emo.shape != new_emo.shape or old_ctx.shape != new_ctx.shape:
        return f"shape mismatch: emo {old_emo.shape} vs {new_emo.shape}, ctx {old_ctx.shape} vs {new_ctx.shape}"
    emo_ok = np.allclose(old_emo, new_emo, atol=1e-4, equal_nan=True)
    ctx_ok = np.allclose(old_ctx, new_ctx, atol=1e-4, equal_nan=True)
    if not (emo_ok and ctx_ok):
        d_emo = np.nanmax(np.abs(old_emo - new_emo)) if old_emo.size else 0.0
        d_ctx = np.nanmax(np.abs(old_ctx - new_ctx)) if old_ctx.size else 0.0
        return f"probs diverge: max|d_emo|={d_emo:.5f} max|d_ctx|={d_ctx:.5f}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-every", type=int, default=25,
                    help="run the probs sanity check on 1/N clips (0=never)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(MERGED / "annotations" / "clips.csv")
    clips = clips[clips.usable == True]  # noqa: E712

    todo = [r for r in clips.itertuples()
            if not (OUT_DIR / f"{r.clip_id}.npz").exists()]
    already = len(clips) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(clips)} usable clips | {already} already cached | "
          f"{len(todo)} to do", flush=True)
    if not todo:
        print("nothing to do")
        return

    from fusion.extraction.perframe import PerFrameExtractor
    ex = PerFrameExtractor()

    failures, check_fails = [], []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        try:
            arrays = ex.extract_embeddings_only(CLIPS_DIR / r.filepath)
        except Exception as e:                       # noqa: BLE001
            failures.append((r.clip_id, str(e)))
            print(f"  FAILED {r.clip_id}: {e}", flush=True)
            continue

        if args.check_every and (i % args.check_every == 0 or i <= 3):
            err = sanity_check(r.clip_id, arrays)
            if err:
                check_fails.append((r.clip_id, err))
                print(f"  SANITY CHECK FAILED {r.clip_id}: {err}", flush=True)

        np.savez_compressed(OUT_DIR / f"{r.clip_id}.npz", **arrays)
        if i % 25 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / max(rate, 1e-9) / 60
            print(f"[{i}/{len(todo)}] {r.clip_id}  "
                  f"({rate:.2f} clips/s, eta {eta:.0f} min)", flush=True)

    n = len(list(OUT_DIR.glob("*.npz")))
    print(f"\ndone: {n}/{len(clips)} clips cached, {len(failures)} failure(s), "
          f"{len(check_fails)}/{min(len(todo), (len(todo)+args.check_every-1)//max(args.check_every,1)+3) if args.check_every else 0} "
          "sanity checks failed")
    if check_fails:
        print("SANITY CHECK FAILURES (investigate before trusting the output):")
        for cid, err in check_fails:
            print(f"  {cid}: {err}")


if __name__ == "__main__":
    main()
