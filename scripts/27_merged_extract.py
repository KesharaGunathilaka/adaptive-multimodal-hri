"""Pass 1 feature extraction for `data/final_merged` (all usable clips, all views).

Writes `data/final_merged/features/perframe/<clip_id>.npz`, one per clip, plus a
`manifest.json` recording which checkpoints produced them.

**Cache reuse.** `data/final` already has 1,440 per-frame caches, and 1,500 of the
2,869 merged clips carry an `old_clip_id` pointing at one. Reusing them halves the
run. But the merge re-encoded some files (DATASET_FIXLIST R4: 726 of 1,016
"phone_1080p" clips are not 1080p), and a re-encode changes frame count / fps —
which would make a reused cache silently wrong. So a cache is reused **only if its
`n_frames` and `fps` match `clips.csv` for the new clip**; otherwise the clip is
re-extracted. Every reuse decision is counted and reported.

Resumable: clips whose output already exists are skipped.

    .venv/Scripts/python scripts/27_merged_extract.py            # everything
    .venv/Scripts/python scripts/27_merged_extract.py --limit 20 # smoke test
    .venv/Scripts/python scripts/27_merged_extract.py --no-reuse # force fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MERGED = ROOT / "data" / "final_merged"
CLIPS_DIR = MERGED / "raw" / "clips"
OUT_DIR = MERGED / "features" / "perframe"
MANIFEST = MERGED / "features" / "manifest.json"
OLD_CACHE = ROOT / "data" / "final" / "features" / "perframe"

FPS_TOL = 0.05          # probed fps can differ in the last decimal place


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reusable_cache(old_clip_id, n_frames, fps) -> Path | None:
    """Return a reusable cache path, or None if absent/inconsistent."""
    if not isinstance(old_clip_id, str) or not old_clip_id:
        return None
    p = OLD_CACHE / f"{old_clip_id}.npz"
    if not p.exists():
        return None
    try:
        with np.load(p) as z:
            if "pose_img" not in z:              # predates the direction-cue cache
                return None
            if int(z["n_frames"]) != int(n_frames):
                return None
            if abs(float(z["fps"]) - float(fps)) > FPS_TOL:
                return None
    except Exception:
        return None
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-reuse", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(MERGED / "annotations" / "clips.csv")
    clips = clips[clips.usable == True]  # noqa: E712

    todo = [r for r in clips.itertuples()
            if not (OUT_DIR / f"{r.clip_id}.npz").exists()]
    already = len(clips) - len(todo)          # count BEFORE --limit truncates
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(clips)} usable clips | {already} already cached | "
          f"{len(todo)} to do"
          + (f" (limited from {len(clips) - already})" if args.limit else ""),
          flush=True)
    if not todo:
        print("nothing to do")
        return

    # Reuse pass first: cheap, and it shrinks the expensive pass.
    reused = rejected = 0
    remaining = []
    if not args.no_reuse:
        import shutil
        for r in todo:
            src = reusable_cache(getattr(r, "old_clip_id", None), r.n_frames, r.fps)
            if src is None:
                remaining.append(r)
                if isinstance(getattr(r, "old_clip_id", None), str):
                    rejected += 1
            else:
                shutil.copyfile(src, OUT_DIR / f"{r.clip_id}.npz")
                reused += 1
        print(f"reused {reused} cache(s) from data/final "
              f"({rejected} rejected: re-encoded or missing pose_img)", flush=True)
    else:
        remaining = todo

    print(f"extracting {len(remaining)} clip(s)...", flush=True)
    failures = []
    if remaining:
        from fusion.extraction.perframe import PerFrameExtractor
        ex = PerFrameExtractor()
        t0 = time.time()
        for i, r in enumerate(remaining, 1):
            try:
                arrays = ex.extract_clip(CLIPS_DIR / r.filepath)
                np.savez_compressed(OUT_DIR / f"{r.clip_id}.npz", **arrays)
            except Exception as e:                       # noqa: BLE001
                failures.append((r.clip_id, str(e)))
                print(f"  FAILED {r.clip_id}: {e}", flush=True)
                continue
            if i % 25 == 0 or i == len(remaining):
                rate = i / (time.time() - t0)
                eta = (len(remaining) - i) / max(rate, 1e-9) / 60
                print(f"[{i}/{len(remaining)}] {r.clip_id}  "
                      f"({rate:.2f} clips/s, eta {eta:.0f} min)", flush=True)

    from fusion.extraction.perframe import EMOTION_CKPT
    from fusion.extraction.windows import GESTURE_CKPT, MOTION_CKPT
    MANIFEST.write_text(json.dumps({
        "version": "final_merged_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "n_clips_cached": len(list(OUT_DIR.glob("*.npz"))),
        "n_reused_from_final": reused,
        "n_reuse_rejected": rejected,
        "failures": failures,
        "checkpoints": {
            "emotion": {"path": str(EMOTION_CKPT.relative_to(ROOT)),
                        "sha256": sha256(EMOTION_CKPT)},
            "gesture": {"path": str(GESTURE_CKPT.relative_to(ROOT)),
                        "sha256": sha256(GESTURE_CKPT)},
            "motion": {"path": str(MOTION_CKPT.relative_to(ROOT)),
                       "sha256": sha256(MOTION_CKPT)},
            "context": "CLIP ViT-B-32 zero-shot (jetson_deploy/hf_cache)",
        },
    }, indent=2))

    n = len(list(OUT_DIR.glob("*.npz")))
    print(f"\ndone: {n}/{len(clips)} clips cached, {len(failures)} failure(s)")
    print(f"-> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
