"""Pass 1 runner: per-frame extraction over every clip in data/annotations/clips.csv.

Writes data/features/perframe/<clip_id>.npz. Resumable — existing npz are
skipped, so it can be re-launched after an interruption.

Run from repo root:  .venv/Scripts/python scripts/02_extract_perframe.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.perframe import PerFrameExtractor  # noqa: E402

OUT = ROOT / "data" / "features" / "perframe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most N clips")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(ROOT / "data" / "annotations" / "clips.csv")
    todo = [(r.clip_id, ROOT / "data" / r.filepath) for r in clips.itertuples()
            if not (OUT / f"{r.clip_id}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(clips)} clips total, {len(todo)} to extract", flush=True)
    if not todo:
        return

    ex = PerFrameExtractor()
    t0 = time.time()
    for i, (clip_id, path) in enumerate(todo, 1):
        try:
            arrays = ex.extract_clip(path)
        except Exception as e:  # noqa: BLE001 — log and continue the batch
            print(f"[{i}/{len(todo)}] FAILED {clip_id}: {e}", flush=True)
            continue
        np.savez_compressed(OUT / f"{clip_id}.npz", **arrays)
        if i % 10 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 60
            print(f"[{i}/{len(todo)}] {clip_id}  ({rate:.2f} clips/s, eta {eta:.0f} min)",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
