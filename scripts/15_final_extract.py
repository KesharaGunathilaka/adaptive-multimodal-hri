"""Pass 1 over `data/final`: one decode per clip -> per-frame cache for all 4 cues.

Same extractor as the fusion pipeline (fusion/extraction/perframe.py), pointed at
the new collection root. Resumable: existing .npz are skipped, so it can be
killed and relaunched. Filters let the deployment camera go first — it is the
only view every scenario has.

    .venv/Scripts/python scripts/15_final_extract.py --context classroom --view realsense_480p
    .venv/Scripts/python scripts/15_final_extract.py --context classroom          # all views
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.final_common import CLIPS_ROOT, load_clips  # noqa: E402
from scripts.realworld_eval.final_unimodal import PERFRAME_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default=None, help="classroom | kitchen")
    ap.add_argument("--view", default=None,
                    help="realsense_480p | phone_1080p | phone_4k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-unmapped", action="store_true",
                    help="also extract folders with no V3 row (e.g. 'dilanka', "
                         "one actor's session that still needs splitting into "
                         "per-row folders — nothing can be scored from it yet)")
    args = ap.parse_args()

    PERFRAME_DIR.mkdir(parents=True, exist_ok=True)
    clips = load_clips()
    if args.context:
        clips = clips[clips.context == args.context]
    if args.view:
        clips = clips[clips.view == args.view]
    if not args.include_unmapped:
        unmapped = clips[clips.v3_row.isna()]
        if len(unmapped):
            print(f"skipping {len(unmapped)} clip(s) in folders with no V3 row: "
                  f"{sorted(unmapped.scenario_dir.unique())}")
            clips = clips[clips.v3_row.notna()]

    todo = [r for r in clips.itertuples()
            if args.force or not (PERFRAME_DIR / f"{r.clip_id}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(clips)} clips selected, {len(todo)} to extract", flush=True)
    if not todo:
        return

    from fusion.extraction.perframe import PerFrameExtractor
    ex = PerFrameExtractor()

    t0, failed = time.time(), []
    for i, r in enumerate(todo, 1):
        try:
            arrays = ex.extract_clip(CLIPS_ROOT / r.filepath)
        except Exception as e:  # noqa: BLE001 — log and continue the batch
            failed.append((r.clip_id, str(e)))
            print(f"[{i}/{len(todo)}] FAILED {r.clip_id}: {e}", flush=True)
            continue
        np.savez_compressed(PERFRAME_DIR / f"{r.clip_id}.npz", **arrays)
        if i % 20 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(todo)}] {r.clip_id}  "
                  f"({rate:.2f} clips/s, eta {(len(todo) - i) / rate / 60:.0f} min)",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed", flush=True)
    for cid, e in failed:
        print(f"  {cid}: {e}")


if __name__ == "__main__":
    main()
