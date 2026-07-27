"""Run ONE finalized model over every real-world clip in data/, independently.

Each invocation loads a single modality's deployed inference code and decodes
every clip itself, writing one per-step prediction CSV per clip:

    results/realworld_eval/predictions/<model>/<context>/<scenario>/<clip_id>.csv

Resumable — clips whose CSV already exists are skipped, so it can be killed and
relaunched. Run one model per process (that is what keeps the four evaluations
independent, and it sidesteps the modality sys.path collisions).

    .venv/Scripts/python scripts/13_realworld_infer.py --model gesture
    .venv/Scripts/python scripts/13_realworld_infer.py --model emotion --limit 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.common import (CHECKPOINTS, MODELS, OUT_ROOT,  # noqa: E402
                                           load_clips, pred_path, sha16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None, help="process at most N clips")
    ap.add_argument("--scenario", default=None, help="restrict to one scenario_id")
    ap.add_argument("--force", action="store_true", help="redo clips that already have a CSV")
    args = ap.parse_args()

    from scripts.realworld_eval.runners import RUNNERS

    clips = load_clips()
    if args.scenario:
        clips = clips[clips.scenario_id == args.scenario]
    todo = [r for _, r in clips.iterrows()
            if args.force or not pred_path(args.model, r).exists()]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[{args.model}] {len(clips)} clips total, {len(todo)} to run", flush=True)
    if not todo:
        return

    runner = RUNNERS[args.model]()
    t0, n_steps = time.time(), 0
    for i, row in enumerate(todo, 1):
        out = pred_path(args.model, row)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            df = runner.run_clip(row["video_path"])
        except Exception as e:  # noqa: BLE001 — log and continue the batch
            print(f"[{i}/{len(todo)}] FAILED {row['clip_id']}: {e}", flush=True)
            continue
        df.insert(0, "clip_id", row["clip_id"])
        df.to_csv(out, index=False)
        n_steps += len(df)
        if i % 10 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(todo)}] {row['clip_id']}  "
                  f"({rate:.2f} clips/s, eta {(len(todo) - i) / rate / 60:.0f} min)",
                  flush=True)

    # provenance: which weights produced this prediction tree
    manifest = OUT_ROOT / "run_manifest.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data[args.model] = {
        "checkpoint": str(CHECKPOINTS[args.model]) if CHECKPOINTS[args.model] else
                      "CLIP ViT-B-32-quickgelu / openai (zero-shot)",
        "sha256_16": sha16(CHECKPOINTS[args.model]),
        "clips_run": len(todo),
        "steps_written": n_steps,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2))
    print(f"done — {n_steps} steps, manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
