"""Regenerate `data/final_merged/annotations/takes.csv` from the current clips.csv.

Why this exists: `takes.csv` is produced by `20_merged_annotations.py`, but that
script rebuilds *every* annotation file. When `clips.csv` is later amended on its
own (de-duplication, person_id backfill), `takes.csv` silently goes stale — the
2026-08-03 audit found it carrying 1,737 rows against 1,736 real takes, the extra
being `S28_F07` take 19 whose `example_clip` (`S28_F07_c052`) no longer exists
anywhere in `clips.csv`, plus person_id filled for only 956 of 1,737 rows while
clips.csv is 100 % complete.

This script reproduces script 20's takes aggregation **exactly** (same groupby,
same columns) but touches nothing else, so it is safe to run any time clips.csv
changes.

    .venv/Scripts/python scripts/24_regen_takes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.merged_common import ANNOT_DIR, CLIPS_CSV  # noqa: E402

TAKES_CSV = ANNOT_DIR / "takes.csv"


def build_takes(clips: pd.DataFrame) -> pd.DataFrame:
    """Identical to 20_merged_annotations.py's takes aggregation."""
    return (clips[clips.usable]
            .groupby(["context", "scenario_dir", "v3_row", "intent", "split_design",
                      "take_index"])
            .agg(n_views=("view", "nunique"),
                 views=("view", lambda s: "+".join(sorted(s))),
                 example_clip=("clip_id", "first"),
                 recorded_at=("recorded_at", "first"),
                 person_id=("person_id",
                            lambda s: s.dropna().iloc[0] if s.notna().any() else None))
            .reset_index())


def main() -> None:
    clips = pd.read_csv(CLIPS_CSV)
    new = build_takes(clips)

    old = pd.read_csv(TAKES_CSV) if TAKES_CSV.exists() else None
    if old is not None:
        key = ["scenario_dir", "take_index"]
        o = set(map(tuple, old[key].dropna().values))
        n = set(map(tuple, new[key].dropna().values))
        for gone in sorted(o - n):
            print(f"  dropped stale take: {gone[0]} take {int(gone[1])}")
        for added in sorted(n - o):
            print(f"  added take: {added[0]} take {int(added[1])}")
        print(f"  person_id known: {old.person_id.notna().sum()} -> "
              f"{new.person_id.notna().sum()} of {len(new)}")

    try:
        new.to_csv(TAKES_CSV, index=False)
    except PermissionError:
        raise SystemExit(f"cannot write {TAKES_CSV} -- close it in Excel and re-run.")

    # every example_clip must exist in clips.csv
    orphans = set(new.example_clip) - set(clips.clip_id)
    print(f"-> {TAKES_CSV.name}  ({len(new)} takes, "
          f"{new.person_id.notna().sum()} with a known actor)")
    print("  OK: every example_clip resolves in clips.csv" if not orphans
          else f"  ERROR: {len(orphans)} orphan example_clip(s): {sorted(orphans)[:5]}")


if __name__ == "__main__":
    main()
