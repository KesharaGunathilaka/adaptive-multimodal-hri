"""Build `data/final_merged/annotations/splits.csv`.

The file that lived at this path was a leftover copy of
`videos/struct/annotations/splits.csv` -- a different, older, now-superseded
dataset (23 scenarios, includes intent F09, pre-merge numbering). It had zero
filepath overlap with `data/final_merged` and nothing in this codebase reads
it; it was never regenerated for this root. This script builds a real one.

The authoritative split is the `split_design` column already verified against
`docs/final_dataset_merged.docx` in `scenarios.csv`/`clips.csv` (each V3 row
is wholly train or wholly test by design; see `docs/DATASET_FIXLIST.md` for
the one row, #58, whose *derived* clips cross that boundary). This is a
one-clip-per-row projection of `clips.csv`, usable clips only, for scripts
that want a lean split lookup without the full label/provenance schema.

    .venv/Scripts/python scripts/23_build_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.merged_common import ANNOT_DIR, CLIPS_CSV  # noqa: E402

SPLITS_CSV = ANNOT_DIR / "splits.csv"

COLUMNS = [
    "clip_id", "v3_row", "context", "scenario_dir", "intent", "action",
    "split_design", "person_id", "filepath", "view", "width", "height",
    "fps", "n_frames", "duration_s", "sha256", "recorded_at",
]


def main() -> None:
    clips = pd.read_csv(CLIPS_CSV)
    usable = clips[clips.usable].copy()

    missing_split = usable.split_design.isna().sum()
    missing_person = usable.person_id.isna().sum()
    if missing_split:
        print(f"  WARNING: {missing_split} usable clips have no split_design")
    if missing_person:
        print(f"  WARNING: {missing_person} usable clips have no person_id")

    out = usable[COLUMNS].sort_values(["v3_row", "clip_id"]).reset_index(drop=True)
    try:
        out.to_csv(SPLITS_CSV, index=False)
    except PermissionError:
        raise SystemExit(f"cannot write {SPLITS_CSV} -- close it in Excel and re-run.")

    print(f"-> {SPLITS_CSV.name}  ({len(out)} usable clips)")
    print(out.split_design.value_counts().to_string())

    actors_by_split = out.groupby("split_design").person_id.apply(lambda s: set(s.dropna()))
    tr, te = actors_by_split.get("train", set()), actors_by_split.get("test", set())
    overlap = sorted(tr & te)
    if overlap:
        print(f"  note: split is scenario-disjoint, not actor-disjoint -- "
              f"{len(overlap)} actor(s) appear in both train and test: {overlap}")


if __name__ == "__main__":
    main()
