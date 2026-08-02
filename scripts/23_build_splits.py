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

Two split columns are written:

  `split_design`  train/test, straight from the V3 table. Scenario-disjoint:
                  the test rows are cue combinations held out *by design*.
  `split`         train/val/test -- the one training code should read.
                  `test` is `split_design == test` untouched; `val` is carved
                  out of train by ACTOR (see VAL_ACTORS), so all clips of a
                  val actor are in val and none are in train.

Why actor-disjoint val: it is what the handover specifies, and it keeps every
train row's cue tuple in training. Holding out *rows* instead would mirror the
test condition more closely but would shrink the tuple coverage that the
2026-07-28 gap decomposition identified as the binding constraint.

VAL_ACTORS = P04 + P03: 319 clips (17.1% of train) covering 30 of 40 train
rows, and removing them empties no train row. P04 was also the val actor in
`data/old`, so the convention carries over.

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
    "split_design", "split", "headline_eval", "person_id", "filepath", "view",
    "resolution_class", "orientation", "width", "height", "fps", "n_frames",
    "duration_s", "agg_span_s", "sha256", "recorded_at",
]

# Actors held out of train to form the validation set (see module docstring).
VAL_ACTORS = ("P04", "P03")

# Aggregation window for clip-level fusion (2026-07-28 decision). Clips shorter
# than this are aggregated over their FULL length rather than padded -- padding
# would invent frames. `agg_span_s` records what was actually used.
AGG_SPAN_S = 4.0


def _resolution_class(w: int, h: int) -> str:
    """Truthful resolution bucket. The `view` label is not reliable: 726 of the
    1,015 'phone_1080p' clips are not 1080p (2026-08-03 audit, N8)."""
    short = min(int(w), int(h))
    for limit, name in ((480, "480p"), (576, "576p"), (720, "720p"),
                        (1080, "1080p"), (2160, "4k")):
        if short <= limit:
            return name
    return "4k+"


def main() -> None:
    clips = pd.read_csv(CLIPS_CSV)
    usable = clips[clips.usable].copy()

    missing_split = usable.split_design.isna().sum()
    missing_person = usable.person_id.isna().sum()
    if missing_split:
        print(f"  WARNING: {missing_split} usable clips have no split_design")
    if missing_person:
        print(f"  WARNING: {missing_person} usable clips have no person_id")

    # train/val/test: test is untouched; val is carved from train by actor.
    usable["split"] = usable.split_design
    is_val = (usable.split_design == "train") & usable.person_id.isin(VAL_ACTORS)
    usable.loc[is_val, "split"] = "val"

    # headline_eval=False marks test clips that are NOT the row's own footage:
    # row #58's 24 clips are row #49's frames (a TRAIN row) re-used with emotion
    # and gesture masked, which also flips the intent F01 -> F06. Scoring them as
    # test partly measures recall of training frames. Report the headline number
    # over headline_eval==True and quote the full-test number beside it.
    usable["headline_eval"] = True
    contaminated = (usable.split == "test") & usable.derived_from_row.notna()
    usable.loc[contaminated, "headline_eval"] = False

    # Truthful capture metadata -- `view` mislabels resolution (audit N8).
    usable["resolution_class"] = [
        _resolution_class(w, h) for w, h in zip(usable.width, usable.height)]
    usable["orientation"] = ["portrait" if h > w else "landscape"
                             for w, h in zip(usable.width, usable.height)]

    # Clips shorter than the aggregation window use their own full length.
    usable["agg_span_s"] = usable.duration_s.clip(upper=AGG_SPAN_S).round(3)

    out = usable[COLUMNS].sort_values(["v3_row", "clip_id"]).reset_index(drop=True)
    try:
        out.to_csv(SPLITS_CSV, index=False)
    except PermissionError:
        raise SystemExit(f"cannot write {SPLITS_CSV} -- close it in Excel and re-run.")

    print(f"-> {SPLITS_CSV.name}  ({len(out)} usable clips)")
    print(out.split_design.value_counts().to_string())
    print("\nsplit (what training code should read):")
    print(out.split.value_counts().to_string())

    _report(out)


def _report(out: pd.DataFrame) -> None:
    """Validate and describe both split columns."""
    actors_by_design = out.groupby("split_design").person_id.apply(lambda s: set(s.dropna()))
    tr, te = actors_by_design.get("train", set()), actors_by_design.get("test", set())
    overlap = sorted(tr & te)
    if overlap:
        print(f"\n  note: split_design is scenario-disjoint, not actor-disjoint -- "
              f"{len(overlap)} actor(s) in both train and test: {overlap}")
        print("        => the test set measures compositional generalisation only,")
        print("           not subject generalisation. State this when reporting.")

    a = out.groupby("split").person_id.apply(lambda s: set(s.dropna()))
    tr_a, va_a = a.get("train", set()), a.get("val", set())
    leak = sorted(tr_a & va_a)
    print(f"\n  val actors: {sorted(va_a)}")
    if leak:
        print(f"  ERROR: val is NOT actor-disjoint from train: {leak}")
    else:
        print("  OK: val is actor-disjoint from train")

    # a take is filmed by one actor, so actor-disjoint implies take-grouped;
    # verify rather than assume (the 3 simultaneous views must not straddle).
    if "scenario_dir" in out.columns:
        per_take = out.groupby(["scenario_dir", "person_id"]).split.nunique()
        bad = per_take[per_take > 1]
        print("  OK: no (scenario, actor) group straddles splits"
              if bad.empty else
              f"  ERROR: {len(bad)} (scenario, actor) groups straddle splits")

    tr_rows = set(out[out.split == "train"].v3_row)
    design_tr = set(out[out.split_design == "train"].v3_row)
    emptied = sorted(design_tr - tr_rows)
    print(f"  train rows retained: {len(tr_rows)}/{len(design_tr)}"
          + (f"  ERROR emptied: {emptied}" if emptied else "  (none emptied)"))
    print(f"  val row coverage: {out[out.split == 'val'].v3_row.nunique()}/{len(design_tr)}")

    te = out[out.split == "test"]
    excl = te[~te.headline_eval]
    print(f"\n  headline test set: {te.headline_eval.sum()} of {len(te)} clips"
          f"  ({len(excl)} excluded as train footage"
          + (f", rows {sorted(excl.v3_row.unique())})" if len(excl) else ")"))
    print("  -> quote the headline number over headline_eval==True, and the "
          "full-test number beside it")

    print(f"\n  resolution_class: {out.resolution_class.value_counts().to_dict()}")
    print(f"  orientation     : {out.orientation.value_counts().to_dict()}")
    short = (out.duration_s < AGG_SPAN_S).sum()
    print(f"  clips shorter than the {AGG_SPAN_S:g}s window: {short} "
          f"(min {out.duration_s.min():.2f}s) -- aggregated over their full "
          f"length, see agg_span_s")


if __name__ == "__main__":
    main()
