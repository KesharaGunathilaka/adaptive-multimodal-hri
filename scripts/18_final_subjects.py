"""Recover per-clip actor identity for `data/final`, which shipped without it.

The validation split must be actor-disjoint (handover §7.3) — all clips of a
person entirely in train or entirely in val — so every clip needs a `person_id`.
Two populations:

* **migrated clips** (`source=curated_clip`): the old root already knows their
  subject. `data/old/annotations/clips.csv` is keyed by the same `clip_id`, so
  these resolve automatically and per clip. Note subjects vary *within* a
  scenario folder there — a folder is not an actor.
* **2026-07-25 takes** (`source=raw_take_20260725`): nothing on disk records who
  performed them. `--template` writes the rows that need a human answer.

Usage:
    .venv/Scripts/python scripts/18_final_subjects.py --template   # ask
    # fill the person_id column in subjects_pending.csv, then:
    .venv/Scripts/python scripts/18_final_subjects.py --apply      # merge
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.final_common import ANNOT_DIR, CLIPS_CSV  # noqa: E402

OLD_CLIPS = ROOT / "data" / "old" / "annotations" / "clips.csv"
PENDING = ANNOT_DIR / "subjects_pending.csv"
SUBJECTS = ANNOT_DIR / "subjects.csv"


def resolve_migrated() -> pd.Series:
    """clip_id -> person_id for clips that came from the pre-2026-07-25 root."""
    old = pd.read_csv(OLD_CLIPS)
    return old.set_index("clip_id")["person_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.template or args.apply):
        raise SystemExit("pass --template or --apply")

    clips = pd.read_csv(CLIPS_CSV)
    known = resolve_migrated()
    clips["person_id"] = clips.clip_id.map(known)

    resolved = clips.person_id.notna()
    print(f"{resolved.sum()}/{len(clips)} clips resolved from data/old "
          f"({clips.loc[resolved, 'person_id'].nunique()} actors: "
          f"{sorted(clips.loc[resolved, 'person_id'].unique())})")

    todo = clips[~resolved]
    if args.template:
        # One row per (folder, take) rather than per file: the three camera views
        # of a take are the same person by construction, so asking per file would
        # triple the manual work and invite inconsistent answers.
        t = (todo.groupby(["context", "scenario_dir", "take_index"])
                 .agg(n_views=("clip_id", "size"),
                      views=("view", lambda s: "+".join(sorted(set(s)))),
                      example_clip=("clip_id", "first"),
                      v3_row=("v3_row", "first"),
                      intent=("intent", "first"))
                 .reset_index())
        t["person_id"] = ""
        t["note"] = ""
        t.loc[t.scenario_dir == "dilanka", "note"] = (
            "folder is not a V3 row — one actor's session spanning several "
            "scenarios; needs splitting into per-row folders before it can be used")
        t.to_csv(PENDING, index=False)
        print(f"\n{len(t)} take(s) need a person_id -> {PENDING}")
        print("Fill the person_id column (e.g. P10). If one actor did a whole "
              "folder, paste the same id down its rows; if a whole session was "
              "one actor, paste it down every row.")
        print("\nTakes per folder:")
        print(t.groupby(["context", "scenario_dir"]).size().to_string())
        return

    # --apply
    if not PENDING.exists():
        raise SystemExit(f"{PENDING} missing — run with --template first")
    filled = pd.read_csv(PENDING)
    filled["person_id"] = filled.person_id.fillna("").astype(str).str.strip()
    blank = filled[filled.person_id == ""]
    if len(blank):
        print(f"WARNING: {len(blank)} take(s) still blank; those clips will have "
              f"no person_id and must be excluded from supervised splits:")
        print(blank.groupby(["context", "scenario_dir"]).size().to_string())

    key = ["context", "scenario_dir", "take_index"]
    m = filled[filled.person_id != ""].set_index(key)["person_id"]
    idx = pd.MultiIndex.from_frame(clips[key])
    clips["person_id"] = clips.person_id.fillna(pd.Series(m.reindex(idx).to_numpy(),
                                                          index=clips.index))

    out = clips[["clip_id", "context", "scenario_dir", "view", "take_index",
                 "source", "person_id"]]
    out.to_csv(SUBJECTS, index=False)
    print(f"\n-> {SUBJECTS}")
    print(f"{out.person_id.notna().sum()}/{len(out)} clips have an actor; "
          f"{out.person_id.nunique()} actors total")
    print(out.groupby("person_id").size().to_string())


if __name__ == "__main__":
    main()
