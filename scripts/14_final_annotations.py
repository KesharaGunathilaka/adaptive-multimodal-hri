"""Build the annotation tables for `data/final`, which shipped without any.

Writes two files:

    data/final/annotations/scenarios_v3.csv   the 62-row Final_Dataset V3 table
    data/final/annotations/clips.csv          every video, probed + joined to GT

The join is `scenario folder number == V3 row number`. That is asserted, not
assumed: 21 of the 40 folders hold clips migrated from `data/raw`, whose old
scenario ID is still in the filename, so their cue tuple can be looked up in
`data/labels.csv` and compared against the V3 row the folder name points at.
Any disagreement is a hard failure — a silent mismatch here would mis-score
every downstream evaluation.

    .venv/Scripts/python scripts/14_final_annotations.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.final_common import (  # noqa: E402
    ANNOT_DIR, CLIPS_CSV, CLIPS_ROOT, SCENARIOS_CSV, VIDEO_EXT,
    _CURATED_RE, _SCENARIO_RE, classify_view, parse_v3_table, probe)

# the pre-2026-07-25 root moved to data/old when data/final was created
OLD_LABELS = ROOT / "data" / "old" / "labels.csv"


def verify_folder_numbering(v3: pd.DataFrame) -> None:
    """Assert scenario-folder number == V3 row, using the migrated clips."""
    old = pd.read_csv(OLD_LABELS).set_index("scenario_id")
    v3i = v3.set_index("v3_row")
    checked = mismatched = 0

    for folder in sorted(CLIPS_ROOT.glob("*/S*_F*")):
        m = _SCENARIO_RE.match(folder.name)
        if not m:
            continue
        row = int(m.group(1))
        old_ids = {c.group(1) for f in folder.iterdir()
                   if (c := _CURATED_RE.match(f.stem))}
        if not old_ids or row not in v3i.index:
            continue
        want = v3i.loc[row]
        for oid in sorted(old_ids):
            if oid not in old.index:
                continue
            checked += 1
            got = old.loc[oid]
            same = (str(got.emotion).lower() == want.emotion_v3
                    and str(got.context).lower() == want.context)
            if not same:
                mismatched += 1
                print(f"  MISMATCH {folder.name} <- {oid}: "
                      f"labels.csv=({got.context},{got.emotion}) "
                      f"V3#{row}=({want.context},{want.emotion_v3})")
    if mismatched:
        raise SystemExit(f"folder->V3 numbering disagrees on {mismatched} scenario(s); "
                         "resolve before scoring anything")
    print(f"  verified folder number == V3 row on {checked} migrated scenario(s)")


def build_inventory(v3: pd.DataFrame) -> pd.DataFrame:
    v3i = v3.set_index("v3_row")
    rows, unreadable = [], []

    files = sorted(p for p in CLIPS_ROOT.rglob("*")
                   if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    print(f"  probing {len(files)} video files ...")
    for i, p in enumerate(files, 1):
        rel = p.relative_to(CLIPS_ROOT)
        context = rel.parts[0]
        scenario_dir = rel.parts[1]
        info = probe(p)
        if info is None:
            unreadable.append(str(rel))
            continue

        m = _SCENARIO_RE.match(scenario_dir)
        v3_row = int(m.group(1)) if m else None
        cur = _CURATED_RE.match(p.stem)

        rec = {
            "clip_id": p.stem,
            "context": context,
            "scenario_dir": scenario_dir,
            "v3_row": v3_row,
            "subject_dir": None if m else scenario_dir,   # e.g. 'dilanka'
            "source": "curated_clip" if cur else "raw_take_20260725",
            "old_scenario_id": cur.group(1) if cur else None,
            "filepath": str(rel).replace("\\", "/"),
            **info,
        }
        rec["view"] = classify_view(info["width"], info["height"], info["fps"])

        if v3_row is not None and v3_row in v3i.index:
            r = v3i.loc[v3_row]
            rec.update({
                "intent": r.intent, "split_design": r.split_design,
                "emotion_v3": r.emotion_v3,
                # pandas hands back NaN, not None, for the masked rows
                "gt_emotion": None if pd.isna(r.gt_emotion) else r.gt_emotion,
                "emotion_masked": bool(r.emotion_masked),
                "gesture_v3": r.gesture_v3, "motion_v3": r.motion_v3,
                "scenario_desc": r.scenario_desc,
            })
        else:
            rec.update({
                "intent": None, "split_design": None, "emotion_v3": None,
                "gt_emotion": None, "emotion_masked": False,
                "gesture_v3": None, "motion_v3": None, "scenario_desc": None,
            })

        # why a clip is unscoreable, spelled out rather than silently dropped
        notes = []
        if rec["gt_emotion"] is None:
            notes.append("emotion [MISSING] by design" if rec["emotion_masked"]
                         else "no V3 row for this folder")
        rec["caveat"] = "; ".join(notes)
        rec["scoreable"] = rec["gt_emotion"] is not None

        rows.append(rec)
        if i % 400 == 0:
            print(f"    {i}/{len(files)}")

    if unreadable:
        print(f"  WARNING: {len(unreadable)} unreadable file(s): {unreadable[:5]}")

    df = pd.DataFrame(rows)
    # best-effort pairing of the three synchronised views: camera clocks are
    # offset by ~25 min, so pair by recording order within a folder, not time.
    df["take_index"] = (df.sort_values("clip_id")
                          .groupby(["context", "scenario_dir", "view"])
                          .cumcount())
    return df.sort_values(["context", "scenario_dir", "view", "clip_id"]).reset_index(drop=True)


def main() -> None:
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    print("V3 table:")
    v3 = parse_v3_table()
    print(f"  parsed {len(v3)} scenario rows from Final_Dataset.docx")
    v3.to_csv(SCENARIOS_CSV, index=False)
    print(f"  -> {SCENARIOS_CSV}")

    print("Verifying folder numbering:")
    verify_folder_numbering(v3)

    print("Inventory:")
    clips = build_inventory(v3)
    clips.to_csv(CLIPS_CSV, index=False)
    print(f"  -> {CLIPS_CSV}  ({len(clips)} videos)")

    print("\nBy view:")
    print(clips.groupby(["view", "source"]).size().to_string())
    print("\nScoreable for emotion:")
    print(clips.scoreable.value_counts().to_string())
    print("\nGround-truth emotion distribution (scoreable only):")
    print(clips[clips.scoreable].gt_emotion.value_counts().to_string())
    unsc = clips[~clips.scoreable]
    if len(unsc):
        print("\nUnscoreable folders:")
        print(unsc.groupby(["scenario_dir", "caveat"]).size().to_string())


if __name__ == "__main__":
    main()
