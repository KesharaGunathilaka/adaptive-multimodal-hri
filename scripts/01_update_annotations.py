"""Sync data/annotations/clips.csv to the curated clip set in data/raw/clips.

The team manually removed low-quality clips (and all of S27_F06) from data/raw/clips;
this script drops the corresponding rows from clips.csv, merges the subject-level
split columns from videos/struct/annotations/splits.csv, and writes data/labels.csv
(one row per scenario, V3-corrected intent labels per docs/DATASET_STATUS.md).

Run from repo root:  .venv/Scripts/python scripts/01_update_annotations.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Scenario -> V3 table row / corrected intent / action / notes.
# Source of truth: docs/Final_Dataset.docx (V3) + docs/DATASET_STATUS.md mapping.
V3_MAP = {
    # scenario_id: (v3_row, intent_v3, action, status, note)
    "S01_F04": (7,  "F04", "A05", "train", ""),
    "S02_F01": (1,  "F01", "A01", "train", "direction: toward robot"),
    "S03_F05": (10, "F05", "A06", "train", ""),
    "S04_F04": (8,  "F04", "A05", "train", ""),
    "S05_F02": (14, "F07", "A08", "train", "V3 relabel F02->F07 (V2 #5 -> V3 #14); folder name keeps F02"),
    "S06_F08": (16, "F08", "A07", "train", ""),
    "S07_F03": (5,  "F03", "A04", "train", ""),
    "S08_F06": (12, "F06", "A11", "train", "gesture [MISSING] by design (hands occupied)"),
    "S09_F02": (3,  "F02", "A14", "train", ""),
    "S11_F05": (11, "F05", "A06", "train", ""),
    "S12_F01": (2,  "F01", "A01", "train", ""),
    "S18_F01": (32, "F01", "A01", "train", ""),
    "S19_F02": (34, "F02", "A02/A03", "train", ""),
    "S20_F03": (36, "F03", "A04", "train", ""),
    "S21_F04": (38, "F04", "A05", "train", "collision pair with S28; S21 is the training scenario for sad+thumbs_down+sit"),
    "S22_F05": (40, "F05", "A06", "train", ""),
    "S23_F08": (46, "F08", "A07", "train", "low clip count after curation"),
    "S24_F07": (44, "F07", "A08", "train", ""),
    "S25_F09": (48, "F09", "A09", "train", "direction: toward exit"),
    "S26_F02": (35, "F02", "A02/A03", "train", ""),
    "S28_F10": (51, "F10", "A12", "recombination_pool",
                "V3 #51 expects gesture=none but recordings directed thumbs_down -> excluded from "
                "supervised training (collides with S21/F04); reserved as cue-vector source for "
                "recombination. Clips whose gesture model output is 'idle' may be promoted to F10 "
                "training after extraction."),
    "S29_F03": (37, "F03", "A13", "train", ""),
}


def main() -> None:
    clips = pd.read_csv(DATA / "annotations" / "clips.csv")
    n_before = len(clips)

    exists = clips["filepath"].map(lambda p: (DATA / p).is_file())
    removed = clips[~exists]
    clips = clips[exists].copy()
    print(f"clips.csv: {n_before} -> {len(clips)} rows ({len(removed)} curated out)")
    rm_counts = removed["scenario_id"].value_counts().sort_index()
    print("removed per scenario:\n", rm_counts.to_string())

    splits = pd.read_csv(ROOT / "videos" / "struct" / "annotations" / "splits.csv",
                         usecols=["clip_id", "split_subject"])
    clips = clips.merge(splits, on="clip_id", how="left")
    missing_split = clips["split_subject"].isna().sum()
    if missing_split:
        print(f"WARNING: {missing_split} clips had no split_subject in struct splits.csv")

    clips.to_csv(DATA / "annotations" / "clips.csv", index=False)

    # labels.csv — one row per scenario
    scen = pd.read_csv(DATA / "annotations" / "scenarios.csv")
    scen["scenario_id"] = scen["Scenario ID"] + "_" + scen["Intent"]
    counts = clips.groupby("scenario_id").size()

    rows = []
    for sid, (v3_row, intent, action, status, note) in V3_MAP.items():
        src = scen[scen["scenario_id"] == sid]
        if src.empty:
            print(f"WARNING: {sid} not in scenarios.csv")
            continue
        s = src.iloc[0]
        missing = ""
        for cue, col in [("emotion", "Intended Emotion"), ("gesture", "Intended Gesture"),
                         ("motion", "Intended Motion")]:
            if str(s[col]).strip().upper() == "[MISSING]":
                missing = cue
        rows.append({
            "scenario_id": sid,
            "v3_row": v3_row,
            "split": "train",           # all recorded scenarios are V3 Train rows
            "status": status,           # train | recombination_pool
            "context": s["Context"],
            "emotion": s["Intended Emotion"],
            "gesture": s["Intended Gesture"],
            "motion": s["Intended Motion"],
            "missing": missing,
            "intent": intent,           # V3-corrected
            "intent_recorded": s["Intent"],  # what the folder/direction sheet said
            "action": action,
            "n_clips": int(counts.get(sid, 0)),
            "note": note,
        })
    labels = pd.DataFrame(rows)
    labels.to_csv(DATA / "labels.csv", index=False)
    print(f"\nlabels.csv: {len(labels)} scenarios, "
          f"{int(labels[labels.status == 'train'].n_clips.sum())} training clips, "
          f"{int(labels[labels.status != 'train'].n_clips.sum())} reserved")
    print(labels[["scenario_id", "v3_row", "intent", "status", "n_clips"]].to_string(index=False))


if __name__ == "__main__":
    main()
