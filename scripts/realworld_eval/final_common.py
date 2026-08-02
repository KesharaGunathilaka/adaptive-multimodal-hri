"""Paths, V3 label authority and clip inventory for the `data/final` root.

**SUPERSEDED 2026-07-31 — this module reads the OLD label table.** The current
root is `data/final_merged` with `docs/final_dataset_merged.docx`; use
`scripts/realworld_eval/merged_common.py` instead. The two tables disagree in
ways that silently change every score: `final_dataset_merged.docx` deleted intent
**F09** (relabelling rows #18/#19/#48/#49/#61 to F01), retired row **#30**, added
row **#63**, unmasked **#57** and **#63**, and moved **#32** to Test / **#52** to
Train. `data/final/annotations/*.csv` still describe the pre-merge design.

Kept so the 2026-07-27 results under `results/realworld_eval_final/` remain
reproducible. Do not use it to produce new numbers.

`data/final` is the post-2026-07-25 collection root and is numbered differently
from `data/`: **its scenario folders are named by Final_Dataset.docx V3 row
number**, not by the old recording IDs. `data/final/raw/clips/classroom/S07_F04`
holds the clips recorded as `S01_F04`, and V3 row #7 is
`classroom, neutral, raise hand, sit` — verified against all 21 folders that
carry migrated clips (see docs/DATASET_STATUS.md §2).

That makes the .docx the ground-truth table for everything under `data/final`,
including the 2026-07-25 takes that no CSV has ever described. This module
extracts it and joins it to a probed per-file inventory.
"""
from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.warn(
    "final_common reads data/final + Final_Dataset.docx, the SUPERSEDED label "
    "table (F09 still present, #30 live, #63 absent). New work must import "
    "scripts.realworld_eval.merged_common, which reads data/final_merged. "
    "See docs/DECISIONS.md 2026-07-31.",
    DeprecationWarning, stacklevel=2)

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "final"
CLIPS_ROOT = FINAL / "raw" / "clips"
ANNOT_DIR = FINAL / "annotations"
DOCX = ROOT / "docs" / "Final_Dataset.docx"

SCENARIOS_CSV = ANNOT_DIR / "scenarios_v3.csv"
CLIPS_CSV = ANNOT_DIR / "clips.csv"

# Overlay videos land on the external disk; the numeric outputs stay in-repo.
E_ROOT = Path("E:/emotion_inference")
ANNOTATED_DIR = E_ROOT / "annotated"

OUT_ROOT = ROOT / "results" / "realworld_eval_final"
PRED_DIR = OUT_ROOT / "predictions" / "emotion"
SCORE_DIR = OUT_ROOT / "scoring"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]

# V3 wording -> deployed class name. '[MISSING]' means the row masks emotion by
# design, so there is no target to score against (V3 rows #22, #25, #30, #49,
# #57, #58) — those clips are predicted and reported, never scored.
GT_EMOTION = {
    "happy": "Happy", "sad": "Sad", "angry": "Anger", "anger": "Anger",
    "disgust": "Disgust", "surprise": "Surprise", "fear": "Fear",
    "neutral": "Neutral", "[missing]": None,
}

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
_CURATED_RE = re.compile(r"^(S\d+_F\d+)_c(\d+)$", re.I)   # migrated from data/raw
_SCENARIO_RE = re.compile(r"^S(\d+)_F(\d+)$", re.I)       # folder name = V3 row


# ── V3 table ────────────────────────────────────────────────────────────────
def parse_v3_table() -> pd.DataFrame:
    """The 62-row scenario table out of Final_Dataset.docx.

    Word stores a table row as <w:tr>, a cell as <w:tc> and visible text as
    <w:t>; the whole table is the only 13-column one in the document.
    """
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")
    cell_text = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)

    rows = []
    for tr in re.findall(r"<w:tr[ >](.*?)</w:tr>", xml, re.S):
        cells = re.split(r"</w:tc>", tr)[:-1]
        if len(cells) != 13:
            continue
        vals = [" ".join(cell_text.findall(c)).strip() for c in cells]
        if not vals[0].isdigit():          # header row
            continue
        # row #41 reads 'n o ne' in the source document
        norm = lambda s: re.sub(r"\s+", " ", s).replace("n o ne", "none").strip()
        rows.append({
            "v3_row": int(vals[0]),
            "split_design": norm(vals[1]).lower(),
            "context": norm(vals[2]).lower(),
            "scenario_desc": norm(vals[3]),
            "emotion_v3": norm(vals[4]).lower(),
            "gesture_v3": norm(vals[5]).lower(),
            "motion_v3": norm(vals[6]).lower(),
            "missing_v3": norm(vals[7]).replace("—", "").strip(),
            "intent": norm(vals[8]).upper(),
            "action": norm(vals[11]),
        })
    df = pd.DataFrame(rows).sort_values("v3_row").reset_index(drop=True)
    df["gt_emotion"] = df.emotion_v3.map(lambda e: GT_EMOTION.get(e))
    df["emotion_masked"] = df.emotion_v3.eq("[missing]")
    return df


# ── inventory ───────────────────────────────────────────────────────────────
def classify_view(width: int, height: int, fps: float) -> str:
    """Which of the three synchronised cameras a file came from.

    Named by probed geometry rather than filename: the phone IMG_#### ranges
    overlap between the two handsets, but the formats never do.
    """
    long_side = max(width, height)
    if long_side <= 640:
        return "realsense_480p"     # 640x480 @15 — the deployment camera
    if long_side <= 1920:
        return "phone_1080p"        # 1920x1080 @30
    return "phone_4k"               # 3840x2160 @60


def probe(path: Path) -> dict | None:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    if not ok or w == 0 or h == 0:
        return None
    return {"width": w, "height": h, "fps": round(fps, 3), "n_frames": n,
            "duration_s": round(n / fps, 3) if fps else 0.0}


def load_clips(csv: Path = CLIPS_CSV) -> pd.DataFrame:
    if not csv.exists():
        raise SystemExit(f"{csv} missing — run scripts/14_final_annotations.py first")
    return pd.read_csv(csv)


def pred_path(row) -> Path:
    return PRED_DIR / row["context"] / row["scenario_dir"] / f"{row['clip_id']}.csv"


def annotated_path(row) -> Path:
    return ANNOTATED_DIR / row["context"] / row["scenario_dir"] / f"{row['clip_id']}_emotion.mp4"


def prob_cols() -> list[str]:
    return [f"p_{c}" for c in EMOTION_LABELS]
