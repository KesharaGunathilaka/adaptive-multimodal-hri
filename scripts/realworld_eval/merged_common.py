"""Paths, V3 label authority and inventory helpers for the `data/final_merged` root.

`data/final_merged` (2026-07-29, renamed from 'final dataset merged') supersedes
`data/final`. Two things changed and both matter downstream:

1. **The table lost F09.** `docs/final_dataset_merged.docx` deletes the Farewell
   intent, relabels rows #18/#19/#48/#49/#61 F09 -> F01, blanks row #30 and adds
   row #63. That removes every cue collision: under the old table four rows were
   separable only by walking direction, which no perception model outputs, and
   that capped classroom test accuracy at 0.900. There are now **zero** colliding
   observable tuples across the 62 live rows.

2. **Every clip was renamed** to `S{v3_row}_F{intent}_c{NNN}`. The old filename
   was the only record of a clip's provenance — which recording session it came
   from, and (for the migrated ones) its `data/old` clip_id and actor. That
   mapping is recovered here by SHA-256 against `data/final` and `data/old`, not
   by filename, and is written into `clips.csv` so it is never lost again.

The .docx remains the ground-truth table; `TABLE_OVERRIDES` below is the only
place where this code disagrees with it, and every entry carries a reason.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data" / "final_merged"
CLIPS_ROOT = MERGED / "raw" / "clips"
ANNOT_DIR = MERGED / "annotations"
DOCX = ROOT / "docs" / "final_dataset_merged.docx"

SCENARIOS_CSV = ANNOT_DIR / "scenarios.csv"
CLIPS_CSV = ANNOT_DIR / "clips.csv"
DERIVED_CSV = ANNOT_DIR / "derived_rows.csv"
TAKES_CSV = ANNOT_DIR / "takes.csv"
RETIRED_CSV = ANNOT_DIR / "retired_rows.csv"
INTEGRITY_MD = ANNOT_DIR / "INTEGRITY.md"

# the roots we recover provenance from
FINAL = ROOT / "data" / "final"
OLD_CLIPS_CSV = ROOT / "data" / "old" / "annotations" / "clips.csv"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]

# V3 wording -> deployed class name; '[missing]' means no target to score against.
GT_EMOTION = {
    "happy": "Happy", "sad": "Sad", "angry": "Anger", "anger": "Anger",
    "disgust": "Disgust", "surprise": "Surprise", "fear": "Fear",
    "neutral": "Neutral", "[missing]": None,
}

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
CHANNELS = ("context", "emotion", "gesture", "motion")

_SCENARIO_RE = re.compile(r"^S(\d+)_F(\d+)$", re.I)       # folder name = V3 row
_MERGED_CLIP_RE = re.compile(r"^S(\d+)_F(\d+)_c(\d+)$", re.I)
_OLD_CURATED_RE = re.compile(r"^(S\d+_F\d+)_c(\d+)$", re.I)   # data/old clip_id
_TS_RE = re.compile(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_ ](\d{2})[_]?(\d{2})[_]?(\d{2})")
_TS_COMPACT_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")

# ── deliberate disagreements with the .docx ─────────────────────────────────
# Applied by parse_v3_table() and echoed into scenarios.csv's override_note,
# so nothing here is ever silent. Empty as of 2026-07-31: the document itself was
# corrected (row #30 deleted, #57 and #63 unmasked, #63 moved to the end, the
# Motion legend fixed, stale cross-references repaired), so the table now parses
# straight through. Add an entry only when the document cannot be fixed at source,
# and always with a `_note` saying why.
TABLE_OVERRIDES: dict[int, dict] = {}


# ── V3 table ────────────────────────────────────────────────────────────────
def _tidy(s: str) -> str:
    return re.sub(r"\s+", " ", s).replace("n o ne", "none").strip()


def _enum(s: str) -> str:
    """Enum cells only: Word sprinkles stray spaces inside words ('F0 1', 's tand')."""
    return re.sub(r"\s+", "", s).lower()


def parse_v3_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(live rows, retired rows) out of final_dataset_merged.docx.

    Word stores a table row as <w:tr>, a cell as <w:tc>, visible text as <w:t>;
    the scenario table is the only 13-column one in the document.
    """
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")
    cell_text = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)

    rows = []
    for tr in re.findall(r"<w:tr[ >](.*?)</w:tr>", xml, re.S):
        cells = re.split(r"</w:tc>", tr)[:-1]
        if len(cells) != 13:
            continue
        vals = [_tidy(" ".join(cell_text.findall(c))) for c in cells]
        if not vals[0].isdigit():          # header row
            continue
        rows.append({
            "v3_row": int(vals[0]),
            "split_design": _enum(vals[1]),
            "context": _enum(vals[2]),
            "scenario_desc": vals[3],
            "emotion_v3": _enum(vals[4]),
            "gesture_v3": _enum(vals[5]).replace("bothhandsup", "both hands up")
                                        .replace("raisehand", "raise hand")
                                        .replace("thumbsup", "thumbs up")
                                        .replace("thumbsdown", "thumbs down"),
            "motion_v3": _enum(vals[6]).replace("stepback", "step back"),
            "missing_v3": _tidy(vals[7]).replace("—", "").strip(),
            "intent": _enum(vals[8]).upper(),
            "goal": _tidy(vals[9]).replace("—", "").strip(),
            "why": vals[10],
            "action": _enum(vals[11]).upper(),
            "test_tags": _tidy(vals[12]).replace("—", "").strip(),
        })

    df = pd.DataFrame(rows).sort_values("v3_row").reset_index(drop=True)
    df["override_note"] = ""

    # a row emptied in the document is retired, not missing
    blank = df.split_design.eq("") & df.context.eq("")
    retired = df[blank].copy()
    df = df[~blank].copy()

    for row, ov in TABLE_OVERRIDES.items():
        note = ov.get("_note", "")
        tgt = retired if ov.get("_retire") else df
        m = tgt.v3_row.eq(row)
        if not m.any():
            continue
        for k, v in ov.items():
            if not k.startswith("_"):
                tgt.loc[m, k] = v
        tgt.loc[m, "override_note"] = note

    df["gt_emotion"] = df.emotion_v3.map(GT_EMOTION.get)
    df["emotion_masked"] = df.emotion_v3.eq("[missing]")

    # Designed missingness is declared two ways and BOTH count: the cue column can
    # read [MISSING], or the Missing column can name the channel while the cue
    # column still holds a real value (#6/#24/#56/#63 name a room but mask context).
    miss = df.missing_v3.map(
        lambda s: {x.strip() for x in _enum(s).split(",") if x.strip()})
    for ch in CHANNELS:
        col = "context" if ch == "context" else f"{ch}_v3"
        df[f"{ch}_masked"] = df[col].eq("[missing]") | miss.map(lambda s, c=ch: c in s)

    # what a perception stack can actually observe, after masking
    df["observed_tuple"] = [
        tuple("?" if r[f"{c}_masked"] else r["context" if c == "context" else f"{c}_v3"]
              for c in CHANNELS)
        for _, r in df.iterrows()
    ]
    return df.reset_index(drop=True), retired.reset_index(drop=True)


# ── inventory ───────────────────────────────────────────────────────────────
def classify_view(width: int, height: int, fps: float = 0.0) -> str:
    """Which of the three synchronised cameras a file came from.

    Named by probed geometry, not filename: the phone IMG_#### ranges overlap
    between the two handsets, but the formats never do.
    """
    long_side = max(width, height)
    if long_side <= 640:
        return "realsense_480p"     # 640x480 @15 — the deployment camera
    if long_side <= 1920:
        return "phone_1080p"
    return "phone_4k"


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


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    m = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            m.update(blk)
    return m.hexdigest()


def recorded_at(name: str) -> str | None:
    """ISO timestamp out of an original camera filename, if it carries one."""
    for rx in (_TS_COMPACT_RE, _TS_RE):
        m = rx.search(name)
        if m:
            y, mo, d, h, mi, s = m.groups()
            return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
    return None


def video_files(root: Path = CLIPS_ROOT) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXT)


# ── loaders ─────────────────────────────────────────────────────────────────
def load_clips(csv: Path = CLIPS_CSV, usable_only: bool = True) -> pd.DataFrame:
    if not csv.exists():
        raise SystemExit(f"{csv} missing — run scripts/20_merged_annotations.py first")
    df = pd.read_csv(csv)
    return df[df.usable] if usable_only else df


def load_scenarios(csv: Path = SCENARIOS_CSV) -> pd.DataFrame:
    if not csv.exists():
        raise SystemExit(f"{csv} missing — run scripts/20_merged_annotations.py first")
    return pd.read_csv(csv)
