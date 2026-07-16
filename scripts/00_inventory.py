"""Verify the curated dataset: open every clip in data/raw/clips with OpenCV,
read first + last frame, record fps/frames/resolution, flag corrupt files and
scenarios with fewer than 50 clips. Writes data/inventory.csv.

Run from repo root:  .venv/Scripts/python scripts/00_inventory.py
"""
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_CLIPS = 50


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    info = {"readable": False, "fps": None, "frames": None, "width": None, "height": None}
    if ok:
        info["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 3)
        info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok_first, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(info["frames"] - 1, 0))
        ok_last, _ = cap.read()
        info["readable"] = bool(ok_first and ok_last)
    cap.release()
    return info


def main() -> None:
    clips = pd.read_csv(DATA / "annotations" / "clips.csv")
    rows = []
    for r in clips.itertuples():
        p = DATA / r.filepath
        rec = {"clip_id": r.clip_id, "scenario_id": r.scenario_id,
               "person_id": r.person_id, "filepath": r.filepath}
        rec.update(probe(p) if p.is_file() else {"readable": False})
        rows.append(rec)
    inv = pd.DataFrame(rows)
    inv.to_csv(DATA / "inventory.csv", index=False)

    bad = inv[~inv["readable"]]
    print(f"{len(inv)} clips probed, {len(bad)} unreadable")
    if len(bad):
        print(bad[["clip_id", "filepath"]].to_string(index=False))
    counts = inv.groupby("scenario_id").size().sort_index()
    low = counts[counts < MIN_CLIPS]
    print(f"\nscenarios under {MIN_CLIPS} clips:")
    print(low.to_string() if len(low) else "  none")
    fps = inv["fps"].value_counts()
    print("\nfps distribution:\n", fps.to_string())


if __name__ == "__main__":
    main()
