"""
Extract face crops from the real-world HRI intent dataset for fine-tuning.

Reads the dataset's subject-disjoint split (splits.csv, column ``split_subject``)
and, for every train/val clip, samples frames evenly, detects the face
(full-range MediaPipe detector with close-range fallback) and saves the tight
crop as a JPEG in RAF-DB ImageFolder layout:

    data/realworld/{train,val}/<1-7>/<clip_id>_f<frame>.jpg

Folder numbers follow RAF-DB (1=Surprise ... 7=Neutral, see config.EMOTION_LABELS)
so ImageFolder assigns identical class indices to RAF-DB and real-world data.
Test-split clips are deliberately NOT extracted — they stay unseen for the
clip-level evaluation in scripts/evaluate_realworld.py.

Run (from modalities/emotion/):
    python scripts/extract_realworld_faces.py
    python scripts/extract_realworld_faces.py --frames 12 --limit 20   # smoke test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from config import DATA_DIR, EMOTION_LABELS, ROOT

DATASET_DIR = os.path.abspath(os.path.join(
    ROOT, "..", "..", "videos", "struct"))
OUT_ROOT = os.path.join(DATA_DIR, "realworld")

# Dataset "Intended Emotion" -> RAF-DB folder number (1-based index in EMOTION_LABELS)
INTENDED_TO_FOLDER = {
    "surprise": 1, "fear": 2, "disgust": 3, "happy": 4,
    "sad": 5, "angry": 6, "neutral": 7,
}


def load_split_clips():
    ann = os.path.join(DATASET_DIR, "annotations")
    splits = pd.read_csv(os.path.join(ann, "splits.csv"))
    scenarios = pd.read_csv(os.path.join(ann, "scenarios.csv"))
    splits["scenario"] = splits["scenario_id"].str.split("_").str[0]
    merged = splits.merge(
        scenarios[["Scenario ID", "Intended Emotion"]],
        left_on="scenario", right_on="Scenario ID", how="left")
    merged["folder"] = merged["Intended Emotion"].str.strip().str.lower().map(
        INTENDED_TO_FOLDER)
    merged = merged[merged["folder"].notna() & merged["split_subject"].isin(["train", "val"])]
    return merged.reset_index(drop=True)


def make_face_detectors():
    fd = mp.solutions.face_detection
    return [
        fd.FaceDetection(model_selection=1, min_detection_confidence=0.4),
        fd.FaceDetection(model_selection=0, min_detection_confidence=0.5),
    ]


def detect_best_face(detectors, rgb):
    for det in detectors:
        results = det.process(rgb)
        if results.detections:
            return max(results.detections, key=lambda d: d.score[0])
    return None


def extract_clip(video_path, out_dir, clip_id, detectors, n_frames):
    """Save up to n_frames face crops from one clip. Returns crops saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sorted(set(np.linspace(0, max(0, frame_count - 1),
                                    min(n_frames, max(1, frame_count))).astype(int)))
    saved = 0
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        det = detect_best_face(detectors, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if det is None:
            continue
        box = det.location_data.relative_bounding_box
        x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
        bw, bh = int(box.width * w), int(box.height * h)
        face = frame[y:y + bh, x:x + bw]
        if face.size == 0 or min(face.shape[:2]) < 16:
            continue
        cv2.imwrite(os.path.join(out_dir, f"{clip_id}_f{idx:03d}.jpg"), face,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12,
                    help="Frames sampled evenly per clip (default 12)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N clips (smoke test)")
    args = ap.parse_args()

    clips = load_split_clips()
    if args.limit:
        clips = clips.head(args.limit)
    print(f"Extracting {len(clips)} train/val clips -> {OUT_ROOT}")

    detectors = make_face_detectors()
    counts = {}
    for i, row in clips.iterrows():
        split, folder = row["split_subject"], int(row["folder"])
        out_dir = os.path.join(OUT_ROOT, split, str(folder))
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(DATASET_DIR, row["filepath"])
        n = extract_clip(video_path, out_dir, row["clip_id"], detectors, args.frames)
        counts[(split, folder)] = counts.get((split, folder), 0) + n
        if (i + 1) % 100 == 0 or (i + 1) == len(clips):
            print(f"  [{i + 1}/{len(clips)}] clips processed")

    print("\nCrops saved per split/class:")
    for split in ["train", "val"]:
        total = 0
        for folder in range(1, 8):
            n = counts.get((split, folder), 0)
            total += n
            print(f"  {split}/{folder} ({EMOTION_LABELS[folder - 1]:<8}): {n}")
        print(f"  {split} total: {total}\n")


if __name__ == "__main__":
    main()
