"""Extract and cache face-crop IMAGES for every clip in `data/final_merged`.

Why this is a separate pass: the Pass-1 per-frame cache
(`data/final_merged/features/perframe/*.npz`, from `scripts/27_merged_extract.py`)
stores emotion_probs -- the DEPLOYED model's OUTPUT -- not the input face crop
images, because that's all Pass-1 needed. Fine-tuning needs the crops
themselves, and re-scoring a NEW checkpoint needs them too (script 35), so this
pass decodes video ONCE and both training and later comparison reuse the cache
-- no third full-dataset decode.

Only face DETECTION is needed here (not Holistic/gesture/motion/context), so
this is faster per clip than the original 4-modality Pass-1.

Caches up to `MAX_CROPS` evenly-spaced frames per clip (not every frame --
233K total frames vs a ~15-frame/clip subsample is a large redundancy cut for
near-zero information loss, and keeps fine-tuning time reasonable; see
docs/WORKLOG.md 2026-08-04 for the throughput reasoning). Crops are saved at
240x240 (Resize(240) is the original transform pipeline's first step, leaving
room for RandomCrop(224) augmentation at train time).

    .venv/Scripts/python scripts/33_extract_emotion_crops.py
    .venv/Scripts/python scripts/33_extract_emotion_crops.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from scripts.realworld_eval.merged_common import CLIPS_ROOT  # noqa: E402
from scripts.realworld_eval.merged_unimodal import load_clips  # noqa: E402

EMO_DIR = ROOT / "modalities" / "emotion"
OUT_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"
MAX_SIDE = 640
CROP_SIZE = 240
MAX_CROPS = 15          # evenly-spaced frames with a detected face, per clip


def uniform_indices(n, target):
    if n <= target:
        return np.arange(n)
    return np.round(np.linspace(0, n - 1, target)).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import mediapipe as mp
    emo = load_module("hri_emo_crops", EMO_DIR / "inference" / "video.py")
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    todo = [r for r in clips.itertuples() if not (OUT_DIR / f"{r.clip_id}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{clips.clip_id.nunique()} clips | {len(todo)} to extract", flush=True)
    if not todo:
        return

    t0, failures = time.time(), []
    for i, r in enumerate(todo, 1):
        try:
            path = CLIPS_ROOT / r.filepath
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError("cannot open")
            all_crops, all_idx = [], []
            t = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                scale = MAX_SIDE / max(h, w)
                small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
                box = emo.detect_face_box(face_det, frame, small)
                if box is not None:
                    x, y, bw, bh = box
                    crop = frame[y:y + bh, x:x + bw]
                    if crop.size:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        rgb = cv2.resize(rgb, (CROP_SIZE, CROP_SIZE))
                        all_crops.append(rgb)
                        all_idx.append(t)
                t += 1
            cap.release()

            if all_crops:
                sel = uniform_indices(len(all_crops), MAX_CROPS)
                crops = np.stack([all_crops[j] for j in sel]).astype(np.uint8)
                idx = np.array([all_idx[j] for j in sel], dtype=np.int64)
            else:
                crops = np.zeros((0, CROP_SIZE, CROP_SIZE, 3), np.uint8)
                idx = np.zeros((0,), np.int64)
            np.savez_compressed(OUT_DIR / f"{r.clip_id}.npz", crops=crops,
                               frame_idx=idx, n_frames_total=t)
        except Exception as e:  # noqa: BLE001
            failures.append((r.clip_id, str(e)))
            print(f"  FAILED {r.clip_id}: {e}", flush=True)
            continue
        if i % 100 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / max(rate, 1e-9) / 60
            print(f"[{i}/{len(todo)}] {r.clip_id}  ({rate:.2f} clips/s, eta {eta:.0f} min)",
                  flush=True)

    n = len(list(OUT_DIR.glob("*.npz")))
    print(f"\ndone: {n}/{clips.clip_id.nunique()} clips cached, {len(failures)} failure(s)")


if __name__ == "__main__":
    main()
