"""Refresh `emotion_probs` (+ `face_valid`) in every Pass-1 per-frame cache
using the PROMOTED emotion checkpoint, in place -- gesture_feats, joints25,
context_probs etc. are left untouched.

Why this is needed and why the crop cache from script 33 is NOT enough: that
cache only kept 15 evenly-spaced frames per clip (for fast fine-tuning), so
re-scoring just those would leave most frames' `emotion_probs` computed by the
OLD checkpoint -- windows would silently mix old-model and new-model
predictions, corrupting the mean-pooled cue vector. This script re-runs face
detection + the NEW checkpoint over EVERY frame, exactly matching what Pass-1
(`fusion/extraction/perframe.py`) originally did, so the refreshed cache is
fully consistent (no video decoding was skippable here -- the face box
per frame isn't cached anywhere, only the crops for a 15-frame subsample were).

    .venv/Scripts/python scripts/37_refresh_emotion_cache.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from scripts.realworld_eval.merged_common import CLIPS_ROOT  # noqa: E402
from scripts.realworld_eval.merged_unimodal import PERFRAME_DIR, load_clips  # noqa: E402

EMO_DIR = ROOT / "modalities" / "emotion"
EMOTION_CKPT = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2.pth"  # now-promoted
MAX_SIDE = 640
BATCH = 64


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import mediapipe as mp
    emo = load_module("hri_emo_refresh", EMO_DIR / "inference" / "video.py")
    model = emo.build_model().to(device).eval()
    model.load_state_dict(torch.load(EMOTION_CKPT, map_location=device, weights_only=True))
    transform = emo.get_transform()
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    print(f"device={device}  checkpoint={EMOTION_CKPT.name}", flush=True)

    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    print(f"{clips.clip_id.nunique()} clips to refresh", flush=True)

    t0, failures = time.time(), []
    for i, r in enumerate(clips.itertuples(), 1):
        npz_path = PERFRAME_DIR / f"{r.clip_id}.npz"
        if not npz_path.exists():
            failures.append((r.clip_id, "no per-frame cache"))
            continue
        try:
            old = dict(np.load(npz_path))
            path = CLIPS_ROOT / r.filepath
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError("cannot open")

            T = int(old["n_frames"])
            emotion_probs = np.full((T, 7), np.nan, np.float32)
            face_valid = np.zeros(T, dtype=bool)
            crops, crop_idx = [], []
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
                        crops.append(crop)
                        crop_idx.append(t)
                        face_valid[t] = True
                t += 1
            cap.release()

            if crops:
                from PIL import Image
                out = []
                with torch.no_grad():
                    for j in range(0, len(crops), BATCH):
                        batch = torch.stack([
                            transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                            for c in crops[j:j + BATCH]]).to(device)
                        out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
                probs = np.concatenate(out)
                emotion_probs[crop_idx] = probs

            old["emotion_probs"] = emotion_probs
            old["face_valid"] = face_valid
            np.savez_compressed(npz_path, **old)
        except Exception as e:  # noqa: BLE001
            failures.append((r.clip_id, str(e)))
            print(f"  FAILED {r.clip_id}: {e}", flush=True)
            continue
        if i % 100 == 0 or i == len(clips):
            rate = i / (time.time() - t0)
            eta = (len(clips) - i) / max(rate, 1e-9) / 60
            print(f"[{i}/{len(clips)}] {r.clip_id}  ({rate:.2f} clips/s, eta {eta:.0f} min)",
                  flush=True)

    print(f"\ndone: {len(clips) - len(failures)}/{len(clips)} refreshed, "
          f"{len(failures)} failure(s)")
    for cid, e in failures[:10]:
        print(f"  {cid}: {e}")


if __name__ == "__main__":
    main()
