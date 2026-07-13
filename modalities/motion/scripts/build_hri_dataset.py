"""
build_hri_dataset.py

Extracts labeled training windows from the hri-multimodal-intent-v1.0.0
real-world dataset, for fine-tuning MotionLSTM on top of the NTU-only
baseline. Mirrors build_dataset.py's windowing scheme but sources frames
from MediaPipe pose estimation on the recorded clips instead of raw NTU
skeleton files.

Label source: each clip inherits its scenario's "Intended Motion" field
(hri-multimodal-intent-v1.0.0/annotations/scenarios.csv), mapped onto the
same 4-class taxonomy as build_dataset.py (sitting/standing/walking/
stepping_back). Scenario S19 ("Move backward (run)") is excluded — it
doesn't map cleanly onto any of the 4 classes.

Frame-rate handling: clips in this dataset were recorded at a mix of fps
(15/23/24/25/29/30/31 — including within the same scenario). Each clip's
extracted position sequence is linearly resampled onto a uniform 30 Hz grid
before windowing, so window semantics (30 frames =~ 1s) match what the
model was trained on regardless of source capture rate.

Usage:
    python build_hri_dataset.py --split train
    python build_hri_dataset.py --split val
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from collections import Counter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SRC_DIR, "..", "..", "..", "videos", "struct")
OUT_DIR = os.path.join(SRC_DIR, "..", "data", "processed", "hri_finetune")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
STEP_SIZE = 10
TARGET_FPS = 30.0
MIN_FRAMES = WINDOW_SIZE

MOTION_LABELS = {0: "sitting", 1: "standing", 2: "walking", 3: "stepping_back"}
MOTION_MAP = {
    "sitting": 0, "stand": 1,
    "walking": 2, "walk": 2, "walk (toward)": 2,
    "stepping back": 3,
}

# Same 14-joint subset / mapping / axis correction as inference.py, video_demo.py
MP_TO_NTU = {0: 3, 11: 4, 12: 8, 13: 5, 14: 9, 15: 6, 16: 10,
             23: 12, 24: 16, 25: 13, 26: 17, 27: 14, 28: 18}
JOINT_SUBSET = [0, 1, 2, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17]
HIP_L_IDX, HIP_R_IDX = JOINT_SUBSET.index(12), JOINT_SUBSET.index(16)
SHOULDER_L_IDX, SHOULDER_R_IDX = JOINT_SUBSET.index(4), JOINT_SUBSET.index(8)


def normalize_skeleton(joints: np.ndarray) -> np.ndarray:
    hip_center = (joints[HIP_L_IDX] + joints[HIP_R_IDX]) / 2.0
    j = joints - hip_center
    shoulder_dist = np.linalg.norm(j[SHOULDER_L_IDX] - j[SHOULDER_R_IDX])
    if shoulder_dist > 0.05:
        j = j / shoulder_dist
    return j.astype(np.float32)


def get_joints(world_lms) -> np.ndarray:
    """MediaPipe world landmarks -> (25,3) NTU-layout joints, axis-corrected
    (negate X,Y — see coordinate-convention investigation)."""
    joints_25 = np.zeros((25, 3), dtype=np.float32)
    for mp_idx, ntu_idx in MP_TO_NTU.items():
        lm = world_lms[mp_idx]
        joints_25[ntu_idx] = [-lm.x, -lm.y, lm.z]
    joints_25[0] = (joints_25[12] + joints_25[16]) / 2
    joints_25[1] = (joints_25[4] + joints_25[8]) / 2
    joints_25[2] = joints_25[1] * 0.5 + joints_25[3] * 0.5
    return joints_25


def extract_clip_positions(video_path: str, pose) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Returns (T,42) normalized positions and (T,) timestamps (seconds)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    positions, timestamps = [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_world_landmarks:
            joints_25 = get_joints(result.pose_world_landmarks.landmark)
            joints_14 = joints_25[JOINT_SUBSET]
            positions.append(normalize_skeleton(joints_14).flatten())
            timestamps.append(frame_idx / fps)
        frame_idx += 1
    cap.release()
    if len(positions) < 2:
        return None, None
    return np.stack(positions), np.array(timestamps)


def resample_to_30fps(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    duration = timestamps[-1] - timestamps[0]
    n_out = max(int(round(duration * TARGET_FPS)) + 1, 2)
    grid = np.linspace(timestamps[0], timestamps[-1], n_out)
    out = np.zeros((n_out, positions.shape[1]), dtype=np.float32)
    for d in range(positions.shape[1]):
        out[:, d] = np.interp(grid, timestamps, positions[:, d])
    return out


def positions_to_windows(positions: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]
    feats = np.concatenate([positions, vel], axis=-1)  # (T,84)
    T = feats.shape[0]
    X_windows, y_windows = [], []
    for start in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
        X_windows.append(feats[start:start + WINDOW_SIZE])
        y_windows.append(label)
    if not X_windows:
        return np.empty((0, WINDOW_SIZE, 84)), np.empty((0,), dtype=np.int64)
    return np.stack(X_windows).astype(np.float32), np.array(y_windows, dtype=np.int64)


def load_usable_clips() -> pd.DataFrame:
    scenarios = pd.read_csv(os.path.join(DATASET_ROOT, "annotations", "scenarios.csv"))
    splits = pd.read_csv(os.path.join(DATASET_ROOT, "annotations", "splits.csv"))

    scenarios["motion_norm"] = scenarios["Intended Motion"].str.strip().str.lower()
    scenarios["class4"] = scenarios["motion_norm"].map(MOTION_MAP)
    splits["scenario_base"] = splits["scenario_id"].str.split("_").str[0]

    merged = splits.merge(scenarios[["Scenario ID", "class4"]],
                           left_on="scenario_base", right_on="Scenario ID", how="left")
    return merged[merged.class4.notna() & (merged.frame_count >= MIN_FRAMES)]


def build(split: str):
    print(f"\n{'='*60}\nhri-multimodal-intent-v1.0.0 -> Fine-tune windows ({split})\n{'='*60}")

    df = load_usable_clips()
    df = df[df.split_scenario == split]
    print(f"Clips in split '{split}': {len(df)}")
    print("Per-class clip counts:")
    counts = df.class4.map(MOTION_LABELS).value_counts()
    print(counts)

    pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5, static_image_mode=False)

    X_all, y_all = [], []
    skipped = 0
    t0 = time.time()

    for i, row in enumerate(df.itertuples()):
        video_path = os.path.join(DATASET_ROOT, row.filepath)
        pos, ts = extract_clip_positions(video_path, pose)
        if pos is None:
            skipped += 1
            continue
        resampled = resample_to_30fps(pos, ts)
        X_w, y_w = positions_to_windows(resampled, int(row.class4))
        if len(X_w) > 0:
            X_all.append(X_w)
            y_all.append(y_w)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(df)}]  elapsed={time.time()-t0:.0f}s")

    pose.close()
    print(f"\nDone in {time.time()-t0:.0f}s | clips processed={len(df)-skipped} | skipped={skipped}")

    if not X_all:
        print("ERROR: no windows collected.")
        sys.exit(1)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"\nTotal windows: {len(y):,}  |  X={X.shape}  y={y.shape}")
    window_counts = Counter(y.tolist())
    for idx, name in MOTION_LABELS.items():
        print(f"  {idx} {name:<15}: {window_counts.get(idx, 0):>7,}")

    np.save(os.path.join(OUT_DIR, f"X_{split}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y_{split}.npy"), y)
    print(f"\nSaved to {OUT_DIR}/X_{split}.npy, y_{split}.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()
    build(args.split)
