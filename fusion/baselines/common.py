"""Shared data loading / metrics for the fusion baselines.

Splits are actor-disjoint via the `split_subject` column (P01/P02/P06 train,
P04 val, P03/P05/P07-09 test — see docs/DATASET_STATUS.md). Rows with
status != 'train' (S28 recombination pool) are excluded from all supervised
sets. NaN cue probabilities (runtime-missing) are zero-filled, with the four
*_obs flags appended so models can tell "missing" from "uniform".
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FEATURES = REPO / "data" / "features" / "features_v1.parquet"

INTENTS = [f"F{i:02d}" for i in range(1, 11)]

EMO_COLS = ["emo_Surprise", "emo_Fear", "emo_Disgust", "emo_Happy",
            "emo_Sad", "emo_Anger", "emo_Neutral"]
GES_COLS = ["ges_idle", "ges_wave", "ges_point", "ges_thumbs_up",
            "ges_thumbs_down", "ges_beckoning", "ges_raise_hand",
            "ges_both_hands_up"]
MOT_COLS = ["mot_sitting", "mot_standing", "mot_walking", "mot_stepping_back"]
CTX_COLS = ["ctx_classroom", "ctx_kitchen", "ctx_hospital", "ctx_cloth_store",
            "ctx_museum"]
OBS_COLS = ["emo_obs", "ges_obs", "mot_obs", "ctx_obs"]
PROB_COLS = EMO_COLS + GES_COLS + MOT_COLS + CTX_COLS

MODALITY_COLS = {"emotion": EMO_COLS, "gesture": GES_COLS,
                 "motion": MOT_COLS, "context": CTX_COLS}


def load_windows():
    df = pd.read_parquet(FEATURES)
    df = df[df.status == "train"].copy()          # S28 pool excluded
    df["y"] = df.intent.map(INTENTS.index)
    return df


def split(df):
    return {name: df[df.split_subject == name] for name in ("train", "val", "test")}


def xy(frame, cols=None):
    cols = cols or PROB_COLS
    X = frame[cols].fillna(0.0).to_numpy(np.float32)
    X = np.concatenate([X, frame[OBS_COLS].to_numpy(np.float32)], axis=1)
    return X, frame["y"].to_numpy()


def clip_vote(frame, window_pred):
    """Majority vote per clip -> (clip_true, clip_pred)."""
    t = frame.assign(pred=window_pred)
    g = t.groupby("clip_id")
    true = g["y"].first()
    pred = g["pred"].agg(lambda s: s.value_counts().idxmax())
    return true.to_numpy(), pred.to_numpy()


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             labels=list(range(10)),
                                             zero_division=0)), 4)}


def evaluate(frame, window_pred):
    win = metrics(frame["y"].to_numpy(), window_pred)
    ct, cp = clip_vote(frame, window_pred)
    clip = metrics(ct, cp)
    return {"window": win, "clip": clip, "n_windows": len(frame),
            "n_clips": frame.clip_id.nunique()}
