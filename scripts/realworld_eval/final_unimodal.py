"""Ground-truth maps and paths for scoring ALL FOUR unimodal models on `data/final`.

`final_common.py` covers the emotion-only evaluation. This module adds the other
three cues so one pass over the clips answers "is each perception model good
enough to feed fusion?".

Two kinds of "no ground truth" must never be conflated:

* **designed-missing** — the V3 row's Missing column names the cue (`[missing]`
  in the cue column). The cue is absent *in the pixels on purpose*; there is no
  target, so the clip is excluded from that cue's accuracy. What we DO measure
  on those rows is the *observation rate*: if row #22 occludes the face, the
  emotion model should mostly fail to find one. A designed-missing row whose cue
  is still observed 95% of the time means the recording did not realise the
  design.
* **runtime-missing** — the model found nothing in a window (no face, no pose)
  on a row that does have a target. That is a perception failure and is reported
  as coverage, separately from accuracy on the windows that did fire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .final_common import (CLIPS_ROOT, FINAL, OUT_ROOT, SCENARIOS_CSV,  # noqa: F401
                           load_clips, parse_v3_table)

PERFRAME_DIR = FINAL / "features" / "perframe"
WINDOWS_PARQUET = FINAL / "features" / "unimodal_windows.parquet"
UNI_DIR = OUT_ROOT / "unimodal"

# Native class orders — must match the deployed heads exactly (MODEL_AUDIT.md).
EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GESTURE_LABELS = ["idle", "wave", "point", "thumbs_up", "thumbs_down",
                  "beckoning", "raise_hand", "both_hands_up"]
MOTION_LABELS = ["sitting", "standing", "walking", "stepping_back"]
CONTEXT_LABELS = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]

LABELS = {"emotion": EMOTION_LABELS, "gesture": GESTURE_LABELS,
          "motion": MOTION_LABELS, "context": CONTEXT_LABELS}
MODALITIES = list(LABELS)

# V3 table wording -> native class name. None = designed-missing (no target).
GT_MAPS = {
    "emotion": {"happy": "Happy", "sad": "Sad", "angry": "Anger",
                "anger": "Anger", "disgust": "Disgust", "surprise": "Surprise",
                "fear": "Fear", "neutral": "Neutral", "[missing]": None},
    # 'none' is a real class (the model calls it 'idle'), NOT a missing cue.
    "gesture": {"wave": "wave", "point": "point", "thumbs up": "thumbs_up",
                "thumbs down": "thumbs_down", "beckoning": "beckoning",
                "raise hand": "raise_hand", "both hands up": "both_hands_up",
                "none": "idle", "[missing]": None},
    # V3 says 'run' nowhere in the motion column; rows describing running set
    # motion=walk, so the 4-class head covers the table with no gap.
    "motion": {"sit": "sitting", "stand": "standing", "walk": "walking",
               "step back": "stepping_back", "[missing]": None},
    "context": {"classroom": "classroom", "kitchen": "kitchen"},
}

# clips.csv column holding each cue's V3 wording
CUE_COL = {"emotion": "emotion_v3", "gesture": "gesture_v3",
           "motion": "motion_v3", "context": "context"}

PROB_PREFIX = {"emotion": "emo", "gesture": "ges", "motion": "mot",
               "context": "ctx"}


def prob_cols(modality: str) -> list[str]:
    return [f"{PROB_PREFIX[modality]}_{c}" for c in LABELS[modality]]


def obs_col(modality: str) -> str:
    return f"{PROB_PREFIX[modality]}_obs"


def add_targets(clips: pd.DataFrame) -> pd.DataFrame:
    """Attach `gt_<modality>` (native class name or NaN) and `missing_<modality>`.

    Designed-missingness has TWO sources in the V3 table and both are needed:

    * the cue column reading `[missing]` — how emotion and gesture express it;
    * the `Missing` column naming the cue — the only way **context** can express
      it, because a context cell always names a real room (V3 #6, #24 and #56 read
      `classroom`/`kitchen` while their Missing column says `context`, meaning the
      room-identity sensor is offline, not that the room is unknown).

    Reading only the cue column silently leaves the context token observed on
    those rows, which is the opposite of what the scenario specifies.
    """
    out = clips.copy()
    if "missing_v3" not in out.columns:
        v3 = pd.read_csv(SCENARIOS_CSV)[["v3_row", "missing_v3"]]
        out = out.merge(v3, on="v3_row", how="left")
    named = (out.missing_v3.fillna("").astype(str).str.lower()
                .str.replace(" ", "").str.split(","))
    for m in MODALITIES:
        words = out[CUE_COL[m]].astype(str).str.strip().str.lower()
        out[f"gt_{m}"] = words.map(GT_MAPS[m])
        out[f"missing_{m}"] = words.eq("[missing]") | named.map(lambda lst, m=m: m in lst)
        # a cue the table declares missing has no target, however it was declared
        out.loc[out[f"missing_{m}"], f"gt_{m}"] = None
    return out


def clip_pool(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Windows -> one row per clip by mean of softmax over observed windows.

    Mean-pooling is the handover §5.2 contract and the 2026-07-26 pooling study
    found the top ten aggregators statistically tied, so this is not a tuning
    knob worth revisiting.
    """
    cols, obs = prob_cols(modality), obs_col(modality)
    fired = frame[frame[obs]]
    if fired.empty:
        return pd.DataFrame(columns=["clip_id", "pred", "n_obs", "n_windows"])
    g = fired.groupby("clip_id")[cols].mean()
    pred = np.asarray(LABELS[modality])[g.to_numpy().argmax(axis=1)]
    n_obs = fired.groupby("clip_id").size()
    n_win = frame.groupby("clip_id").size()
    return pd.DataFrame({"clip_id": g.index, "pred": pred,
                         "n_obs": n_obs.reindex(g.index).to_numpy(),
                         "n_windows": n_win.reindex(g.index).to_numpy()})


def score(y_true, y_pred, labels=None) -> dict:
    """Accuracy + macro-F1 over the classes that actually occur.

    `labels` is ignored for the macro average on purpose: averaging over a
    model's full head would charge every subset for classes the V3 table never
    asks for in that subset (kitchen has no 'hospital', a single scenario has
    one emotion), making per-view and per-split numbers incomparable. sklearn's
    default — the union of truth and prediction — still charges the model for
    classes it invents.
    """
    from sklearn.metrics import accuracy_score, f1_score
    if len(y_true) == 0:
        return {"n": 0, "acc": None, "macro_f1": None, "n_classes": 0}
    return {"n": int(len(y_true)),
            "acc": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
            "n_classes": int(len(set(y_true)))}
