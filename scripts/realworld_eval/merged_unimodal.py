"""Ground-truth maps and paths for scoring all four unimodal models on
`data/final_merged`.

Mirrors `final_unimodal.py` (same window table shape, same scoring contract),
but reads the annotation columns `20_merged_annotations.py` already computed
(`gt_emotion`, `*_masked`) rather than re-deriving them, and joins the split
columns `23_build_splits.py` added (`split`, `headline_eval`,
`resolution_class`, `agg_span_s`) instead of raw `view` / `split_design`.

Designed-missing vs runtime-missing (same distinction as `data/final`, restated
because it is easy to get backwards):

* **designed-missing** (`*_masked` in clips.csv) — the V3 row says this cue is
  absent *in the pixels on purpose*. No target; excluded from that cue's
  accuracy. What we DO measure is the *observation rate* on those rows — if the
  model still finds a face 99% of the time on a "face occluded" row, the
  recording did not realise the design (see `docs/DATASET_FIXLIST.md` and
  `docs/methodology/04_missing_cues.md`).
* **runtime-missing** — the model found nothing on a row that HAS a target.
  A perception failure, reported as coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .merged_common import ANNOT_DIR, CLIPS_CSV, MERGED  # noqa: F401

PERFRAME_DIR = MERGED / "features" / "perframe"
WINDOWS_PARQUET = MERGED / "features" / "unimodal_windows.parquet"
SPLITS_CSV = ANNOT_DIR / "splits.csv"
UNI_DIR = MERGED.parent.parent / "results" / "realworld_eval_merged" / "unimodal"

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GESTURE_LABELS = ["idle", "wave", "point", "thumbs_up", "thumbs_down",
                  "beckoning", "raise_hand", "both_hands_up"]
MOTION_LABELS = ["sitting", "standing", "walking", "stepping_back"]
CONTEXT_LABELS = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]

LABELS = {"emotion": EMOTION_LABELS, "gesture": GESTURE_LABELS,
          "motion": MOTION_LABELS, "context": CONTEXT_LABELS}
MODALITIES = list(LABELS)

# clips.csv wording (already lower-cased/tidied by 20_merged_annotations.py) ->
# native class name used by each deployed head.
GT_MAPS = {
    "emotion": {"happy": "Happy", "sad": "Sad", "angry": "Anger",
                "anger": "Anger", "disgust": "Disgust", "surprise": "Surprise",
                "fear": "Fear", "neutral": "Neutral"},
    "gesture": {"wave": "wave", "point": "point", "thumbs up": "thumbs_up",
                "thumbs down": "thumbs_down", "beckoning": "beckoning",
                "raise hand": "raise_hand", "both hands up": "both_hands_up",
                "idle": "idle"},          # V3 already spells this 'idle', not 'none'
    "motion": {"sit": "sitting", "stand": "standing", "walk": "walking",
               "step back": "stepping_back"},   # 'run' does not occur (N.B. audit N10)
    "context": {"classroom": "classroom", "kitchen": "kitchen"},
}

CUE_COL = {"emotion": "emotion_v3", "gesture": "gesture_v3",
           "motion": "motion_v3", "context": "context"}
MASK_COL = {"emotion": "emotion_masked", "gesture": "gesture_masked",
           "motion": "motion_masked", "context": "context_masked"}

PROB_PREFIX = {"emotion": "emo", "gesture": "ges", "motion": "mot", "context": "ctx"}


def prob_cols(modality: str) -> list[str]:
    return [f"{PROB_PREFIX[modality]}_{c}" for c in LABELS[modality]]


def obs_col(modality: str) -> str:
    return f"{PROB_PREFIX[modality]}_obs"


def load_clips() -> pd.DataFrame:
    """Usable clips joined with the split/headline/resolution columns.

    `clips.csv` is the label authority (V3 wording, masks); `splits.csv` is the
    eval authority (`split`, `headline_eval`, `resolution_class`,
    `agg_span_s`) — join rather than duplicate so the two can never drift.
    """
    clips = pd.read_csv(CLIPS_CSV)
    clips = clips[clips.usable == True].copy()  # noqa: E712
    splits = pd.read_csv(SPLITS_CSV)[
        ["clip_id", "split", "headline_eval", "resolution_class",
         "orientation", "agg_span_s"]]
    out = clips.merge(splits, on="clip_id", how="inner")
    for m in MODALITIES:
        col = CUE_COL[m]
        words = out[col].astype(str).str.strip().str.lower()
        out[f"gt_{m}"] = words.map(GT_MAPS[m].get)
        out.loc[out[MASK_COL[m]], f"gt_{m}"] = None
    return out


def clip_pool(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Windows -> one row per clip, mean-softmax over OBSERVED windows.

    Mean-pooling is the handover §5.2 contract; the pooling-method study on
    `data/final` found the candidates statistically tied, so this is not a
    tuning knob (see `docs/methodology/06_fusion_model.md`).
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


def score(y_true, y_pred) -> dict:
    """Accuracy + macro-F1 over the classes that occur in y_true (see
    final_unimodal.score's docstring for why `labels` is never passed to
    sklearn's average — it would charge a subset for classes it can't contain)."""
    from sklearn.metrics import accuracy_score, f1_score
    if len(y_true) == 0:
        return {"n": 0, "acc": None, "macro_f1": None, "n_classes": 0}
    return {"n": int(len(y_true)),
            "acc": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
            "n_classes": int(len(set(y_true)))}
