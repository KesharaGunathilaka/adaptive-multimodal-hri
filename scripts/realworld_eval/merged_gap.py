"""Shared building blocks for gap-decomposition-style experiments on
`data/final_merged` — factored out of `scripts/29_merged_gap_decomposition.py`
so `scripts/30_merged_recombination.py` (and any future re-run of the
decomposition) can reuse them instead of re-deriving.

Everything here operates on CLIP-LEVEL tables in the shape
`fusion.baselines.common` expects (`PROB_COLS` + `OBS_COLS` + `y`), built by
mean-pooling `data/final_merged/features/unimodal_windows.parquet` per clip
(the clip-level aggregation adopted 2026-07-28, see
`docs/methodology/07_evaluation.md` §7.2) or by one-hotting the V3 table's
ground truth ("oracle" cues, a diagnostic never available at deployment).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.baselines.rule_based import rule_intent  # noqa: E402
from scripts.realworld_eval.merged_unimodal import (  # noqa: E402
    CONTEXT_LABELS, EMOTION_LABELS, GESTURE_LABELS, MOTION_LABELS,
    WINDOWS_PARQUET, load_clips, obs_col, prob_cols)

OUT_DIR = ROOT / "results" / "realworld_eval_merged"
LABELS = {"emo": EMOTION_LABELS, "ges": GESTURE_LABELS,
          "mot": MOTION_LABELS, "ctx": CONTEXT_LABELS}
GT_COL = {"emo": "gt_emotion", "ges": "gt_gesture", "mot": "gt_motion",
         "ctx": "gt_context"}
MASK_COL = {"emo": "emotion_masked", "ges": "gesture_masked",
           "mot": "motion_masked", "ctx": "context_masked"}
FULL_NAME = {"emo": "emotion", "ges": "gesture", "mot": "motion", "ctx": "context"}


def load_clips_and_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    clips = load_clips()
    clips = clips[clips.v3_row.notna()].copy()
    windows = pd.read_parquet(WINDOWS_PARQUET)
    windows = windows[windows.clip_id.isin(set(clips.clip_id))]
    return clips, windows


# ── ceiling ──────────────────────────────────────────────────────────────────
def check_ceiling(clips: pd.DataFrame) -> dict:
    tup_cols = ["context", "emotion_v3", "gesture_v3", "motion_v3"]
    scen = clips.drop_duplicates("v3_row")[tup_cols + ["v3_row", "intent"]]
    g = scen.groupby(tup_cols).intent.nunique()
    weighted = clips.groupby(tup_cols).intent.agg(lambda s: s.value_counts().iloc[0])
    total = clips.groupby(tup_cols).size()
    return {"n_colliding_tuples": int((g > 1).sum()),
            "clip_weighted_ceiling": round(float(weighted.sum() / total.sum()), 4),
            "n_clips": int(total.sum())}


# ── REAL / ORACLE clip-level tables ─────────────────────────────────────────
def build_real(clips: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Clip-level mean-softmax vectors from the deployed models' predictions."""
    out = clips[["clip_id", "v3_row", "context", "intent", "split",
                 "headline_eval", "source"]].copy()
    for m in ("emo", "ges", "mot", "ctx"):
        cols = prob_cols(FULL_NAME[m])
        obs = obs_col(FULL_NAME[m])
        fired = windows[windows[obs]]
        g = fired.groupby("clip_id")[cols].mean()
        out = out.merge(g, left_on="clip_id", right_index=True, how="left")
        out[f"{m}_obs"] = out.clip_id.isin(g.index).astype(float)
        out[cols] = out[cols].fillna(0.0)
    return out


def build_oracle(clips: pd.DataFrame) -> pd.DataFrame:
    """Clip-level ONE-HOT vectors from the V3 table's true labels.

    `out` and `clips` share row order/length, so one-hot columns are filled by
    row POSITION via a class-index lookup (an earlier per-row clip_id join was
    O(n^2) over 2,869 rows).
    """
    out = clips[["clip_id", "v3_row", "context", "intent", "split",
                 "headline_eval", "source"]].copy()
    for m in ("emo", "ges", "mot", "ctx"):
        names = LABELS[m]
        idx_of = {name: i for i, name in enumerate(names)}
        cols = [f"{m}_{c}" for c in names]
        onehot = np.zeros((len(clips), len(names)), dtype=np.float32)
        masked = clips[MASK_COL[m]].to_numpy()
        gt = clips[GT_COL[m]].to_numpy()
        for i, (label, is_masked) in enumerate(zip(gt, masked)):
            if is_masked or label is None or (isinstance(label, float) and pd.isna(label)):
                continue
            if label in idx_of:
                onehot[i, idx_of[label]] = 1.0
        out[cols] = onehot
        out[f"{m}_obs"] = (~masked).astype(float)
    return out


def to_common_schema(tbl: pd.DataFrame) -> pd.DataFrame:
    out = tbl.copy()
    out["y"] = out.intent.map(common.INTENTS.index)
    return out


def eval_clip(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {"n": int(len(y_true)),
            "acc": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             labels=list(range(10)),
                                             zero_division=0)), 4)}


def agg(runs: list[dict]) -> dict:
    return {"acc_mean": round(float(np.mean([r["acc"] for r in runs])), 4),
            "acc_std": round(float(np.std([r["acc"] for r in runs])), 4),
            "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in runs])), 4),
            "macro_f1_std": round(float(np.std([r["macro_f1"] for r in runs])), 4),
            "n": runs[0]["n"]}


# ── rule-based ───────────────────────────────────────────────────────────────
# rule_intent() takes 4 plain labels and has no notion of "missing". An
# unhandled missing cue would show up as an all-zero one-hot/mean row; argmax
# of an all-zero vector silently returns class index 0 (e.g. "Surprise" for
# emotion) -- an arbitrary substitution, not a reasoned default. Missing cues
# instead fall back to the V3 table's own stated safe fallback (Final_Dataset
# §2.6): emotion=Neutral, gesture=idle, motion=standing. Context is the
# exception -- `context_masked` means the ROOM-IDENTITY SENSOR is offline, not
# that a rule system with prior knowledge (e.g. a fixed installation) doesn't
# know the room, so the rule path always uses the clip's true `context`
# column; the fusion model has no such prior and must mask context like any
# other cue via `ctx_obs`.
DEFAULT = {"emo": "Neutral", "ges": "idle", "mot": "standing"}


def rule_predict(tbl: pd.DataFrame) -> np.ndarray:
    def pick(pref):
        cols = [f"{pref}_{c}" for c in LABELS[pref]]
        probs = tbl[cols].to_numpy()
        obs = tbl[f"{pref}_obs"].to_numpy().astype(bool)
        labels = np.array(LABELS[pref])[probs.argmax(1)]
        return np.where(obs, labels, DEFAULT[pref])

    emo, ges, mot = pick("emo"), pick("ges"), pick("mot")
    ctx = tbl["context"].to_numpy()
    preds = []
    for e, g, m, c in zip(emo, ges, mot, ctx):
        e = "anger" if e == "Anger" else e.lower()
        intent = rule_intent(e, g, m, c)
        # F09 was folded into F01 when the table was merged (docs/DECISIONS.md
        # 2026-08-03) -- rule_based.py still predicts it (data/old needs that
        # branch), so remap here rather than edit the shared function.
        if intent == "F09":
            intent = "F01"
        preds.append(common.INTENTS.index(intent) if intent in common.INTENTS
                     else common.INTENTS.index("F05"))
    return np.array(preds)
