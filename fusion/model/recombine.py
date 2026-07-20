"""Cue recombination: synthesize training windows for V3 train rows that were
never recorded (incl. all F10 rows) by combining REAL cue vectors from
different windows (handover §8.2, Final_Dataset §1).

Per synthetic row spec we sample, independently per cue, a real probability
vector from a pool of train-subject windows whose argmax matches the intended
cue class ('none' -> gesture idle; [MISSING] -> obs=0/zeros). Labels follow
the V3 table. S28's recombination_pool windows ARE valid pool sources (their
cue vectors are real observations; only their intent label is disputed).

Skipped: V3 #18 (classroom happy+wave+walk F09) — without the direction cue it
is cue-identical to recorded #1/S02 (F01); synthesizing it would inject
irreducible label noise. Logged in DECISIONS.md.
"""
import numpy as np

from ..baselines.common import (CTX_COLS, EMO_COLS, GES_COLS, INTENTS,
                                MOT_COLS, OBS_COLS, PROB_COLS)

# (v3_row, context, emotion, gesture, motion, missing_cues, intent)
# 'none' gesture = idle class; None cue value = [MISSING]
SYNTH_ROWS = [
    (4,  "classroom", "fear",    "both_hands_up", "stepping_back", (),           "F02"),
    (6,  "classroom", "neutral", "beckoning",     "sitting",       ("context",), "F03"),
    (9,  "classroom", "sad",     None,            "sitting",       ("gesture",), "F04"),
    (13, "classroom", "anger",   "wave",          "standing",      (),           "F06"),
    (15, "classroom", "anger",   "thumbs_down",   "sitting",       (),           "F07"),
    (17, "classroom", "disgust", "thumbs_down",   "sitting",       (),           "F08"),
    (19, "classroom", "neutral", "wave",          "walking",       (),           "F09"),
    (20, "classroom", "sad",     "idle",          "sitting",       (),           "F10"),
    (21, "classroom", "sad",     "idle",          "standing",      (),           "F10"),
    (33, "kitchen",   "happy",   "thumbs_down",   "standing",      (),           "F01"),
    (39, "kitchen",   "sad",     "beckoning",     "standing",      (),           "F04"),
    (41, "kitchen",   "neutral", "idle",          "standing",      (),           "F05"),
    (42, "kitchen",   "disgust", "point",         "stepping_back", (),           "F06"),  # S27 curated out
    (43, "kitchen",   "neutral", None,            "walking",       ("gesture",), "F06"),
    (45, "kitchen",   "anger",   "point",         "standing",      (),           "F07"),
    (47, "kitchen",   "disgust", "thumbs_down",   "standing",      (),           "F08"),
    (49, "kitchen",   None,      "wave",          "walking",       ("emotion",), "F09"),
    (50, "kitchen",   "sad",     "idle",          "sitting",       (),           "F10"),
    (51, "kitchen",   "sad",     "idle",          "standing",      (),           "F10"),
]

_CUE_COLS = {"emotion": EMO_COLS, "gesture": GES_COLS,
             "motion": MOT_COLS, "context": CTX_COLS}
_CUE_OBS = {"emotion": 0, "gesture": 1, "motion": 2, "context": 3}


def _class_of(col):
    return col.split("_", 1)[1].lower()


def build_pools(df):
    """Per (cue, class): array of real probability vectors from train subjects.
    Includes recombination_pool rows (S28) — sources, not labels."""
    src = df[df.split_subject == "train"]
    pools = {}
    for cue, cols in _CUE_COLS.items():
        obs = src[f"{cue[:3]}_obs"].to_numpy(bool) if False else \
            src[OBS_COLS[_CUE_OBS[cue]]].to_numpy(bool)
        vals = src[cols].fillna(0.0).to_numpy(np.float32)[obs]
        am = vals.argmax(1)
        for i, col in enumerate(cols):
            pool = vals[am == i]
            if len(pool):
                pools[(cue, _class_of(col))] = pool
    return pools


def generate(df, n_per_row=400, seed=0):
    """-> (X [N,24], obs [N,4], y [N]) synthetic samples, plus a skip report."""
    rng = np.random.default_rng(seed)
    pools = build_pools(df)
    X, OBS, Y, report = [], [], [], []
    for row in SYNTH_ROWS:
        v3, ctx, emo, ges, mot, missing, intent = row
        spec = {"emotion": emo, "gesture": ges, "motion": mot, "context": ctx}
        needed = {c: v for c, v in spec.items() if v is not None}
        absent = [c for c, v in needed.items() if (c, v) not in pools]
        if absent:
            report.append(f"row #{v3}: no pool for {absent} — skipped")
            continue
        x = np.zeros((n_per_row, 24), np.float32)
        obs = np.ones((n_per_row, 4), np.float32)
        col_off = {"emotion": 0, "gesture": 7, "motion": 15, "context": 19}
        for cue, cols in _CUE_COLS.items():
            k = _CUE_OBS[cue]
            if spec[cue] is None or cue in missing:
                obs[:, k] = 0.0
                continue
            pool = pools[(cue, spec[cue])]
            pick = rng.integers(0, len(pool), n_per_row)
            x[:, col_off[cue]:col_off[cue] + len(cols)] = pool[pick]
        X.append(x)
        OBS.append(obs)
        Y.append(np.full(n_per_row, INTENTS.index(intent), np.int64))
        report.append(f"row #{v3}: {n_per_row} samples ({intent})")
    return np.concatenate(X), np.concatenate(OBS), np.concatenate(Y), report
