"""Rule-based intent baseline: hand-coded if/else over argmax cue labels,
mirroring the Final_Dataset V3 §2.6 labeling rubric as closely as possible
WITHOUT the direction cue (no model outputs it — DECISIONS.md 2026-07-16).

Where the rubric needs direction (wave toward robot=F01 vs toward exit=F09;
point at object=F03 vs at robot=F06), the rule takes the more common reading
and that ambiguity becomes measurable failure — which is the point of this
baseline (handover §7.2: its failures on conflict/missing rows are a thesis
result, not a bug).
"""
import numpy as np

from .common import CTX_COLS, EMO_COLS, GES_COLS, INTENTS, MOT_COLS

EMO = [c.split("_", 1)[1].lower() for c in EMO_COLS]     # surprise..neutral
GES = [c.split("_", 1)[1] for c in GES_COLS]             # idle..both_hands_up
MOT = [c.split("_", 1)[1] for c in MOT_COLS]
CTX = [c.split("_", 1)[1] for c in CTX_COLS]


def rule_intent(emo, ges, mot, ctx):
    """Argmax cue labels -> intent code (V3 §2.6 rubric, direction-free)."""
    emo = "anger" if emo == "anger" else emo

    if ges == "both_hands_up":
        if emo in ("fear", "surprise"):
            return "F02"                       # alarm / hazard recoil
        if emo == "anger":
            return "F07"                       # venting, stationary
        return "F05"                           # neutral stretch (V3 #26)
    if emo == "fear":
        return "F02"                           # fear escalates regardless
    if ges == "beckoning":
        return "F04" if emo == "sad" else "F03"
    if ges == "raise_hand":
        if ctx == "classroom":
            return "F05" if emo == "happy" else "F04"
        return "F01"                           # kitchen social (V3 #52)
    if ges == "wave":
        if emo == "anger":
            return "F06"                       # dismissal wave
        # direction unavailable: toward-robot greet vs toward-exit farewell
        return "F09" if mot == "walking" and emo != "happy" else "F01"
    if ges == "point":
        if emo == "anger":
            return "F07" if mot != "walking" else "F06"
        if emo == "disgust":
            return "F06"                       # push-aside (V3 #42)
        if mot == "walking":
            return "F03"                       # point at object en route
        return "F05"                           # task-embedded point
    if ges == "thumbs_up":
        if emo == "anger":
            return "F07"                       # sarcasm
        if emo == "disgust":
            return "F08"                       # sarcasm
        return "F01"
    if ges == "thumbs_down":
        if emo == "anger":
            return "F07"
        if emo == "disgust":
            return "F08"
        if emo == "happy":
            return "F01"                       # playful
        return "F04"                           # sad/neutral distress
    # ges == idle (none)
    if emo == "sad":
        return "F10"                           # disengagement
    if mot == "walking":
        return "F06"                           # approach, hands empty/occupied
    return "F05"


def predict(frame):
    emo = np.array(EMO)[frame[EMO_COLS].fillna(0).to_numpy().argmax(1)]
    ges = np.array(GES)[frame[GES_COLS].fillna(0).to_numpy().argmax(1)]
    mot = np.array(MOT)[frame[MOT_COLS].fillna(0).to_numpy().argmax(1)]
    ctx = np.array(CTX)[frame[CTX_COLS].fillna(0).to_numpy().argmax(1)]
    return np.array([INTENTS.index(rule_intent(e, g, m, c))
                     for e, g, m, c in zip(emo, ges, mot, ctx)])
