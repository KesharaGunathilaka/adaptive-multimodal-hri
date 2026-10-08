"""PHASE 2 — the perception-aware intent taxonomy (v4) and its go/no-go gate.

Phase 1 (`results/fusion_v2/CUE_RELIABILITY.md`) measured, on held-out footage:

  gesture  0.999 design / 0.864 test   -> most reliable discriminator
  context  0.972-0.995                 -> reliable, binary modifier
  emotion  0.752 val / 0.762 test      -> moderate; top confusions
                                          Neutral->Sad .146, Disgust->Anger .125,
                                          Fear<->Surprise (test)
  motion   0.435                       -> UNUSABLE: standing->walking .44,
                                          stepping_back recall .149; even the
                                          static/approach/retreat merge is 0.507

So v4 keys intents on GESTURE first, EMOTION SUPERCLASS second, CONTEXT as a
modifier, and **excludes motion entirely** (plan P1). The two emotion pairs that
carry boundaries in v3 are merged away (plan P2): `hostile={Anger,Disgust}` and
`alarm={Fear,Surprise}`.

The v3 rubric put boundaries on all three top confusions:
  * `thumbs_up`/`thumbs_down`: Anger->F07 vs Disgust->F08   (Disgust->Anger .125)
  * `idle`: Sad->F10 vs Neutral->F05/F06                    (Neutral->Sad .146)
  * `fear`->F02 regardless                                   (Fear->Surprise, test)
v4 splits F07/F08 on gesture POLARITY instead of emotion polarity -- the reliable
cue does the work.

Gate (decided on VAL, which is held out from cue fine-tuning; test is reported for
information only and is NOT used to iterate the design):
    projected measured-cue ceiling >= 0.90  AND  rules-on-measured-cues >= 0.80

    .venv/Scripts/python scripts/91_taxonomy_v4.py
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.model import MODALITIES  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402

OUT = ROOT / "results" / "fusion_v2" / "TAXONOMY_V4.md"
OUTJSON = ROOT / "results" / "fusion_v2" / "taxonomy_v4.json"
PREFIX = {"emotion": "emo", "gesture": "ges", "motion": "mot", "context": "ctx"}
CLS = {"emotion": ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
       "gesture": ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
                   "raise_hand", "both_hands_up"],
       "context": ["classroom", "kitchen"]}
EMO_SUPER = {"Fear": "alarm", "Surprise": "alarm", "Anger": "hostile",
             "Disgust": "hostile", "Happy": "positive", "Sad": "low",
             "Neutral": "neutral"}


def rule_intent_v4(emo_super: str, ges: str, ctx: str) -> str:
    """v4 rubric: gesture-first, emotion-superclass second, context modifier.
    Motion is deliberately absent -- it is 0.435 accurate and carries almost no
    intent signal (per-cue oracle ROI measured its value at -0.013)."""
    if emo_super == "alarm":
        return "F02"                                   # emergency, any gesture
    if ges == "both_hands_up":
        return "F07" if emo_super == "hostile" else "F05"
    if ges == "beckoning":
        return "F04" if emo_super == "low" else "F03"
    if ges == "raise_hand":                            # the T04 context flip
        return "F04" if ctx == "classroom" else "F01"
    if ges == "wave":
        return "F06" if emo_super == "hostile" else "F01"
    if ges == "point":
        return "F06" if emo_super == "hostile" else "F03"
    if ges == "thumbs_up":                             # polarity split on GESTURE
        return "F07" if emo_super == "hostile" else "F01"
    if ges == "thumbs_down":
        return "F08" if emo_super == "hostile" else "F04"
    if emo_super == "low":                             # ges == idle
        return "F10"
    if emo_super == "hostile":
        return "F07"
    return "F05"


INTENTS_V4 = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
ACTION_OF = {"F01": "A01 greet/acknowledge", "F02": "A02 emergency response",
             "F03": "A04 go to task", "F04": "A05 offer help",
             "F05": "A06 do not interrupt", "F06": "A11 yield/move aside",
             "F07": "A08 de-escalate", "F08": "A07 offer break",
             "F10": "A12 encourage/re-engage"}


def measured_cues(frame):
    out = {}
    for m in ("emotion", "gesture", "context"):
        p = PREFIX[m]
        cols = [f"{p}_{c}" for c in CLS[m] if f"{p}_{c}" in frame.columns]
        have = [c.split("_", 1)[1] for c in cols]
        out[m] = np.array([have[i] for i in frame[cols].fillna(0).to_numpy().argmax(1)])
    return out


def main() -> None:
    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    idx = clips.set_index("clip_id")

    # ---- v4 ground-truth labels from ANNOTATED cues ----
    gtc = {m: idx.reindex(real.clip_id.to_numpy())[f"gt_{m}"].to_numpy()
           for m in ("emotion", "gesture", "context")}
    ok = np.ones(len(real), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok &= pd.notna(gtc[m]) & ~mk
    y4 = np.array([rule_intent_v4(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                   if o else None
                   for e, g, c, o in zip(gtc["emotion"], gtc["gesture"],
                                         gtc["context"], ok)], dtype=object)
    real = real.assign(y4=y4, y4_ok=ok)

    mc = measured_cues(real)
    pred_rules = np.array([rule_intent_v4(EMO_SUPER.get(e, "neutral"), g, c)
                           for e, g, c in zip(mc["emotion"], mc["gesture"], mc["context"])])
    mtuple = list(zip([EMO_SUPER.get(e, "neutral") for e in mc["emotion"]],
                      mc["gesture"], mc["context"]))

    L = ["# Phase 2 — perception-aware taxonomy v4 and gate", "",
         "Keyed on **gesture** (most reliable), **emotion superclass** "
         "(`alarm={Fear,Surprise}`, `hostile={Anger,Disgust}`, `positive`, `low`, "
         "`neutral`) and **context**. **Motion is excluded entirely** — 0.435 "
         "accurate, and its measured value to intent is −0.013.", "",
         "## Intents and actions (all actionably distinct — plan P4)", "",
         "| intent | robot action |", "|---|---|"]
    for i in INTENTS_V4:
        L.append(f"| {i} | {ACTION_OF[i]} |")
    L += ["", "## Gate", "",
          "| split | n | v4 ceiling (measured cues) | v4 rules (measured cues) | "
          "v3 rules (reference) |", "|---|---|---|---|---|"]

    payload, gate = {}, {}
    for split in ("val", "test"):
        sel = (real.split == split).to_numpy() & real.y4_ok.to_numpy()
        if sel.sum() == 0:
            continue
        yy = real.y4.to_numpy()[sel]
        # measured-cue ceiling: best achievable given the measured tuples
        g = defaultdict(Counter)
        for t, lab in zip([mtuple[i] for i in np.flatnonzero(sel)], yy):
            g[t][lab] += 1
        tot = sum(sum(c.values()) for c in g.values())
        ceil = sum(max(c.values()) for c in g.values()) / tot
        racc = float((pred_rules[sel] == yy).mean())
        v3r = float((G.rule_predict(real[sel]) ==
                     real[sel].y.to_numpy()).mean())
        L.append(f"| {split} | {int(sel.sum())} | **{ceil:.4f}** | **{racc:.4f}** | "
                 f"{v3r:.4f} |")
        payload[split] = dict(n=int(sel.sum()), ceiling=ceil, rules=racc, v3_rules=v3r)
        gate[split] = (ceil, racc)
        print(f"{split}: n={int(sel.sum())} ceiling={ceil:.4f} rules_v4={racc:.4f} "
              f"(v3 rules {v3r:.4f})", flush=True)

    cv, rv = gate.get("val", (0, 0))
    passed = cv >= 0.90 and rv >= 0.80
    L += ["", f"**Gate decided on val: ceiling {cv:.4f} (need ≥0.90), rules {rv:.4f} "
          f"(need ≥0.80) → {'PASS' if passed else 'FAIL'}**", "",
          "Test is shown for information only; the design was not iterated against it.",
          ""]

    # class balance + per-class rules recall on val
    sel = (real.split == "val").to_numpy() & real.y4_ok.to_numpy()
    yy = real.y4.to_numpy()[sel]; pp = pred_rules[sel]
    L += ["## v4 class balance and rule recall (val)", "",
          "| intent | n | rules recall |", "|---|---|---|"]
    for i in INTENTS_V4:
        n = int((yy == i).sum())
        L.append(f"| {i} | {n} | "
                 f"{(pp[yy == i] == i).mean():.3f} |" if n else f"| {i} | 0 | — |")
    L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    OUTJSON.write_text(json.dumps(dict(gate=payload, passed=bool(passed),
                                       intents=INTENTS_V4), indent=2))
    print(f"\nGATE {'PASS' if passed else 'FAIL'}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
