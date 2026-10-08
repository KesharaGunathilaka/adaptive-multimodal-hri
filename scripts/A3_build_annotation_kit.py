"""WP3a - build the human annotation kit.

Purpose: establish that the recorded footage actually conveys the intent the data
table specifies, and produce an INDEPENDENT ground truth against which fusion and
rules can be compared without the rule's built-in advantage.

Design decisions that make the study valid:

  * Clips are copied under ANONYMOUS ids (`clip_001.mp4` ...). Nothing in the
    filename reveals the scenario, the intent or the cue signature.
  * The sample is BALANCED across the nine intents, so inter-annotator agreement
    is not inflated by one dominant class.
  * Presentation order is SHUFFLED, so annotators cannot infer that consecutive
    clips share a scenario.
  * The instruction sheet describes each intent by its MEANING - what the person
    wants from the robot - and deliberately never mentions cues, emotions or
    gestures. If we told annotators "angry + thumbs up = frustration" we would
    only be testing whether humans can apply our rubric, which proves nothing.
  * An `unclear` option is offered so annotators are not forced to guess; the
    rate of `unclear` is itself a measure of how legible the footage is.

    .venv/Scripts/python scripts/A3_build_annotation_kit.py --n-clips 200
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402

VIDEO_BASE = ROOT / "data" / "final_merged" / "raw" / "clips"
KIT = ROOT / "results" / "fusion_v2" / "annotation_kit"
EMO_SUPER = {"Fear": "alarm", "Surprise": "alarm", "Anger": "hostile",
             "Disgust": "hostile", "Happy": "positive", "Sad": "low",
             "Neutral": "neutral"}
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]

# Plain-language definitions. NOTE: intentionally phrased in terms of what the
# person WANTS, never in terms of the cues that produce it.
INTENT_DESC = {
    "F01": ("Greeting or positive acknowledgement",
            "The person is greeting the robot, saying hello or goodbye, or "
            "signalling approval / that something went well."),
    "F02": ("Emergency or danger",
            "Something is wrong or unsafe - the person is alarmed, startled, "
            "panicking, or recoiling from a hazard. The robot should treat this "
            "as urgent."),
    "F03": ("Asking the robot to do a task",
            "The person wants the robot to come over, fetch something, or attend "
            "to an object or place. A request directed outward, at a task."),
    "F04": ("Asking for help",
            "The person is struggling or in difficulty and wants assistance for "
            "THEMSELVES. Distress or being stuck, rather than delegating a task."),
    "F05": ("Busy - does not want to be interrupted",
            "The person is absorbed in their own activity, or dealing with "
            "someone else. The robot should stay out of the way."),
    "F06": ("Wants the robot to move / give space",
            "The person needs to get past, or is waving the robot away. About "
            "physical space, not about a task."),
    "F07": ("Frustration or agitation",
            "The person is annoyed, irritated or venting - but nothing is "
            "physically dangerous. The robot should calm things down."),
    "F08": ("Wants a break or relief",
            "The person is fed up with, or repulsed by, the current activity and "
            "wants it to stop or change."),
    "F10": ("Discouraged / has given up",
            "The person has disengaged - low energy, defeated, no longer trying. "
            "They are not asking for anything directly."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-copy", action="store_true",
                    help="write the manifest and forms but do not copy video files")
    args = ap.parse_args()

    splits = pd.read_csv(ROOT / "data/final_merged/annotations/splits.csv")
    # `clips.csv` alone carries only gt_emotion; the other gt_* columns are
    # derived by the shared loader, so use it rather than the raw csv.
    clips, _ = G.load_clips_and_windows()
    idx = clips.set_index("clip_id")
    te = splits[splits.split == "test"].copy()

    # v4 label per clip, from the designed cue signature
    sys.path.insert(0, str(ROOT / "scripts"))
    gt = {m: idx.reindex(te.clip_id.to_numpy())[f"gt_{m}"].to_numpy()
          for m in ("emotion", "gesture", "context")}
    ok = np.ones(len(te), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(te.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok &= pd.notna(gt[m]) & ~mk

    def v4(e, g, c):
        s = EMO_SUPER.get(str(e), "neutral")
        if s == "alarm":
            return "F02"
        if g == "both_hands_up":
            return "F07" if s == "hostile" else "F05"
        if g == "beckoning":
            return "F04" if s == "low" else "F03"
        if g == "raise_hand":
            return "F04" if c == "classroom" else "F01"
        if g == "wave":
            return "F06" if s == "hostile" else "F01"
        if g == "point":
            return "F06" if s == "hostile" else "F03"
        if g == "thumbs_up":
            return "F07" if s == "hostile" else "F01"
        if g == "thumbs_down":
            return "F08" if s == "hostile" else "F04"
        if s == "low":
            return "F10"
        if s == "hostile":
            return "F07"
        return "F05"

    te = te[ok].reset_index(drop=True)
    te["v4_intent"] = [v4(e, g, c) for e, g, c in
                       zip(gt["emotion"][ok], gt["gesture"][ok], gt["context"][ok])]

    # balanced sample across intents, then shuffle
    rng = np.random.default_rng(args.seed)
    per = max(1, args.n_clips // len(INTENTS))
    picks = []
    for it in INTENTS:
        sub = te[te.v4_intent == it]
        if len(sub) == 0:
            continue
        take = min(per, len(sub))
        picks.append(sub.sample(take, random_state=args.seed))
    sample = pd.concat(picks, ignore_index=True)
    if len(sample) < args.n_clips:                     # top up from the remainder
        rest = te[~te.clip_id.isin(sample.clip_id)]
        if len(rest):
            sample = pd.concat([sample, rest.sample(
                min(args.n_clips - len(sample), len(rest)),
                random_state=args.seed)], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample["anon_id"] = [f"clip_{i+1:03d}" for i in range(len(sample))]

    KIT.mkdir(parents=True, exist_ok=True)
    vids = KIT / "videos"
    vids.mkdir(exist_ok=True)
    n_copied, missing = 0, []
    for r in sample.itertuples():
        src = VIDEO_BASE / r.filepath
        if not src.exists():
            missing.append(r.clip_id); continue
        if not args.no_copy:
            shutil.copy2(src, vids / f"{r.anon_id}.mp4")
        n_copied += 1

    # KEY: this file maps anonymous ids back to the truth. Annotators must NOT see it.
    sample[["anon_id", "clip_id", "v3_row", "scenario_dir", "v4_intent",
            "person_id"]].to_csv(KIT / "KEY_do_not_share.csv", index=False)

    # blank response form - one per annotator
    for who in ("annotator_A", "annotator_B", "annotator_C"):
        with open(KIT / f"responses_{who}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["anon_id", "intent", "confidence_1to5", "notes"])
            for a in sample.anon_id:
                w.writerow([a, "", "", ""])

    lines = [
        "# Intent annotation - instructions", "",
        f"You will watch {len(sample)} short video clips. In each one a person is",
        "interacting with (or ignoring) a robot in a room.", "",
        "**Your task:** for each clip, decide what the person most likely WANTS",
        "from the robot, and write the matching code in `responses_<you>.csv`.", "",
        "Please:", "",
        "- Watch each clip once, at normal speed. Trust your first impression.",
        "- Judge only what you SEE. There is no audio and no context beyond the clip.",
        "- Work through them in the order given; do not go back and revise.",
        "- Do not discuss clips with the other annotators until everyone is finished.",
        "- If you genuinely cannot tell, write `unclear` rather than guessing.",
        "- Also give a confidence from 1 (guessing) to 5 (certain).", "",
        "## The intent codes", "",
        "| code | short name | what it means |", "|---|---|---|",
    ]
    for c in INTENTS:
        short, desc = INTENT_DESC[c]
        lines.append(f"| **{c}** | {short} | {desc} |")
    lines += [
        "| **unclear** | Cannot tell | The clip does not let you decide. |", "",
        "## Distinctions people find tricky", "",
        "- **F03 vs F04** - both are requests. F03 asks the robot to do something",
        "  about a TASK or object; F04 asks for help for the PERSON themselves.",
        "- **F05 vs F10** - both mean 'not asking for anything'. F05 is busy and",
        "  engaged; F10 is defeated and has given up.",
        "- **F06 vs F03** - F06 is about physical space (move aside); F03 is about",
        "  a task (come and do this).",
        "- **F07 vs F08** - F07 is annoyance at the situation; F08 is wanting the",
        "  current activity to stop.",
        "- **F02** is only for genuine urgency or danger, not ordinary annoyance.",
        "", "## Files", "",
        "- `videos/` - the clips, named `clip_001.mp4` and so on.",
        f"- `responses_annotator_A.csv` - your answer sheet (use YOUR letter).", "",
        "Thank you - this takes about 60-90 minutes.", "",
    ]
    (KIT / "INSTRUCTIONS.md").write_text(chr(10).join(lines), encoding="utf-8")

    dist = sample.v4_intent.value_counts().sort_index()
    print(f"sampled {len(sample)} clips, {n_copied} videos "
          f"{'located' if args.no_copy else 'copied'}")
    if missing:
        print(f"  WARNING: {len(missing)} clips had no video file")
    print("\nintent balance:")
    print(dist.to_string())
    print(f"\nkit -> {KIT}")
    print("  INSTRUCTIONS.md, videos/, responses_annotator_{A,B,C}.csv")
    print("  KEY_do_not_share.csv  <-- keep this away from the annotators")


if __name__ == "__main__":
    main()
