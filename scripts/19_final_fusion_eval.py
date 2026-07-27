"""Run the DEPLOYED fusion head on `data/final` — does fusion v1 transfer?

Fusion v1 was trained and evaluated on `data/old`, where the emotion and motion
models had training overlap with the clips. This script feeds the new collection
to the frozen deployed checkpoint (`jetson_deploy/fusion/fusion_attn.pt`) without
retraining, so the number it prints is the honest transfer result, and compares
every masking condition against the table-imposed ceiling from
`scripts/17_cue_ambiguity.py`.

No video decoding: reads the window table built by scripts/16_final_unimodal_eval.py.

    .venv/Scripts/python scripts/19_final_fusion_eval.py --context classroom
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines.common import (CTX_COLS, EMO_COLS, GES_COLS,  # noqa: E402
                                     INTENTS, MOT_COLS, OBS_COLS, PROB_COLS)
from fusion.model.model import AttentionFusion  # noqa: E402
from scripts.realworld_eval.final_common import OUT_ROOT, load_clips  # noqa: E402
from scripts.realworld_eval.final_unimodal import (WINDOWS_PARQUET,  # noqa: E402
                                                   add_targets)

CKPT = ROOT / "jetson_deploy" / "fusion" / "fusion_attn.pt"
CFG = ROOT / "jetson_deploy" / "fusion" / "fusion_config.json"
CEILING = OUT_ROOT / "ambiguity" / "ceiling_classroom.json"
OUT_DIR = OUT_ROOT / "fusion_transfer"

CUES = ["emotion", "gesture", "motion", "context"]
CUE_COLS = {"emotion": EMO_COLS, "gesture": GES_COLS,
            "motion": MOT_COLS, "context": CTX_COLS}


def build_inputs(windows: pd.DataFrame, clips: pd.DataFrame):
    """Window table -> (x[N,24], obs[N,4], y[N], meta) in the trained column order.

    The window table already uses each model's native class order, which is the
    order fusion_config.json pins, so the columns line up by name.
    """
    meta_cols = ["intent", "v3_row", "scenario_dir", "split_design", "source",
                 "view", "missing_v3"] + [f"missing_{c}" for c in CUES]
    m = clips.set_index("clip_id")[meta_cols]
    df = windows.join(m, on="clip_id")
    df = df[df.intent.notna()].copy()

    x = df[PROB_COLS].fillna(0.0).to_numpy(np.float32).copy()
    obs = df[OBS_COLS].to_numpy(np.float32).copy()

    # designed-missing overrides observation: the table says this cue is offline,
    # so it must reach the model the same way a detector failure would (obs=0 and
    # a zeroed block), which is exactly how it was trained.
    for k, cue in enumerate(CUES):
        flag = df[f"missing_{cue}"].to_numpy(bool)
        obs[flag, k] = 0.0
        idx = [PROB_COLS.index(c) for c in CUE_COLS[cue]]
        x[np.ix_(flag, idx)] = 0.0

    y = df.intent.map(INTENTS.index).to_numpy(np.int64)
    return x, obs, y, df


def predict(x, obs, mask_cues=()) -> np.ndarray:
    model = AttentionFusion(missing_mode="exclude")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    x, obs = x.copy(), obs.copy()
    for cue in mask_cues:
        k = CUES.index(cue)
        obs[:, k] = 0.0
        idx = [PROB_COLS.index(c) for c in CUE_COLS[cue]]
        x[np.ix_(np.ones(len(x), bool), idx)] = 0.0
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(obs))
    return logits.argmax(1).numpy()


def clip_vote(df: pd.DataFrame, pred: np.ndarray):
    t = df.assign(pred=pred)
    g = t.groupby("clip_id")
    truth = g.intent.first().map(INTENTS.index)
    voted = g.pred.agg(lambda s: s.value_counts().idxmax())
    return truth, voted, g.v3_row.first(), g.split_design.first(), g.source.first()


def metrics(y, p) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {"n": int(len(y)),
            "acc": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro",
                                             zero_division=0)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="classroom")
    ap.add_argument("--view", default="realsense_480p")
    args = ap.parse_args()

    clips = add_targets(load_clips())
    clips = clips[clips.v3_row.notna() & (clips.context == args.context)]
    if args.view:
        clips = clips[clips.view == args.view]
    windows = pd.read_parquet(WINDOWS_PARQUET)
    windows = windows[windows.clip_id.isin(set(clips.clip_id))]

    x, obs, y, df = build_inputs(windows, clips)
    print(f"{len(df)} windows / {df.clip_id.nunique()} clips")
    print(f"observed rates: {dict(zip(CUES, obs.mean(0).round(3)))}")

    ceil = json.loads(CEILING.read_text())["recorded"] if CEILING.exists() else {}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "checkpoint": str(CKPT.relative_to(ROOT)),
           "config": json.loads(CFG.read_text())["config"],
           "context": args.context, "view": args.view, "conditions": {}}

    rows = []
    for name, mask in [("none", ()), ("emotion", ("emotion",)),
                       ("gesture", ("gesture",)), ("motion", ("motion",)),
                       ("context", ("context",)),
                       ("emotion+gesture", ("emotion", "gesture"))]:
        pred = predict(x, obs, mask)
        truth, voted, v3, split, source = clip_vote(df, pred)
        cond = {"overall": metrics(truth, voted)}
        for s in ("train", "test"):
            sel = split == s
            if sel.any():
                cond[s] = metrics(truth[sel], voted[sel])
                c = ceil.get(s, {}).get("conditions", {}).get(name, {})
                cond[s]["ceiling"] = c.get("ceiling")
        # held-out clips only (no emotion/motion training overlap)
        sel = source == "raw_take_20260725"
        if sel.any():
            cond["held_out_only"] = metrics(truth[sel], voted[sel])
        out["conditions"][name] = cond
        rows.append((name, cond))
        print(f"  mask {name:16s} overall={cond['overall']['acc']:.3f}  "
              f"train={cond.get('train', {}).get('acc')} "
              f"(ceil {cond.get('train', {}).get('ceiling')})  "
              f"test={cond.get('test', {}).get('acc')} "
              f"(ceil {cond.get('test', {}).get('ceiling')})  "
              f"held-out={cond.get('held_out_only', {}).get('acc')}")

    # per-row breakdown, unmasked
    pred = predict(x, obs)
    truth, voted, v3, split, source = clip_vote(df, pred)
    per_row = (pd.DataFrame({"v3_row": v3.astype(int), "split": split,
                             "source": source, "true": truth, "pred": voted})
                 .assign(correct=lambda d: d.true == d.pred)
                 .groupby(["v3_row", "split", "source"])
                 .agg(n=("correct", "size"), acc=("correct", "mean"),
                      true_intent=("true", lambda s: INTENTS[s.iloc[0]]),
                      top_pred=("pred", lambda s: INTENTS[s.value_counts().index[0]]))
                 .round(3).reset_index().sort_values("acc"))
    out["per_v3_row"] = per_row.to_dict("records")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"transfer_{args.context}.json").write_text(json.dumps(out, indent=2))
    per_row.to_csv(OUT_DIR / f"per_row_{args.context}.csv", index=False)

    L = [f"# Fusion v1 transfer to `data/final` — {args.context}", "",
         f"Generated {out['generated']} · frozen checkpoint "
         f"`{out['checkpoint']}` · **no retraining** · view {args.view}", "",
         "Fusion v1 reported 0.939 clip accuracy on `data/old`. The emotion and "
         "motion models had training overlap with those clips, so the "
         "`held-out` column below — the 2026-07-25 takes only — is the number "
         "that estimates real generalisation.", "",
         "| Cue(s) masked | overall | train-design | ceiling | test-design | "
         "ceiling | held-out only |", "|---|---|---|---|---|---|---|"]
    for name, c in rows:
        L.append(f"| {name} | {c['overall']['acc']} | "
                 f"{c.get('train', {}).get('acc')} | "
                 f"{c.get('train', {}).get('ceiling')} | "
                 f"{c.get('test', {}).get('acc')} | "
                 f"{c.get('test', {}).get('ceiling')} | "
                 f"{c.get('held_out_only', {}).get('acc')} |")
    L += ["", "## Per V3 row (no masking beyond the table's own)", "",
          per_row.to_markdown(index=False)]
    (OUT_DIR / f"TRANSFER_{args.context}.md").write_text("\n".join(L), encoding="utf8")
    print(f"\n-> {OUT_DIR / f'TRANSFER_{args.context}.md'}")


if __name__ == "__main__":
    main()
