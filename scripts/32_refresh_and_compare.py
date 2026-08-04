"""Refresh `unimodal_windows.parquet` with a NEW checkpoint and report a clean
before/after comparison against the currently-deployed checkpoint's numbers.

Does NOT touch the deployed checkpoint or the deployed parquet -- writes to a
separate `unimodal_windows_candidate.parquet` and a separate report, so this
can be run freely and compared before any promotion decision.

    .venv/Scripts/python scripts/32_refresh_and_compare.py --motion-ckpt modalities/motion/checkpoints/best_model_finetuned_merged.pt
    .venv/Scripts/python scripts/32_refresh_and_compare.py --gesture-ckpt modalities/gesture/checkpoints/best_TCN_finetuned_merged.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval.merged_unimodal import (  # noqa: E402
    LABELS, MODALITIES, PERFRAME_DIR, PROB_PREFIX, UNI_DIR, load_clips)

OUT_DIR = ROOT / "results" / "realworld_eval_merged"
BASELINE_JSON = UNI_DIR / "unimodal_final_merged.json"


def build_windows(clips, fz):
    rows, missing = [], []
    t0 = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        npz_path = PERFRAME_DIR / f"{r.clip_id}.npz"
        if not npz_path.exists():
            missing.append(r.clip_id)
            continue
        npz = np.load(npz_path)
        for w in fz.featurize_clip(npz):
            row = {"clip_id": r.clip_id, "window_idx": w["window_idx"], "t_end": w["t_end"]}
            for m, key in [("emotion", "emo_probs"), ("gesture", "ges_probs"),
                           ("motion", "mot_probs"), ("context", "ctx_probs")]:
                p, pref = w[key], PROB_PREFIX[m]
                row[f"{pref}_obs"] = p is not None
                row[f"{pref}_cov"] = round(float(w[f"{pref}_cov"]), 3)
                for i, c in enumerate(LABELS[m]):
                    row[f"{pref}_{c}"] = float(p[i]) if p is not None else np.nan
            rows.append(row)
        if n % 300 == 0:
            print(f"  [{n}/{len(clips)}] {len(rows)} windows "
                  f"({n/(time.time()-t0):.1f} clips/s)", flush=True)
    if missing:
        print(f"  WARNING: {len(missing)} clip(s) with no cache")
    return pd.DataFrame(rows)


def clip_pool(frame, modality):
    cols = [f"{PROB_PREFIX[modality]}_{c}" for c in LABELS[modality]]
    obs = f"{PROB_PREFIX[modality]}_obs"
    fired = frame[frame[obs]]
    if fired.empty:
        return pd.DataFrame(columns=["clip_id", "pred"])
    g = fired.groupby("clip_id")[cols].mean()
    pred = np.asarray(LABELS[modality])[g.to_numpy().argmax(1)]
    return pd.DataFrame({"clip_id": g.index, "pred": pred})


def score(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    if len(y_true) == 0:
        return {"n": 0, "acc": None, "macro_f1": None}
    return {"n": int(len(y_true)), "acc": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4)}


def evaluate_modality(modality, windows, clips):
    """All breakdowns are restricted to the TEST split first, THEN split by
    source/context -- computing curated_clip/raw_take over all splits mixed
    (train dominates by clip count) silently produces a number that looks like
    a generalisation estimate but mostly isn't. Diagnosed 2026-08-04 after the
    motion fine-tune comparison: an all-splits `raw_take` figure showed a
    misleading +0.19 "improvement" that vanished (became a regression) once
    restricted to test-split raw_take clips."""
    gt = f"gt_{modality}"
    mask_col = f"{modality}_masked"
    meta = clips.set_index("clip_id")
    cols = [gt, mask_col, "context", "split", "headline_eval", "source"]
    w = windows.join(meta[cols], on="clip_id")
    s = w[w[gt].notna() & ~w[mask_col]]
    pooled = clip_pool(s, modality).join(meta[cols], on="clip_id")

    te = pooled[pooled.split == "test"]
    hl = te[te.headline_eval]
    curated = te[te.source == "curated_clip"]
    raw = te[te.source.astype(str).str.startswith("raw_take")]
    return {
        "headline": score(hl[gt].to_numpy(), hl.pred.to_numpy()),
        "full_test": score(te[gt].to_numpy(), te.pred.to_numpy()),
        "test_curated_clip": score(curated[gt].to_numpy(), curated.pred.to_numpy()),
        "test_raw_take": score(raw[gt].to_numpy(), raw.pred.to_numpy()),
        "val": score(pooled[pooled.split == "val"][gt].to_numpy(),
                     pooled[pooled.split == "val"].pred.to_numpy()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion-ckpt", default=None)
    ap.add_argument("--gesture-ckpt", default=None)
    ap.add_argument("--motion-ckpt", default=None)
    ap.add_argument("--tag", default="candidate")
    args = ap.parse_args()
    changed = [m for m, c in [("emotion", args.emotion_ckpt), ("gesture", args.gesture_ckpt),
                              ("motion", args.motion_ckpt)] if c]
    if not changed:
        raise SystemExit("pass at least one of --emotion-ckpt/--gesture-ckpt/--motion-ckpt")

    from fusion.extraction.windows import WindowFeaturizer
    kwargs = {}
    if args.gesture_ckpt:
        kwargs["gesture_ckpt"] = Path(args.gesture_ckpt)
    if args.motion_ckpt:
        kwargs["motion_ckpt"] = Path(args.motion_ckpt)
    fz = WindowFeaturizer(**kwargs)

    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    print(f"{clips.clip_id.nunique()} clips -- rebuilding windows with NEW "
          f"checkpoint(s) for: {changed}", flush=True)
    windows = build_windows(clips, fz)
    out_parquet = ROOT / "data" / "final_merged" / "features" / \
        f"unimodal_windows_{args.tag}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_parquet, index=False)
    print(f"-> {out_parquet} ({len(windows)} windows)")

    # BEFORE is recomputed through the identical evaluate_modality() over the
    # DEPLOYED-checkpoint parquet (not read from unimodal_final_merged.json,
    # which used the same all-splits-mixed curated_clip/raw_take bug this
    # script just fixed) -- guarantees before/after are apples-to-apples.
    deployed_parquet = ROOT / "data" / "final_merged" / "features" / "unimodal_windows.parquet"
    windows_before = pd.read_parquet(deployed_parquet)
    windows_before = windows_before[windows_before.clip_id.isin(set(clips.clip_id))]

    print(f"\n{'modality':10}{'metric':22}{'BEFORE':>10}{'AFTER':>10}{'DELTA':>9}")
    report = {}
    for m in changed:
        old = evaluate_modality(m, windows_before, clips)
        new = evaluate_modality(m, windows, clips)
        report[m] = {"before": old, "after": new}
        rows = [
            ("headline acc", old["headline"]["acc"], new["headline"]["acc"]),
            ("headline macro-F1", old["headline"]["macro_f1"], new["headline"]["macro_f1"]),
            ("val acc", old["val"]["acc"], new["val"]["acc"]),
            ("test curated_clip acc", old["test_curated_clip"]["acc"],
             new["test_curated_clip"]["acc"]),
            ("test raw_take acc", old["test_raw_take"]["acc"], new["test_raw_take"]["acc"]),
        ]
        for label, before, after in rows:
            d = (round(after - before, 4) if before is not None and after is not None
                else None)
            arrow = "  ^" if (d or 0) > 0.001 else ("  v" if (d or 0) < -0.001 else "")
            print(f"{m:10}{label:22}{before if before is not None else '-':>10}"
                 f"{after if after is not None else '-':>10}"
                 f"{d if d is not None else '-':>9}{arrow}")

    (OUT_DIR / f"refresh_compare_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT_DIR / f'refresh_compare_{args.tag}.json'}")

    for m in changed:
        h = report[m]["after"]["headline"]
        with start_run("03_diagnostics", f"final_merged__unimodal_{m}_finetuned",
                       dataset="final_merged", split_kind="scenarios", cues="real",
                       params={"model": m, "checkpoint_tag": args.tag},
                       notes="after fine-tune on complete dataset; compare to "
                            f"final_merged__unimodal_{m}") as run:
            run.log_metrics({"headline_clip_acc": h["acc"], "headline_clip_macro_f1": h["macro_f1"],
                             "test_raw_take_acc": report[m]["after"]["test_raw_take"]["acc"]})


if __name__ == "__main__":
    main()
