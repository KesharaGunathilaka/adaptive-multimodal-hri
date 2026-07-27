"""Pass 2 over `data/final`: score all four unimodal models against the V3 cues.

Answers one question — *is each perception model good enough to feed fusion?* —
by separating the three ways a cue can be wrong:

1. **accuracy** on clips that have a target and where the model fired,
2. **coverage** (runtime-missing): windows where the model produced nothing,
3. **designed-missing realism**: on rows the table marks `[missing]`, the cue
   *should* be unobservable; a high observation rate there means the recording
   did not realise the design.

    .venv/Scripts/python scripts/16_final_unimodal_eval.py --context classroom
    .venv/Scripts/python scripts/16_final_unimodal_eval.py --context classroom --rebuild-windows
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

from scripts.realworld_eval.final_common import load_clips  # noqa: E402
from scripts.realworld_eval.final_unimodal import (  # noqa: E402
    LABELS, MODALITIES, PERFRAME_DIR, PROB_PREFIX, UNI_DIR, WINDOWS_PARQUET,
    add_targets, clip_pool, obs_col, prob_cols, score)


# ── window table ────────────────────────────────────────────────────────────
def build_windows(clips: pd.DataFrame) -> pd.DataFrame:
    """Run the shared WindowFeaturizer over every cached clip."""
    from fusion.extraction.windows import WindowFeaturizer
    fz = WindowFeaturizer()

    # the featurizer's own label order must match what we score against
    for m, got in [("gesture", fz.gesture_labels), ("motion", fz.motion_labels)]:
        if list(got) != LABELS[m]:
            raise SystemExit(f"{m} label order drifted: {got} != {LABELS[m]}")

    rows, missing_cache = [], []
    t0 = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        npz_path = PERFRAME_DIR / f"{r.clip_id}.npz"
        if not npz_path.exists():
            missing_cache.append(r.clip_id)
            continue
        npz = np.load(npz_path)
        for w in fz.featurize_clip(npz):
            row = {"clip_id": r.clip_id, "window_idx": w["window_idx"],
                   "t_end": w["t_end"]}
            for m, key in [("emotion", "emo_probs"), ("gesture", "ges_probs"),
                           ("motion", "mot_probs"), ("context", "ctx_probs")]:
                p, pref = w[key], PROB_PREFIX[m]
                row[f"{pref}_obs"] = p is not None
                row[f"{pref}_cov"] = round(float(w[f"{pref}_cov"]), 3)
                for i, c in enumerate(LABELS[m]):
                    row[f"{pref}_{c}"] = float(p[i]) if p is not None else np.nan
            rows.append(row)
        if n % 100 == 0:
            print(f"  [{n}/{len(clips)}] {len(rows)} windows "
                  f"({n / (time.time() - t0):.1f} clips/s)", flush=True)

    if missing_cache:
        print(f"  WARNING: {len(missing_cache)} clip(s) have no per-frame cache "
              f"— run scripts/15_final_extract.py: {missing_cache[:5]}")
    return pd.DataFrame(rows)


# ── reporting ───────────────────────────────────────────────────────────────
def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in labels], columns=labels)


def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": p.round(3), "recall": r.round(3),
                         "f1": f.round(3), "support": s}, index=labels)


def evaluate(modality: str, windows: pd.DataFrame, clips: pd.DataFrame) -> dict:
    labels = LABELS[modality]
    gt, miss, obs = f"gt_{modality}", f"missing_{modality}", obs_col(modality)
    meta = clips.set_index("clip_id")

    cols = [gt, miss, "view", "scenario_dir", "v3_row", "split_design",
            "intent", "source"]
    w = windows.join(meta[cols], on="clip_id")

    # 1. designed-missing realism — the cue must be hard to observe there
    des = w[w[miss]]
    designed = {"n_clips": int(des.clip_id.nunique()),
                "n_windows": int(len(des)),
                "observation_rate": round(float(des[obs].mean()), 3) if len(des) else None,
                "per_scenario": (des.groupby("scenario_dir")[obs].mean().round(3)
                                 .to_dict() if len(des) else {})}

    # 2. scoreable rows
    s = w[w[gt].notna() & ~w[miss]].copy()
    if s.empty:
        return {"designed_missing": designed, "note": "no scoreable clips"}

    coverage = {"window_observation_rate": round(float(s[obs].mean()), 3),
                "clips_never_observed": int((~s.groupby("clip_id")[obs].any()).sum()),
                "per_view": s.groupby("view")[obs].mean().round(3).to_dict()}

    # window-level: only windows where the model fired
    fired = s[s[obs]]
    win = score(fired[gt].to_numpy(),
                np.asarray(labels)[fired[prob_cols(modality)].to_numpy().argmax(1)],
                labels)

    # clip-level: mean-softmax pool over observed windows
    pooled = clip_pool(s, modality).join(meta[cols[:1] + cols[2:]], on="clip_id")
    clip = score(pooled[gt].to_numpy(), pooled.pred.to_numpy(), labels)
    pooled["correct"] = pooled.pred == pooled[gt]

    def by(col):
        return {k: score(g[gt].to_numpy(), g.pred.to_numpy(), labels)
                for k, g in pooled.groupby(col)}

    # `curated_clip` = migrated from data/old, i.e. the clips the emotion model
    # was fine-tuned on. Splitting by source is the only way to tell a genuinely
    # held-out number from one inflated by training-set overlap.
    by_scen = (pooled.groupby(["scenario_dir", "v3_row", "intent", "split_design",
                               "source", gt])
                     .agg(n=("correct", "size"), acc=("correct", "mean"),
                          top_error=("pred", lambda s: _top_error(s, pooled, gt)))
                     .round(3).reset_index()
                     .sort_values("acc"))

    return {"designed_missing": designed, "coverage": coverage,
            "window": win, "clip": clip, "by_view": by("view"),
            "by_split_design": by("split_design"), "by_source": by("source"),
            "per_class": per_class(pooled[gt], pooled.pred, labels),
            "confusion": confusion(pooled[gt], pooled.pred, labels),
            "per_scenario": by_scen}


def _top_error(preds: pd.Series, pooled: pd.DataFrame, gt: str) -> str:
    """Most frequent wrong prediction in a scenario group, with its count."""
    truth = pooled.loc[preds.index, gt]
    wrong = preds[preds != truth]
    if wrong.empty:
        return ""
    top = wrong.value_counts()
    return f"{top.index[0]}×{top.iloc[0]}"


def write_report(res: dict, clips: pd.DataFrame, context: str, out_md: Path) -> None:
    L = [f"# Unimodal results on `data/final` — {context}",
         "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} · "
             f"{clips.clip_id.nunique()} clips · "
             f"views: {', '.join(sorted(clips.view.unique()))}",
         "",
         "Clip-level = mean of softmax over the windows where the model fired, "
         "then argmax. Rows the V3 table marks `[missing]` for a cue are excluded "
         "from that cue's accuracy and reported separately.",
         "", "## 1. Headline", "",
         "| Modality | Clips scored | Clip acc | Clip macro-F1 | Window acc | "
         "Window macro-F1 | Windows observed |", "|---|---|---|---|---|---|---|"]
    for m in MODALITIES:
        r = res[m]
        if "clip" not in r:
            L.append(f"| {m} | 0 | — | — | — | — | — |")
            continue
        L.append(f"| {m} | {r['clip']['n']} | {r['clip']['acc']} | "
                 f"{r['clip']['macro_f1']} | {r['window']['acc']} | "
                 f"{r['window']['macro_f1']} | "
                 f"{r['coverage']['window_observation_rate']} |")

    L += ["", "## 2. Train-design vs test-design rows", "",
          "| Modality | train acc | train macro-F1 | test acc | test macro-F1 |",
          "|---|---|---|---|---|"]
    for m in MODALITIES:
        b = res[m].get("by_split_design", {})
        tr, te = b.get("train", {}), b.get("test", {})
        L.append(f"| {m} | {tr.get('acc')} | {tr.get('macro_f1')} | "
                 f"{te.get('acc')} | {te.get('macro_f1')} |")

    L += ["", "## 2b. Held-out vs training-overlap clips", "",
          "`curated_clip` rows were migrated from `data/old` — the emotion model "
          "was fine-tuned on them, so its number there is not a generalisation "
          "estimate. `raw_take_20260725` is the genuinely unseen collection.", "",
          "| Modality | curated_clip acc | macro-F1 | raw_take_20260725 acc | macro-F1 |",
          "|---|---|---|---|---|"]
    for m in MODALITIES:
        b = res[m].get("by_source", {})
        cu, rw = b.get("curated_clip", {}), b.get("raw_take_20260725", {})
        L.append(f"| {m} | {cu.get('acc')} (n={cu.get('n')}) | "
                 f"{cu.get('macro_f1')} | {rw.get('acc')} (n={rw.get('n')}) | "
                 f"{rw.get('macro_f1')} |")

    L += ["", "## 3. By camera view (clip-level)", "",
          "| Modality | " + " | ".join(sorted(clips.view.unique())) + " |",
          "|---" * (len(clips.view.unique()) + 1) + "|"]
    views = sorted(clips.view.unique())
    for m in MODALITIES:
        b = res[m].get("by_view", {})
        cells = [f"{b[v]['acc']} (n={b[v]['n']})" if v in b else "—" for v in views]
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L += ["", "## 4. Designed-missing rows — was the cue really unobservable?", "",
          "Observation rate is the fraction of windows where the model still "
          "produced an output. Low is what the design asks for.", "",
          "| Modality | Clips | Observation rate | Per scenario |",
          "|---|---|---|---|"]
    for m in MODALITIES:
        d = res[m]["designed_missing"]
        if not d["n_clips"]:
            continue
        per = ", ".join(f"{k}={v}" for k, v in sorted(d["per_scenario"].items()))
        L.append(f"| {m} | {d['n_clips']} | {d['observation_rate']} | {per} |")

    for m in MODALITIES:
        r = res[m]
        if "per_class" not in r:
            continue
        L += ["", f"## 5.{MODALITIES.index(m) + 1} {m} detail", "",
              "**Per class (clip-level)**", "",
              r["per_class"].to_markdown(), "",
              "**Confusion (rows = truth)**", "",
              r["confusion"].to_markdown(), "",
              "**Worst scenarios**", "",
              r["per_scenario"].head(10).to_markdown(index=False), "",
              f"Coverage: {r['coverage']['clips_never_observed']} clip(s) never "
              f"observed; per-view observation "
              f"{r['coverage']['per_view']}"]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf8")
    print(f"  -> {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default=None)
    ap.add_argument("--view", default=None)
    ap.add_argument("--rebuild-windows", action="store_true")
    args = ap.parse_args()

    clips = add_targets(load_clips())
    clips = clips[clips.v3_row.notna()]            # drop unmapped folders
    if args.context:
        clips = clips[clips.context == args.context]
    if args.view:
        clips = clips[clips.view == args.view]
    cached = {p.stem for p in PERFRAME_DIR.glob("*.npz")}
    clips = clips[clips.clip_id.isin(cached)]
    print(f"{len(clips)} clips with a per-frame cache")
    if clips.empty:
        raise SystemExit("nothing to score — run scripts/15_final_extract.py first")

    if args.rebuild_windows or not WINDOWS_PARQUET.exists():
        print("Building window table:")
        windows = build_windows(clips)
        WINDOWS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        windows.to_parquet(WINDOWS_PARQUET, index=False)
        print(f"  -> {WINDOWS_PARQUET} ({len(windows)} windows)")
    else:
        windows = pd.read_parquet(WINDOWS_PARQUET)
        print(f"reusing {WINDOWS_PARQUET} ({len(windows)} windows) "
              "— pass --rebuild-windows to regenerate")
    windows = windows[windows.clip_id.isin(set(clips.clip_id))]

    res = {m: evaluate(m, windows, clips) for m in MODALITIES}

    tag = args.context or "all"
    UNI_DIR.mkdir(parents=True, exist_ok=True)
    write_report(res, clips, tag, UNI_DIR / f"UNIMODAL_{tag}.md")

    jsonable = {m: {k: v for k, v in r.items()
                    if not isinstance(v, pd.DataFrame)} for m, r in res.items()}
    (UNI_DIR / f"unimodal_{tag}.json").write_text(json.dumps(jsonable, indent=2))
    for m, r in res.items():
        if "per_scenario" in r:
            r["per_scenario"].to_csv(UNI_DIR / f"per_scenario_{m}_{tag}.csv", index=False)

    print("\n== headline (clip-level) ==")
    for m in MODALITIES:
        c = res[m].get("clip")
        print(f"  {m:8s} acc={c['acc'] if c else '—'} "
              f"macroF1={c['macro_f1'] if c else '—'} n={c['n'] if c else 0}")


if __name__ == "__main__":
    main()
