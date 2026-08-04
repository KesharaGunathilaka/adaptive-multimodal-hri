"""Pass 2 + evaluation over `data/final_merged`: score all four unimodal models
against the V3 cues, now that the complete (classroom + kitchen, train + test)
dataset exists.

Builds the window table from the Pass-1 caches (`scripts/27_merged_extract.py`),
mean-pools to clip level, and reports accuracy split every way that matters for
"should we fine-tune this model?":

  * headline (`headline_eval=True` only — excludes row #58's 24 train-derived
    test clips per DECISIONS 2026-08-03)
  * train / val / test (the `split` column from `23_build_splits.py`)
  * source: `curated_clip` (migrated from `data/old`, i.e. clips emotion/motion
    were fine-tuned on) vs the `raw_take_*` sessions that are genuinely unseen
    by every model — the only way to avoid quoting a fine-tune-inflated number
  * resolution_class / context — the confounds documented in DATASET_FIXLIST N8/N9

Logs one MLflow run per modality (experiment `03_diagnostics`) plus a combined
run, so this sits next to the `data/old` and `data/final` numbers already
backfilled.

    .venv/Scripts/python scripts/28_merged_unimodal_eval.py
    .venv/Scripts/python scripts/28_merged_unimodal_eval.py --rebuild-windows
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

from scripts.realworld_eval.merged_unimodal import (  # noqa: E402
    LABELS, MODALITIES, PERFRAME_DIR, PROB_PREFIX, UNI_DIR, WINDOWS_PARQUET,
    clip_pool, load_clips, obs_col, prob_cols, score)


# ── window table (Pass 2) ───────────────────────────────────────────────────
def build_windows(clips: pd.DataFrame) -> pd.DataFrame:
    from fusion.extraction.windows import WindowFeaturizer
    fz = WindowFeaturizer()
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
        if n % 200 == 0 or n == len(clips):
            print(f"  [{n}/{len(clips)}] {len(rows)} windows "
                  f"({n / (time.time() - t0):.1f} clips/s)", flush=True)

    if missing_cache:
        print(f"  WARNING: {len(missing_cache)} clip(s) have no per-frame cache: "
              f"{missing_cache[:5]}")
    return pd.DataFrame(rows)


# ── reporting ────────────────────────────────────────────────────────────────
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
    gt, mask_col, obs = f"gt_{modality}", f"{modality}_masked", obs_col(modality)
    meta = clips.set_index("clip_id")

    cols = [gt, mask_col, "context", "resolution_class", "orientation",
            "scenario_dir", "v3_row", "split", "split_design", "intent",
            "source", "headline_eval"]
    w = windows.join(meta[cols], on="clip_id")

    des = w[w[mask_col]]
    designed = {"n_clips": int(des.clip_id.nunique()), "n_windows": int(len(des)),
                "observation_rate": round(float(des[obs].mean()), 3) if len(des) else None,
                "per_scenario": (des.groupby("scenario_dir")[obs].mean().round(3)
                                 .to_dict() if len(des) else {})}

    s = w[w[gt].notna() & ~w[mask_col]].copy()
    if s.empty:
        return {"designed_missing": designed, "note": "no scoreable clips"}

    coverage = {"window_observation_rate": round(float(s[obs].mean()), 3),
                "clips_never_observed": int((~s.groupby("clip_id")[obs].any()).sum())}

    fired = s[s[obs]]
    win = score(fired[gt].to_numpy(),
                np.asarray(labels)[fired[prob_cols(modality)].to_numpy().argmax(1)])

    pooled = clip_pool(s, modality).join(meta[cols[:1] + cols[2:]], on="clip_id")
    pooled["correct"] = pooled.pred == pooled[gt]
    clip = score(pooled[gt].to_numpy(), pooled.pred.to_numpy())
    # headline = TEST split only, and excluding row #58's train-derived clips.
    # (headline_eval is True by default for train/val rows too -- it only ever
    # turns False for those 24 clips -- so it must be AND-ed with split=='test',
    # never used alone, or the "headline" number silently includes training data.)
    hl = pooled[(pooled.split == "test") & pooled.headline_eval]
    headline = score(hl[gt].to_numpy(), hl.pred.to_numpy())

    def by(col):
        return {k: score(g[gt].to_numpy(), g.pred.to_numpy())
                for k, g in pooled.groupby(col)}

    by_scen = (pooled.groupby(["scenario_dir", "v3_row", "intent", "split",
                               "context", "source", gt])
                     .agg(n=("correct", "size"), acc=("correct", "mean"))
                     .round(3).reset_index().sort_values("acc"))

    return {"designed_missing": designed, "coverage": coverage,
            "window": win, "clip": clip, "headline": headline,
            "by_split": by("split"), "by_source": by("source"),
            "by_context": by("context"), "by_resolution": by("resolution_class"),
            "per_class": per_class(pooled[gt], pooled.pred, labels),
            "confusion": confusion(pooled[gt], pooled.pred, labels),
            "per_scenario": by_scen}


def write_report(res: dict, clips: pd.DataFrame, out_md: Path) -> None:
    L = [f"# Unimodal results on `data/final_merged` — complete dataset "
         f"(classroom + kitchen, train + test)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} · "
         f"{clips.clip_id.nunique()} clips · "
         f"{clips.v3_row.nunique()} V3 rows · "
         f"contexts: {', '.join(sorted(clips.context.unique()))}",
         "",
         "Clip-level = mean of softmax over windows where the model fired, then "
         "argmax. **headline** additionally excludes row #58's 24 test clips "
         "that are row #49's train footage re-used with masked cues "
         "(`docs/DECISIONS.md` 2026-08-03, `headline_eval=False`).", "",
         "## 1. Headline (test split, headline_eval only)", "",
         "| Modality | Clips | Clip acc | Clip macro-F1 | Window acc | "
         "Window macro-F1 | Obs. rate |", "|---|---|---|---|---|---|---|"]
    for m in MODALITIES:
        r = res[m]
        if "clip" not in r:
            L.append(f"| {m} | 0 | — | — | — | — | — |")
            continue
        h = r["headline"]
        L.append(f"| {m} | {h['n']} | {h['acc']} | {h['macro_f1']} | "
                 f"{r['window']['acc']} | {r['window']['macro_f1']} | "
                 f"{r['coverage']['window_observation_rate']} |")

    L += ["", "## 2. Train / val / test (all test clips, incl. #58's 24)", "",
          "| Modality | train acc | train F1 | val acc | val F1 | "
          "test acc | test F1 |", "|---|---|---|---|---|---|---|"]
    for m in MODALITIES:
        b = res[m].get("by_split", {})
        tr, va, te = b.get("train", {}), b.get("val", {}), b.get("test", {})
        L.append(f"| {m} | {tr.get('acc')} | {tr.get('macro_f1')} | "
                 f"{va.get('acc')} | {va.get('macro_f1')} | "
                 f"{te.get('acc')} | {te.get('macro_f1')} |")

    L += ["", "## 3. Held-out vs fine-tune-overlap clips (the number that matters)", "",
          "`curated_clip` = migrated from `data/old` — emotion and motion were "
          "fine-tuned on (a subset of) these; their number here is not a "
          "generalisation estimate. The `raw_take_*` sessions are genuinely "
          "unseen by every model.", "",
          "| Modality | curated_clip acc (n) | F1 | raw_take (all) acc (n) | F1 |",
          "|---|---|---|---|---|"]
    for m in MODALITIES:
        b = res[m].get("by_source", {})
        cu = b.get("curated_clip", {})
        raw_ns = [v["n"] for k, v in b.items() if k.startswith("raw_take") and v.get("n")]
        raw_correct = sum(v["acc"] * v["n"] for k, v in b.items()
                          if k.startswith("raw_take") and v.get("n"))
        raw_n = sum(raw_ns)
        raw_acc = round(raw_correct / raw_n, 4) if raw_n else None
        L.append(f"| {m} | {cu.get('acc')} (n={cu.get('n', 0)}) | "
                 f"{cu.get('macro_f1')} | {raw_acc} (n={raw_n}) | — |")

    L += ["", "## 4. By context (classroom vs kitchen — now both complete)", "",
          "| Modality | classroom acc | F1 | kitchen acc | F1 |",
          "|---|---|---|---|---|"]
    for m in MODALITIES:
        b = res[m].get("by_context", {})
        cl, ki = b.get("classroom", {}), b.get("kitchen", {})
        L.append(f"| {m} | {cl.get('acc')} (n={cl.get('n', 0)}) | "
                 f"{cl.get('macro_f1')} | {ki.get('acc')} (n={ki.get('n', 0)}) | "
                 f"{ki.get('macro_f1')} |")

    L += ["", "## 5. By resolution class (confound: also correlated with context, N8/N9)", "",
          "| Modality | " + " | ".join(sorted(clips.resolution_class.unique())) + " |",
          "|---" * (len(clips.resolution_class.unique()) + 1) + "|"]
    rescls = sorted(clips.resolution_class.unique())
    for m in MODALITIES:
        b = res[m].get("by_resolution", {})
        cells = [f"{b[v]['acc']} (n={b[v]['n']})" if v in b and b[v].get("n") else "—"
                 for v in rescls]
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L += ["", "## 6. Designed-missing rows — did the cue come out missing?", "",
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
        L += ["", f"## 7.{MODALITIES.index(m) + 1} {m} detail", "",
              "**Per class (clip-level, all test)**", "",
              r["per_class"].to_markdown(), "",
              "**Confusion (rows = truth)**", "",
              r["confusion"].to_markdown(), "",
              "**Worst scenarios**", "",
              r["per_scenario"].head(10).to_markdown(index=False)]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"  -> {out_md}")


def log_mlflow(res: dict) -> None:
    sys.path.insert(0, str(ROOT))
    from fusion.tracking import start_run

    for m in MODALITIES:
        r = res[m]
        if "clip" not in r:
            continue
        with start_run("03_diagnostics", f"final_merged__unimodal_{m}",
                       dataset="final_merged", split_kind="scenarios", cues="real",
                       params={"model": m, "aggregation": "clip_mean_window"},
                       notes="all views, both contexts, headline excludes row #58") as run:
            run.log_metrics({
                "test_clip_acc": r["clip"]["acc"],
                "test_clip_macro_f1": r["clip"]["macro_f1"],
                "headline_clip_acc": r["headline"]["acc"],
                "headline_clip_macro_f1": r["headline"]["macro_f1"],
                "test_window_acc": r["window"]["acc"],
                "test_window_macro_f1": r["window"]["macro_f1"],
                "window_observation_rate": r["coverage"]["window_observation_rate"],
            })
            for src in ("curated_clip",):
                b = r.get("by_source", {}).get(src)
                if b and b.get("n"):
                    run.log_metrics({f"{src}_acc": b["acc"], f"{src}_n": b["n"]})
            for ctx in ("classroom", "kitchen"):
                b = r.get("by_context", {}).get(ctx)
                if b and b.get("n"):
                    run.log_metrics({f"{ctx}_acc": b["acc"], f"{ctx}_n": b["n"]})
            run.log_table(r["per_scenario"], f"per_scenario_{m}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-windows", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    cached = {p.stem for p in PERFRAME_DIR.glob("*.npz")}
    clips = clips[clips.clip_id.isin(cached)]
    print(f"{len(clips)} clips with a per-frame cache", flush=True)
    if clips.empty:
        raise SystemExit("nothing to score — run scripts/27_merged_extract.py first")

    if args.rebuild_windows or not WINDOWS_PARQUET.exists():
        print("Building window table:", flush=True)
        windows = build_windows(clips)
        WINDOWS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        windows.to_parquet(WINDOWS_PARQUET, index=False)
        print(f"  -> {WINDOWS_PARQUET} ({len(windows)} windows)", flush=True)
    else:
        windows = pd.read_parquet(WINDOWS_PARQUET)
        print(f"reusing {WINDOWS_PARQUET} ({len(windows)} windows) "
              "-- pass --rebuild-windows to regenerate", flush=True)
    windows = windows[windows.clip_id.isin(set(clips.clip_id))]

    res = {m: evaluate(m, windows, clips) for m in MODALITIES}

    UNI_DIR.mkdir(parents=True, exist_ok=True)
    write_report(res, clips, UNI_DIR / "UNIMODAL_final_merged.md")

    jsonable = {m: {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}
                for m, r in res.items()}
    (UNI_DIR / "unimodal_final_merged.json").write_text(json.dumps(jsonable, indent=2))
    for m, r in res.items():
        if "per_scenario" in r:
            r["per_scenario"].to_csv(UNI_DIR / f"per_scenario_{m}.csv", index=False)

    if not args.no_mlflow:
        log_mlflow(res)
        print("logged to MLflow (03_diagnostics) -- "
              "run scripts/25_export_runs.py to update the committed ledger")

    print("\n== headline (clip-level, headline_eval, test split) ==")
    for m in MODALITIES:
        h = res[m].get("headline")
        print(f"  {m:8s} acc={h['acc'] if h else '—'} "
              f"macroF1={h['macro_f1'] if h else '—'} n={h['n'] if h else 0}")


if __name__ == "__main__":
    main()
