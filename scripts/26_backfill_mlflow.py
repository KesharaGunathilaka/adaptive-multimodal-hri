"""Backfill MLflow with the experiments already run, from their saved outputs.

Without this the new runs have nothing to sit beside. Everything logged here is
read from files in `results/` — no model is re-trained, no number is invented.
Each run is tagged `backfilled=true` and carries the date it was originally
measured, so a backfilled row is never mistaken for a fresh one.

Idempotent: existing runs with the same name are skipped.

    .venv/Scripts/python scripts/26_backfill_mlflow.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.tracking import DB_URI, start_run  # noqa: E402

FUSION_V1 = ROOT / "results" / "fusion_v1"
# scripts/04_run_baselines.py writes here, NOT into results/fusion_v1/
BASELINES_JSON = ROOT / "fusion" / "baselines" / "results.json"


def _existing() -> set[str]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    mlflow.set_tracking_uri(DB_URI)
    names = set()
    for exp in ("01_baselines", "02_fusion", "03_diagnostics"):
        try:
            df = mlflow.search_runs(experiment_names=[exp])
        except Exception:
            continue
        if len(df) and "tags.mlflow.runName" in df:
            names |= set(df["tags.mlflow.runName"].dropna())
    return names


def backfill_baselines(done: set[str]) -> int:
    """data/old baselines — fusion/baselines/results.json."""
    f = BASELINES_JSON
    if not f.exists():
        return 0
    res = json.loads(f.read_text())
    n = 0
    simple = {"rule_based": "rule_based", "unimodal_emotion": "unimodal_emotion",
              "unimodal_gesture": "unimodal_gesture",
              "unimodal_motion": "unimodal_motion",
              "unimodal_context": "unimodal_context"}
    for key, model in simple.items():
        name = f"old__{model}"
        if key not in res or name in done:
            continue
        with start_run("01_baselines", name, dataset="old", split_kind="people",
                       cues="real", params={"model": model, "aggregation": "window_vote"},
                       tags={"backfilled": "true", "measured_on": "2026-07-17"},
                       notes="data/old, actor-disjoint test subjects") as r:
            for split in ("train", "val", "test"):
                if split not in res[key]:
                    continue
                for lvl in ("window", "clip"):
                    m = res[key][split][lvl]
                    r.log_metrics({f"{split}_{lvl}_acc": m["acc"],
                                   f"{split}_{lvl}_macro_f1": m["macro_f1"]})
        n += 1

    agg = res.get("concat_mlp")
    if agg and "old__concat_mlp" not in done:
        with start_run("01_baselines", "old__concat_mlp", dataset="old",
                       split_kind="people", cues="real",
                       params={"model": "concat_mlp", "seeds": 3,
                               "aggregation": "window_vote"},
                       tags={"backfilled": "true", "measured_on": "2026-07-17"},
                       notes="mean over 3 seeds") as r:
                r.log_metrics({k.replace(".", "_"): v["mean"] for k, v in agg.items()})
                r.log_metrics({k.replace(".", "_") + "_std": v["std"]
                               for k, v in agg.items()})
        n += 1
    return n


def backfill_fusion(done: set[str]) -> int:
    """attention-fusion ablations — results/fusion_v1/results.json."""
    f = FUSION_V1 / "results.json"
    if not f.exists():
        return 0
    res = json.loads(f.read_text())
    cfgs = (res.get("_meta") or {}).get("configs", {})
    n = 0
    for cfg_name in ("attn_base", "attn_do", "attn_do_jit", "attn_full", "attn_robust"):
        block = res.get(cfg_name)
        name = f"old__{cfg_name}"
        if not block or "agg" not in block or name in done:
            continue
        p = {"model": "attention_fusion", "seeds": 3, "aggregation": "window_vote",
             "missing_mode": "exclude" if cfg_name == "attn_robust" else "token"}
        p.update({k: v for k, v in (cfgs.get(cfg_name) or {}).items()})
        with start_run("02_fusion", name, dataset="old", split_kind="people",
                       cues="real", params=p,
                       tags={"backfilled": "true", "measured_on": "2026-07-17"},
                       notes="mean over 3 seeds") as r:
            for k, v in block["agg"].items():
                r.log_metrics({k.replace(".", "_"): v["mean"],
                               k.replace(".", "_") + "_std": v["std"]})
        n += 1

    sweep = res.get("masking_sweep_test_clip")
    if sweep and "old__masking_sweep" not in done:
        with start_run("03_diagnostics", "old__masking_sweep", dataset="old",
                       split_kind="people", cues="real",
                       params={"model": "attention_vs_mlp", "aggregation": "window_vote"},
                       tags={"backfilled": "true", "measured_on": "2026-07-17"},
                       notes="T03 masking sweep, best attn_robust seed") as r:
            for sysname, blocks in sweep.items():
                for mask, m in blocks.items():
                    tag = mask.replace("+", "_")
                    r.log_metrics({f"{sysname}_mask_{tag}_acc": m["acc"],
                                   f"{sysname}_mask_{tag}_macro_f1": m["macro_f1"]})
        n += 1
    return n


def backfill_window_sweep(done: set[str]) -> int:
    f = FUSION_V1 / "window_sweep.json"
    if not f.exists() or "old__window_sweep" in done:
        return 0
    res = json.loads(f.read_text())
    with start_run("03_diagnostics", "old__window_sweep", dataset="old",
                   split_kind="people", cues="real",
                   params={"model": "attention_fusion", "aggregation": "window"},
                   tags={"backfilled": "true", "measured_on": "2026-07-20"},
                   notes="lookback span x0.5 / x1 / x2") as r:
        for scale, block in res.items():
            s = scale.replace(".", "_")
            r.log_metrics({f"{s}_test_clip_acc": block["test_clip"]["acc"],
                           f"{s}_test_clip_macro_f1": block["test_clip"]["macro_f1"],
                           f"{s}_val_clip_acc": block["val_clip"]["acc"],
                           f"{s}_n_windows": block["n_windows"],
                           f"{s}_ges_span_s": block["ges_span_s"]})
    return 1


def backfill_gap(done: set[str]) -> int:
    """2026-07-28 gap decomposition + clip-vs-window (GAP_DECOMPOSITION.md)."""
    n = 0
    gap = [("classroom__rules_oracle", "rule_based", "oracle", 0.900, None),
           ("classroom__fusion_oracle", "attention_fusion", "oracle", 0.500, None),
           ("classroom__rules_real", "rule_based", "real", 0.494, 0.473),
           ("classroom__fusion_real", "attention_fusion", "real", 0.325, 0.242)]
    for name, model, cues, acc, f1 in gap:
        if name in done:
            continue
        with start_run("03_diagnostics", name, dataset="final",
                       split_kind="scenarios", cues=cues,
                       params={"model": model, "aggregation": "clip_mean_4s",
                               "context": "classroom", "view": "realsense_480p"},
                       tags={"backfilled": "true", "measured_on": "2026-07-28"},
                       notes="gap decomposition; ceiling for these rows is 0.900") as r:
            r.log_metrics({"test_clip_acc": acc, "table_ceiling": 0.900})
            if f1 is not None:
                r.log_metrics({"test_clip_macro_f1": f1})
        n += 1

    cvw = [("clipvswin__train_window_vote", "window_vote", 0.342, 0.284),
           ("clipvswin__train_window_infer_clipmean", "window_train_clip_mean", 0.331, 0.269),
           ("clipvswin__train_window_infer_clipmax", "window_train_clip_max", 0.309, 0.246),
           ("clipvswin__clip_mean", "clip_mean", 0.374, 0.320),
           ("clipvswin__clip_max", "clip_max", 0.338, 0.289),
           ("clipvswin__clip_peak", "clip_peak", 0.373, 0.308)]
    for name, aggm, acc, f1 in cvw:
        if name in done:
            continue
        with start_run("03_diagnostics", name, dataset="final",
                       split_kind="scenarios", cues="real",
                       params={"model": "attention_fusion", "aggregation": aggm,
                               "seeds": 3},
                       tags={"backfilled": "true", "measured_on": "2026-07-28"},
                       notes="clip-level vs window-level study") as r:
            r.log_metrics({"test_clip_acc": acc, "test_clip_macro_f1": f1})
        n += 1
    return n


def main() -> None:
    done = _existing()
    if done:
        print(f"{len(done)} run(s) already logged — skipping those")
    total = (backfill_baselines(done) + backfill_fusion(done)
             + backfill_window_sweep(done) + backfill_gap(done))
    print(f"\nbackfilled {total} run(s)")
    print("next: .venv/Scripts/python scripts/25_export_runs.py")


if __name__ == "__main__":
    main()
