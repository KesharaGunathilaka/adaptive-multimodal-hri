"""Export every MLflow run to `results/EXPERIMENTS.csv` (committed) + a readable
Markdown summary.

`mlruns.db` / `mlruns/` are per-machine and gitignored, exactly like checkpoints.
This export is what crosses machines: run it before committing, and the other PC
sees every number through git without any tracking store to merge.

    .venv/Scripts/python scripts/25_export_runs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.tracking import DB_URI, EXPERIMENTS  # noqa: E402

OUT_CSV = ROOT / "results" / "EXPERIMENTS.csv"
OUT_MD = ROOT / "results" / "EXPERIMENTS.md"

# columns worth putting in the readable summary, in order
HEADLINE = ["experiment", "run_name", "dataset", "split_kind", "cues",
            "model", "aggregation", "seed",
            "test_clip_acc", "test_clip_macro_f1", "val_clip_acc"]


def main() -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(DB_URI)
    frames = []
    for exp in EXPERIMENTS:
        try:
            df = mlflow.search_runs(experiment_names=[exp])
        except Exception:
            continue
        if len(df):
            df = df.copy()
            df["experiment"] = exp
            frames.append(df)

    if not frames:
        print("no runs found -- nothing to export")
        return

    runs = pd.concat(frames, ignore_index=True)
    runs["run_name"] = runs.get("tags.mlflow.runName")

    keep = ["experiment", "run_name", "start_time", "status"]
    keep += sorted(c for c in runs.columns if c.startswith(("params.", "metrics.")))
    keep += [c for c in ("tags.git_commit", "tags.machine", "tags.notes")
             if c in runs.columns]
    out = runs[[c for c in keep if c in runs.columns]].copy()
    out.columns = [c.replace("params.", "").replace("metrics.", "")
                   .replace("tags.", "") for c in out.columns]
    out = out.sort_values(["experiment", "start_time"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # readable summary
    cols = [c for c in HEADLINE if c in out.columns]
    md = ["# Experiment ledger", "",
          f"{len(out)} runs · exported from `mlruns.db` by `scripts/25_export_runs.py`.",
          "",
          "`mlruns.db` is per-machine and gitignored; **this file is the "
          "cross-machine record**. Full parameter/metric set in "
          "`EXPERIMENTS.csv`.", ""]
    for exp in out.experiment.unique():
        sub = out[out.experiment == exp][cols]
        md += [f"## {exp}", "", sub.to_markdown(index=False), ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"-> {OUT_CSV.relative_to(ROOT)}  ({len(out)} runs, {len(out.columns)} cols)")
    print(f"-> {OUT_MD.relative_to(ROOT)}")
    print(out.groupby("experiment").size().to_string())


if __name__ == "__main__":
    main()
