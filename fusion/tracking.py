"""MLflow experiment tracking for the fusion pipeline.

Storage model (2026-08-03 decision, mirrors the checkpoint-manifest pattern):
  * `mlruns.db` + `mlruns/` live in the repo root and are **gitignored** — they
    are per-machine, like checkpoints.
  * `scripts/25_export_runs.py` dumps every run to `results/EXPERIMENTS.csv`,
    which **is committed** — so either machine sees all numbers through git
    without any store to merge.

Why a SQLite backend: MLflow 3.x put the plain-file store in maintenance mode
and refuses it without an env-var opt-out. SQLite is a single local file, which
suits "local store + committed export" fine.

ENVIRONMENT WARNING: installing mlflow upgrades **protobuf**, which *breaks
MediaPipe* — `SymbolDatabase.GetPrototype` disappears and every Holistic call on
a real frame dies. A blank-frame smoke test does NOT catch it (no landmark
protos are built). Keep `protobuf==3.20.3` pinned; mlflow 3.15 works with it.

---
The three-axis discipline from `docs/methodology/07_evaluation.md` is *enforced*
here: `start_run` requires `dataset`, `split_kind` and `cues`, because a number
without those three is uninterpretable in this project.

    from fusion.tracking import start_run

    with start_run("04_recombination", "attn_recomb_seed0",
                   dataset="final_merged", split_kind="scenarios", cues="real",
                   params={"seed": 0, "dropout_p": 0.3}) as run:
        ...
        run.log_metrics({"test_clip_acc": 0.41, "test_clip_macro_f1": 0.38})
        run.log_checkpoint(ckpt_path)
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB_URI = "sqlite:///" + (REPO / "mlruns.db").as_posix()
ARTIFACT_ROOT = (REPO / "mlruns").as_uri()

# The experiments (groups of runs) this project uses.
EXPERIMENTS = (
    "01_baselines",       # rule-based, unimodal, concat-MLP
    "02_fusion",          # attention fusion + ablations
    "03_diagnostics",     # oracle/gap decomposition, clip-vs-window, sweeps
    "04_recombination",   # rubric-driven augmentation
)

DATASETS = ("old", "final", "final_merged")
SPLIT_KINDS = ("people", "scenarios")     # what is held out in the test split
CUE_SOURCES = ("real", "oracle")          # model predictions vs table labels


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        return False


def checkpoint_hashes() -> dict:
    """The deployed-checkpoint manifest, so a run records which weights fed it."""
    man = REPO / "docs" / "checkpoint_manifest.sha256"
    out = {}
    if man.exists():
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            h, _, path = line.partition(" ")
            out[Path(path.strip().lstrip("*")).name] = h[:12]
    return out


class Run:
    """Thin wrapper so callers never import mlflow directly."""

    def __init__(self, mlflow):
        self._mlflow = mlflow

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params({k: v for k, v in params.items() if v is not None})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items()
                 if v is not None and not isinstance(v, str)}
        self._mlflow.log_metrics(clean, step=step)

    def log_tags(self, tags: dict) -> None:
        self._mlflow.set_tags(tags)

    def log_checkpoint(self, path) -> None:
        """Fusion checkpoints are ~288 KB, small enough to store directly."""
        self._mlflow.log_artifact(str(path), artifact_path="checkpoint")

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, obj: dict, name: str) -> None:
        self._mlflow.log_dict(obj, name)

    def log_table(self, df, name: str) -> None:
        """Log a DataFrame (per-row accuracy, masking sweep, ...) as CSV."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            df.to_csv(p, index=False)
            self._mlflow.log_artifact(str(p), artifact_path="tables")


@contextmanager
def start_run(experiment: str, run_name: str, *, dataset: str, split_kind: str,
              cues: str, params: dict | None = None, tags: dict | None = None,
              notes: str | None = None):
    """Open an MLflow run with this project's mandatory metadata.

    `dataset`, `split_kind` and `cues` are required: methodology 07 §7.1 says no
    number in this project is interpretable without all three, so the tracker
    refuses to record one that omits them.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    if split_kind not in SPLIT_KINDS:
        raise ValueError(f"split_kind must be one of {SPLIT_KINDS}, got {split_kind!r}")
    if cues not in CUE_SOURCES:
        raise ValueError(f"cues must be one of {CUE_SOURCES}, got {cues!r}")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(DB_URI)
    try:
        mlflow.set_experiment(experiment)
    except Exception:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"dataset": dataset, "split_kind": split_kind,
                           "cues": cues, **(params or {})})
        mlflow.set_tags({
            "git_commit": _git_commit(),
            "git_dirty": str(_git_dirty()),
            "machine": platform.node(),
            "checkpoints": json.dumps(checkpoint_hashes()),
            **({"notes": notes} if notes else {}),
            **(tags or {}),
        })
        yield Run(mlflow)
