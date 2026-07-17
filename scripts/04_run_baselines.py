"""Run all fusion baselines (rule-based, 4x unimodal logreg, concat-MLP x3 seeds)
and write fusion/baselines/results.json + RESULTS.md.

Run from repo root:  .venv/Scripts/python scripts/04_run_baselines.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common, learned, rule_based  # noqa: E402

OUT = ROOT / "fusion" / "baselines"


def main() -> None:
    df = common.load_windows()
    splits = common.split(df)
    print({k: (v.clip_id.nunique(), len(v)) for k, v in splits.items()})

    results = {}

    results["rule_based"] = {
        name: common.evaluate(f, rule_based.predict(f))
        for name, f in splits.items()}
    print("rule_based:", results["rule_based"]["test"])

    for mod in common.MODALITY_COLS:
        preds = learned.unimodal_logreg(splits, mod)
        results[f"unimodal_{mod}"] = {
            name: common.evaluate(f, preds[name]) for name, f in splits.items()}
        print(f"unimodal_{mod}:", results[f"unimodal_{mod}"]["test"])

    seed_runs = []
    for seed in (0, 1, 2):
        preds, val_acc = learned.train_concat_mlp(splits, seed=seed)
        seed_runs.append({name: common.evaluate(f, preds[name])
                          for name, f in splits.items()})
        print(f"concat_mlp seed{seed}: val_win_acc={val_acc:.4f}",
              seed_runs[-1]["test"])
    results["concat_mlp_seeds"] = seed_runs
    agg = {}
    for split_name in ("train", "val", "test"):
        for lvl in ("window", "clip"):
            for m in ("acc", "macro_f1"):
                vals = [r[split_name][lvl][m] for r in seed_runs]
                agg[f"{split_name}.{lvl}.{m}"] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4)}
    results["concat_mlp"] = agg

    results["_meta"] = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "features": "features_v1.parquet",
                        "note": ("val/test are actor-disjoint subject splits of the 22 "
                                 "recorded TRAIN scenarios; the V3 test scenarios are "
                                 "not recorded yet. F10 has no supervised rows "
                                 "(S28 pooled).")}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Baseline results (features_v1, window-level & clip-level)", "",
             results["_meta"]["note"], "",
             "| model | test win acc | test win F1 | test clip acc | test clip F1 |",
             "|---|---|---|---|---|"]
    for key in ["rule_based", "unimodal_emotion", "unimodal_gesture",
                "unimodal_motion", "unimodal_context"]:
        t = results[key]["test"]
        lines.append(f"| {key} | {t['window']['acc']:.3f} | {t['window']['macro_f1']:.3f} "
                     f"| {t['clip']['acc']:.3f} | {t['clip']['macro_f1']:.3f} |")
    a = results["concat_mlp"]
    lines.append(
        f"| concat_mlp (3 seeds) | {a['test.window.acc']['mean']:.3f}±{a['test.window.acc']['std']:.3f} "
        f"| {a['test.window.macro_f1']['mean']:.3f}±{a['test.window.macro_f1']['std']:.3f} "
        f"| {a['test.clip.acc']['mean']:.3f}±{a['test.clip.acc']['std']:.3f} "
        f"| {a['test.clip.macro_f1']['mean']:.3f}±{a['test.clip.macro_f1']['std']:.3f} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT / 'results.json'} and RESULTS.md")


if __name__ == "__main__":
    main()
