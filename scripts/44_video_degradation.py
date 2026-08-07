"""Phase 2 robustness study — REAL pixel degradation, not simulated cue noise.

`scripts/43_robustness_battery.py` (Phase 1, part B) corrupted PROBABILITY
VECTORS with Gaussian log-prob noise -- a clean, controlled proxy, but a
proxy. It also inherits a documented honesty problem: the V3 table's
"designed-missing" rows were checked in `scripts/42_scenario_test_report.py`
and found to have 100% detection anyway -- nothing about them was actually
hard for the detectors. Neither experiment puts real sensor degradation in
front of the actual perception stack.

This script does that: blur / darken / downsample-then-upsample / JPEG-
recompress the ACTUAL VIDEO FRAMES before Holistic, face detection, or CLIP
ever see them (`fusion/extraction/perframe.py`'s new `frame_transform` hook),
run the full Pass-1 -> Pass-2 pipeline on the corrupted pixels, pool to
clip-level, and compare fusion vs rules on cues that were genuinely harder to
read -- not cues that were merely told they were harder to read.

SCOPE: a stratified SUBSAMPLE of the headline test clips (`N_CLIPS`, default
70), sampled once (fixed seed) and reused identically across every condition.

RUN AS ONE SUBPROCESS PER CONDITION (2026-08-06 crash post-mortem): the first
version of this script loaded one `PerFrameExtractor`/`WindowFeaturizer` and
looped over all 12 conditions in a single process. It leaked memory across
~430 clip decodes (MediaPipe Holistic + OpenCV + CUDA context accumulating
with no release point) and crashed with an OOM inside OpenCV followed by a
segfault -- and lost every completed condition's results, because they were
only written to disk at the very end. Fixed by:
  * `--condition NAME` runs exactly one condition as its own OS process, so
    all resources are guaranteed released when it exits (no cross-condition
    accumulation possible);
  * each condition's result is written to its own JSON file IMMEDIATELY on
    completion (`video_degradation_<condition>.json`), so a crash in
    condition N does not lose conditions 1..N-1;
  * the 3-seed `full` fusion model is trained once and cached to disk
    (`_phase2_full_model.pt`) so 8 conditions don't each pay ~2 min retrain
    cost redundantly;
  * `--combine` merges whatever per-condition JSON files exist into the final
    report -- safe to run after a partial crash to see what was recovered.

    .venv/Scripts/python scripts/44_video_degradation.py --condition clean
    .venv/Scripts/python scripts/44_video_degradation.py --condition blur_9
    ...
    .venv/Scripts/python scripts/44_video_degradation.py --combine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.extraction.perframe import PerFrameExtractor  # noqa: E402
from fusion.extraction.windows import WindowFeaturizer  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval.merged_gap import rule_predict_fair  # noqa: E402
from scripts.realworld_eval.merged_common import CLIPS_ROOT  # noqa: E402

OUT_DIR = G.OUT_DIR
SEED = 0
N_CLIPS = 70                            # trimmed from 144 -- see script docstring
SEEDS = (0, 1, 2)
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True)
MODEL_CACHE = OUT_DIR / "_phase2_full_model.pt"
RESULT_PREFIX = "video_degradation_"


def make_transform(kind: str, severity: float):
    if kind == "clean":
        return None
    if kind == "blur":
        k = int(severity)
        k = k if k % 2 == 1 else k + 1
        return lambda f: cv2.GaussianBlur(f, (k, k), 0)
    if kind == "dark":
        return lambda f: np.clip(f.astype(np.float32) * severity, 0, 255).astype(np.uint8)
    if kind == "downsample":
        s = severity
        def _fn(f, s=s):
            h, w = f.shape[:2]
            small = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return _fn
    if kind == "jpeg":
        q = int(severity)
        def _fn(f, q=q):
            ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
            return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else f
        return _fn
    raise ValueError(kind)


# trimmed from 12 to 8 conditions to fit a sane wall-clock budget (see
# docstring's crash post-mortem -- ~16s/clip observed, so 8 conditions x 70
# clips is ~2.5h sequential, tractable as background jobs; 12x144 was ~8h+)
CONDITIONS = [
    ("clean", None),
    ("blur", 9), ("blur", 21),
    ("dark", 0.35), ("dark", 0.15),
    ("downsample", 0.35), ("downsample", 0.15),
    ("jpeg", 10),
]
COND_LABELS = {("clean" if k == "clean" else f"{k}_{s}"): (k, s) for k, s in CONDITIONS}


def pick_subsample(clips: pd.DataFrame, hl: pd.DataFrame) -> pd.DataFrame:
    rng_seed = SEED
    pool = clips[clips.clip_id.isin(set(hl.clip_id))].copy()
    per_class = max(1, N_CLIPS // pool.intent.nunique())
    picks = []
    for _, g in pool.groupby("intent"):
        n = min(len(g), per_class)
        picks.append(g.sample(n=n, random_state=rng_seed))
    out = pd.concat(picks)
    if len(out) > N_CLIPS:
        out = out.sample(n=N_CLIPS, random_state=rng_seed)
    return out


def clip_level_table(sub_clips, extractor, featurizer, transform) -> pd.DataFrame:
    rows = []
    for r in sub_clips.itertuples():
        try:
            npz = extractor.extract_clip(CLIPS_ROOT / r.filepath, frame_transform=transform)
            wins = featurizer.featurize_clip(npz)
        except Exception as e:  # pragma: no cover
            print(f"    FAILED {r.clip_id}: {e}", flush=True)
            continue
        row = {"clip_id": r.clip_id, "v3_row": r.v3_row, "context": r.context,
              "intent": r.intent}
        for pref, key, cols in (("emo", "emo_probs", common.EMO_COLS),
                               ("ges", "ges_probs", common.GES_COLS),
                               ("mot", "mot_probs", common.MOT_COLS),
                               ("ctx", "ctx_probs", common.CTX_COLS)):
            vecs = [w[key] for w in wins if w.get(key) is not None]
            if vecs:
                row.update(zip(cols, np.mean(vecs, axis=0)))
                row[f"{pref}_obs"] = 1.0
            else:
                row.update({c: 0.0 for c in cols})
                row[f"{pref}_obs"] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df["y"] = df.intent.map(common.INTENTS.index)
    return df


def get_or_train_models(splits, extra, device):
    if MODEL_CACHE.exists():
        print(f"loading cached `full` models from {MODEL_CACHE}", flush=True)
        states = torch.load(MODEL_CACHE, map_location=device, weights_only=True)
        models = []
        for sd in states:
            m = AttentionFusion(missing_mode="exclude").to(device)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
        return models
    print("training `full` fusion model (3 seeds, no cache found)...", flush=True)
    models = []
    for seed in SEEDS:
        m, va = T.train_fusion(splits, seed=seed, dropout_p=FULL_CFG["dropout_p"],
                               jitter_sigma=FULL_CFG["jitter_sigma"], extra=extra,
                               device=device, missing_mode="exclude",
                               select_masked=FULL_CFG["select_masked"])
        models.append(m)
        print(f"  seed{seed}: val={va:.4f}", flush=True)
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save([m.state_dict() for m in models], MODEL_CACHE)
    return models


def run_one_condition(label: str, device) -> None:
    kind, severity = COND_LABELS[label]
    t0 = time.time()

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval]
    sub = pick_subsample(clips, hl)
    print(f"[{label}] {len(sub)} clips (stratified, of {len(hl)} headline test)",
         flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    Xr, obsr, yr, _ = generate(pools, n_per_combo=100, seed=0)
    models = get_or_train_models(splits, (Xr, obsr, yr), device)

    print(f"[{label}] loading extractor + featurizer...", flush=True)
    extractor = PerFrameExtractor(device=device)
    featurizer = WindowFeaturizer(device=device)

    tf = make_transform(kind, severity) if kind != "clean" else None
    tbl = clip_level_table(sub, extractor, featurizer, tf)
    if tbl.empty:
        print(f"[{label}] NO CLIPS EXTRACTED", flush=True)
        return

    X, obs = T.frame_arrays(tbl)
    y_true = tbl.y.to_numpy()
    preds = np.stack([T._eval_arrays(m, X, obs, device) for m in models])
    fusion_pred = np.array([np.bincount(preds[:, i], minlength=10).argmax()
                            for i in range(preds.shape[1])])
    rule_pred = rule_predict_fair(tbl, use_predicted_context=False)

    row = {
        "condition": label, "kind": kind, "severity": severity,
        "n_clips": len(tbl),
        "fusion_acc": round(float((fusion_pred == y_true).mean()), 4),
        "rule_acc": round(float((rule_pred == y_true).mean()), 4),
        "emo_obs_rate": round(float(tbl.emo_obs.mean()), 3),
        "ges_obs_rate": round(float(tbl.ges_obs.mean()), 3),
        "mot_obs_rate": round(float(tbl.mot_obs.mean()), 3),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{RESULT_PREFIX}{label}.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8")
    print(f"[{label}] DONE n={row['n_clips']} fusion={row['fusion_acc']} "
         f"rules={row['rule_acc']} obs={row['emo_obs_rate']}/"
         f"{row['ges_obs_rate']}/{row['mot_obs_rate']} ({row['elapsed_s']:.0f}s)",
         flush=True)


def combine() -> None:
    results = []
    for label in COND_LABELS:
        p = OUT_DIR / f"{RESULT_PREFIX}{label}.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
    if not results:
        print("no per-condition result files found yet", flush=True)
        return
    order = list(COND_LABELS)
    results.sort(key=lambda r: order.index(r["condition"]))
    (OUT_DIR / "video_degradation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, N_CLIPS)
    print(f"combined {len(results)}/{len(COND_LABELS)} conditions -> "
         f"{OUT_DIR / 'VIDEO_DEGRADATION.md'}", flush=True)


def write_report(results, n_sub):
    L = [
        "# Phase 2 — real video degradation (not simulated)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        f"stratified subsample of **{n_sub} clips per condition** (of the "
        "979-clip headline test set — full re-extraction across every "
        "condition was not tractable; see script docstring, including a "
        "2026-08-06 crash post-mortem: the first version leaked memory across "
        "conditions and lost results on crash, fixed by running one "
        "subprocess per condition with incremental result writing) · same "
        "`full`-recipe fusion model as elsewhere (cached across conditions), "
        "3-seed majority vote.",
        "",
        "Unlike `ROBUSTNESS_BATTERY.md` part B (which corrupts probability "
        "vectors post-hoc), this corrupts the ACTUAL VIDEO PIXELS before "
        "Holistic/face-detection/CLIP ever run.",
        "",
        f"**{len(results)}/{len(COND_LABELS)} conditions completed.**",
        "",
        "| Condition | n | Fusion acc | Rule acc | emo obs | ges obs | mot obs |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        L.append(f"| {r['condition']} | {r['n_clips']} | {r['fusion_acc']} | "
                 f"{r['rule_acc']} | {r['emo_obs_rate']} | {r['ges_obs_rate']} | "
                 f"{r['mot_obs_rate']} |")
    if results:
        clean = next((r for r in results if r["condition"] == "clean"), results[0])
        others = [r for r in results if r is not clean]
        if others:
            worst = min(others, key=lambda r: r["fusion_acc"] - r["rule_acc"])
            L += ["",
                 f"Clean baseline: fusion {clean['fusion_acc']} vs rules "
                 f"{clean['rule_acc']} (delta "
                 f"{clean['fusion_acc']-clean['rule_acc']:+.4f}).",
                 f"Largest fusion-vs-rules gap under degradation: "
                 f"`{worst['condition']}` — fusion {worst['fusion_acc']} vs "
                 f"rules {worst['rule_acc']} "
                 f"(delta {worst['fusion_acc']-worst['rule_acc']:+.4f})."]
    (OUT_DIR / "VIDEO_DEGRADATION.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(COND_LABELS))
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("\n".join(COND_LABELS))
        return
    if args.combine:
        combine()
        return
    if not args.condition:
        raise SystemExit("pass --condition NAME (--list to see names) or --combine")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_one_condition(args.condition, device)


if __name__ == "__main__":
    main()
