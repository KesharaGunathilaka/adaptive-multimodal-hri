"""Cue attribution report — the CAM-analogue for `AttentionFusion` (see
`fusion/model/attribution.py`'s docstring for why literal Class Activation
Mapping does not apply to a 24-dim probability-vector input, and why
attention-weight extraction is the faithful transplant of the idea).

Two outputs:
  1. AGGREGATE attribution — mean CLS->modality attention, overall and broken
     down by predicted intent. Answers "which cue does the model lean on for
     each decision type" — the thesis figure `06_fusion_model.md` flagged as
     unexercised.
  2. QUALITATIVE spotlight — the same gesture-conflict rows
     `scripts/42_scenario_test_report.py`'s T02 used (same gesture, different
     emotion -> different intent). Shows whether the model's attention
     visibly SHIFTS toward the disambiguating cue (usually emotion) on these
     rows, which is the concrete "the network is not a black box" evidence
     rules' interpretability advantage is meant to beat.

    .venv/Scripts/python scripts/47_cue_attribution.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model import train as T  # noqa: E402
from fusion.model.attribution import TOKEN_NAMES, batched_cls_attention  # noqa: E402
from fusion.model.recombine_merged import build_pools, generate  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SEEDS = (0, 1, 2)
OUT_DIR = G.OUT_DIR
FULL_CFG = dict(dropout_p=0.3, jitter_sigma=0.15, select_masked=True)

# same conflict families as scripts/42_scenario_test_report.py's T02, trimmed
# to the clearest 2-3 rows per gesture for a compact qualitative table
SPOTLIGHT_ROWS = {
    "thumbs_down": [8, 15, 17, 33],     # sad->F04, angry->F07, disgust->F08, happy->F01
    "both hands up": [3, 14, 26],       # surprise/fear->F02, angry->F07, neutral->F05
    "wave": [1, 13, 22],                # happy->F01, angry->F06, missing-emo->F01
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    splits = {s: real[real.split == s] for s in ("train", "val", "test")}
    hl = real[(real.split == "test") & real.headline_eval].copy()
    print(f"{clips.clip_id.nunique()} clips · headline test {len(hl)}", flush=True)

    pools = build_pools(real[real.split == "train"], clips)
    X, obs, y, rep = generate(pools, n_per_combo=100, seed=0)
    print(f"recombination: {rep.n_generated} synthetic samples", flush=True)

    print("training `full` model (3 seeds, incumbent architecture)...", flush=True)
    models = []
    for seed in SEEDS:
        m, va = T.train_fusion(splits, seed=seed, dropout_p=FULL_CFG["dropout_p"],
                               jitter_sigma=FULL_CFG["jitter_sigma"],
                               extra=(X, obs, y), device=device,
                               missing_mode="exclude",
                               select_masked=FULL_CFG["select_masked"])
        models.append(m)
        print(f"  seed{seed}: val={va:.4f}", flush=True)

    Xh, obsh = T.frame_arrays(hl)
    y_true = hl.y.to_numpy()

    # ── 1. aggregate attribution (average over the 3-seed ensemble) ────────
    all_w = torch.stack([batched_cls_attention(m, Xh, obsh, device) for m in models])
    w = all_w.mean(dim=0)                       # [N, n_layers, 5]
    last_layer = w[:, -1, 1:].numpy()            # [N, 4] -- modality weights only

    overall = {m: round(float(last_layer[:, i].mean()), 4)
              for i, m in enumerate(TOKEN_NAMES[1:])}
    print(f"\noverall attribution (last layer): {overall}", flush=True)

    per_intent = {}
    for intent_idx, intent_name in enumerate(common.INTENTS):
        sel = y_true == intent_idx
        if sel.sum() == 0:
            continue
        per_intent[intent_name] = {m: round(float(last_layer[sel, i].mean()), 4)
                                   for i, m in enumerate(TOKEN_NAMES[1:])}

    # ── 2. qualitative spotlight on conflict rows ───────────────────────────
    hl_rows = hl.copy()
    spot_records = []
    for gesture, rows in SPOTLIGHT_ROWS.items():
        for v3row in rows:
            sub = hl_rows[hl_rows.v3_row == v3row]
            if sub.empty:
                continue
            idx = sub.index
            pos = [hl_rows.index.get_loc(i) for i in idx]
            attn = last_layer[pos].mean(axis=0)
            meta = clips[clips.v3_row == v3row].iloc[0]
            spot_records.append({
                "gesture": gesture, "v3_row": int(v3row), "context": meta.context,
                "emotion": meta.emotion_v3, "motion": meta.motion_v3,
                "intent": meta.intent, "n_clips": len(sub),
                "attn_emotion": round(float(attn[0]), 3),
                "attn_gesture": round(float(attn[1]), 3),
                "attn_motion": round(float(attn[2]), 3),
                "attn_context": round(float(attn[3]), 3),
            })

    results = {"overall": overall, "per_intent": per_intent, "spotlight": spot_records}
    (OUT_DIR / "cue_attribution_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    write_report(overall, per_intent, spot_records)
    print(f"\n({time.time()-t0:.0f}s) -> {OUT_DIR / 'CUE_ATTRIBUTION.md'}")


def write_report(overall, per_intent, spot):
    L = [
        "# Cue attribution — the CAM-analogue for AttentionFusion",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged` · "
        "headline test · self-attention `full` model, 3-seed ensemble mean · "
        "CLS token's last-layer attention weight onto each modality token "
        "(`fusion/model/attribution.py`).",
        "",
        "**Note on 'CAM' terminology**: this is NOT Class Activation Mapping "
        "(that needs spatial feature maps a 24-dim vector doesn't have). It is "
        "the model's own attention weights — the mixing coefficients the "
        "classifier head actually reads from, extracted directly rather than "
        "approximated. `fusion_zoo.py`'s `ChannelAttentionFusion` is the "
        "unrelated SE/CBAM-style head that happened to share the initials.",
        "",
        "## Overall — which cue does the model lean on, on average?",
        "",
        "| Modality | Mean attention |",
        "|---|---|",
    ]
    for m, v in overall.items():
        L.append(f"| {m} | {v} |")

    L += ["", "## Per predicted intent", "",
         "| Intent | emotion | gesture | motion | context |",
         "|---|---|---|---|---|"]
    for intent, d in per_intent.items():
        L.append(f"| {intent} | {d['emotion']} | {d['gesture']} | "
                 f"{d['motion']} | {d['context']} |")

    L += ["", "## Qualitative spotlight — same gesture, different emotion",
         "",
         "Does the model's attention visibly shift toward the disambiguating "
         "cue when the same gesture means different things?", "",
         "| Gesture | Row | Context | Emotion | Motion | Intent | n | "
         "attn(emo) | attn(ges) | attn(mot) | attn(ctx) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in spot:
        L.append(f"| {r['gesture']} | #{r['v3_row']} | {r['context']} | "
                 f"{r['emotion']} | {r['motion']} | {r['intent']} | {r['n_clips']} | "
                 f"{r['attn_emotion']} | {r['attn_gesture']} | {r['attn_motion']} | "
                 f"{r['attn_context']} |")
    (OUT_DIR / "CUE_ATTRIBUTION.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
