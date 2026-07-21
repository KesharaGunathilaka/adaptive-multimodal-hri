"""G3 spotlight (handover §8.3): same gesture, different intent. Shows the
deployed fusion model's per-scenario predictions for the gesture families
where emotion/context/motion flips the meaning.

Real recorded scenarios are evaluated clip-level on TEST-subject clips (fall
back to val/train subjects when a scenario has no test-subject clips — marked
in the table). Unrecorded V3 rows are evaluated on synthetic cue combinations
(recombination pools, 200 samples).

Writes results/fusion_v1/G3_SPOTLIGHT.md.

Run from repo root:  .venv/Scripts/python scripts/09_g3_spotlight.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.baselines import common  # noqa: E402
from fusion.model import recombine  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from fusion.model.train import _eval_arrays, frame_arrays  # noqa: E402

OUT = ROOT / "results" / "fusion_v1" / "G3_SPOTLIGHT.md"

# (family, source, id, cue summary, expected intent)
# source: 'real' -> scenario_id in parquet; 'synth' -> V3 row in recombine.SYNTH_ROWS
SPOTLIGHT = [
    ("raise hand", "real",  "S01_F04", "classroom + neutral + sit", "F04"),
    ("raise hand", "real",  "S11_F05", "classroom + happy + sit",   "F05"),
    ("thumbs down", "real", "S04_F04", "classroom + sad + sit",     "F04"),
    ("thumbs down", "synth", 15,       "classroom + angry + sit",   "F07"),
    ("thumbs down", "synth", 17,       "classroom + disgust + sit", "F08"),
    ("thumbs down", "synth", 33,       "kitchen + happy + stand",   "F01"),
    ("thumbs down", "real", "S21_F04", "kitchen + sad + sit",       "F04"),
    ("wave", "real",  "S02_F01", "classroom + happy + walk",        "F01"),
    ("wave", "real",  "S25_F09", "kitchen + happy + walk (exit)",   "F09"),
    ("wave", "synth", 13,       "classroom + angry + stand",        "F06"),
    ("both hands up", "real", "S09_F02", "classroom + surprise + stand", "F02"),
    ("both hands up", "real", "S05_F02", "classroom + angry + stand",    "F07"),
    ("both hands up", "real", "S24_F07", "kitchen + angry + stand",      "F07"),
    ("both hands up", "real", "S19_F02", "kitchen + fear + step back",   "F02"),
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionFusion(missing_mode="exclude").to(device)
    model.load_state_dict(torch.load(
        ROOT / "jetson_deploy" / "fusion" / "fusion_attn.pt",
        map_location=device, weights_only=True))
    model.eval()

    df = pd.read_parquet(common.FEATURES)
    df["y"] = df.intent.map(common.INTENTS.index)
    full = df.copy()

    # synthetic samples per needed row
    synth_rows = {r[0]: r for r in recombine.SYNTH_ROWS}
    lines = ["# G3 spotlight — same gesture, different intent",
             "", "Model: jetson_deploy/fusion/fusion_attn.pt (attn exclude, deployed).",
             "", "| family | case | cues | expected | predicted | agreement | source |",
             "|---|---|---|---|---|---|---|"]

    for fam, src, ident, cues, expected in SPOTLIGHT:
        if src == "real":
            g = df[df.scenario_id == ident]
            sub, note = g[g.split_subject == "test"], "real (test subjects)"
            if sub.empty:
                sub, note = g[g.split_subject == "val"], "real (val subjects)"
            if sub.empty:
                sub, note = g, "real (train subjects!)"
            pred = _eval_arrays(model, *frame_arrays(sub), device)
            ct, cp = common.clip_vote(sub, pred)
            votes = Counter(common.INTENTS[i] for i in cp)
            top, n = votes.most_common(1)[0]
            agree = n / len(cp)
            lines.append(f"| {fam} | {ident} | {cues} | {expected} "
                         f"| {top} ({n}/{len(cp)} clips) | {agree:.2f} | {note} |")
        else:
            row = synth_rows[ident]
            X, obs, y, _ = _generate_single(full, row)
            with torch.no_grad():
                p = model(torch.from_numpy(X).to(device),
                          torch.from_numpy(obs).to(device))
                pi = p.argmax(1).cpu().numpy()
            votes = Counter(common.INTENTS[i] for i in pi)
            top, n = votes.most_common(1)[0]
            lines.append(f"| {fam} | V3 #{ident} | {cues} | {expected} "
                         f"| {top} ({n}/{len(pi)} samples) | {n/len(pi):.2f} | synthetic |")

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _generate_single(full_df, row):
    import fusion.model.recombine as R
    saved = R.SYNTH_ROWS
    R.SYNTH_ROWS = [row]
    try:
        return R.generate(full_df, n_per_row=200, seed=1)
    finally:
        R.SYNTH_ROWS = saved


if __name__ == "__main__":
    main()
