"""How much intent information does the V3 table itself carry, per masking condition?

Fusion cannot beat the label table. If two scenario rows share a cue tuple but
carry different intents, no model reading only those cues can separate them — the
best possible strategy is to always answer the majority intent of that group.
This script computes that **table-imposed ceiling**, weighted by how many clips
each row actually has on disk, for:

  * all four cues present,
  * each single cue masked (T03 single-drop),
  * each pair masked (T03 pair-drop).

It also lists every ambiguous cue tuple, which is what turns a disappointing
masking number into a documented, expected one: a fusion model scoring 0.67 with
emotion masked against a 0.75 ceiling is near-optimal, not broken.

    .venv/Scripts/python scripts/17_cue_ambiguity.py --context classroom
    .venv/Scripts/python scripts/17_cue_ambiguity.py            # both contexts
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.final_common import (CLIPS_CSV, SCENARIOS_CSV,  # noqa: E402
                                                 OUT_ROOT)

CUES = ["context", "emotion_v3", "gesture_v3", "motion_v3"]
SHORT = {"context": "context", "emotion_v3": "emotion",
         "gesture_v3": "gesture", "motion_v3": "motion"}
OUT_DIR = OUT_ROOT / "ambiguity"


def clip_counts(view: str | None, context: str | None) -> pd.Series:
    """Clips per V3 row on disk — the weights for the ceiling."""
    cl = pd.read_csv(CLIPS_CSV)
    cl = cl[cl.v3_row.notna()]
    if view:
        cl = cl[cl.view == view]
    if context:
        cl = cl[cl.context == context]
    return cl.groupby(cl.v3_row.astype(int)).size()


def ceiling(rows: pd.DataFrame, keep: list[str]) -> tuple[float | None, int]:
    """Best achievable clip accuracy given only `keep`, plus clips lost to ties."""
    total = int(rows.n_clips.sum())
    if total == 0:
        return None, 0
    best = sum(g.groupby("intent").n_clips.sum().max() for _, g in rows.groupby(keep))
    return round(best / total, 4), total - int(best)


def ambiguous(rows: pd.DataFrame, keep: list[str]) -> list[dict]:
    out = []
    for key, g in rows.groupby(keep):
        by_intent = g.groupby("intent").n_clips.sum()
        if len(by_intent) < 2:
            continue
        out.append({
            "cues": dict(zip(keep, key if isinstance(key, tuple) else (key,))),
            "v3_rows": sorted(int(r) for r in g.v3_row),
            "intents": {k: int(v) for k, v in by_intent.items()},
            "clips_lost": int(by_intent.sum() - by_intent.max()),
        })
    return sorted(out, key=lambda d: -d["clips_lost"])


def analyse(rows: pd.DataFrame) -> dict:
    res = {"n_rows": int(len(rows)), "n_clips": int(rows.n_clips.sum()),
           "conditions": {}, "ambiguities": {}}
    conds = [("none", CUES)]
    conds += [(SHORT[c], [k for k in CUES if k != c]) for c in CUES]
    conds += [(f"{SHORT[a]}+{SHORT[b]}", [k for k in CUES if k not in (a, b)])
              for a, b in itertools.combinations(CUES, 2)]
    for name, keep in conds:
        acc, lost = ceiling(rows, keep)
        res["conditions"][name] = {"ceiling": acc, "clips_unreachable": lost}
        amb = ambiguous(rows, keep)
        if amb:
            res["ambiguities"][name] = amb
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default=None, help="classroom | kitchen")
    ap.add_argument("--view", default="realsense_480p",
                    help="weight by this view's clip counts ('' = all views)")
    args = ap.parse_args()

    v3 = pd.read_csv(SCENARIOS_CSV)
    if args.context:
        v3 = v3[v3.context == args.context]
    counts = clip_counts(args.view or None, args.context)
    v3 = v3.assign(n_clips=v3.v3_row.map(counts).fillna(0).astype(int))

    recorded = v3[v3.n_clips > 0]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "context": args.context or "all", "view": args.view or "all",
           "designed_all_rows": analyse(v3.assign(n_clips=1)),
           "recorded": {s: analyse(g) for s, g in recorded.groupby("split_design")}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.context or "all"
    (OUT_DIR / f"ceiling_{tag}.json").write_text(json.dumps(out, indent=2))

    L = [f"# Table-imposed intent ceiling — {tag}", "",
         f"Generated {out['generated']} · weights = clips on disk "
         f"(view: {out['view']})", "",
         "A fusion model reading only the cues left in a masking condition cannot "
         "exceed these numbers, because the V3 table maps the same cue tuple to "
         "different intents on some rows. Compare every masking result against "
         "the matching ceiling before calling a model weak.", ""]
    for split, r in out["recorded"].items():
        L += [f"## Recorded `{split}` rows — {r['n_rows']} rows, {r['n_clips']} clips",
              "", "| Cue(s) masked | Ceiling | Clips unreachable |", "|---|---|---|"]
        for name, c in r["conditions"].items():
            L.append(f"| {name} | {c['ceiling']} | {c['clips_unreachable']} |")
        L.append("")
        for name, amb in r["ambiguities"].items():
            L += [f"**Ambiguous under `{name}`**", ""]
            for a in amb:
                cues = ", ".join(f"{k.replace('_v3', '')}={v}" for k, v in a["cues"].items())
                L.append(f"- rows {a['v3_rows']} → {a['intents']} "
                         f"({cues}) — {a['clips_lost']} clips unreachable")
            L.append("")

    (OUT_DIR / f"CEILING_{tag}.md").write_text("\n".join(L), encoding="utf8")
    print(f"-> {OUT_DIR / f'CEILING_{tag}.md'}")
    for split, r in out["recorded"].items():
        print(f"\n{split}: {r['n_rows']} rows / {r['n_clips']} clips")
        for name, c in r["conditions"].items():
            print(f"  mask {name:18s} ceiling={c['ceiling']}  "
                  f"unreachable={c['clips_unreachable']}")


if __name__ == "__main__":
    main()
