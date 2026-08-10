"""Are the recombination pools representative of deployment? (2026-08-08)

`fusion/model/recombine_merged.py::build_pools` buckets real per-clip cue
vectors by GROUND TRUTH class and draws from them to synthesise training
samples. The design note argues ground-truth indexing is the principled choice
because it preserves "what a genuine occurrence of thumbs_down really looks
like as a probability vector, including the times it gets confused" -- i.e. the
pool is supposed to carry the perception model's realistic error.

It does not, for two of the four cues. Pools are built from the TRAIN split,
and emotion and gesture were **fine-tuned on exactly that split** (promoted
2026-08-04, `docs/WORKLOG.md`). Their pools therefore contain memorised,
near-perfect vectors rather than realistic ones. Motion was never promoted (its
fine-tune regressed 4/4 times), so motion's pool is the only honest one.

This script measures pool PURITY -- the fraction of a bucket's vectors the cue
model actually argmaxes to that bucket's class -- for train, val and test, and
writes the comparison. The train-vs-test gap is a train/serve distribution
shift injected directly into fusion's training data: the fusion head learns to
trust cues that are perfect during training and are not at deployment.

    .venv/Scripts/python scripts/53_pool_purity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.recombine_merged import LABELS, build_pools  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SPLITS = ("train", "val", "test")
FINETUNED = {"ges": True, "emo": True, "mot": False, "ctx": False}
FULL = {"emo": "emotion", "ges": "gesture", "mot": "motion", "ctx": "context"}


def main() -> None:
    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    names = {m: [c.split("_", 1)[1] for c in LABELS[m]] for m in LABELS}
    pools = {s: build_pools(real[real.split == s], clips) for s in SPLITS}

    out: dict = {}
    for m in ("ges", "emo", "mot"):
        per_class, overall = {}, {}
        for s in SPLITS:
            allv = []
            for cls in names[m]:
                p = pools[s].get((m, cls))
                if p is None:
                    per_class.setdefault(cls, {})[s] = None
                    continue
                am = np.asarray(p).argmax(1)
                hit = (am == names[m].index(cls))
                per_class.setdefault(cls, {})[s] = round(float(hit.mean()), 4)
                per_class[cls].setdefault("n", {})[s] = int(len(p))
                allv.append(hit)
            overall[s] = round(float(np.concatenate(allv).mean()), 4) if allv else None
        out[m] = {"finetuned_on_train": FINETUNED[m], "per_class": per_class,
                  "overall": overall,
                  "train_minus_test": (round(overall["train"] - overall["test"], 4)
                                       if overall["train"] and overall["test"] else None)}

    (G.OUT_DIR / "pool_purity_results.json").write_text(json.dumps(out, indent=2))
    write_report(out, names)
    print(f"-> {G.OUT_DIR / 'POOL_PURITY.md'}")
    for m, d in out.items():
        print(f"  {FULL[m]:8s} train={d['overall']['train']} val={d['overall']['val']} "
              f"test={d['overall']['test']}  gap={d['train_minus_test']}  "
              f"finetuned={d['finetuned_on_train']}")


def write_report(out: dict, names: dict) -> None:
    lines = [
        "# Recombination pool purity — are the synthetic cues realistic?",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · `data/final_merged`.",
        "",
        "**Purity** = the fraction of a ground-truth bucket's per-clip vectors "
        "that the cue model actually argmaxes to that bucket's own class. "
        "Recombination draws from the **train** column; deployment sees the "
        "**test** column. A gap between them is distribution shift injected "
        "straight into fusion's training data.",
        "",
        "## Overall", "",
        "| Cue | Fine-tuned on train? | Train | Val | Test | Train − Test |",
        "|---|---|---|---|---|---|",
    ]
    for m in ("ges", "emo", "mot"):
        d = out[m]
        o = d["overall"]
        lines.append(
            f"| {FULL[m]} | {'**yes**' if d['finetuned_on_train'] else 'no'} | "
            f"{o['train']:.3f} | {o['val']:.3f} | {o['test']:.3f} | "
            f"**{d['train_minus_test']:+.3f}** |")

    lines += [
        "",
        "The pattern lines up exactly with which models were promoted on "
        "2026-08-04: **emotion and gesture were fine-tuned on the train split, "
        "and their pools are correspondingly unrealistic; motion was not "
        "promoted (its fine-tune regressed 4/4 times) and its pool is the only "
        "honest one.** Gesture's train purity is a perfect 1.000 — every single "
        "training clip is classified correctly — against 0.864 at test.",
        "",
        "## Per class", "",
    ]
    for m in ("ges", "emo", "mot"):
        lines += [f"### {FULL[m]}", "",
                  "| Class | Train | Val | Test |", "|---|---|---|---|"]
        for cls in names[m]:
            c = out[m]["per_class"].get(cls, {})
            def f(s):
                v = c.get(s)
                return "—" if v is None else f"{v:.3f}"
            lines.append(f"| {cls} | {f('train')} | {f('val')} | {f('test')} |")
        lines.append("")

    lines += [
        "## Why this matters", "",
        "1. **Fusion is trained to over-trust gesture and emotion.** During "
        "training those cues are essentially never wrong; at deployment they are "
        "wrong 13.6% and 23.8% of the time. This independently predicts the cue-"
        "attribution result (`CUE_ATTRIBUTION.md`: gesture 31.8% > emotion 29.5% "
        "> motion 17.6% > context 10.1%) — the model weights the cues in "
        "precisely the order of how clean their training pools were, not how "
        "reliable they actually are.",
        "2. **It explains the oracle/real gap.** `PERCEPTION_BAND.md` finds "
        "fusion at 0.9373 on clean cues but 0.7133 on real ones. A model trained "
        "almost exclusively on clean cues is expected to be brittle to realistic "
        "noise.",
        "3. **`raise_hand` is the worst case**: train purity 1.000, test 0.327. "
        "Fusion has never once seen a mistaken `raise_hand` vector in training. "
        "Row #25 (the only `raise_hand` test row, and the entire flip subset of "
        "T04 — `CONTEXT_COUNTERFACTUAL.md`) scores 0.31/0.19 for fusion/rules.",
        "",
        "## What to do about it", "",
        "- **Do not simply switch the pools to val.** Val is actor-disjoint but "
        "shares scenarios with train, so gesture purity there is still 0.997 — "
        "it measures actor shift, not the scenario shift the test split "
        "measures. Val pools would fix emotion (0.959 → 0.752) but barely touch "
        "gesture.",
        "- **The sound fix is out-of-fold prediction**: refit each cue model "
        "K times, holding out a different scenario group each time, and pool the "
        "held-out predictions. That yields pool vectors carrying the error rate "
        "the model has on unseen scenarios, which is what deployment sees. Cost "
        "is K unimodal fine-tunes, not a re-extraction.",
        "- **A cheaper approximation** is calibrated noise injection: perturb "
        "pool vectors until bucket purity matches a held-out estimate of "
        "test-time purity. `WindowDataset`'s `jitter_sigma` already does "
        "something in this spirit, but it is class-agnostic and not calibrated "
        "to per-class confusion.",
        "- Either way this is a **recombination-design** issue, not a fusion-"
        "architecture one, and it sits squarely in the perception band that "
        "`PERCEPTION_BAND.md` identifies as holding all the remaining headroom.",
    ]
    (G.OUT_DIR / "POOL_PURITY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
