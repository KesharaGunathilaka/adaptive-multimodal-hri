"""PHASE 1 — characterise the FROZEN cue models: what can they actually
discriminate?

The v2 design principle is that intent boundaries must only be keyed on cue
distinctions the models get right (plan P1/P2). This measures, per cue:

  * the confusion matrix and per-class precision/recall (Wilson CIs)
  * the most-confused class PAIRS -- the distinctions that must NOT carry an
    intent boundary
  * accuracy at coarser granularities (candidate superclasses, plus a greedy
    merge that reports what each merge buys)

Uses **train+val only**. The recorded test split is untouched so it stays a
clean evaluation set for Phase 5.

    .venv/Scripts/python scripts/90_cue_reliability.py
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.model import MODALITIES  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402

OUT = ROOT / "results" / "fusion_v2" / "CUE_RELIABILITY.md"
OUTJSON = ROOT / "results" / "fusion_v2" / "cue_reliability.json"
PREFIX = {"emotion": "emo", "gesture": "ges", "motion": "mot", "context": "ctx"}
CLS = {"emotion": ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
       "gesture": ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
                   "raise_hand", "both_hands_up"],
       "motion": ["sitting", "standing", "walking", "stepping_back"],
       "context": ["classroom", "kitchen"]}
# candidate superclass groupings to evaluate (plan P2)
GROUPS = {
    "emotion": {"alarm": ["Fear", "Surprise"], "hostile": ["Anger", "Disgust"],
                "low": ["Sad"], "positive": ["Happy"], "neutral": ["Neutral"]},
    "gesture": {"summon": ["beckoning", "raise_hand"], "greet": ["wave"],
                "deictic": ["point"], "appraise_pos": ["thumbs_up"],
                "appraise_neg": ["thumbs_down"], "alarm_pose": ["both_hands_up"],
                "none": ["idle"]},
    "motion": {"static": ["sitting", "standing"], "approach": ["walking"],
               "retreat": ["stepping_back"]},
    "context": {"classroom": ["classroom"], "kitchen": ["kitchen"]},
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    idx = clips.set_index("clip_id")
    # The cue models were FINE-TUNED on `curated_clip` footage, which sits in the
    # train split -- measuring reliability there reports ~0.92 emotion against
    # ~0.75 at deployment. The taxonomy must be designed against the HONEST
    # number, so the design set excludes fine-tune footage, and every cue is also
    # reported per-split/per-source so the contamination is visible.
    src = idx.reindex(real.clip_id.to_numpy())["source"].to_numpy().astype(str)
    real = real.assign(source=src)
    design = real[real.split.isin(["train", "val"]) &
                  (real.source != "curated_clip")].reset_index(drop=True)
    print(f"design set (train+val, fine-tune footage excluded): {len(design)} clips; "
          f"test held out\n", flush=True)

    print("=== contamination check: accuracy by split / source ===", flush=True)
    for m in MODALITIES:
        p, names = PREFIX[m], CLS[m]
        cols = [f"{p}_{c}" for c in names if f"{p}_{c}" in real.columns]
        have = [c.split("_", 1)[1] for c in cols]
        pr = np.array([have[i] for i in real[cols].fillna(0).to_numpy().argmax(1)])
        gt = idx.reindex(real.clip_id.to_numpy())[f"gt_{m}"].to_numpy()
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok = real[f"{p}_obs"].to_numpy().astype(bool) & ~mk & pd.notna(gt)
        parts = []
        for lbl, sel in (("train/finetuned", (real.split == "train").to_numpy() & (real.source == "curated_clip").to_numpy()),
                         ("train/other", (real.split == "train").to_numpy() & (real.source != "curated_clip").to_numpy()),
                         ("val", (real.split == "val").to_numpy()),
                         ("test", (real.split == "test").to_numpy())):
            s = ok & sel
            parts.append(f"{lbl}={(gt[s].astype(str)==pr[s]).mean():.3f}(n={int(s.sum())})"
                         if s.sum() else f"{lbl}=n/a")
        print(f"  {m:<8} " + "  ".join(parts), flush=True)
    print("", flush=True)

    L = ["# Phase 1 — cue-model reliability (frozen models)", "",
         f"Measured on the **train+val** recorded clips only ({len(design)} clips); "
         f"the test split is untouched. Ground truth = the clip's annotated cue class; "
         f"prediction = argmax of the deployed cue model's clip-pooled output. "
         f"Rows where the cue is designed-missing or unobserved are excluded.", ""]
    payload = {}

    for m in MODALITIES:
        p, names = PREFIX[m], CLS[m]
        cols = [f"{p}_{c}" for c in names if f"{p}_{c}" in design.columns]
        have = [c.split("_", 1)[1] for c in cols]
        pred = np.array([have[i] for i in design[cols].fillna(0).to_numpy().argmax(1)])
        gt = idx.reindex(design.clip_id.to_numpy())[f"gt_{m}"].to_numpy()
        msk = idx.reindex(design.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        obs = design[f"{p}_obs"].to_numpy().astype(bool)
        keep = obs & ~msk & pd.notna(gt)
        g, q = gt[keep].astype(str), pred[keep]
        acc = float((g == q).mean())
        lo, hi = wilson(int((g == q).sum()), int(keep.sum()))
        L += [f"## {m} — accuracy {acc:.3f} (95% CI {lo:.3f}–{hi:.3f}, n={int(keep.sum())})",
              ""]

        present = [c for c in names if c in set(g) | set(q)]
        cm = pd.DataFrame(0, index=[f"true_{c}" for c in present], columns=present)
        for a, b in zip(g, q):
            if a in present and b in present:
                cm.loc[f"true_{a}", b] += 1
        L += ["| | " + " | ".join(present) + " |",
              "|" + "---|" * (len(present) + 1)]
        for r in cm.index:
            L.append(f"| **{r}** | " + " | ".join(str(v) for v in cm.loc[r]) + " |")
        L.append("")

        # per-class recall / precision
        L += ["| class | n | recall | precision |", "|---|---|---|---|"]
        percls = {}
        for c in present:
            n = int((g == c).sum())
            rec = float((q[g == c] == c).mean()) if n else float("nan")
            pn = int((q == c).sum())
            prec = float((g[q == c] == c).mean()) if pn else float("nan")
            percls[c] = dict(n=n, recall=rec, precision=prec)
            L.append(f"| {c} | {n} | {rec:.3f} | {prec:.3f} |")
        L.append("")

        # most-confused ordered pairs -> distinctions that must not carry a boundary
        pairs = []
        for a in present:
            na = int((g == a).sum())
            if na == 0:
                continue
            for b in present:
                if a == b:
                    continue
                k = int(((g == a) & (q == b)).sum())
                if k:
                    pairs.append((a, b, k, k / na))
        pairs.sort(key=lambda t: -t[2])
        L += ["**Most-confused distinctions** (true → predicted; these must not be the "
              "sole discriminator between two intents):", "",
              "| true | predicted | n | rate |", "|---|---|---|---|"]
        for a, b, k, r in pairs[:8]:
            L.append(f"| {a} | {b} | {k} | {r:.3f} |")
        if not pairs:
            L.append("| — | — | 0 | — |")
        L.append("")

        # granularity: candidate superclasses
        grp = GROUPS[m]
        of = {c: gname for gname, cs in grp.items() for c in cs}
        gg = np.array([of.get(x, x) for x in g]); qq = np.array([of.get(x, x) for x in q])
        sacc = float((gg == qq).mean())
        slo, shi = wilson(int((gg == qq).sum()), len(gg))
        L += [f"**Superclass granularity** {dict((k2, v) for k2, v in grp.items())}", "",
              f"- accuracy **{sacc:.3f}** (95% CI {slo:.3f}–{shi:.3f}) vs "
              f"{acc:.3f} fine-grained → **{sacc-acc:+.3f}**", ""]
        # what each individual merge buys
        merges = []
        for gname, cs in grp.items():
            if len(cs) < 2:
                continue
            of1 = {c: gname for c in cs}
            g1 = np.array([of1.get(x, x) for x in g]); q1 = np.array([of1.get(x, x) for x in q])
            merges.append((gname, cs, float((g1 == q1).mean()) - acc))
        if merges:
            L += ["| merge | classes | accuracy gain |", "|---|---|---|"]
            for gname, cs, d in sorted(merges, key=lambda t: -t[2]):
                L.append(f"| {gname} | {'+'.join(cs)} | {d:+.3f} |")
            L.append("")
        payload[m] = dict(accuracy=acc, ci=[lo, hi], n=int(keep.sum()),
                          per_class=percls, superclass_accuracy=sacc,
                          top_confusions=[dict(true=a, pred=b, n=k, rate=r)
                                          for a, b, k, r in pairs[:8]],
                          merge_gains={gn: d for gn, _, d in merges})
        print(f"{m:<8} fine={acc:.3f}  superclass={sacc:.3f}  ({sacc-acc:+.3f})", flush=True)

    L += ["## Design implications (plan P1/P2)", "",
          "Distinctions listed under *most-confused* above are the ones an intent "
          "boundary must not rest on. Merges with a large accuracy gain are the "
          "superclasses the new rubric should key on instead.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    OUTJSON.write_text(json.dumps(payload, indent=2))
    print(f"\n-> {OUT}\n-> {OUTJSON}")


if __name__ == "__main__":
    main()
