"""WP3b - analyse the human annotations.

Answers three questions the manuscript needs:

  1. Do annotators agree with EACH OTHER?  (Cohen's / Fleiss' kappa)
     If they do not, the intent taxonomy is not legible to humans and no model
     comparison against human labels would mean anything.
  2. Do they agree with the DATA TABLE?    (accuracy of human consensus vs v4 label)
     This is the ecological-validity claim: does the recorded footage actually
     convey the intent the table specifies?
  3. Scored against HUMAN labels rather than table labels, does fusion still beat
     rules? This removes the rule cascade's built-in advantage, because the human
     labels were produced with no knowledge of the rubric.

Run after annotators fill `results/fusion_v2/annotation_kit/responses_*.csv`.

    .venv/Scripts/python scripts/A3_analyse_annotations.py
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KIT = ROOT / "results" / "fusion_v2" / "annotation_kit"
OUT = ROOT / "results" / "fusion_v2" / "A3_HUMAN_VALIDATION.md"
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]


def cohen_kappa(a, b, labels):
    """unweighted Cohen's kappa between two label sequences."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = 0.0
    for l in labels:
        pe += (np.mean(a == l)) * (np.mean(b == l))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(mat):
    """mat: [n_items, n_categories] counts of raters per category."""
    n_items, n_cat = mat.shape
    n_raters = mat.sum(1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] < 2:
        return float("nan")
    n = n_raters[0]
    p_j = mat.sum(0) / (n_items * n)
    P_i = ((mat ** 2).sum(1) - n) / (n * (n - 1))
    return (P_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-annotators", type=int, default=2)
    args = ap.parse_args()

    key_path = KIT / "KEY_do_not_share.csv"
    if not key_path.exists():
        sys.exit("run scripts/A3_build_annotation_kit.py first")
    key = pd.read_csv(key_path)

    resp = {}
    for f in sorted(KIT.glob("responses_*.csv")):
        d = pd.read_csv(f)
        d["intent"] = d["intent"].astype(str).str.strip().str.upper()
        d.loc[d.intent.isin(["", "NAN", "NONE"]), "intent"] = ""
        filled = d[d.intent != ""]
        if len(filled) == 0:
            continue
        resp[f.stem.replace("responses_", "")] = d.set_index("anon_id")
        print(f"{f.name}: {len(filled)}/{len(d)} answered", flush=True)
    if len(resp) < args.min_annotators:
        sys.exit(f"need >= {args.min_annotators} completed response files; "
                 f"found {len(resp)}. Annotators still to complete the kit.")

    ids = [a for a in key.anon_id
           if all(a in r.index and str(r.loc[a, "intent"]).strip() != ""
                  for r in resp.values())]
    print(f"\n{len(ids)} clips answered by all {len(resp)} annotators", flush=True)
    who = sorted(resp)
    A = {w: np.array([str(resp[w].loc[a, "intent"]).strip().upper() for a in ids])
         for w in who}
    truth = key.set_index("anon_id").loc[ids, "v4_intent"].to_numpy()

    L = ["# WP3 - human validation of the intent taxonomy", "",
         f"{len(ids)} test clips, {len(resp)} annotators, video only - no cue "
         f"labels, no scenario identity, no rubric. Clips were presented under "
         f"anonymous ids in shuffled order.", ""]

    # ---- 1. inter-annotator agreement ----
    labels = INTENTS + ["UNCLEAR"]
    L += ["## 1. Do the annotators agree with each other?", "",
          "| pair | raw agreement | Cohen's kappa |", "|---|---|---|"]
    for x, y in combinations(who, 2):
        L.append(f"| {x} vs {y} | {np.mean(A[x] == A[y]):.3f} | "
                 f"{cohen_kappa(A[x], A[y], labels):.3f} |")
    mat = np.zeros((len(ids), len(labels)), int)
    for i, a in enumerate(ids):
        for w in who:
            v = A[w][i]
            if v in labels:
                mat[i, labels.index(v)] += 1
    fk = fleiss_kappa(mat)
    L += ["", f"Fleiss' kappa over all annotators: **{fk:.3f}** "
          f"({'substantial' if fk > 0.6 else 'moderate' if fk > 0.4 else 'fair/poor'} "
          f"agreement by the Landis-Koch scale).", ""]
    unclear = float(np.mean(np.concatenate([A[w] for w in who]) == "UNCLEAR"))
    L += [f"`unclear` responses: **{unclear:.1%}** of all judgements - a direct "
          f"measure of how legible the footage is.", ""]

    # ---- 2. consensus vs the data table ----
    cons, tie = [], 0
    for i in range(len(ids)):
        votes = Counter(A[w][i] for w in who if A[w][i] != "UNCLEAR")
        if not votes:
            cons.append("UNCLEAR"); continue
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            tie += 1
        cons.append(top[0][0])
    cons = np.array(cons)
    decided = cons != "UNCLEAR"
    agree = float(np.mean(cons[decided] == truth[decided]))
    L += ["## 2. Do the annotators agree with the data table?", "",
          f"Human consensus matches the table's intent on **{agree:.1%}** of the "
          f"{int(decided.sum())} clips where a consensus exists "
          f"({tie} ties broken arbitrarily).", "",
          "| intent | clips | human-table agreement |", "|---|---|---|"]
    for c in INTENTS:
        m = decided & (truth == c)
        if m.sum() == 0:
            continue
        L.append(f"| {c} | {int(m.sum())} | {np.mean(cons[m] == c):.3f} |")
    L += ["", "Low agreement for a particular intent means the footage does not "
          "convey that intent as designed - a dataset finding, independent of any "
          "model.", ""]

    # ---- 3. models scored against human labels ----
    pred_path = ROOT / "results" / "fusion_v2" / "a3_model_predictions.csv"
    if pred_path.exists():
        mp = pd.read_csv(pred_path).set_index("clip_id")
        cid = key.set_index("anon_id").loc[ids, "clip_id"].to_numpy()
        have = [i for i, c in enumerate(cid) if c in mp.index and decided[i]]
        if have:
            hn = cons[have]
            tb = truth[have]
            fu = mp.loc[cid[have], "fusion"].to_numpy()
            ru = mp.loc[cid[have], "rules"].to_numpy()
            L += ["## 3. Scored against HUMAN labels instead of the table", "",
                  f"On the {len(have)} annotated clips with a human consensus:", "",
                  "| ground truth used | fusion | rules |", "|---|---|---|",
                  f"| the data table | {np.mean(fu == tb):.4f} | "
                  f"{np.mean(ru == tb):.4f} |",
                  f"| **human consensus** | **{np.mean(fu == hn):.4f}** | "
                  f"**{np.mean(ru == hn):.4f}** |", "",
                  "The second row is the non-circular comparison: the human labels "
                  "were produced with no knowledge of the rubric the rule cascade "
                  "implements.", ""]
    else:
        L += ["## 3. Scored against HUMAN labels", "",
              "Pending - write model predictions for the annotated clips to "
              "`results/fusion_v2/a3_model_predictions.csv` with columns "
              "`clip_id,fusion,rules`, then re-run this script.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    print(chr(10).join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
