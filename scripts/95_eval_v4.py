"""PHASE 5 - full T01-T05 evaluation of the v4 fusion engine.

Configuration fixed by `92_fusion_v4.py`: softmax-super representation, no motion
token, exhaustive v4 cue-combination augmentation (300/combo, every TEST
combination excluded), cue-dropout 0.0, plain val selection. 10 seeds.

Test-type membership is DERIVED from each clip's ground-truth v4 cue signature, so
it cannot drift out of sync with the taxonomy:

  T01 full-cue fusion       every cue observed and none designed-missing
  T02 cue conflict          emotion superclass contradicts gesture valence
  T03 missing-cue           a cue designed-missing or unobserved, PLUS a masking
                            sweep (each cue forced missing at eval time)
  T04 context generalization  rows whose intent depends on context (raise_hand),
                            plus a per-context breakdown
  T05 rules comparison      whole test set, fusion vs the v4 rule cascade

Also: per-class recall, person-disjoint subset (actors never in training),
McNemar vs rules.

    .venv/Scripts/python scripts/95_eval_v4.py --n-seeds 10
"""
from __future__ import annotations

import argparse
import io
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.generic_train import _eval_arrays, modality_slices  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402

OUT = ROOT / "results" / "fusion_v2" / "FINAL_EVAL_V4.md"
PREFIX = {"emotion": "emo", "gesture": "ges", "motion": "mot", "context": "ctx"}
CLS = {"emotion": ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
       "gesture": ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
                   "raise_hand", "both_hands_up"],
       "context": ["classroom", "kitchen"]}
EMO_SUPER = {"Fear": "alarm", "Surprise": "alarm", "Anger": "hostile",
             "Disgust": "hostile", "Happy": "positive", "Sad": "low",
             "Neutral": "neutral"}
SUPERS = ["alarm", "hostile", "positive", "low", "neutral"]
GES, CTX = CLS["gesture"], CLS["context"]
MODS = ["emotion", "gesture", "context"]
DIMS = {"emotion": len(SUPERS), "gesture": len(GES), "context": len(CTX)}
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
Y_IDX = {c: i for i, c in enumerate(INTENTS)}
# emotion-superclass vs gesture-valence contradictions (T02)
CONFLICT = {("hostile", "thumbs_up"), ("hostile", "wave"),
            ("positive", "thumbs_down"), ("low", "thumbs_up")}


def rule_intent_v4(emo_super, ges, ctx):
    if emo_super == "alarm":
        return "F02"
    if ges == "both_hands_up":
        return "F07" if emo_super == "hostile" else "F05"
    if ges == "beckoning":
        return "F04" if emo_super == "low" else "F03"
    if ges == "raise_hand":
        return "F04" if ctx == "classroom" else "F01"
    if ges == "wave":
        return "F06" if emo_super == "hostile" else "F01"
    if ges == "point":
        return "F06" if emo_super == "hostile" else "F03"
    if ges == "thumbs_up":
        return "F07" if emo_super == "hostile" else "F01"
    if ges == "thumbs_down":
        return "F08" if emo_super == "hostile" else "F04"
    if emo_super == "low":
        return "F10"
    if emo_super == "hostile":
        return "F07"
    return "F05"


class _DS(Dataset):
    def __init__(self, X, obs, y, extra=None):
        if extra is not None:
            X = np.concatenate([X, extra[0]]); obs = np.concatenate([obs, extra[1]])
            y = np.concatenate([y, extra[2]])
        self.X, self.obs, self.y = X, obs, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]), torch.from_numpy(self.obs[i]),
                int(self.y[i]))


def train_one(Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(_DS(Xtr, otr, ytr, extra), batch_size=512, shuffle=True)
    model = AttentionFusion(d=64, n_heads=4, n_layers=2, dropout=0.3,
                            n_classes=len(INTENTS), missing_mode="exclude",
                            modality_dims=DIMS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(
        weight=torch.as_tensor(cw, dtype=torch.float32, device=dev),
        label_smoothing=0.05)
    best, state, bad = -1.0, None, 0
    for _ in range(80):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            ce(model(xb.to(dev), ob.to(dev)), yb.to(dev)).backward()
            opt.step()
        a = float((_eval_arrays(model, Xva, ova, dev) == yva).mean())
        if a > best:
            best, bad = a, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-per-combo", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    idx = clips.set_index("clip_id")
    gtc = {m: idx.reindex(real.clip_id.to_numpy())[f"gt_{m}"].to_numpy() for m in MODS}
    designed_missing = np.zeros(len(real), bool)
    ok = np.ones(len(real), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        designed_missing |= mk
        ok &= pd.notna(gtc[m]) & ~mk
    lab = np.array([rule_intent_v4(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                    if o else "NA"
                    for e, g, c, o in zip(gtc["emotion"], gtc["gesture"],
                                          gtc["context"], ok)])
    sig = np.array([(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                    for e, g, c in zip(gtc["emotion"], gtc["gesture"], gtc["context"])],
                   dtype=object)
    keep = ok
    real = real[keep].reset_index(drop=True)
    sig, lab = sig[keep], lab[keep]
    dm = designed_missing[keep]
    y = np.array([Y_IDX[v] for v in lab], np.int64)
    person = idx.reindex(real.clip_id.to_numpy())["person_id"].to_numpy()

    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()
    train_persons = set(pd.Series(person[tr_m]).dropna())

    # ---- representation: softmax-super, no motion ----
    E = np.zeros((len(real), len(SUPERS)), np.float32)
    for c in CLS["emotion"]:
        col = f"emo_{c}"
        if col in real.columns:
            E[:, SUPERS.index(EMO_SUPER[c])] += real[col].fillna(0).to_numpy(np.float32)

    def norm(v):
        s = v.sum(1, keepdims=True)
        return np.divide(v, s, out=np.zeros_like(v), where=s > 0)

    feat = {"emotion": norm(E),
            "gesture": norm(real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy(np.float32)),
            "context": norm(real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy(np.float32))}
    X = np.concatenate([feat[m] for m in MODS], 1).astype(np.float32)
    O = np.stack([real[f"{PREFIX[m]}_obs"].to_numpy(np.float32) for m in MODS], 1)

    # ---- augmentation: every v4 combo except those used by TEST ----
    TEST_C = {tuple(t) for t in sig[te_m]}
    ALL_C = list(itertools.product(SUPERS, GES, CTX))
    # Two protocols, both reported:
    #   strict      augmentation excludes every TEST cue-combination -> measures
    #               compositional transfer to combinations never seen in any form
    #   full_rubric augmentation enumerates the whole authored rubric -> the way a
    #               deployed system would actually be built (you teach it every
    #               combination you defined). No test CLIP or SCENARIO is used;
    #               only the combination is enumerated and labelled by the rubric.
    PROTOCOLS = {"strict": [c for c in ALL_C if c not in TEST_C],
                 "full_rubric": ALL_C}
    rng = np.random.default_rng(0)
    keyof = {m: np.array([EMO_SUPER.get(str(v), "neutral") if m == "emotion" else str(v)
                          for v in gtc[m][keep]], dtype=object) for m in MODS}
    pools = {}
    for m in MODS:
        for cls in set(keyof[m][tr_m]):
            s = tr_m & (keyof[m] == cls) & O[:, MODS.index(m)].astype(bool)
            if s.sum():
                pools[(m, cls)] = feat[m][s]
    def build_aug(combos):
        Xs, Os, Ys = [], [], []
        for es, g, c in combos:
            need = {"emotion": es, "gesture": g, "context": c}
            if any((m, need[m]) not in pools for m in MODS):
                continue
            n = args.n_per_combo
            Xs.append(np.concatenate(
                [pools[(m, need[m])][rng.integers(0, len(pools[(m, need[m])]), n)]
                 for m in MODS], 1))
            Os.append(np.ones((n, len(MODS)), np.float32))
            Ys.append(np.full(n, Y_IDX[rule_intent_v4(es, g, c)], np.int64))
        return (np.concatenate(Xs).astype(np.float32),
                np.concatenate(Os).astype(np.float32), np.concatenate(Ys))
    print(f"train={tr_m.sum()} val={va_m.sum()} test={te_m.sum()}", flush=True)

    cnt = np.bincount(y[tr_m], minlength=len(INTENTS)).astype(np.float64)
    inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
    inv = inv / inv[inv > 0].mean()

    RES = {}
    for proto, combos in PROTOCOLS.items():
        extra = build_aug(combos)
        models, seed_acc = [], []
        for s in range(args.n_seeds):
            mdl = train_one(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], s,
                            inv, extra, dev)
            models.append(mdl)
            p = _eval_arrays(mdl, X[te_m], O[te_m], dev)
            seed_acc.append(float((p == y[te_m]).mean()))
        Pm = np.stack([_eval_arrays(m, X[te_m], O[te_m], dev) for m in models])
        RES[proto] = dict(models=models, seed_acc=seed_acc,
                          fus=np.array([Counter(Pm[:, j]).most_common(1)[0][0]
                                        for j in range(Pm.shape[1])]),
                          n_combos=len(combos), n_rows=len(extra[2]))
        print(f"[{proto}] combos={len(combos)} rows={len(extra[2])} "
              f"acc={np.mean(seed_acc):.4f} +/- {np.std(seed_acc):.4f}", flush=True)
    # rules: argmax cue class -> superclass -> rubric (Phase-2 gate convention)
    emo_arg = np.array([CLS["emotion"][i] for i in
                        real[[f"emo_{c}" for c in CLS["emotion"]]].fillna(0)
                        .to_numpy().argmax(1)])
    ges_arg = np.array([GES[i] for i in
                        real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy().argmax(1)])
    ctx_arg = np.array([CTX[i] for i in
                        real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy().argmax(1)])
    rules = np.array([Y_IDX[rule_intent_v4(EMO_SUPER.get(e, "neutral"), g, c)]
                      for e, g, c in zip(emo_arg, ges_arg, ctx_arg)])

    yte, rte, ste = y[te_m], rules[te_m], sig[te_m]
    dm_te, obs_te = dm[te_m], O[te_m]
    rul_ok = rte == yte
    full_cue = obs_te.all(1) & ~dm_te
    conflict = np.array([(t[0], t[1]) in CONFLICT for t in ste])
    ctxdep = np.array([t[1] == "raise_hand" for t in ste])
    disj = ~pd.Series(person[te_m]).isin(train_persons).to_numpy()
    sl = modality_slices(DIMS, MODS)
    fallback = {"emotion": "neutral", "gesture": "idle", "context": "classroom"}
    from scipy import stats as st

    def detail(proto):
        r = RES[proto]
        fus, models, seed_acc = r["fus"], r["models"], r["seed_acc"]
        fus_ok = fus == yte

        def line(nm, m):
            if m.sum() == 0:
                return f"| {nm} | 0 | - | - | - |"
            return (f"| {nm} | {int(m.sum())} | {fus_ok[m].mean():.4f} | "
                    f"{rul_ok[m].mean():.4f} | "
                    f"{fus_ok[m].mean()-rul_ok[m].mean():+.4f} |")

        D = [f"## `{proto}` protocol - detail", "",
             f"Augmentation over {r['n_combos']} combinations ({r['n_rows']} synthetic "
             f"rows); {len(seed_acc)} seeds. Per-seed **{np.mean(seed_acc):.4f} +/- "
             f"{np.std(seed_acc):.4f}**; ensemble **{fus_ok.mean():.4f}**; "
             f"v4 rules **{rul_ok.mean():.4f}**.", "",
             "| slice | n | fusion | rules | delta |", "|---|---|---|---|---|",
             line("**Overall (T05 - rules comparison)**", np.ones(len(yte), bool)),
             line("T01 full-cue fusion", full_cue),
             line("T02 cue conflict", conflict),
             line("T04 context-dependent rows", ctxdep),
             line("- classroom", np.array([t[2] == "classroom" for t in ste])),
             line("- kitchen", np.array([t[2] == "kitchen" for t in ste])),
             line("Person-disjoint (unseen actor)", disj),
             line("Person-overlap", ~disj), "",
             "**T03 - masking sweep** (each cue forced missing at eval)", "",
             "| masked cue | fusion | rules |", "|---|---|---|"]
        for k, m in enumerate(MODS):
            X2, O2 = X[te_m].copy(), O[te_m].copy()
            X2[:, sl[m]] = 0.0; O2[:, k] = 0.0
            Pm = np.stack([_eval_arrays(md, X2, O2, dev) for md in models])
            fm = np.array([Counter(Pm[:, j]).most_common(1)[0][0]
                           for j in range(Pm.shape[1])])
            rr = np.array([Y_IDX[rule_intent_v4(
                fallback["emotion"] if m == "emotion" else EMO_SUPER.get(e, "neutral"),
                fallback["gesture"] if m == "gesture" else g,
                fallback["context"] if m == "context" else c)]
                for e, g, c in zip(emo_arg[te_m], ges_arg[te_m], ctx_arg[te_m])])
            D.append(f"| {m} | {(fm == yte).mean():.4f} | {(rr == yte).mean():.4f} |")
        D += ["", "**Per-class recall**", "", "| intent | n | fusion | rules |",
              "|---|---|---|---|"]
        for i, nm in enumerate(INTENTS):
            m = yte == i
            if m.sum() == 0:
                D.append(f"| {nm} | 0 | - | - |")
                continue
            D.append(f"| {nm} | {int(m.sum())} | {(fus[m] == i).mean():.3f} | "
                     f"{(rte[m] == i).mean():.3f} |")
        n10 = int((fus_ok & ~rul_ok).sum()); n01 = int((~fus_ok & rul_ok).sum())
        pv = (float(st.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
              if n10 + n01 else 1.0)
        D += ["", f"McNemar vs rules: fusion-only-correct **{n10}**, "
              f"rules-only-correct **{n01}**, p = **{pv:.4g}** "
              f"({'significant' if pv < 0.05 else 'not significant'}).", "", "---", ""]
        return D

    proto_rows = []
    for proto, r in RES.items():
        fo = r["fus"] == yte
        proto_rows.append(
            f"| `{proto}` | {r['n_combos']} | {r['n_rows']} | "
            f"{np.mean(r['seed_acc']):.4f} +/- {np.std(r['seed_acc']):.4f} | "
            f"{fo.mean():.4f} | {fo[conflict].mean():.4f} |")

    L = ["# Phase 5 - v4 fusion, full T01-T05 evaluation", "",
         f"softmax-super representation (emotion superclasses / gesture / context), "
         f"no motion token, cue-dropout 0.0, plain val selection, "
         f"{args.n_seeds} seeds. Recorded test n={int(te_m.sum())}.", "",
         "## Augmentation protocol (reported both ways)", "",
         "The only conflict family in the test set is `hostile+thumbs_up` "
         f"({int(conflict.sum())} of {int(te_m.sum())} clips) and it appears NOWHERE "
         "in the recorded training data. Excluding test cue-combinations from "
         "augmentation therefore makes T02 a test of transfer to a conflict "
         "combination never seen in any form. Enumerating the full authored rubric is "
         "how a deployed system is actually built - no test clip, scenario or actor is "
         "used, only the combination is enumerated and labelled by the rubric, and "
         "materialised from real TRAIN cue vectors.", "",
         "| protocol | combos | synth rows | per-seed acc | ensemble | T02 conflict |",
         "|---|---|---|---|---|---|"] + proto_rows + ["",
         f"v4 rules on the same clips: **{rul_ok.mean():.4f}** "
         f"(T02 conflict {rul_ok[conflict].mean():.4f}).", "",
         "T03 has no rows as a *slice*: clips with a designed-missing cue cannot "
         "receive a v4 ground-truth label (the rubric needs that cue), so missing-cue "
         "behaviour is measured by the masking sweep in each protocol below. T04's "
         "context-dependent branch (`raise_hand`) has no test clips, so context "
         "generalization is reported as the per-context breakdown.", "", "---", ""]
    for proto in RES:
        L += detail(proto)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
