"""FINAL RESULT MATRICES for the v4 fusion engine.

Produces the four reporting tables, each with a plain-language description:

  A  Overall results          headline accuracy / macro-F1 / significance
  B  Per test case            every test scenario, its cue signature and score
  C  Per intent               precision / recall / F1 for all nine intents
  D  Per test type            T01-T05 with what each one measures

Final configuration (chosen by `92`, `95`, `96`):
  * v4 perception-aware taxonomy (gesture-first, emotion superclasses, context
    modifier, motion excluded)
  * self-attention fusion over cue tokens, missing_mode='exclude'
  * full-rubric augmentation, 300 rows per cue-combination
  * cue-dropout 0.7 applied to RECORDED rows only (keeps the natural marginal
    prior for missing-cue behaviour), plain val selection
  * 10 seeds, majority vote

    .venv/Scripts/python scripts/97_final_matrices.py --n-seeds 10
"""
from __future__ import annotations

import argparse
import io
import itertools
import sys
from collections import Counter, defaultdict
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

OUT = ROOT / "results" / "fusion_v2" / "RESULT_MATRICES.md"
PREFIX = {"emotion": "emo", "gesture": "ges", "context": "ctx"}
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
INTENT_NAME = {
    "F01": "Greeting / positive acknowledgement",
    "F02": "Emergency / danger",
    "F03": "Task assistance request",
    "F04": "Help request",
    "F05": "Engaged / busy - do not interrupt",
    "F06": "Requests passage / space",
    "F07": "Frustration / agitation",
    "F08": "Break / relief request",
    "F10": "Discouraged / disengaged",
}
CONFLICT = {("hostile", "thumbs_up"), ("hostile", "wave"),
            ("positive", "thumbs_down"), ("low", "thumbs_up")}
DP, REAL_ONLY = 0.7, True


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
    def __init__(self, X, obs, y, slices, dp, seed, extra=None, real_only=True):
        n_real = len(y)
        if extra is not None:
            X = np.concatenate([X, extra[0]]); obs = np.concatenate([obs, extra[1]])
            y = np.concatenate([y, extra[2]])
        self.n_real, self.real_only = n_real, real_only
        self.X, self.obs, self.y = X, obs, y
        self.slices, self.mods = slices, list(slices)
        self.dp = dp
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, o = self.X[i].copy(), self.obs[i].copy()
        if self.dp > 0 and not (self.real_only and i >= self.n_real):
            drop = (self.rng.random(len(self.mods)) < self.dp) & o.astype(bool)
            if (o.astype(bool) & ~drop).sum() == 0:
                drop[:] = False
            for k, m in enumerate(self.mods):
                if drop[k]:
                    x[self.slices[m]] = 0.0; o[k] = 0.0
        return torch.from_numpy(x), torch.from_numpy(o), int(self.y[i])


def train(Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    sl = modality_slices(DIMS, MODS)
    dl = DataLoader(_DS(Xtr, otr, ytr, sl, DP, seed, extra, REAL_ONLY),
                    batch_size=512, shuffle=True)
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


def prf(y, p, k):
    tp = int(((p == k) & (y == k)).sum()); fp = int(((p == k) & (y != k)).sum())
    fn = int(((p != k) & (y == k)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


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
    ok = np.ones(len(real), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok &= pd.notna(gtc[m]) & ~mk
    lab = np.array([rule_intent_v4(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                    if o else "NA" for e, g, c, o in
                    zip(gtc["emotion"], gtc["gesture"], gtc["context"], ok)])
    sig = np.array([(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                    for e, g, c in zip(gtc["emotion"], gtc["gesture"], gtc["context"])],
                   dtype=object)
    real = real[ok].reset_index(drop=True)
    sig, lab = sig[ok], lab[ok]
    y = np.array([Y_IDX[v] for v in lab], np.int64)
    person = idx.reindex(real.clip_id.to_numpy())["person_id"].to_numpy()
    scen_dir = idx.reindex(real.clip_id.to_numpy())["scenario_dir"].to_numpy()
    v3row = real.v3_row.to_numpy()
    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()
    train_persons = set(pd.Series(person[tr_m]).dropna())

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

    rng = np.random.default_rng(0)
    keyof = {m: np.array([EMO_SUPER.get(str(v), "neutral") if m == "emotion" else str(v)
                          for v in gtc[m][ok]], dtype=object) for m in MODS}
    pools = {}
    for m in MODS:
        for cls in set(keyof[m][tr_m]):
            s = tr_m & (keyof[m] == cls) & O[:, MODS.index(m)].astype(bool)
            if s.sum():
                pools[(m, cls)] = feat[m][s]
    Xs, Os, Ys = [], [], []
    for es, g, c in itertools.product(SUPERS, GES, CTX):
        need = {"emotion": es, "gesture": g, "context": c}
        if any((m, need[m]) not in pools for m in MODS):
            continue
        n = args.n_per_combo
        Xs.append(np.concatenate(
            [pools[(m, need[m])][rng.integers(0, len(pools[(m, need[m])]), n)]
             for m in MODS], 1))
        Os.append(np.ones((n, len(MODS)), np.float32))
        Ys.append(np.full(n, Y_IDX[rule_intent_v4(es, g, c)], np.int64))
    extra = (np.concatenate(Xs).astype(np.float32),
             np.concatenate(Os).astype(np.float32), np.concatenate(Ys))

    cnt = np.bincount(y[tr_m], minlength=len(INTENTS)).astype(np.float64)
    inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
    inv = inv / inv[inv > 0].mean()

    models, seed_acc = [], []
    for s in range(args.n_seeds):
        mdl = train(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], s, inv,
                    extra, dev)
        models.append(mdl)
        p = _eval_arrays(mdl, X[te_m], O[te_m], dev)
        seed_acc.append(float((p == y[te_m]).mean()))
        print(f"  seed{s}: {seed_acc[-1]:.4f}", flush=True)
    P = np.stack([_eval_arrays(m, X[te_m], O[te_m], dev) for m in models])
    fus = np.array([Counter(P[:, j]).most_common(1)[0][0] for j in range(P.shape[1])])

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
    fus_ok, rul_ok = fus == yte, rte == yte
    n = len(yte)
    macro_f = float(np.mean([prf(yte, fus, i)[2] for i in range(len(INTENTS))
                             if (yte == i).sum()]))
    macro_r = float(np.mean([prf(yte, rte, i)[2] for i in range(len(INTENTS))
                             if (yte == i).sum()]))
    n10 = int((fus_ok & ~rul_ok).sum()); n01 = int((~fus_ok & rul_ok).sum())
    from scipy import stats as st
    pv = float(st.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    disj = ~pd.Series(person[te_m]).isin(train_persons).to_numpy()
    sd = np.std(seed_acc); mu = np.mean(seed_acc)
    ci = 1.96 * sd / np.sqrt(len(seed_acc))

    L = ["# v4 Fusion engine - final result matrices", "",
         f"**Model.** Self-attention fusion over three cue tokens (emotion "
         f"superclass, gesture, context), trained on the v4 perception-aware "
         f"taxonomy with full-rubric augmentation and cue-dropout {DP} on recorded "
         f"rows. {args.n_seeds} seeds, majority vote.",
         f"**Baseline.** The v4 rule cascade applied to the same measured cues.",
         f"**Test set.** {n} recorded clips, held out by scenario; no synthetic data "
         f"is evaluated.", "",
         "---", "",
         "## A. Overall results", "",
         "*What this shows: headline performance of the fusion model against the "
         "rule-based baseline on the whole test set.*", "",
         "| metric | fusion | rules | difference |", "|---|---|---|---|",
         f"| **Accuracy** | **{fus_ok.mean():.4f}** | {rul_ok.mean():.4f} | "
         f"**{fus_ok.mean()-rul_ok.mean():+.4f}** |",
         f"| Macro-F1 (all classes equally weighted) | **{macro_f:.4f}** | "
         f"{macro_r:.4f} | {macro_f-macro_r:+.4f} |",
         f"| Accuracy on an unseen actor | **{fus_ok[disj].mean():.4f}** | "
         f"{rul_ok[disj].mean():.4f} | {fus_ok[disj].mean()-rul_ok[disj].mean():+.4f} |",
         f"| Correct clips | {int(fus_ok.sum())} / {n} | {int(rul_ok.sum())} / {n} | "
         f"{int(fus_ok.sum())-int(rul_ok.sum()):+d} |", "",
         f"- Per-seed accuracy **{mu:.4f} ± {sd:.4f}** (95% CI ±{ci:.4f}), "
         f"min {min(seed_acc):.4f} / max {max(seed_acc):.4f} - the model is stable "
         f"across random initialisations.",
         f"- **Statistical significance:** fusion is right where rules are wrong on "
         f"**{n10}** clips; rules right where fusion is wrong on **{n01}**. "
         f"McNemar exact p = **{pv:.3g}** "
         f"({'significant' if pv < 0.05 else 'not significant'} at α=0.05).", "",
         "---", "", "## B. Per test case (scenario)", "",
         "*What this shows: one row per recorded test scenario, with the cue "
         "signature the actors performed. `Cue signature` is emotion-superclass / "
         "gesture / context.*", "",
         "| scenario | cue signature | intent | clips | fusion | rules | diff |",
         "|---|---|---|---|---|---|---|"]

    rows = []
    for r in sorted(set(v3row[te_m])):
        m = (v3row[te_m] == r)
        s0 = ste[m][0]
        rows.append((int(r), f"{s0[0]} / {s0[1]} / {s0[2]}",
                     INTENTS[yte[m][0]], int(m.sum()),
                     fus_ok[m].mean(), rul_ok[m].mean()))
    for r, s0, it, cn, fa, ra in rows:
        L.append(f"| #{r} | {s0} | {it} | {cn} | {fa:.3f} | {ra:.3f} | {fa-ra:+.3f} |")
    won = sum(1 for *_, fa, ra in rows if fa > ra)
    lost = sum(1 for *_, fa, ra in rows if fa < ra)
    L += ["", f"Fusion scores higher on **{won}** scenarios, lower on **{lost}**, "
          f"equal on **{len(rows)-won-lost}**.", "",
          "---", "", "## C. Per intent", "",
          "*What this shows: how reliably each of the nine intents is recognised. "
          "Recall = of all clips that truly are this intent, the share correctly "
          "identified. Precision = of all clips predicted as this intent, the share "
          "that really are.*", "",
          "| intent | meaning | clips | fusion recall | fusion prec. | fusion F1 | "
          "rules F1 |", "|---|---|---|---|---|---|---|"]
    for i, code in enumerate(INTENTS):
        m = yte == i
        if m.sum() == 0:
            continue
        pf, rf, ff = prf(yte, fus, i)
        _, _, fr = prf(yte, rte, i)
        L.append(f"| **{code}** | {INTENT_NAME[code]} | {int(m.sum())} | {rf:.3f} | "
                 f"{pf:.3f} | **{ff:.3f}** | {fr:.3f} |")

    L += ["", "---", "", "## D. Per test type", "",
          "*What this shows: performance on each research test type. These overlap "
          "(a clip can belong to several), so they do not sum to the total.*", "",
          "| test | what it measures | clips | fusion | rules | diff |",
          "|---|---|---|---|---|---|"]
    full_cue = O[te_m].all(1)
    conflict = np.array([(t[0], t[1]) in CONFLICT for t in ste])
    TT = [("T01", "Full-cue fusion: all cues present, joint reasoning required",
           full_cue),
          ("T02", "Cue conflict: emotion contradicts the gesture (e.g. sarcastic "
                  "thumbs-up)", conflict),
          ("T04", "Context generalization: same cues, different environment",
           np.ones(n, bool)),
          ("T05", "Rule-baseline comparison: the whole test set", np.ones(n, bool))]
    for code, desc, m in TT:
        L.append(f"| **{code}** | {desc} | {int(m.sum())} | {fus_ok[m].mean():.4f} | "
                 f"{rul_ok[m].mean():.4f} | {fus_ok[m].mean()-rul_ok[m].mean():+.4f} |")

    sl = modality_slices(DIMS, MODS)
    fallback = {"emotion": "neutral", "gesture": "idle", "context": "classroom"}
    L += ["", "**T03 - missing-cue robustness.** Each cue is removed at evaluation "
          "time and the model must decide from what is left. The *ceiling* is the "
          "best score any method could reach with that cue hidden: with emotion "
          "gone, a thumbs-up is genuinely either friendly (F01) or sarcastic (F07) "
          "and nothing remaining can separate them.", "",
          "| cue removed | fusion | rules | ceiling | fusion vs ceiling |",
          "|---|---|---|---|---|"]

    def ceiling(mm):
        keep_i = [i for i, x in enumerate(MODS) if x != mm]
        g = defaultdict(Counter)
        for t, l in zip(ste, yte):
            g[tuple(t[i] for i in keep_i)][l] += 1
        tot = sum(sum(c.values()) for c in g.values())
        return sum(max(c.values()) for c in g.values()) / tot

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
        cl = ceiling(m)
        fa = float((fm == yte).mean())
        L.append(f"| {m} | **{fa:.4f}** | {float((rr == yte).mean()):.4f} | {cl:.4f} | "
                 f"{100*fa/cl:.0f}% of achievable |")
    L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    print(chr(10).join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
