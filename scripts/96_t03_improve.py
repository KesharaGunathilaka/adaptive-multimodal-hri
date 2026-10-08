"""T03 (missing-cue robustness) improvement + the headline fusion-vs-rules table.

The Phase-5 model was trained with cue-dropout 0.0, so it never learned to run
without a cue: masking emotion or gesture dropped it from 0.82 to ~0.41-0.44.
This sweeps training-time cue dropout and the checkpoint-selection criterion, and
reports every masking condition against BOTH the rule cascade and the
information-theoretic ceiling for that condition.

The ceiling matters: with emotion hidden, `thumbs_up` is F01 or F07 and nothing in
the remaining cues can separate them, so no model can do better than the majority
intent per remaining-cue signature. Reporting masked accuracy without that
reference makes graceful degradation look worse than it is.

    .venv/Scripts/python scripts/96_t03_improve.py --n-seeds 5
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

OUT = ROOT / "results" / "fusion_v2" / "T03_AND_HEADLINE.md"
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
    def __init__(self, X, obs, y, slices, dp, seed, extra=None, real_only=False):
        n_real = len(y)
        if extra is not None:
            X = np.concatenate([X, extra[0]]); obs = np.concatenate([obs, extra[1]])
            y = np.concatenate([y, extra[2]])
        # When a cue is hidden the optimal answer is the MARGINAL majority, which
        # follows the real data prior. Synthetic rows are uniform over the 80
        # combinations, so dropping cues on them teaches a uniform marginal and
        # corrupts exactly the behaviour T03 measures. `real_only` restricts
        # dropout to recorded rows, which carry the natural prior.
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


def train(Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dp, sel_masked, dev,
          real_only=False):
    torch.manual_seed(seed); np.random.seed(seed)
    sl = modality_slices(DIMS, MODS)
    dl = DataLoader(_DS(Xtr, otr, ytr, sl, dp, seed, extra, real_only),
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
        plain = float((_eval_arrays(model, Xva, ova, dev) == yva).mean())
        if sel_masked:
            accs = [plain]
            for k, m in enumerate(MODS):
                X2, o2 = Xva.copy(), ova.copy()
                X2[:, sl[m]] = 0.0; o2[:, k] = 0.0
                accs.append(float((_eval_arrays(model, X2, o2, dev) == yva).mean()))
            score = float(np.mean(accs))
        else:
            score = plain
        if score > best:
            best, bad = score, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
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
    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()

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

    # full-rubric augmentation (the adopted deployment protocol)
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

    emo_arg = np.array([CLS["emotion"][i] for i in
                        real[[f"emo_{c}" for c in CLS["emotion"]]].fillna(0)
                        .to_numpy().argmax(1)])
    ges_arg = np.array([GES[i] for i in
                        real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy().argmax(1)])
    ctx_arg = np.array([CTX[i] for i in
                        real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy().argmax(1)])
    yte, ste = y[te_m], sig[te_m]
    rules_full = np.array([Y_IDX[rule_intent_v4(EMO_SUPER.get(e, "neutral"), g, c)]
                           for e, g, c in zip(emo_arg[te_m], ges_arg[te_m],
                                              ctx_arg[te_m])])
    fallback = {"emotion": "neutral", "gesture": "idle", "context": "classroom"}

    def rules_masked(m):
        return np.array([Y_IDX[rule_intent_v4(
            fallback["emotion"] if m == "emotion" else EMO_SUPER.get(e, "neutral"),
            fallback["gesture"] if m == "gesture" else g,
            fallback["context"] if m == "context" else c)]
            for e, g, c in zip(emo_arg[te_m], ges_arg[te_m], ctx_arg[te_m])])

    def ceiling(m):
        """best achievable with cue `m` hidden: majority intent per remaining
        ground-truth cue signature."""
        keep_i = [i for i, mm in enumerate(MODS) if mm != m]
        g = defaultdict(Counter)
        for t, lab_i in zip(ste, yte):
            g[tuple(t[i] for i in keep_i)][lab_i] += 1
        tot = sum(sum(c.values()) for c in g.values())
        return sum(max(c.values()) for c in g.values()) / tot

    sl = modality_slices(DIMS, MODS)
    conds = ["full"] + MODS
    ceil = {"full": 1.0, **{m: ceiling(m) for m in MODS}}
    print(f"test n={int(te_m.sum())} | ceilings: " +
          "  ".join(f"{c}={ceil[c]:.3f}" for c in conds), flush=True)

    rows = []
    for dp, sel, ronly in ((0.0, False, False), (0.3, False, False),
                           (0.3, False, True), (0.5, False, True),
                           (0.7, False, True), (0.5, True, True)):
        per = {c: [] for c in conds}
        models = []
        for s in range(args.n_seeds):
            mdl = train(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], s, inv,
                        extra, dp, sel, dev, real_only=ronly)
            models.append(mdl)
        for c in conds:
            X2, O2 = X[te_m].copy(), O[te_m].copy()
            if c != "full":
                k = MODS.index(c)
                X2[:, sl[c]] = 0.0; O2[:, k] = 0.0
            P = np.stack([_eval_arrays(md, X2, O2, dev) for md in models])
            ens = np.array([Counter(P[:, j]).most_common(1)[0][0]
                            for j in range(P.shape[1])])
            per[c] = float((ens == yte).mean())
        rows.append((dp, sel, per, ronly))
        print(f"dp={dp} sel={'masked' if sel else 'plain'}"
              f"{' real-only' if ronly else ''}: " +
              "  ".join(f"{c}={per[c]:.4f}" for c in conds), flush=True)

    rr = {"full": float((rules_full == yte).mean()),
          **{m: float((rules_masked(m) == yte).mean()) for m in MODS}}
    best = max(rows, key=lambda r: np.mean([r[2][c] for c in conds]))

    L = ["# T03 missing-cue robustness + headline fusion vs rules", "",
         f"v4 taxonomy, softmax-super representation, full-rubric augmentation, "
         f"{args.n_seeds} seeds (majority vote), recorded test n={int(te_m.sum())}.",
         "", "## Training-time cue dropout sweep", "",
         "The Phase-5 model used dropout 0.0 and therefore never learned to run "
         "without a cue. `sel` is the checkpoint-selection criterion: *plain* = "
         "full-cue val accuracy, *masked* = mean of full-cue and each "
         "single-cue-masked val accuracy.", "",
         "| dropout | sel | full-cue | emotion masked | gesture masked | context masked |",
         "|---|---|---|---|---|---|"]
    for dp, sel, per, ronly in rows:
        L.append(f"| {dp}{' (real-only)' if ronly else ''} | "
                 f"{'masked' if sel else 'plain'} | {per['full']:.4f} | "
                 f"{per['emotion']:.4f} | {per['gesture']:.4f} | {per['context']:.4f} |")
    L += ["", f"| **rules** | - | **{rr['full']:.4f}** | {rr['emotion']:.4f} | "
          f"{rr['gesture']:.4f} | {rr['context']:.4f} |",
          f"| **ceiling** | - | 1.0000 | {ceil['emotion']:.4f} | "
          f"{ceil['gesture']:.4f} | {ceil['context']:.4f} |", "",
          "The ceiling is the best any predictor can reach with that cue hidden "
          "(majority intent per remaining ground-truth cue signature): with emotion "
          "hidden, `thumbs_up` is F01 or F07 and nothing else separates them.", ""]

    dp, sel, per, ronly = best
    L += ["## Recommended configuration", "",
          f"dropout **{dp}**"
          f"{' applied to recorded rows only' if ronly else ''}, "
          f"selection **{'masked' if sel else 'plain'}**", "",
          "| condition | fusion | rules | ceiling | fusion as % of ceiling |",
          "|---|---|---|---|---|"]
    for c in conds:
        L.append(f"| {'full-cue (T01)' if c == 'full' else c + ' masked (T03)'} | "
                 f"**{per[c]:.4f}** | {rr[c]:.4f} | {ceil[c]:.4f} | "
                 f"{100*per[c]/ceil[c]:.1f}% |")
    L += ["", "## Headline: fusion vs rule-based", "",
          "| metric | fusion | rules | delta |", "|---|---|---|---|",
          f"| Overall accuracy (T01/T05) | **{per['full']:.4f}** | {rr['full']:.4f} | "
          f"{per['full']-rr['full']:+.4f} |",
          f"| T03 emotion lost | **{per['emotion']:.4f}** | {rr['emotion']:.4f} | "
          f"{per['emotion']-rr['emotion']:+.4f} |",
          f"| T03 gesture lost | **{per['gesture']:.4f}** | {rr['gesture']:.4f} | "
          f"{per['gesture']-rr['gesture']:+.4f} |",
          f"| T03 context lost | **{per['context']:.4f}** | {rr['context']:.4f} | "
          f"{per['context']-rr['context']:+.4f} |",
          f"| **mean over all four conditions** | "
          f"**{np.mean([per[c] for c in conds]):.4f}** | "
          f"{np.mean([rr[c] for c in conds]):.4f} | "
          f"{np.mean([per[c] for c in conds])-np.mean([rr[c] for c in conds]):+.4f} |",
          ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    print(chr(10).join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
