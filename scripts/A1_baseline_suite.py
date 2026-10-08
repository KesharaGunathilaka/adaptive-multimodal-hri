"""WP1 - comparison against published multimodal-fusion methods.

Every method sees the IDENTICAL v4 pipeline: same labels, same cue features, same
full-rubric augmentation, same seed list, same recorded test set. The only thing
that varies is the fusion mechanism, so the comparison is like-for-like.

Methods (the neural ones are faithful ports of `fusion/model/fusion_zoo.py`,
re-parameterised for v4's 3-cue / 5-8-2 layout instead of the legacy 24-dim one):

  rules            the v4 rule cascade (non-learned reference)
  unimodal-*       one cue alone -> intent (floor per modality)
  late-product     independent per-cue classifiers, probabilities multiplied
  concat-mlp       early fusion: concatenate all cues + obs flags -> MLP
  gbt              gradient-boosted trees on the concatenated cue vector
  gmu              Gated Multimodal Unit (Arevalo et al., 2017)
  lmf              Low-rank Multimodal Fusion (Liu et al., 2018)
  cross-attn       learned-query cross-attention
  channel-attn     squeeze-and-excitation over modality channels
  self-attn        OURS - self-attention over cue tokens

    .venv/Scripts/python scripts/A1_baseline_suite.py --n-seeds 10
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
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
from scripts.realworld_eval.stats import bootstrap_diff, ci95_over_seeds, mcnemar  # noqa: E402

OUT = ROOT / "results" / "fusion_v2" / "A1_BASELINE_SUITE.md"
OUTJSON = ROOT / "results" / "fusion_v2" / "a1_baseline_suite.json"
OUT_NOAUG = ROOT / "results" / "fusion_v2" / "A1_BASELINE_NOAUG.md"
OUTJSON_NOAUG = ROOT / "results" / "fusion_v2" / "a1_baseline_noaug.json"
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
TOTAL = sum(DIMS.values())
OFF = {}
_i = 0
for _m in MODS:
    OFF[_m] = (_i, _i + DIMS[_m]); _i += DIMS[_m]
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
Y_IDX = {c: i for i, c in enumerate(INTENTS)}
NC = len(INTENTS)
DP = 0.7


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


# ----------------------------- architectures -----------------------------
class ConcatMLPFusion(nn.Module):
    """Early fusion: concatenate every cue vector and the observed flags."""

    def __init__(self, d=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOTAL + len(MODS), d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, NC))

    def forward(self, x, obs):
        return self.net(torch.cat([x, obs], dim=1))


class GMUFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al. 2017)."""

    def __init__(self, d=64, dropout=0.2):
        super().__init__()
        self.enc = nn.ModuleDict({m: nn.Linear(DIMS[m], d) for m in MODS})
        self.gate = nn.ModuleDict({m: nn.Linear(TOTAL, d) for m in MODS})
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout),
                                  nn.Linear(d, NC))

    def forward(self, x, obs):
        fused = 0.0
        for k, m in enumerate(MODS):
            a, b = OFF[m]
            h = torch.tanh(self.enc[m](x[:, a:b]))
            z = torch.sigmoid(self.gate[m](x)) * obs[:, k:k + 1]
            fused = fused + z * h
        return self.head(fused)


class LMFusion(nn.Module):
    """Low-rank Multimodal Fusion (Liu et al. 2018)."""

    def __init__(self, d=48, rank=4, dropout=0.2):
        super().__init__()
        self.rank, self.d = rank, d
        self.factors = nn.ModuleDict(
            {m: nn.Linear(DIMS[m] + 1, rank * d) for m in MODS})
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout),
                                  nn.Linear(d, NC))

    def forward(self, x, obs):
        B = x.shape[0]
        prod = x.new_ones(B, self.rank, self.d)
        for k, m in enumerate(MODS):
            a, b = OFF[m]
            xa = torch.cat([x[:, a:b], x.new_ones(B, 1)], dim=1)
            f = self.factors[m](xa).view(B, self.rank, self.d)
            f = torch.where(obs[:, k].view(B, 1, 1).bool(), f, torch.ones_like(f))
            prod = prod * f
        return self.head(prod.sum(dim=1))


class CrossAttentionFusion(nn.Module):
    """Learned-query cross-attention into the modality tokens."""

    def __init__(self, d=64, n_heads=4, n_layers=2, n_queries=1, ff=128, dropout=0.2):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(DIMS[m], d) for m in MODS})
        self.mod_emb = nn.Parameter(torch.randn(len(MODS), d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True),
            "norm1": nn.LayerNorm(d), "norm2": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff, d))}) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NC))

    def forward(self, x, obs):
        B = x.shape[0]
        toks = torch.stack([self.proj[m](x[:, OFF[m][0]:OFF[m][1]]) + self.mod_emb[k]
                            for k, m in enumerate(MODS)], dim=1)
        pad = ~obs.bool()
        # a row with every cue masked would make attention undefined
        pad = torch.where(pad.all(1, keepdim=True), torch.zeros_like(pad), pad)
        q = self.query.expand(B, -1, -1)
        for L in self.layers:
            a, _ = L["attn"](q, toks, toks, key_padding_mask=pad)
            q = L["norm1"](q + a)
            q = L["norm2"](q + L["ff"](q))
        return self.head(q.mean(dim=1))


class ChannelAttentionFusion(nn.Module):
    """Squeeze-and-excitation over modality channels."""

    def __init__(self, d=64, reduction=2, dropout=0.2):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(DIMS[m], d) for m in MODS})
        c = len(MODS)
        self.se = nn.Sequential(nn.Linear(c, max(1, c // reduction)), nn.ReLU(),
                                nn.Linear(max(1, c // reduction), c))
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm(c * d),
                                  nn.Dropout(dropout), nn.Linear(c * d, NC))

    def forward(self, x, obs):
        maps = torch.stack([torch.relu(self.proj[m](x[:, OFF[m][0]:OFF[m][1]]))
                            for m in MODS], dim=1)
        gate = torch.sigmoid(self.se(maps.mean(dim=2))) * obs
        return self.head(maps * gate.unsqueeze(-1))


def build_model(name, dev):
    if name == "self-attn":
        return AttentionFusion(d=64, n_heads=4, n_layers=2, dropout=0.3,
                               n_classes=NC, missing_mode="exclude",
                               modality_dims=DIMS).to(dev)
    return {"concat-mlp": ConcatMLPFusion, "gmu": GMUFusion, "lmf": LMFusion,
            "cross-attn": CrossAttentionFusion,
            "channel-attn": ChannelAttentionFusion}[name]().to(dev)


# ----------------------------- training -----------------------------
class _DS(Dataset):
    def __init__(self, X, obs, y, slices, seed, extra):
        n_real = len(y)
        X = np.concatenate([X, extra[0]]); obs = np.concatenate([obs, extra[1]])
        y = np.concatenate([y, extra[2]])
        self.n_real = n_real
        self.X, self.obs, self.y = X, obs, y
        self.slices, self.mods = slices, list(slices)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, o = self.X[i].copy(), self.obs[i].copy()
        if i < self.n_real:                       # dropout on recorded rows only
            drop = (self.rng.random(len(self.mods)) < DP) & o.astype(bool)
            if (o.astype(bool) & ~drop).sum() == 0:
                drop[:] = False
            for k, m in enumerate(self.mods):
                if drop[k]:
                    x[self.slices[m]] = 0.0; o[k] = 0.0
        return torch.from_numpy(x), torch.from_numpy(o), int(self.y[i])


def train(name, Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    sl = modality_slices(DIMS, MODS)
    dl = DataLoader(_DS(Xtr, otr, ytr, sl, seed, extra), batch_size=512, shuffle=True)
    model = build_model(name, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.as_tensor(cw, dtype=torch.float32,
                                                    device=dev),
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


def macro_f1(y, p):
    f = []
    for k in sorted(set(y)):
        tp = ((p == k) & (y == k)).sum(); fp = ((p == k) & (y != k)).sum()
        fn = ((p != k) & (y == k)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-per-combo", type=int, default=300)
    ap.add_argument("--no-aug", action="store_true",
                    help="train every method WITHOUT cue-combination augmentation, "
                         "to show the augmentation is a general lever rather than "
                         "something tuned to favour our model")
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
    real = real[ok].reset_index(drop=True)
    lab = lab[ok]
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
    if args.no_aug:
        d = extra[0].shape[1]
        extra = (np.zeros((0, d), np.float32), np.zeros((0, len(MODS)), np.float32),
                 np.zeros((0,), np.int64))
    cnt = np.bincount(y[tr_m], minlength=NC).astype(np.float64)
    inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
    inv = inv / inv[inv > 0].mean()
    yte = y[te_m]
    print(f"train={tr_m.sum()} val={va_m.sum()} test={te_m.sum()} "
          f"synth={len(extra[2])}", flush=True)

    preds, results = {}, {}

    # ---- rules ----
    emo_arg = np.array([CLS["emotion"][i] for i in
                        real[[f"emo_{c}" for c in CLS["emotion"]]].fillna(0)
                        .to_numpy().argmax(1)])
    ges_arg = np.array([GES[i] for i in
                        real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy().argmax(1)])
    ctx_arg = np.array([CTX[i] for i in
                        real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy().argmax(1)])
    preds["rules"] = np.array([Y_IDX[rule_intent_v4(EMO_SUPER.get(e, "neutral"), g, c)]
                               for e, g, c in zip(emo_arg[te_m], ges_arg[te_m],
                                                  ctx_arg[te_m])])
    results["rules"] = dict(seed_accs=None)

    # ---- unimodal floors + late product (sklearn logistic regression) ----
    from sklearn.linear_model import LogisticRegression
    tr_full_X = np.concatenate([X[tr_m], extra[0]])
    tr_full_y = np.concatenate([y[tr_m], extra[2]])   # extra is empty under --no-aug
    probs_per_mod = {}
    for m in MODS:
        a, b = OFF[m]
        lr = LogisticRegression(max_iter=2000)
        lr.fit(tr_full_X[:, a:b], tr_full_y)
        preds[f"unimodal-{m}"] = lr.predict(X[te_m][:, a:b])
        results[f"unimodal-{m}"] = dict(seed_accs=None)
        pp = lr.predict_proba(X[te_m][:, a:b])
        full = np.full((len(pp), NC), 1e-9)
        full[:, lr.classes_] = pp
        probs_per_mod[m] = full
    prod = np.ones((int(te_m.sum()), NC))
    for k, m in enumerate(MODS):
        w = O[te_m][:, k:k + 1]
        prod *= np.where(w > 0, probs_per_mod[m], 1.0)
    preds["late-product"] = prod.argmax(1)
    results["late-product"] = dict(seed_accs=None)

    # ---- GBT ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    gb_acc = []
    gb_last = None
    for s in range(min(args.n_seeds, 3)):
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=s)
        gb.fit(np.concatenate([tr_full_X, np.concatenate([O[tr_m], extra[1]])], 1),
               tr_full_y)
        p = gb.predict(np.concatenate([X[te_m], O[te_m]], 1))
        gb_acc.append(float((p == yte).mean())); gb_last = p
    preds["gbt"] = gb_last
    results["gbt"] = dict(seed_accs=gb_acc)

    # ---- neural fusion methods ----
    for name in ("concat-mlp", "gmu", "lmf", "cross-attn", "channel-attn",
                 "self-attn"):
        accs, P = [], []
        for s in range(args.n_seeds):
            mdl = train(name, X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m],
                        s, inv, extra, dev)
            p = _eval_arrays(mdl, X[te_m], O[te_m], dev)
            accs.append(float((p == yte).mean())); P.append(p)
        P = np.stack(P)
        preds[name] = np.array([Counter(P[:, j]).most_common(1)[0][0]
                                for j in range(P.shape[1])])
        results[name] = dict(seed_accs=accs)
        print(f"{name:<14} {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)

    # ---- report ----
    ours = preds["self-attn"] == yte
    L = ["# WP1 - comparison against published fusion methods", "",
         f"All methods share the v4 taxonomy, cue features, full-rubric "
         f"augmentation ({len(extra[2])} synthetic rows), seed list and the same "
         f"{int(te_m.sum())} recorded test clips. Only the fusion mechanism varies. "
         f"Neural methods: {args.n_seeds} seeds, majority vote.", "",
         "| method | accuracy | macro-F1 | per-seed mean +/- std | vs ours (p) |",
         "|---|---|---|---|---|"]
    order = ["rules", "unimodal-emotion", "unimodal-gesture", "unimodal-context",
             "late-product", "concat-mlp", "gbt", "channel-attn", "lmf", "gmu",
             "cross-attn", "self-attn"]
    payload = {}
    for name in order:
        if name not in preds:
            continue
        p = preds[name]
        acc = float((p == yte).mean())
        sa = results[name]["seed_accs"]
        per = (f"{np.mean(sa):.4f} +/- {np.std(sa):.4f}" if sa else "-")
        if name == "self-attn":
            sig = "-"
        else:
            mc = mcnemar(ours, p == yte)
            sig = f"{mc['p_value']:.3g}"
        star = " **(ours)**" if name == "self-attn" else ""
        L.append(f"| `{name}`{star} | **{acc:.4f}** | {macro_f1(yte, p):.4f} | {per} | "
                 f"{sig} |")
        payload[name] = dict(accuracy=acc, macro_f1=macro_f1(yte, p),
                             seed_accs=sa)
    L += ["", "`p` is a McNemar exact test of that method against ours on the same "
          "clips. Unimodal floors, late-product and rules are deterministic given "
          "the cue features, so they have no seed variance.", ""]
    out_md = OUT_NOAUG if args.no_aug else OUT
    out_js = OUTJSON_NOAUG if args.no_aug else OUTJSON
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(chr(10).join(L), encoding="utf-8")
    out_js.write_text(json.dumps(payload, indent=2, default=float))
    print(chr(10).join(L))
    print(f"-> {out_md}")


if __name__ == "__main__":
    main()
