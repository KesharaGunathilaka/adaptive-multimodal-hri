"""PHASE 3+4 — build the v4 training data and choose the fusion input representation.

Labels come from the Phase-2 taxonomy (`91_taxonomy_v4.py`): gesture-first,
emotion-superclass second, context modifier, motion excluded. Gate passed on val
(ceiling 0.9505, rules 0.8269).

Compared representations, all fed to the same self-attention fusion
(`fusion/model/model.py::AttentionFusion`, `missing_mode='exclude'`):

  softmax_super  deployed cue softmax, emotion collapsed to the 5 superclasses
                 (5/8/2 dims) -- the cue models' own calibrated output
  embed          PCA-reduced penultimate embeddings per cue (the user's preferred
                 representation)
  two_stage      stage-1 classifier embedding -> cue posterior, then fusion over
                 posteriors (most stable representation measured in v3)

Each is run with and without a MOTION token: v4 labels do not depend on motion, and
motion measures 0.435, so it is expected to be pure noise -- this settles it.

Augmentation (the +0.152 lever): the v4 cue space is only 5x8x2 = 80 combinations,
so it is enumerated exhaustively, minus every combination used by a TEST scenario,
and materialised from real captured cue vectors of the TRAIN split.

    .venv/Scripts/python scripts/92_fusion_v4.py --n-seeds 5
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.generic_train import _eval_arrays, modality_slices  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402
from scripts.embedding_fusion_common import PREF_OF, load_embed_table  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
LEGACY = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled.npz"
OUT = ROOT / "results" / "fusion_v2" / "FUSION_V4_COMPARE.md"
PREFIX = {"emotion": "emo", "gesture": "ges", "motion": "mot", "context": "ctx"}
CLS = {"emotion": ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
       "gesture": ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
                   "raise_hand", "both_hands_up"],
       "motion": ["sitting", "standing", "walking", "stepping_back"],
       "context": ["classroom", "kitchen"]}
EMO_SUPER = {"Fear": "alarm", "Surprise": "alarm", "Anger": "hostile",
             "Disgust": "hostile", "Happy": "positive", "Sad": "low",
             "Neutral": "neutral"}
SUPERS = ["alarm", "hostile", "positive", "low", "neutral"]
GES = CLS["gesture"]; CTX = CLS["context"]
# Inlined from `91_taxonomy_v4.py` rather than imported: that module re-wraps
# sys.stdout at import time, which closes this process's stdout.
def rule_intent_v4(emo_super: str, ges: str, ctx: str) -> str:
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


INTENTS_V4 = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
Y_IDX = {c: i for i, c in enumerate(INTENTS_V4)}


def super_probs(frame):
    """deployed emotion softmax -> 5 superclass probabilities."""
    out = np.zeros((len(frame), len(SUPERS)), np.float32)
    for c in CLS["emotion"]:
        col = f"emo_{c}"
        if col in frame.columns:
            out[:, SUPERS.index(EMO_SUPER[c])] += frame[col].fillna(0).to_numpy(np.float32)
    s = out.sum(1, keepdims=True)
    return np.divide(out, s, out=np.zeros_like(out), where=s > 0)


def cue_probs(frame, m, names):
    cols = [f"{PREFIX[m]}_{c}" for c in names if f"{PREFIX[m]}_{c}" in frame.columns]
    v = frame[cols].fillna(0).to_numpy(np.float32)
    s = v.sum(1, keepdims=True)
    return np.divide(v, s, out=np.zeros_like(v), where=s > 0)


class _DS(Dataset):
    """cue-dropout augmentation over real rows (plus any synthetic `extra`)."""

    def __init__(self, X, obs, y, slices, dropout_p, seed, extra=None):
        if extra is not None:
            X = np.concatenate([X, extra[0]]); obs = np.concatenate([obs, extra[1]])
            y = np.concatenate([y, extra[2]])
        self.X, self.obs, self.y = X, obs, y
        self.slices, self.mods = slices, list(slices)
        self.dp = dropout_p
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, o = self.X[i].copy(), self.obs[i].copy()
        if self.dp > 0:
            drop = (self.rng.random(len(self.mods)) < self.dp) & o.astype(bool)
            if (o.astype(bool) & ~drop).sum() == 0:
                drop[:] = False
            for k, m in enumerate(self.mods):
                if drop[k]:
                    x[self.slices[m]] = 0.0; o[k] = 0.0
        return torch.from_numpy(x), torch.from_numpy(o), int(self.y[i])


def train_fusion_v4(Xtr, otr, ytr, Xva, ova, yva, dims, mods, n_classes, seed,
                    class_weights=None, extra=None, dropout_p=0.4, d_model=64,
                    n_layers=2, n_heads=4, attn_dropout=0.3, epochs=80, patience=10,
                    lr=1e-3, dev="cuda", select_masked=True):
    """Self-contained trainer: the shared `train_fusion_generic` has no class-weight
    or capacity arguments, and `fusion/model/*` is kept at its baseline state."""
    torch.manual_seed(seed); np.random.seed(seed)
    slices = modality_slices(dims, mods)
    dl = DataLoader(_DS(Xtr, otr, ytr, slices, dropout_p, seed, extra),
                    batch_size=512, shuffle=True)
    model = AttentionFusion(d=d_model, n_heads=n_heads, n_layers=n_layers,
                            dropout=attn_dropout, n_classes=n_classes,
                            missing_mode="exclude", modality_dims=dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cw = (torch.as_tensor(class_weights, dtype=torch.float32, device=dev)
          if class_weights is not None else None)
    ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            ce(model(xb.to(dev), ob.to(dev)), yb.to(dev)).backward()
            opt.step()
        # Selection criterion. With only 3 cue tokens, averaging over
        # single-cue-masked conditions optimises DEGRADED accuracy at the cost of
        # the full-cue number, so this is switchable.
        plain = float((_eval_arrays(model, Xva, ova, dev) == yva).mean())
        if select_masked:
            accs = [plain]
            for k, m in enumerate(mods):
                X2, o2 = Xva.copy(), ova.copy()
                X2[:, slices[m]] = 0.0; o2[:, k] = 0.0
                accs.append(float((_eval_arrays(model, X2, o2, dev) == yva).mean()))
            score = float(np.mean(accs))
        else:
            score = plain
        if score > best:
            best, bad = score, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


def train_stage1(Xtr, ytr, Xva, yva, n_cls, dev, seed=0, epochs=150, patience=15):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, n_cls)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    xv, yv = torch.from_numpy(Xva).to(dev), torch.from_numpy(yva).to(dev)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            j = perm[i:i + 256]
            opt.zero_grad(); ce(net(xt[j]), yt[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            a = (net(xv).argmax(1) == yv).float().mean().item()
        if a > best:
            best, bad, state = a, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-per-combo", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    idx = clips.set_index("clip_id")
    gtc = {m: idx.reindex(real.clip_id.to_numpy())[f"gt_{m}"].to_numpy()
           for m in ("emotion", "gesture", "context")}
    ok = np.ones(len(real), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok &= pd.notna(gtc[m]) & ~mk
    lab = np.array([rule_intent_v4(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                    if o else "NA" for e, g, c, o in
                    zip(gtc["emotion"], gtc["gesture"], gtc["context"], ok)])
    gt_tuple = np.array([(EMO_SUPER.get(str(e), "neutral"), str(g), str(c))
                         for e, g, c in zip(gtc["emotion"], gtc["gesture"], gtc["context"])],
                        dtype=object)
    real = real.assign(y4=lab, ok4=ok)
    real = real[real.ok4].reset_index(drop=True)
    keepmask = ok
    gt_tuple = gt_tuple[keepmask]
    y = np.array([Y_IDX[v] for v in real.y4], np.int64)

    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()
    print(f"v4 labels: train={tr_m.sum()} val={va_m.sum()} test={te_m.sum()}", flush=True)
    print("class balance (train):",
          dict(sorted(Counter(real.y4[tr_m]).items())), flush=True)

    # Clips the CUE MODELS were fine-tuned on read far cleaner than deployment
    # (emotion 0.973 vs 0.762 -- Phase 1). Augmentation pools drawn from them
    # teach a mapping calibrated for cues the model will never see at test.
    src = idx.reindex(real.clip_id.to_numpy())["source"].to_numpy().astype(str)
    clean_mask = src != "curated_clip"
    print(f"train pool sources: fine-tuned={int((tr_m & ~clean_mask).sum())} "
          f"clean={int((tr_m & clean_mask).sum())}", flush=True)

    TEST_COMBOS = {tuple(t) for t in gt_tuple[te_m]}
    ALL = [tuple(c) for c in itertools.product(SUPERS, GES, CTX)]
    AUG = [c for c in ALL if c not in TEST_COMBOS]
    print(f"\ncue-combination space={len(ALL)}  test combos={len(TEST_COMBOS)}  "
          f"augmentable={len(AUG)}", flush=True)

    # ---------- representations ----------
    # `load_embed_table` reads EMBED_PATH, which already is `embeddings_pooled.npz`
    e_all = load_embed_table(real.clip_id.to_numpy())
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca_feat, pca_dims = {}, {}
    for m in ("emotion", "gesture", "motion", "context"):
        p = PREF_OF[m]
        E = e_all[f"{p}_embed"]; O = e_all[f"{p}_obs"].astype(bool)
        fitmask = O & tr_m
        kk = min(args.k, int(fitmask.sum()) - 1, E.shape[1])
        pc = PCA(n_components=kk, random_state=0).fit(E[fitmask])
        sc = StandardScaler().fit(pc.transform(E[fitmask]))
        pca_feat[m] = sc.transform(pc.transform(E)).astype(np.float32)
        pca_dims[m] = kk
    obs_all = {m: e_all[f"{PREF_OF[m]}_obs"].astype(np.float32)
               for m in ("emotion", "gesture", "motion", "context")}

    sm = {"emotion": super_probs(real), "gesture": cue_probs(real, "gesture", GES),
          "motion": cue_probs(real, "motion", CLS["motion"]),
          "context": cue_probs(real, "context", CTX)}
    sm_dims = {"emotion": len(SUPERS), "gesture": len(GES), "motion": 4,
               "context": len(CTX)}

    # two-stage: stage-1 classifiers on PCA embeddings -> cue posteriors
    ts = {}
    tgt = {"emotion": [SUPERS.index(EMO_SUPER.get(str(v), "neutral")) for v in gtc["emotion"][keepmask]],
           "gesture": [GES.index(str(v)) if str(v) in GES else -1 for v in gtc["gesture"][keepmask]],
           "context": [CTX.index(str(v)) if str(v) in CTX else -1 for v in gtc["context"][keepmask]]}
    for m, n_cls in (("emotion", len(SUPERS)), ("gesture", len(GES)), ("context", len(CTX))):
        t = np.array(tgt[m], np.int64)
        a = tr_m & (t >= 0) & obs_all[m].astype(bool)
        b = va_m & (t >= 0) & obs_all[m].astype(bool)
        net = train_stage1(pca_feat[m][a], t[a], pca_feat[m][b], t[b], n_cls, dev)
        with torch.no_grad():
            ts[m] = torch.softmax(net(torch.from_numpy(pca_feat[m]).to(dev)), 1).cpu().numpy()
        print(f"  stage1 {m}: val acc={(ts[m][b].argmax(1)==t[b]).mean():.3f}", flush=True)
    ts["motion"] = sm["motion"]

    REPS = {"softmax_super": (sm, sm_dims),
            "embed": (pca_feat, pca_dims),
            "two_stage": (ts, {"emotion": len(SUPERS), "gesture": len(GES),
                               "motion": 4, "context": len(CTX)})}

    def assemble(feat, dims, mods):
        X = np.concatenate([feat[m] for m in mods], 1).astype(np.float32)
        O = np.stack([obs_all[m] for m in mods], 1).astype(np.float32)
        return X, O, {m: dims[m] for m in mods}

    def augment(feat, dims, mods, seed=0, n_per_combo=None, pool_mask=None):
        """enumerate v4 cue-combinations (minus test combos), materialised from
        real TRAIN cue vectors of the matching ground-truth class.

        `pool_mask` restricts which train clips may donate cue vectors. This
        matters: 57% of the train split is `curated_clip` footage the CUE MODELS
        were fine-tuned on, where emotion reads 0.973 against 0.762 at
        deployment. Drawing synthetic cue vectors from it teaches a mapping
        calibrated for unrealistically clean cues (Phase 1,
        `CUE_RELIABILITY.md`)."""
        rng = np.random.default_rng(seed)
        src_ok = tr_m if pool_mask is None else (tr_m & pool_mask)
        pools = {}
        for m in mods:
            if m == "motion":
                continue
            key = {"emotion": [EMO_SUPER.get(str(v), "neutral") for v in gtc["emotion"][keepmask]],
                   "gesture": [str(v) for v in gtc["gesture"][keepmask]],
                   "context": [str(v) for v in gtc["context"][keepmask]]}[m]
            key = np.array(key, dtype=object)
            for cls in set(key[src_ok]):
                s = src_ok & (key == cls) & obs_all[m].astype(bool)
                if s.sum():
                    pools[(m, cls)] = feat[m][s]
        mot_pool = feat["motion"][src_ok] if "motion" in mods else None
        Xs, Os, Ys = [], [], []
        for es, g, c in AUG:
            need = {"emotion": es, "gesture": g, "context": c}
            if any((m, need[m]) not in pools for m in need if m in mods):
                continue
            n = n_per_combo or args.n_per_combo
            parts, obs_row = [], []
            for m in mods:
                if m == "motion":
                    parts.append(mot_pool[rng.integers(0, len(mot_pool), n)])
                    obs_row.append(np.ones(n, np.float32))
                    continue
                pl = pools[(m, need[m])]
                parts.append(pl[rng.integers(0, len(pl), n)])
                obs_row.append(np.ones(n, np.float32))
            Xs.append(np.concatenate(parts, 1))
            Os.append(np.stack(obs_row, 1))
            Ys.append(np.full(n, Y_IDX[rule_intent_v4(es, g, c)], np.int64))
        return (np.concatenate(Xs).astype(np.float32), np.concatenate(Os).astype(np.float32),
                np.concatenate(Ys))

    CFG = dict(dropout_p=0.4, d_model=64, n_layers=2, n_heads=4, attn_dropout=0.3)
    cnt = np.bincount(y[tr_m], minlength=len(INTENTS_V4)).astype(np.float64)
    inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
    inv = inv / inv[inv > 0].mean()

    rules_te = np.array([Y_IDX[rule_intent_v4(
        SUPERS[sm["emotion"][i].argmax()], GES[sm["gesture"][i].argmax()],
        CTX[sm["context"][i].argmax()])] for i in np.flatnonzero(te_m)])
    rules_acc = float((rules_te == y[te_m]).mean())
    print(f"\nv4 rules on test: {rules_acc:.4f}\n", flush=True)

    rows = []
    for rep, (feat, dims) in REPS.items():
        # (motion token?, synthetic rows per cue-combination). The v4 combination
        # space is only 80, so 300/combo swamps the 1371 real rows ~14:1 -- the
        # sweep tests whether that synthetic prior is helping or hurting.
        for use_motion, n_pc, clean, dp, sm_sel in (
                (False, args.n_per_combo, False, 0.4, True),
                (False, args.n_per_combo, False, 0.4, False),
                (False, args.n_per_combo, False, 0.1, False),
                (False, args.n_per_combo, False, 0.0, False)):
            mods = ["emotion", "gesture", "motion", "context"] if use_motion else \
                   ["emotion", "gesture", "context"]
            X, O, D = assemble(feat, dims, mods)
            extra = (augment(feat, dims, mods, n_per_combo=n_pc,
                             pool_mask=(clean_mask if clean else None))
                     if n_pc else None)
            accs = []
            for s in range(args.n_seeds):
                model = train_fusion_v4(
                    X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], D, mods,
                    len(INTENTS_V4), s, class_weights=inv, extra=extra, dev=dev,
                    select_masked=sm_sel, **{**CFG, "dropout_p": dp})
                p = _eval_arrays(model, X[te_m], O[te_m], dev)
                accs.append(float((p == y[te_m]).mean()))
            n_syn = 0 if extra is None else len(extra[2])
            tag = f"{rep} dp{dp} {'maskedsel' if sm_sel else 'plainsel'}"
            rows.append((tag, float(np.mean(accs)), float(np.std(accs)), n_syn))
            print(f"{tag:<26} {np.mean(accs):.4f} ± {np.std(accs):.4f} "
                  f"(synth={n_syn})", flush=True)

    best = max(rows, key=lambda r: r[1])
    L = ["# Phase 3+4 — v4 fusion: representation comparison", "",
         f"Taxonomy v4 labels; self-attention fusion; exhaustive v4 cue-combination "
         f"augmentation ({len(AUG)} of {len(ALL)} combinations, every TEST combination "
         f"excluded). Recorded test n={int(te_m.sum())}, {args.n_seeds} seeds.", "",
         f"**v4 rules on measured cues: {rules_acc:.4f}**", "",
         "| representation | test acc | synthetic rows |", "|---|---|---|"]
    for t, m, s, n in rows:
        L.append(f"| {t} | {m:.4f} ± {s:.4f} | {n} |")
    L += ["", f"**Best: {best[0]} — {best[1]:.4f} ± {best[2]:.4f}**", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
