"""WP7 - what does the fusion model actually attend to?

`nn.TransformerEncoderLayer` calls its self-attention with `need_weights=False`,
so the weights are discarded. Forward pre-hooks flip that flag and forward hooks
collect the result, leaving the computation untouched.

For every test clip we take the CLS token's attention over the three cue tokens -
i.e. how much each cue contributed to the pooled representation that drives the
classification - and average it per true intent. The result is an
intent x cue matrix showing which evidence the model relies on for each decision.

The interesting prediction, from the v4 design: intents whose definition turns on
emotion overriding a gesture (F07 frustration, F08 break request) should place
more weight on emotion than intents defined by the gesture itself (F03 task
request, F01 greeting).

Writes a markdown table and a heatmap PNG.

    .venv/Scripts/python scripts/A7_attention_analysis.py --n-seeds 5
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

OUT = ROOT / "results" / "fusion_v2" / "A7_ATTENTION.md"
PNG = ROOT / "results" / "fusion_v2" / "a7_attention_heatmap.png"
EMO_F = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GES = ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
       "raise_hand", "both_hands_up"]
CTX = ["classroom", "kitchen"]
SUPERS = ["alarm", "hostile", "positive", "low", "neutral"]
EMO_SUPER = {"Fear": "alarm", "Surprise": "alarm", "Anger": "hostile",
             "Disgust": "hostile", "Happy": "positive", "Sad": "low",
             "Neutral": "neutral"}
MODS = ["emotion", "gesture", "context"]
DIMS = {"emotion": len(SUPERS), "gesture": len(GES), "context": len(CTX)}
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
INTENT_NAME = {"F01": "Greeting", "F02": "Emergency", "F03": "Task request",
               "F04": "Help request", "F05": "Busy", "F06": "Give space",
               "F07": "Frustration", "F08": "Break request", "F10": "Disengaged"}
Y_IDX = {c: i for i, c in enumerate(INTENTS)}
NC = len(INTENTS)
DP = 0.7
# intents whose v4 definition requires emotion to override the gesture
EMOTION_LED = {"F02", "F06", "F07", "F08", "F10"}


def rule_intent_v4(s, g, c):
    if s == "alarm":
        return "F02"
    if g == "both_hands_up":
        return "F07" if s == "hostile" else "F05"
    if g == "beckoning":
        return "F04" if s == "low" else "F03"
    if g == "raise_hand":
        return "F04" if c == "classroom" else "F01"
    if g == "wave":
        return "F06" if s == "hostile" else "F01"
    if g == "point":
        return "F06" if s == "hostile" else "F03"
    if g == "thumbs_up":
        return "F07" if s == "hostile" else "F01"
    if g == "thumbs_down":
        return "F08" if s == "hostile" else "F04"
    if s == "low":
        return "F10"
    if s == "hostile":
        return "F07"
    return "F05"


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
        if i < self.n_real:
            drop = (self.rng.random(len(self.mods)) < DP) & o.astype(bool)
            if (o.astype(bool) & ~drop).sum() == 0:
                drop[:] = False
            for k, m in enumerate(self.mods):
                if drop[k]:
                    x[self.slices[m]] = 0.0; o[k] = 0.0
        return torch.from_numpy(x), torch.from_numpy(o), int(self.y[i])


def train(Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    sl = modality_slices(DIMS, MODS)
    dl = DataLoader(_DS(Xtr, otr, ytr, sl, seed, extra), batch_size=512, shuffle=True)
    model = AttentionFusion(d=64, n_heads=4, n_layers=2, dropout=0.3, n_classes=NC,
                            missing_mode="exclude", modality_dims=DIMS).to(dev)
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


def capture_attention(model, X, O, dev, bs=512):
    """CLS-token attention over the cue tokens, averaged over layers.

    `TransformerEncoderLayer` calls self-attention with `need_weights=False`, so
    the weights are never returned. Forward PRE-hooks flip that flag on the way in
    and forward hooks collect the result on the way out; the computation itself is
    untouched. (Reassigning `self_attn.forward` does not work here - the call is
    not routed through the instance attribute.)"""
    store, handles = [], []

    def pre(mod, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return args, kwargs

    def post(mod, args, kwargs, out):
        if out[1] is not None:
            store.append(out[1].detach().cpu())

    for layer in model.encoder.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(post, with_kwargs=True))
    try:
        model.eval()
        cls_rows = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                store.clear()
                xb = torch.from_numpy(X[i:i + bs]).to(dev)
                ob = torch.from_numpy(O[i:i + bs]).to(dev)
                model(xb, ob)
                if not store:
                    return None
                w = torch.stack(store, 0).mean(0)       # [B, n_tok, n_tok]
                cls_rows.append(w[:, 0, 1:].numpy())     # CLS row -> cue columns
        return np.concatenate(cls_rows)
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-per-combo", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    idx = clips.set_index("clip_id")
    gt = {m: idx.reindex(real.clip_id.to_numpy())[f"gt_{m}"].to_numpy() for m in MODS}
    ok = np.ones(len(real), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok &= pd.notna(gt[m]) & ~mk
    real = real[ok].reset_index(drop=True)
    gt = {m: v[ok] for m, v in gt.items()}
    sup = np.array([EMO_SUPER.get(str(e), "neutral") for e in gt["emotion"]],
                   dtype=object)
    y = np.array([Y_IDX[rule_intent_v4(s, str(g), str(c))]
                  for s, g, c in zip(sup, gt["gesture"], gt["context"])], np.int64)

    def norm(v):
        s = v.sum(1, keepdims=True)
        return np.divide(v, s, out=np.zeros_like(v), where=s > 0)

    Ef = norm(real[[f"emo_{c}" for c in EMO_F]].fillna(0).to_numpy(np.float32))
    Es = np.zeros((len(real), len(SUPERS)), np.float32)
    for j, c in enumerate(EMO_F):
        Es[:, SUPERS.index(EMO_SUPER[c])] += Ef[:, j]
    feat = {"emotion": norm(Es),
            "gesture": norm(real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy(np.float32)),
            "context": norm(real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy(np.float32))}
    X = np.concatenate([feat[m] for m in MODS], 1).astype(np.float32)
    O = np.stack([real[f"{p}_obs"].to_numpy(np.float32) for p in
                  ("emo", "ges", "ctx")], 1)
    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()

    keyof = {"emotion": sup, "gesture": gt["gesture"].astype(object),
             "context": gt["context"].astype(object)}
    rng = np.random.default_rng(0)
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
    cnt = np.bincount(y[tr_m], minlength=NC).astype(np.float64)
    inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
    inv = inv / inv[inv > 0].mean()

    per_seed = []
    for s in range(args.n_seeds):
        mdl = train(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], s, inv,
                    extra, dev)
        A = capture_attention(mdl, X[te_m], O[te_m], dev)
        if A is None:
            sys.exit("could not capture attention weights")
        per_seed.append(A)
        print(f"  seed{s}: captured {A.shape}", flush=True)
    A = np.mean(np.stack(per_seed), axis=0)       # [n_test, 3]
    A = A / np.clip(A.sum(1, keepdims=True), 1e-9, None)
    yte = y[te_m]

    rows = []
    for i, c in enumerate(INTENTS):
        m = yte == i
        if m.sum() == 0:
            continue
        rows.append((c, int(m.sum()), *A[m].mean(0)))

    L = ["# WP7 - what the fusion model attends to", "",
         f"CLS-token attention over the three cue tokens, averaged over layers, "
         f"over {args.n_seeds} seeds, and over the test clips of each intent. Rows "
         f"sum to 1: they show the share of evidence the model drew from each cue "
         f"when producing the representation it classified from.", "",
         "| intent | meaning | clips | emotion | gesture | context |",
         "|---|---|---|---|---|---|"]
    for c, n, e, g, x in rows:
        L.append(f"| **{c}** | {INTENT_NAME[c]} | {n} | {e:.3f} | {g:.3f} | {x:.3f} |")
    led = [r for r in rows if r[0] in EMOTION_LED]
    ges = [r for r in rows if r[0] not in EMOTION_LED]
    if led and ges:
        e_led = float(np.mean([r[2] for r in led]))
        e_ges = float(np.mean([r[2] for r in ges]))
        L += ["", "## Does the model use the evidence the taxonomy says it should?", "",
              "The v4 rubric defines F02/F06/F07/F08/F10 by emotion overriding or "
              "qualifying the gesture, while F01/F03/F04/F05 are led by the gesture "
              "itself. If the model learned the intended structure, the first group "
              "should draw more on the emotion token.", "",
              f"- emotion-defined intents: mean emotion attention **{e_led:.3f}**",
              f"- gesture-defined intents: mean emotion attention **{e_ges:.3f}**",
              f"- difference: **{e_led - e_ges:+.3f}**", "",
              ("The model does place more weight on emotion exactly where the "
               "taxonomy requires it, which is evidence that it learned the intended "
               "decision structure rather than a surface correlation."
               if e_led > e_ges + 0.02 else
               "The difference is small, so this attention read-out does NOT support "
               "a claim that the model internalised the intended structure. Attention "
               "weights are a description of the forward pass, not an explanation, "
               "and should not be over-interpreted."), ""]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        M = np.array([[r[2], r[3], r[4]] for r in rows])
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=M.max())
        ax.set_xticks(range(3)); ax.set_xticklabels(["emotion", "gesture", "context"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r[0]} {INTENT_NAME[r[0]]}" for r in rows], fontsize=8)
        for i in range(len(rows)):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=8)
        ax.set_title("Cue attention per intent", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(PNG, dpi=200)
        plt.close(fig)
        L += [f"![cue attention per intent]({PNG.name})", ""]
        print(f"heatmap -> {PNG}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(no heatmap: {e})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    print(chr(10).join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
