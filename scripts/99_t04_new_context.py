"""T04 - context generalization, measured properly on genuinely unseen environments.

The v4 fusion model is trained only on Classroom and Kitchen. The AI scenarios
provide Hospital and Museum: two environments it has never seen. Their clips are
processed by `98_extract_ai_adaptive.py` through the SAME cue pipeline as the
training data, so this measures a change of environment rather than a change of
pipeline.

How the unseen context is handled: the model's context token covers only
classroom/kitchen, so for Hospital/Museum clips the context cue is marked
UNOBSERVED and excluded from attention - the honest treatment of an environment
the system has no representation for. Kitchen AI clips are reported separately as
a control: same generation process, but a context the model DOES know, so any gap
between them isolates the effect of novelty from the effect of AI-generated
footage.

Ground truth: the v4 rubric applied to each scenario's designed cue signature.
None of these scenarios use `raise_hand`, the rubric's only context-dependent
branch, so their labels are context-independent and unambiguous.

    .venv/Scripts/python scripts/99_t04_new_context.py --n-seeds 10
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
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model.generic_train import _eval_arrays, modality_slices  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402

AI_CUES = ROOT / "results" / "fusion_v2" / "ai_clips_adaptive_cues.parquet"
AI_MEAS = ROOT / "results" / "ai_scenarios"
OUT = ROOT / "results" / "fusion_v2" / "T04_NEW_CONTEXT.md"
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
DP, REAL_ONLY = 0.7, True
# designed cue signature per AI scenario (measured + rubric-validated, script 63)
AI_SIG = {
    "S64_F10": ("low", "idle"), "S65_F07": ("hostile", "thumbs_up"),
    "S66_F04": ("low", "thumbs_down"), "S67_F08": ("hostile", "thumbs_up"),
    "S68_F02": ("alarm", "idle"), "S69_F06": ("hostile", "point"),
    "S70_F04": ("low", "thumbs_down"), "S71_F01": ("positive", "wave"),
    "S72_F10": ("low", "idle"), "S73_F10": ("low", "idle"),
}


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


def norm(v):
    s = v.sum(1, keepdims=True)
    return np.divide(v, s, out=np.zeros_like(v), where=s > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-per-combo", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not AI_CUES.exists():
        sys.exit("run scripts/98_extract_ai_adaptive.py first")

    # ---------- recorded data (training) ----------
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

    E = np.zeros((len(real), len(SUPERS)), np.float32)
    for c in CLS["emotion"]:
        col = f"emo_{c}"
        if col in real.columns:
            E[:, SUPERS.index(EMO_SUPER[c])] += real[col].fillna(0).to_numpy(np.float32)
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

    # ---------- AI clips, adaptive pipeline ----------
    ai = pd.read_parquet(AI_CUES)
    ai = ai[ai.scenario_dir.isin(AI_SIG)].reset_index(drop=True)
    Ea = np.zeros((len(ai), len(SUPERS)), np.float32)
    for c in CLS["emotion"]:
        col = f"emo_{c}"
        if col in ai.columns:
            Ea[:, SUPERS.index(EMO_SUPER[c])] += ai[col].fillna(0).to_numpy(np.float32)
    fa = {"emotion": norm(Ea),
          "gesture": norm(ai[[f"ges_{c}" for c in GES]].fillna(0).to_numpy(np.float32)),
          "context": norm(ai[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy(np.float32))}
    Xa = np.concatenate([fa[m] for m in MODS], 1).astype(np.float32)
    Oa = np.stack([ai[f"{PREFIX[m]}_obs"].to_numpy(np.float32) for m in MODS], 1)
    novel = ai.context_folder.isin(["hospital", "museum"]).to_numpy()
    # an unseen environment has no representation in the 2-class context token
    Oa[novel, MODS.index("context")] = 0.0
    ya = np.array([Y_IDX[rule_intent_v4(*AI_SIG[s], "classroom")]
                   for s in ai.scenario_dir], np.int64)
    designed = np.array([s.split("_")[1] for s in ai.scenario_dir])
    agree = float(np.mean([INTENTS[v] == d for v, d in zip(ya, designed)]))
    print(f"AI clips={len(ai)} novel(hosp/museum)={int(novel.sum())} "
          f"control(kitchen)={int((~novel).sum())} | rubric vs folder intent "
          f"agreement={agree:.3f}", flush=True)

    rules_ai = np.array([Y_IDX[rule_intent_v4(
        EMO_SUPER.get(CLS["emotion"][int(np.argmax(
            [ai.iloc[i][f"emo_{c}"] for c in CLS["emotion"]]))], "neutral"),
        GES[int(np.argmax([ai.iloc[i][f"ges_{c}"] for c in GES]))],
        "classroom")] for i in range(len(ai))])

    models, accs = [], []
    for s in range(args.n_seeds):
        mdl = train(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], s, inv,
                    extra, dev)
        models.append(mdl)
        accs.append(float((_eval_arrays(mdl, Xa, Oa, dev) == ya).mean()))
        print(f"  seed{s}: AI acc={accs[-1]:.4f}", flush=True)
    P = np.stack([_eval_arrays(m, Xa, Oa, dev) for m in models])
    fus = np.array([Counter(P[:, j]).most_common(1)[0][0] for j in range(P.shape[1])])
    fo, ro = fus == ya, rules_ai == ya

    def line(nm, m):
        if m.sum() == 0:
            return f"| {nm} | 0 | - | - | - |"
        return (f"| {nm} | {int(m.sum())} | {fo[m].mean():.4f} | {ro[m].mean():.4f} | "
                f"{fo[m].mean()-ro[m].mean():+.4f} |")

    L = ["# T04 - context generalization on unseen environments", "",
         "The v4 model is trained only on **Classroom** and **Kitchen**. These clips "
         "are **Hospital** and **Museum** - environments it has never seen - "
         "processed through the same cue pipeline as the training data, so this "
         "isolates the effect of a new environment. Because the context token only "
         "covers the two known rooms, context is marked *unobserved* for the novel "
         "environments and excluded from attention.", "",
         f"AI Kitchen clips are a **control**: identical generation process, but a "
         f"context the model knows. {args.n_seeds} seeds, majority vote.", "",
         f"Per-seed accuracy over all AI clips: **{np.mean(accs):.4f} +/- "
         f"{np.std(accs):.4f}**.", "",
         "| slice | clips | fusion | rules | diff |", "|---|---|---|---|---|",
         line("**Novel environments (Hospital + Museum)**", novel),
         line("- Hospital", (ai.context_folder == "hospital").to_numpy()),
         line("- Museum", (ai.context_folder == "museum").to_numpy()),
         line("Control: AI Kitchen (known environment)", ~novel),
         line("All AI clips", np.ones(len(ai), bool)), "",
         "## Per scenario", "",
         "| scenario | environment | cue signature | intent | clips | fusion | rules |",
         "|---|---|---|---|---|---|---|"]
    for s in sorted(set(ai.scenario_dir)):
        m = (ai.scenario_dir == s).to_numpy()
        es, g = AI_SIG[s]
        L.append(f"| {s} | {ai.context_folder[m].iloc[0]} | {es} / {g} | "
                 f"{INTENTS[ya[m][0]]} | {int(m.sum())} | {fo[m].mean():.3f} | "
                 f"{ro[m].mean():.3f} |")
    L += ["", "## Reference", "",
          "Recorded test set (known environments): fusion **0.8202**, rules 0.7646.",
          "", f"Novel-environment accuracy of **{fo[novel].mean():.4f}** against "
          f"**{fo[~novel].mean():.4f}** on the AI Kitchen control isolates the cost "
          f"of environment novelty at **{fo[novel].mean()-fo[~novel].mean():+.4f}** - "
          f"the remaining difference from the recorded-test figure is attributable to "
          f"AI-generated footage rather than to the new environment.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    print(chr(10).join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
