"""WP2 - isolate the contribution of the perception-aware taxonomy.

The paper's central claim is that redesigning the INTENT TAXONOMY around what the
frozen cue models can actually discriminate is what unlocks the gain - not the
fusion architecture (WP1 showed several architectures tie at ~0.82). This holds
the model, augmentation budget, seeds and evaluation protocol fixed and varies
only the LABEL SPACE.

Arms - each reverts one v4 design rule back to its v3 form:

  v3            the original rubric: 4 cues incl. motion, fine emotion classes,
                F07/F08 split on anger-vs-disgust
  v4            gesture-first, emotion superclasses, motion excluded,
                F07/F08 split on gesture polarity            <- ours
  v4-emosplit   v4 but F07/F08 split on anger-vs-disgust (Disgust->Anger .125)
  v4-finealarm  v4 but only Fear (not Surprise) triggers F02 (Fear<->Surprise)
  v4-withmotion v4 but motion discriminates point+walking (motion is 0.435)

Every arm gets its own labels, its own full-rubric augmentation and its own rule
baseline. Synthetic budget is held at ~24k rows for every arm so the comparison
is not confounded by training-set size (v3's space is 448 combinations against
v4's 80).

    .venv/Scripts/python scripts/A2_taxonomy_ablation.py --n-seeds 10
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

from fusion.baselines.rule_based import rule_intent as rule_intent_v3  # noqa: E402
from fusion.model.generic_train import _eval_arrays, modality_slices  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402
from scripts.realworld_eval import merged_unimodal as MU  # noqa: E402
from scripts.realworld_eval.stats import mcnemar  # noqa: E402

OUT = ROOT / "results" / "fusion_v2" / "A2_TAXONOMY_ABLATION.md"
OUTJSON = ROOT / "results" / "fusion_v2" / "a2_taxonomy_ablation.json"
EMO_F = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
GES = ["idle", "wave", "point", "thumbs_up", "thumbs_down", "beckoning",
       "raise_hand", "both_hands_up"]
MOT = ["sitting", "standing", "walking", "stepping_back"]
CTX = ["classroom", "kitchen"]
SUPERS = ["alarm", "hostile", "positive", "low", "neutral"]
EMO_SUPER = {"Fear": "alarm", "Surprise": "alarm", "Anger": "hostile",
             "Disgust": "hostile", "Happy": "positive", "Sad": "low",
             "Neutral": "neutral"}
E2R = {"Surprise": "surprise", "Fear": "fear", "Disgust": "disgust",
       "Happy": "happy", "Sad": "sad", "Anger": "anger", "Neutral": "neutral"}
M2R = {"sitting": "sitting", "standing": "standing", "walking": "walking",
       "stepping_back": "step back"}
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
Y_IDX = {c: i for i, c in enumerate(INTENTS)}
NC = len(INTENTS)
DP, SYNTH_BUDGET = 0.7, 24000


# --------------------------- the rubric variants ---------------------------
def _v4_core(emo_super, ges, ctx, *, emo_fine=None, mot=None,
             emosplit=False, finealarm=False, withmotion=False):
    if finealarm:
        if emo_fine == "Fear":
            return "F02"
    elif emo_super == "alarm":
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
        if emo_super == "hostile":
            # v3 used motion here: angry pointing while WALKING means "make way"
            if withmotion and mot == "walking":
                return "F06"
            return "F07" if withmotion else "F06"
        return "F03"
    if ges == "thumbs_up":
        if emo_super != "hostile":
            return "F01"
        # v4 splits F07/F08 on gesture polarity; v3 split it on anger-vs-disgust
        return ("F07" if emo_fine == "Anger" else "F08") if emosplit else "F07"
    if ges == "thumbs_down":
        if emo_super != "hostile":
            return "F04"
        return ("F07" if emo_fine == "Anger" else "F08") if emosplit else "F08"
    if emo_super == "low":
        return "F10"
    if emo_super == "hostile":
        return "F07"
    return "F05"


ARMS = {
    "v3": dict(kind="v3", mods=["emotion", "gesture", "motion", "context"],
               desc="original rubric: 4 cues, fine emotion, anger/disgust split"),
    "v4": dict(kind="v4", mods=["emotion", "gesture", "context"], opts={},
               desc="ours: gesture-first, superclasses, no motion"),
    "v4-emosplit": dict(kind="v4", mods=["emotion", "gesture", "context"],
                        opts=dict(emosplit=True),
                        desc="revert F07/F08 to the anger-vs-disgust split"),
    "v4-finealarm": dict(kind="v4", mods=["emotion", "gesture", "context"],
                         opts=dict(finealarm=True),
                         desc="revert alarm superclass: only Fear triggers F02"),
    "v4-withmotion": dict(kind="v4", mods=["emotion", "gesture", "motion", "context"],
                          opts=dict(withmotion=True),
                          desc="re-admit motion as a discriminator"),
}


def label_of(arm, emo_fine, ges, mot, ctx):
    cfg = ARMS[arm]
    if cfg["kind"] == "v3":
        i = rule_intent_v3(E2R[emo_fine], ges, M2R[mot], ctx)
        return "F01" if i == "F09" else i
    return _v4_core(EMO_SUPER[emo_fine], ges, ctx, emo_fine=emo_fine, mot=mot,
                    **cfg.get("opts", {}))


def dims_for(arm):
    cfg = ARMS[arm]
    d = {}
    for m in cfg["mods"]:
        d[m] = {"emotion": len(EMO_F) if cfg["kind"] == "v3" else len(SUPERS),
                "gesture": len(GES), "motion": len(MOT), "context": len(CTX)}[m]
    return d


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


def train(dims, mods, Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    sl = modality_slices(dims, mods)
    dl = DataLoader(_DS(Xtr, otr, ytr, sl, seed, extra), batch_size=512, shuffle=True)
    model = AttentionFusion(d=64, n_heads=4, n_layers=2, dropout=0.3, n_classes=NC,
                            missing_mode="exclude", modality_dims=dims).to(dev)
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
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clips, windows = G.load_clips_and_windows()
    real = G.to_common_schema(G.build_real(clips, windows))
    idx = clips.set_index("clip_id")
    gt = {m: idx.reindex(real.clip_id.to_numpy())[f"gt_{m}"].to_numpy()
          for m in ("emotion", "gesture", "motion", "context")}
    ok = np.ones(len(real), bool)
    for m in ("emotion", "gesture"):
        mk = idx.reindex(real.clip_id.to_numpy())[MU.MASK_COL[m]].to_numpy().astype(bool)
        ok &= pd.notna(gt[m]) & ~mk
    for m in ("motion", "context"):
        ok &= pd.notna(gt[m])
    real = real[ok].reset_index(drop=True)
    gt = {m: v[ok] for m, v in gt.items()}
    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()

    def norm(v):
        s = v.sum(1, keepdims=True)
        return np.divide(v, s, out=np.zeros_like(v), where=s > 0)

    Efine = norm(real[[f"emo_{c}" for c in EMO_F]].fillna(0).to_numpy(np.float32))
    Esup = np.zeros((len(real), len(SUPERS)), np.float32)
    for j, c in enumerate(EMO_F):
        Esup[:, SUPERS.index(EMO_SUPER[c])] += Efine[:, j]
    FEAT = {"emotion_fine": Efine, "emotion_super": norm(Esup),
            "gesture": norm(real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy(np.float32)),
            "motion": norm(real[[f"mot_{c}" for c in MOT]].fillna(0).to_numpy(np.float32)),
            "context": norm(real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy(np.float32))}
    OBS = {m: real[f"{p}_obs"].to_numpy(np.float32) for m, p in
           (("emotion", "emo"), ("gesture", "ges"), ("motion", "mot"),
            ("context", "ctx"))}
    ARG = {"emotion": np.array([EMO_F[i] for i in Efine.argmax(1)]),
           "gesture": np.array([GES[i] for i in FEAT["gesture"].argmax(1)]),
           "motion": np.array([MOT[i] for i in FEAT["motion"].argmax(1)]),
           "context": np.array([CTX[i] for i in FEAT["context"].argmax(1)])}
    print(f"train={tr_m.sum()} val={va_m.sum()} test={te_m.sum()}", flush=True)

    rows, payload, preds = [], {}, {}
    for arm, cfg in ARMS.items():
        mods = cfg["mods"]
        dims = dims_for(arm)
        efeat = "emotion_fine" if cfg["kind"] == "v3" else "emotion_super"
        f = {m: FEAT[efeat if m == "emotion" else m] for m in mods}
        X = np.concatenate([f[m] for m in mods], 1).astype(np.float32)
        O = np.stack([OBS[m] for m in mods], 1)
        y = np.array([Y_IDX[label_of(arm, e, g, mo, c)] for e, g, mo, c in
                      zip(gt["emotion"], gt["gesture"], gt["motion"], gt["context"])],
                     np.int64)

        # augmentation over this arm's own cue space, budget-matched
        axes = []
        for m in mods:
            axes.append(EMO_F if (m == "emotion" and cfg["kind"] == "v3") else
                        SUPERS if m == "emotion" else
                        {"gesture": GES, "motion": MOT, "context": CTX}[m])
        combos = list(itertools.product(*axes))
        n_per = max(20, SYNTH_BUDGET // max(1, len(combos)))
        keyof = {}
        for m in mods:
            if m == "emotion":
                keyof[m] = (gt["emotion"] if cfg["kind"] == "v3" else
                            np.array([EMO_SUPER[e] for e in gt["emotion"]], dtype=object))
            else:
                keyof[m] = gt[m].astype(object)
        pools = {}
        for m in mods:
            for cls in set(keyof[m][tr_m]):
                s = tr_m & (keyof[m] == cls) & OBS[m].astype(bool)
                if s.sum():
                    pools[(m, cls)] = f[m][s]
        rng = np.random.default_rng(0)
        Xs, Os, Ys = [], [], []
        for combo in combos:
            need = dict(zip(mods, combo))
            if any((m, need[m]) not in pools for m in mods):
                continue
            # label needs all four cues; fill the ones this arm does not model
            e = need["emotion"] if cfg["kind"] == "v3" else None
            if e is None:
                # pick any fine emotion inside the superclass for label purposes
                sup = need["emotion"]
                e = next(k for k, v in EMO_SUPER.items() if v == sup)
            lab = label_of(arm, e, need["gesture"],
                           need.get("motion", "standing"), need.get("context",
                                                                    "classroom"))
            Xs.append(np.concatenate(
                [pools[(m, need[m])][rng.integers(0, len(pools[(m, need[m])]), n_per)]
                 for m in mods], 1))
            Os.append(np.ones((n_per, len(mods)), np.float32))
            Ys.append(np.full(n_per, Y_IDX[lab], np.int64))
        extra = (np.concatenate(Xs).astype(np.float32),
                 np.concatenate(Os).astype(np.float32), np.concatenate(Ys))

        cnt = np.bincount(y[tr_m], minlength=NC).astype(np.float64)
        inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
        inv = inv / inv[inv > 0].mean()
        yte = y[te_m]

        rul = np.array([Y_IDX[label_of(arm, ARG["emotion"][i], ARG["gesture"][i],
                                       ARG["motion"][i], ARG["context"][i])]
                        for i in np.flatnonzero(te_m)])
        # ceiling: majority intent per measured cue signature this arm can see
        gsig = defaultdict(Counter)
        for i, j in enumerate(np.flatnonzero(te_m)):
            gsig[tuple(ARG[m][j] for m in mods)][yte[i]] += 1
        tot = sum(sum(c.values()) for c in gsig.values())
        ceil = sum(max(c.values()) for c in gsig.values()) / tot

        accs, P = [], []
        for s in range(args.n_seeds):
            mdl = train(dims, mods, X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m],
                        y[va_m], s, inv, extra, dev)
            p = _eval_arrays(mdl, X[te_m], O[te_m], dev)
            accs.append(float((p == yte).mean())); P.append(p)
        P = np.stack(P)
        ens = np.array([Counter(P[:, j]).most_common(1)[0][0]
                        for j in range(P.shape[1])])
        preds[arm] = (ens, yte)
        rows.append(dict(arm=arm, desc=cfg["desc"], n_combos=len(combos),
                         n_synth=len(extra[2]), fusion=float((ens == yte).mean()),
                         fusion_f1=macro_f1(yte, ens),
                         rules=float((rul == yte).mean()), ceiling=ceil,
                         seed_mean=float(np.mean(accs)), seed_std=float(np.std(accs))))
        payload[arm] = rows[-1]
        print(f"{arm:<15} fusion={rows[-1]['fusion']:.4f} rules={rows[-1]['rules']:.4f} "
              f"ceiling={ceil:.4f} (combos={len(combos)} synth={len(extra[2])})",
              flush=True)

    base = next(r for r in rows if r["arm"] == "v4")
    L = ["# WP2 - taxonomy ablation", "",
         f"Model, augmentation budget (~{SYNTH_BUDGET} synthetic rows), seed list "
         f"({args.n_seeds} seeds) and evaluation protocol are identical across arms. "
         f"Only the LABEL SPACE changes. Each arm gets its own rule baseline and its "
         f"own measured-cue ceiling, since those are properties of the taxonomy.", "",
         "| taxonomy | what changed | fusion | macro-F1 | rules | ceiling | vs v4 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["arm"] == "v4":
            delta = "-"
        else:
            mc = mcnemar(preds["v4"][0] == preds["v4"][1],
                         preds[r["arm"]][0] == preds[r["arm"]][1])
            delta = f"{r['fusion']-base['fusion']:+.4f} (p={mc['p_value']:.3g})"
        star = " **(ours)**" if r["arm"] == "v4" else ""
        L.append(f"| `{r['arm']}`{star} | {r['desc']} | **{r['fusion']:.4f}** | "
                 f"{r['fusion_f1']:.4f} | {r['rules']:.4f} | {r['ceiling']:.4f} | "
                 f"{delta} |")
    L += ["", "Per-seed: " + ", ".join(
        f"`{r['arm']}` {r['seed_mean']:.4f}+/-{r['seed_std']:.4f}" for r in rows), "",
        "The `ceiling` column is the key diagnostic: it is the best score ANY method "
        "could reach on that taxonomy given the cues the perception stack actually "
        "delivers. A taxonomy that places intent boundaries on unreliable cue "
        "distinctions lowers its own ceiling before any model is trained.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    OUTJSON.write_text(json.dumps(payload, indent=2, default=float))
    print(chr(10).join(L))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
