"""WP5 - robustness studies.

Two experiments, both aimed at the claim the paper actually rests on: that fusion's
advantage over the rule cascade IS robustness to imperfect perception.

  A. Leave-one-actor-out cross-validation
     The headline split holds out scenarios, not people. Here each of the four
     actors with substantial coverage (P01/P02/P04/P06, 490-810 clips over 47-61
     scenarios) is held out entirely, the model trained on the remaining actors,
     and evaluated on the held-out person. This is a much stronger external
     validity claim than a single person-disjoint slice.

  B. Cue-reliability sweep
     A cue's measured vector is replaced, with probability p, by a real vector
     drawn from a DIFFERENT ground-truth class - i.e. a realistic confusion rather
     than random noise. Sweeping p traces how fusion and rules degrade as
     perception worsens, and shows where the crossover is. Because the vectors are
     real, the noise has the same distributional shape as genuine cue errors.

    .venv/Scripts/python scripts/A5_robustness.py --n-seeds 5
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

OUT = ROOT / "results" / "fusion_v2" / "A5_ROBUSTNESS.md"
OUTJSON = ROOT / "results" / "fusion_v2" / "a5_robustness.json"
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
OFF, _i = {}, 0
for _m in MODS:
    OFF[_m] = (_i, _i + DIMS[_m]); _i += DIMS[_m]
INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]
Y_IDX = {c: i for i, c in enumerate(INTENTS)}
NC = len(INTENTS)
DP, ACTORS = 0.7, ["P01", "P02", "P04", "P06"]


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
    person = idx.reindex(real.clip_id.to_numpy())["person_id"].to_numpy()

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
    keyof = {"emotion": sup, "gesture": gt["gesture"].astype(object),
             "context": gt["context"].astype(object)}

    def build_aug(train_mask, seed=0):
        rng = np.random.default_rng(seed)
        pools = {}
        for m in MODS:
            for cls in set(keyof[m][train_mask]):
                s = train_mask & (keyof[m] == cls) & O[:, MODS.index(m)].astype(bool)
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
        return (np.concatenate(Xs).astype(np.float32),
                np.concatenate(Os).astype(np.float32), np.concatenate(Ys))

    def weights(mask):
        cnt = np.bincount(y[mask], minlength=NC).astype(np.float64)
        w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
        return w / w[w > 0].mean()

    def rules_on(Xm, Om):
        e = np.array([SUPERS[i] for i in Xm[:, OFF["emotion"][0]:OFF["emotion"][1]].argmax(1)])
        g = np.array([GES[i] for i in Xm[:, OFF["gesture"][0]:OFF["gesture"][1]].argmax(1)])
        c = np.array([CTX[i] for i in Xm[:, OFF["context"][0]:OFF["context"][1]].argmax(1)])
        return np.array([Y_IDX[rule_intent_v4(a, b, d)] for a, b, d in zip(e, g, c)])

    payload = {}
    L = ["# WP5 - robustness studies", ""]

    # ---------------- A. leave-one-actor-out ----------------
    print("=== leave-one-actor-out ===", flush=True)
    L += ["## A. Leave-one-actor-out cross-validation", "",
          "Each actor is held out entirely; the model is trained on the remaining "
          "actors and evaluated on the held-out person. The headline split holds "
          "out scenarios, so this is a strictly stronger generalization test.", "",
          "| held-out actor | test clips | fusion | rules | diff |",
          "|---|---|---|---|---|"]
    loao = []
    for a in ACTORS:
        te = (person == a)
        pool = ~te
        rng = np.random.default_rng(0)
        vi = rng.random(len(y)) < 0.15
        tr = pool & ~vi
        va = pool & vi
        if te.sum() < 50 or va.sum() < 30:
            continue
        extra = build_aug(tr)
        accs, P = [], []
        for s in range(args.n_seeds):
            mdl = train(X[tr], O[tr], y[tr], X[va], O[va], y[va], s, weights(tr),
                        extra, dev)
            p = _eval_arrays(mdl, X[te], O[te], dev)
            accs.append(float((p == y[te]).mean())); P.append(p)
        P = np.stack(P)
        ens = np.array([Counter(P[:, j]).most_common(1)[0][0]
                        for j in range(P.shape[1])])
        rl = float((rules_on(X[te], O[te]) == y[te]).mean())
        fu = float((ens == y[te]).mean())
        loao.append(dict(actor=a, n=int(te.sum()), fusion=fu, rules=rl,
                         seed_mean=float(np.mean(accs)), seed_std=float(np.std(accs))))
        L.append(f"| {a} | {int(te.sum())} | **{fu:.4f}** | {rl:.4f} | {fu-rl:+.4f} |")
        print(f"  {a}: n={int(te.sum())} fusion={fu:.4f} rules={rl:.4f}", flush=True)
    if loao:
        mf = np.mean([r["fusion"] for r in loao]); mr = np.mean([r["rules"] for r in loao])
        L.append(f"| **mean** | - | **{mf:.4f}** | {mr:.4f} | {mf-mr:+.4f} |")
    payload["loao"] = loao
    L.append("")

    # ---------------- B. cue-reliability sweep ----------------
    print("\n=== cue-reliability sweep ===", flush=True)
    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()
    extra = build_aug(tr_m)
    models = [train(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m], y[va_m], s,
                    weights(tr_m), extra, dev) for s in range(args.n_seeds)]
    yte = y[te_m]

    def corrupt(target, p, seed=0):
        """with prob p, swap a cue vector for a real one of a DIFFERENT class."""
        rng = np.random.default_rng(seed)
        X2 = X[te_m].copy()
        mods = MODS if target == "all" else [target]
        for m in mods:
            a, b = OFF[m]
            src_pool = {}
            for cls in set(keyof[m][tr_m]):
                s = tr_m & (keyof[m] == cls)
                if s.sum():
                    src_pool[cls] = feat[m][s]
            classes = list(src_pool)
            for i, j in enumerate(np.flatnonzero(te_m)):
                if rng.random() >= p:
                    continue
                true_c = keyof[m][j]
                alts = [c for c in classes if c != true_c]
                if not alts:
                    continue
                pool = src_pool[alts[rng.integers(len(alts))]]
                X2[i, a:b] = pool[rng.integers(len(pool))]
        return X2

    L += ["## B. Cue-reliability sweep", "",
          "With probability *p* a cue's measured vector is replaced by a real "
          "vector from a different ground-truth class - a realistic confusion, not "
          "random noise. This traces how each method degrades as perception "
          "worsens.", ""]
    sweep = {}
    for target in ["emotion", "gesture", "all"]:
        L += [f"**Corrupting {target}**", "",
              "| p | fusion | rules | diff |", "|---|---|---|---|"]
        rows = []
        for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            X2 = corrupt(target, p)
            P = np.stack([_eval_arrays(md, X2, O[te_m], dev) for md in models])
            ens = np.array([Counter(P[:, j]).most_common(1)[0][0]
                            for j in range(P.shape[1])])
            fu = float((ens == yte).mean())
            rl = float((rules_on(X2, O[te_m]) == yte).mean())
            rows.append(dict(p=p, fusion=fu, rules=rl))
            L.append(f"| {p:.1f} | **{fu:.4f}** | {rl:.4f} | {fu-rl:+.4f} |")
            print(f"  {target} p={p:.1f}: fusion={fu:.4f} rules={rl:.4f}", flush=True)
        sweep[target] = rows
        L.append("")
    payload["sweep"] = sweep

    # Report the ratio as well as the difference: as p grows both methods approach
    # chance, so the ABSOLUTE gap must shrink even if the relative advantage holds.
    L += ["Note on reading these curves: as *p* grows both methods approach chance, "
          "so the absolute difference necessarily narrows. The relative advantage "
          "(fusion / rules) is the quantity that is comparable across noise levels:",
          "", "| corrupted | p=0.0 | p=0.2 | p=0.4 |", "|---|---|---|---|"]
    for target, rows in sweep.items():
        by_p = {r["p"]: r for r in rows}
        cells = []
        for p_ in (0.0, 0.2, 0.4):
            r = by_p.get(p_)
            cells.append(f"{r['fusion']/r['rules']:.3f}x" if r and r["rules"] else "-")
        L.append(f"| {target} | " + " | ".join(cells) + " |")
    L += ["", "A roughly constant ratio means fusion keeps a proportional advantage "
          "at every level of perception quality, rather than the advantage being an "
          "artefact of one operating point.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    OUTJSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
