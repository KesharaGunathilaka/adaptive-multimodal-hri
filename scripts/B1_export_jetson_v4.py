"""Train, select, and export the v4 fusion model for Jetson deployment.

This is the ONLY point in the whole v4 programme where a checkpoint is actually
persisted to disk -- every evaluation script (92/95/96/97/A1/A2/A5/A7) trains
fresh, in-memory models and discards them. Deployment needs one concrete file.

Selection: trains N seeds with the exact winning recipe from 97_final_matrices.py
/ A1_baseline_suite.py (softmax-super representation, self-attention, d=64
heads=4 layers=2, cue-dropout 0.7 on recorded rows only, full-rubric
augmentation, inverse-frequency class weights), and picks the seed with the best
MASKED validation score (mean of full-cue + each single-cue-masked val accuracy)
-- not the best test score, to avoid selecting on the number we report.

Writes, into jetson_deploy/fusion/:
  fusion_attn_v4.pt       state_dict, loadable by the v4 model.py
  fusion_config_v4.json   arch + label + selection metadata
  fusion_attn_v4.onnx     ONNX export, dynamic batch axis
and prints a PyTorch-vs-ONNX numerical parity check on the real test set.

    .venv/Scripts/python scripts/B1_export_jetson_v4.py --n-seeds 10
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
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

OUT_DIR = ROOT / "jetson_deploy" / "fusion"
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
Y_IDX = {c: i for i, c in enumerate(INTENTS)}
NC = len(INTENTS)
DP = 0.7


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


def masked_val_score(model, Xva, ova, yva, sl, dev):
    accs = [float((_eval_arrays(model, Xva, ova, dev) == yva).mean())]
    for k, m in enumerate(MODS):
        X2, o2 = Xva.copy(), ova.copy()
        X2[:, sl[m]] = 0.0; o2[:, k] = 0.0
        accs.append(float((_eval_arrays(model, X2, o2, dev) == yva).mean()))
    return float(np.mean(accs)), accs[0]


def train_one(Xtr, otr, ytr, Xva, ova, yva, seed, cw, extra, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    sl = modality_slices(DIMS, MODS)
    dl = DataLoader(_DS(Xtr, otr, ytr, sl, seed, extra), batch_size=512, shuffle=True)
    model = AttentionFusion(d=64, n_heads=4, n_layers=2, dropout=0.3, n_classes=NC,
                            missing_mode="exclude", modality_dims=DIMS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.as_tensor(cw, dtype=torch.float32,
                                                    device=dev),
                             label_smoothing=0.05)
    best_score, best_state, bad = -1.0, None, 0
    for _ in range(80):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            ce(model(xb.to(dev), ob.to(dev)), yb.to(dev)).backward()
            opt.step()
        score, _ = masked_val_score(model, Xva, ova, yva, sl, dev)
        if score > best_score:
            best_score, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    return model, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
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
    tr_m = (real.split == "train").to_numpy()
    va_m = (real.split == "val").to_numpy()
    te_m = (real.split == "test").to_numpy()

    def norm(v):
        s = v.sum(1, keepdims=True)
        return np.divide(v, s, out=np.zeros_like(v), where=s > 0)

    E = np.zeros((len(real), len(SUPERS)), np.float32)
    for c in EMO_F:
        col = f"emo_{c}"
        if col in real.columns:
            E[:, SUPERS.index(EMO_SUPER[c])] += real[col].fillna(0).to_numpy(np.float32)
    feat = {"emotion": norm(E),
            "gesture": norm(real[[f"ges_{c}" for c in GES]].fillna(0).to_numpy(np.float32)),
            "context": norm(real[[f"ctx_{c}" for c in CTX]].fillna(0).to_numpy(np.float32))}
    X = np.concatenate([feat[m] for m in MODS], 1).astype(np.float32)
    O = np.stack([real[f"{p}_obs"].to_numpy(np.float32) for p in
                  ("emo", "ges", "ctx")], 1)

    rng = np.random.default_rng(0)
    keyof = {m: np.array([EMO_SUPER.get(str(v), "neutral") if m == "emotion" else str(v)
                          for v in gt[m]], dtype=object) for m in MODS}
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
    print(f"train={tr_m.sum()} val={va_m.sum()} test={te_m.sum()} "
          f"synth={len(extra[2])}", flush=True)

    yte = y[te_m]
    candidates = []
    for s in range(args.n_seeds):
        model, score = train_one(X[tr_m], O[tr_m], y[tr_m], X[va_m], O[va_m],
                                 y[va_m], s, inv, extra, dev)
        acc = float((_eval_arrays(model, X[te_m], O[te_m], dev) == yte).mean())
        candidates.append((s, score, acc, model))
        print(f"  seed{s}: masked_val={score:.4f} test_acc={acc:.4f}", flush=True)

    best_seed, best_val, best_test, best_model = max(candidates, key=lambda c: c[1])
    print(f"\nchosen: seed={best_seed} masked_val={best_val:.4f} "
          f"test_acc={best_test:.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "fusion_attn_v4.pt"
    # NOTE: nn.Module.to() mutates in place and returns self, so `best_model`
    # itself is now on CPU too -- never reuse it against `dev`-device tensors
    # after this point (see embedding_self_attention.py's docstring for the
    # same landmine hit in the friend's pipeline).
    best_model_cpu = best_model.to("cpu").eval()
    torch.save(best_model_cpu.state_dict(), ckpt_path)

    config = {
        "created": pd.Timestamp.now().isoformat(),
        "spec_version": "v4",
        "arch": "AttentionFusion d=64 heads=4 layers=2 ff=128 pool=cls "
                "missing_mode=exclude modality_dims={emotion:5,gesture:8,context:2}",
        "chosen_seed": best_seed,
        "masked_val_score": round(best_val, 4),
        "test_clip_acc": round(best_test, 4),
        "n_seeds_tried": args.n_seeds,
        "training": {"dropout_p": DP, "dropout_scope": "recorded rows only",
                     "augmentation": "full-rubric, all 80 combos, "
                                     f"{args.n_per_combo}/combo, "
                                     f"{len(extra[2])} synthetic rows",
                     "class_weights": "inverse frequency (train split)"},
        "intents": INTENTS,
        "emotion_superclasses": SUPERS,
        "emotion_superclass_map": EMO_SUPER,
        "gesture_labels": GES,
        "context_labels": CTX,
        "input": "x[15]=emo_super(5)|ges(8)|ctx(2), obs[3] float 1/0, order=[emotion,gesture,context]",
        "note_motion": "motion is DELIBERATELY EXCLUDED -- measured 0.435 accuracy "
                       "on held-out footage and -0.013 value to intent accuracy "
                       "(see docs/FUSION_V4_METHOD_GUIDE.md). Do not feed a motion "
                       "cue into this model.",
        "note_context": "the fusion context token only knows classroom/kitchen. "
                        "For any other scene (hospital, museum, ...) mark context "
                        "UNOBSERVED (obs[2]=0) rather than passing a wrong one-hot.",
    }
    (OUT_DIR / "fusion_config_v4.json").write_text(json.dumps(config, indent=2))
    print(f"-> {ckpt_path}\n-> {OUT_DIR / 'fusion_config_v4.json'}")

    # ---------------- ONNX export + numerical parity ----------------
    onnx_path = OUT_DIR.parent / "onnx" / "fusion_attn_v4.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_x = torch.zeros(1, 15, dtype=torch.float32)
    dummy_o = torch.ones(1, 3, dtype=torch.float32)
    torch.onnx.export(
        best_model_cpu, (dummy_x, dummy_o), str(onnx_path),
        input_names=["x", "obs"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17)
    print(f"-> {onnx_path}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        Xte_np, Ote_np = X[te_m].astype(np.float32), O[te_m].astype(np.float32)
        onnx_logits = sess.run(None, {"x": Xte_np, "obs": Ote_np})[0]
        with torch.no_grad():
            torch_logits = best_model_cpu(torch.from_numpy(Xte_np),
                                          torch.from_numpy(Ote_np)).numpy()
        max_abs_diff = float(np.abs(onnx_logits - torch_logits).max())
        onnx_pred = onnx_logits.argmax(1)
        torch_pred = torch_logits.argmax(1)
        agree = float((onnx_pred == torch_pred).mean())
        onnx_acc = float((onnx_pred == yte).mean())
        print(f"\nONNX vs PyTorch parity: max|logit diff|={max_abs_diff:.2e}, "
              f"prediction agreement={agree:.4f}, ONNX test_acc={onnx_acc:.4f} "
              f"(PyTorch {best_test:.4f})")
        if agree < 1.0:
            print("WARNING: ONNX and PyTorch predictions disagree on some clips -- "
                  "investigate before deploying.")
    except ImportError:
        print("\n(onnxruntime not installed here -- skipping parity check; "
              "verify on the Jetson before trusting the .onnx file)")


if __name__ == "__main__":
    main()
