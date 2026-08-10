"""Generalised training loop + augmentation for fusion inputs of ARBITRARY
per-modality width (2026-08-08) -- `fusion/model/datasets.py::WindowDataset`
and `train.py::train_fusion` are hardcoded to the 24-dim (7/8/4/5) probability
layout; this is the embedding-fusion analogue, reused for BOTH the embedding
config and a same-code-path probs baseline so the two are paired fairly
(same training loop, same seeds, only the input differs).

Jitter differs from `WindowDataset`'s: probability jitter perturbs in
log-space and re-normalises (valid only because probabilities sum to 1).
Embeddings have no such constraint, so jitter here is plain additive
Gaussian noise -- meaningful as long as the input is roughly standardised
(the embedding-fusion pipeline z-scores PCA output for exactly this reason).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import AttentionFusion


def modality_slices(dims: dict, order: list[str]) -> dict[str, slice]:
    slices, i = {}, 0
    for m in order:
        slices[m] = slice(i, i + dims[m])
        i += dims[m]
    return slices


class GenericWindowDataset(Dataset):
    def __init__(self, X, obs, y, slices: dict[str, slice], dropout_p=0.0,
                jitter_sigma=0.0, seed=0, extra=None):
        if extra is not None:
            X = np.concatenate([X, extra[0].astype(np.float32)])
            obs = np.concatenate([obs, extra[1].astype(np.float32)])
            y = np.concatenate([y, extra[2].astype(np.int64)])
        self.X, self.obs, self.y = X, obs, y
        self.slices = slices
        self.modalities = list(slices)
        self.dropout_p, self.jitter_sigma = dropout_p, jitter_sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, obs = self.X[i].copy(), self.obs[i].copy()
        n_mod = len(self.modalities)

        if self.dropout_p > 0:
            drop = self.rng.random(n_mod) < self.dropout_p
            drop &= obs.astype(bool)
            while drop.sum() > 2:
                drop[self.rng.choice(np.flatnonzero(drop))] = False
            if (obs.astype(bool) & ~drop).sum() == 0:
                drop[:] = False
            for k, m in enumerate(self.modalities):
                if drop[k]:
                    x[self.slices[m]] = 0.0
                    obs[k] = 0.0

        if self.jitter_sigma > 0:
            for k, m in enumerate(self.modalities):
                if obs[k] and self.rng.random() < 0.5:
                    sl = self.slices[m]
                    x[sl] = x[sl] + self.rng.normal(0, self.jitter_sigma, x[sl].shape)

        return torch.from_numpy(x), torch.from_numpy(obs), int(self.y[i])


def _eval_arrays(model, X, obs, device, bs=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            ob = torch.from_numpy(obs[i:i + bs]).to(device)
            preds.append(model(xb, ob).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def _masked_val_acc(model, Xva, obs_va, yva, slices, device):
    """Same rationale as `train.py::_masked_val_acc`: select on mean of
    unmasked + each single-modality-masked val accuracy, not unmasked alone."""
    modalities = list(slices)
    accs = [float((_eval_arrays(model, Xva, obs_va, device) == yva).mean())]
    for k, m in enumerate(modalities):
        X2, o2 = Xva.copy(), obs_va.copy()
        X2[:, slices[m]] = 0.0
        o2[:, k] = 0.0
        accs.append(float((_eval_arrays(model, X2, o2, device) == yva).mean()))
    return float(np.mean(accs)), accs[0]


def train_fusion_generic(Xtr, obstr, ytr, Xva, obsva, yva, modality_dims: dict,
                         order: list[str], seed=0, dropout_p=0.0, jitter_sigma=0.0,
                         extra=None, epochs=80, patience=10, lr=1e-3, device=None,
                         select_masked=False, missing_mode="exclude"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    slices = modality_slices(modality_dims, order)
    ds = GenericWindowDataset(Xtr, obstr, ytr, slices, dropout_p=dropout_p,
                              jitter_sigma=jitter_sigma, seed=seed, extra=extra)
    dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)

    model = AttentionFusion(missing_mode=missing_mode,
                            modality_dims=modality_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_score, best_acc, best_state, bad = 0.0, 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(device), ob.to(device)), yb.to(device)).backward()
            opt.step()
        if select_masked:
            score, acc = _masked_val_acc(model, Xva, obsva, yva, slices, device)
        else:
            score = acc = float((_eval_arrays(model, Xva, obsva, device) == yva).mean())
        if score > best_score:
            best_score, best_acc, bad = score, acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_acc
