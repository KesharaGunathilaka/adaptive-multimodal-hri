"""Temporal fusion over the window SEQUENCE (R3, Study 2) -- as opposed to
`model.AttentionFusion`'s self-attention over the 4 MODALITY tokens of one
already-pooled vector. Here each timestep's 24-dim cue vector (+4 obs flags)
becomes one token; a transformer encoder runs over the TIME axis.

`sequences.build_sequences` right-aligns every clip's windows (padding at the
START, most-recent real window always at index T-1). That lets a single
`causal=True` model be evaluated at ANY trailing buffer length K (just call
`build_sequences(..., trailing=K)` and feed the shorter [B,K,24] tensor) with
no retraining -- exactly the "how much history does the live system need"
question Study 2 / Study 3 asks. `causal=False` (bidirectional) additionally
sees future-within-the-clip windows and is the offline "how much does dynamics
awareness help with the full clip available" ceiling -- NOT deployable live.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .sequences import SequenceDataset


class TemporalFusion(nn.Module):
    def __init__(self, d=64, n_heads=4, n_layers=2, ff=128, dropout=0.2,
                n_classes=10, max_len=40, causal=False):
        super().__init__()
        self.causal = causal
        self.max_len = max_len
        self.token = nn.Linear(24 + 4, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x, obs, valid):
        """x:[B,T,24] obs:[B,T,4] valid:[B,T] bool (True=real window, right-
        aligned -- valid[:, -1] is the most recent real window whenever the
        clip has any). T may be < self.max_len (trailing-K eval)."""
        B, T, _ = x.shape
        tok = self.token(torch.cat([x, obs], dim=-1)) + self.pos[:, -T:]

        if self.causal:
            # Left-padded + causal: a padding QUERY at position q < T-take can
            # only see keys <= q (causal), which are ALL padding too -- if we
            # also excluded padding KEYS via src_key_padding_mask, every such
            # row would have zero visible keys -> NaN softmax, propagating
            # through later self-attention into the one position we actually
            # read (index -1, always real by construction). Fix: for the
            # causal path, don't mask keys at all -- the causal mask alone
            # guarantees every query sees at least itself, and the leading
            # zero-tokens just become learnable context noise the readout
            # position (always real) can attend past.
            attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            out = self.encoder(tok, mask=attn_mask)
        else:
            pad = ~valid                                # True = ignore (bidi only)
            empty = pad.all(dim=1)
            if empty.any():
                pad = pad.clone()
                pad[empty, -1] = False
            out = self.encoder(tok, src_key_padding_mask=pad)
        if self.causal:
            readout = out[:, -1, :]
        else:
            vf = valid.float().unsqueeze(-1)
            readout = (out * vf).sum(dim=1) / vf.sum(dim=1).clamp_min(1.0)
        return self.head(readout)


def _eval_arrays(model, X, obs, valid, device, bs=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            ob = torch.from_numpy(obs[i:i + bs]).to(device)
            vb = torch.from_numpy(valid[i:i + bs]).to(device)
            preds.append(model(xb, ob, vb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_temporal(seq_splits, seed=0, dropout_p=0.0, jitter_sigma=0.0,
                   causal=False, epochs=80, patience=10, lr=1e-3, device=None,
                   model_kwargs=None, extra=None):
    """seq_splits: {'train': (X,obs,valid,y), 'val': (...), 'test': (...)}
    from `sequences.build_sequences`, all built with the SAME max_len/T.
    `extra`: optional (X,obs,valid,y) synthetic trajectories
    (`recombine_sequences.generate_sequences`), appended to train only."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, obstr, validtr, ytr = seq_splits["train"]
    ds = SequenceDataset(Xtr, obstr, validtr, ytr, dropout_p=dropout_p,
                         jitter_sigma=jitter_sigma, seed=seed, extra=extra)
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)
    Xva, obsva, validva, yva = seq_splits["val"]

    T = Xtr.shape[1]
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("max_len", T)
    model = TemporalFusion(causal=causal, **kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, ob, vb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(device), ob.to(device), vb.to(device)),
                    yb.to(device)).backward()
            opt.step()
        pred = _eval_arrays(model, Xva, obsva, validva, device)
        acc = float((pred == yva).mean())
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_acc
