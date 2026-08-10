"""Per-clip WINDOW-SEQUENCE representation (R3, Study 2) built directly from
`unimodal_windows.parquet` -- no re-extraction needed, the per-window rows
already carry `clip_id`/`window_idx`/`t_end` plus the same 24 prob + 4 obs
columns `fusion.baselines.common` uses for the pooled (R1) representation.

Median clip = 14 windows (2.1-4.3s cues at the deployment 8/30s stride, over a
~4s clip), max 42 -- see WORKLOG 2026-08-05. Sequences are padded/truncated to
`max_len`, keeping the MOST RECENT `max_len` windows when a clip is longer
(consistent with a live system that only ever has a bounded trailing buffer).

Two ways to build the per-clip sample, both produced by `build_sequences`:
  * full sequence   (`trailing=None`) -- every window up to max_len; the
    "offline" R3 input, analogous to R1's clip-pool but order-preserving.
  * trailing window (`trailing=K`)    -- only the LAST K windows; the R3
    live-inference variant, matching what a Jetson stream actually has
    available at any decision point (no future frames).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..baselines.common import OBS_COLS, PROB_COLS

CUE_SLICES = {"emotion": slice(0, 7), "gesture": slice(7, 15),
              "motion": slice(15, 19), "context": slice(19, 24)}


def build_sequences(windows: pd.DataFrame, real: pd.DataFrame, split: str,
                    max_len: int = 40, trailing: int | None = None):
    """windows: raw `unimodal_windows.parquet` rows. real: common-schema
    clip-level table (has clip_id/split/y) from `merged_gap.build_real` +
    `to_common_schema` -- reused only for the split filter and labels, its
    own pooled cue columns are ignored here.

    -> (X[N,T,24] float32, obs[N,T,4] float32, valid[N,T] bool
       [True=real window, False=pad], y[N] int64, clip_ids[N])
    T = min(max_len, longest clip) if trailing is None, else T = trailing.
    """
    sub = real[real.split == split]
    clip_ids = sub.clip_id.to_numpy()
    y = sub.y.to_numpy()
    y_of = dict(zip(sub.clip_id, sub.y))

    w = windows[windows.clip_id.isin(set(clip_ids))].sort_values(
        ["clip_id", "window_idx"])
    groups = {cid: g for cid, g in w.groupby("clip_id")}

    # T must be the SAME for every split built against one model -- it fixes
    # the positional-embedding table size and (for the causal path) which
    # index is "the most recent real window". Using each split's own longest
    # clip here previously gave train T=40 but val T=27, silently training and
    # evaluating the model at different sequence lengths (see WORKLOG
    # 2026-08-05, the causal-collapse bug this caused).
    T = trailing if trailing is not None else max_len

    N = len(clip_ids)
    X = np.zeros((N, T, 24), np.float32)
    obs = np.zeros((N, T, 4), np.float32)
    valid = np.zeros((N, T), bool)
    kept_ids = []
    for i, cid in enumerate(clip_ids):
        g = groups.get(cid)
        if g is None or len(g) == 0:
            kept_ids.append(cid)
            continue                                  # all-padding row (rare)
        probs = g[PROB_COLS].to_numpy(np.float32)
        o = g[OBS_COLS].to_numpy(np.float32)
        probs = np.nan_to_num(probs, nan=0.0)
        n = len(g)
        take = min(n, T)                              # most recent `take` windows
        X[i, T - take:] = probs[n - take:]
        obs[i, T - take:] = o[n - take:]
        valid[i, T - take:] = True
        kept_ids.append(cid)

    return X, obs, valid, y, np.array(kept_ids)


class SequenceDataset(Dataset):
    """Per-timestep modality-dropout / confidence-jitter, drawn independently
    at each real (non-padded) timestep -- simulates transient per-window
    sensor dropouts/noise rather than a single clip-wide event, which is the
    realistic failure mode for a live cue stream.

    `extra`: optional (X, obs, valid, y) tuple of synthetic trajectories from
    `fusion/model/recombine_sequences.py` (Phase 3, 2026-08-06), appended to
    the real data -- the sequence analogue of `WindowDataset`'s `extra`."""

    def __init__(self, X, obs, valid, y, dropout_p=0.0, jitter_sigma=0.0, seed=0,
                extra=None):
        if extra is not None:
            X = np.concatenate([X, extra[0].astype(np.float32)])
            obs = np.concatenate([obs, extra[1].astype(np.float32)])
            valid = np.concatenate([valid, extra[2].astype(bool)])
            y = np.concatenate([y, extra[3].astype(np.int64)])
        self.X, self.obs, self.valid, self.y = X, obs, valid, y
        self.dropout_p, self.jitter_sigma = dropout_p, jitter_sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, obs, valid = self.X[i].copy(), self.obs[i].copy(), self.valid[i]
        T = x.shape[0]

        if self.dropout_p > 0:
            for t in range(T):
                if not valid[t]:
                    continue
                drop = self.rng.random(4) < self.dropout_p
                drop &= obs[t].astype(bool)
                while drop.sum() > 2:
                    drop[self.rng.choice(np.flatnonzero(drop))] = False
                if (obs[t].astype(bool) & ~drop).sum() == 0:
                    drop[:] = False
                for k, m in enumerate(CUE_SLICES):
                    if drop[k]:
                        x[t, CUE_SLICES[m]] = 0.0
                        obs[t, k] = 0.0

        if self.jitter_sigma > 0:
            for t in range(T):
                if not valid[t]:
                    continue
                for k, m in enumerate(CUE_SLICES):
                    if obs[t, k] and self.rng.random() < 0.5:
                        sl = CUE_SLICES[m]
                        logp = np.log(np.clip(x[t, sl], 1e-6, None))
                        logp += self.rng.normal(0, self.jitter_sigma, logp.shape)
                        e = np.exp(logp - logp.max())
                        x[t, sl] = (e / e.sum()).astype(np.float32)

        return (torch.from_numpy(x), torch.from_numpy(obs),
                torch.from_numpy(valid), int(self.y[i]))
