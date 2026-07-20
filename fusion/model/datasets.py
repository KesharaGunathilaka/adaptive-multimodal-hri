"""Fusion training dataset: window rows -> (x[24], obs[4], y) with on-the-fly
train-only augmentation (handover §8.2):

  * modality dropout — each cue independently masked with p (default 0.2),
    capped at 2 dropped cues per sample (Final_Dataset §1); teaches the
    [MISSING] tokens.
  * confidence jitter — Gaussian noise on log-probs, per-cue with prob 0.5;
    simulates sensor noise / miscalibration.

The scenario-designed [MISSING] flags and runtime-missing cues arrive already
encoded as obs=0 from the parquet. Augmentation only ever REMOVES information,
so labels stay valid.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from ..baselines.common import OBS_COLS, PROB_COLS

CUE_SLICES = {"emotion": slice(0, 7), "gesture": slice(7, 15),
              "motion": slice(15, 19), "context": slice(19, 24)}


class WindowDataset(Dataset):
    def __init__(self, frame, dropout_p=0.0, jitter_sigma=0.0, seed=0,
                 extra=None):
        """`frame`: parquet rows. `extra`: optional (X, obs, y) numpy triple of
        synthetic samples (recombination) appended to the real data."""
        X = frame[PROB_COLS].fillna(0.0).to_numpy(np.float32)
        obs = frame[OBS_COLS].to_numpy(np.float32)
        y = frame["y"].to_numpy(np.int64)
        if extra is not None:
            X = np.concatenate([X, extra[0].astype(np.float32)])
            obs = np.concatenate([obs, extra[1].astype(np.float32)])
            y = np.concatenate([y, extra[2].astype(np.int64)])
        self.X, self.obs, self.y = X, obs, y
        self.dropout_p = dropout_p
        self.jitter_sigma = jitter_sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].copy()
        obs = self.obs[i].copy()

        if self.dropout_p > 0:
            drop = self.rng.random(4) < self.dropout_p
            drop &= obs.astype(bool)                 # can't drop what's absent
            while drop.sum() > 2:                    # cap at 2 dropped cues
                drop[self.rng.choice(np.flatnonzero(drop))] = False
            if (obs.astype(bool) & ~drop).sum() == 0:
                drop[:] = False                      # never drop everything
            for k, m in enumerate(CUE_SLICES):
                if drop[k]:
                    x[CUE_SLICES[m]] = 0.0
                    obs[k] = 0.0

        if self.jitter_sigma > 0:
            for k, m in enumerate(CUE_SLICES):
                if obs[k] and self.rng.random() < 0.5:
                    sl = CUE_SLICES[m]
                    logp = np.log(np.clip(x[sl], 1e-6, None))
                    logp += self.rng.normal(0, self.jitter_sigma, logp.shape)
                    e = np.exp(logp - logp.max())
                    x[sl] = (e / e.sum()).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(obs), int(self.y[i])
