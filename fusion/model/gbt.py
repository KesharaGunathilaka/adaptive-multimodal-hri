"""GBT (LightGBM) fusion head for the architecture ablation (Study 1).

Gradient-boosted trees on the 28-dim tabular cue vector (24 cue probabilities +
4 observed flags) — the SAME information the neural heads get via
`common.xy`, so the comparison is apples-to-apples. Trees are a strong,
near-free contender on low-dim tabular fusion and deploy trivially on Jetson.

LightGBM can't use the torch loop's on-the-fly `WindowDataset` augmentation, so
recombination samples and modality-dropout / confidence-jitter are MATERIALIZED
into the training matrix by sampling that exact same dataset for a few passes —
keeping the augmented training distribution identical to what the neural heads
see under the matching config.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from ..baselines import common
from .datasets import WindowDataset


def _materialize(frame, dropout_p, jitter_sigma, extra, seed, n_passes):
    """Sample the shared WindowDataset for `n_passes` epochs -> (X[N,28], y).
    With dropout_p=jitter_sigma=0 and n_passes=1 this is just the raw rows
    (+ recombination `extra`), matching the neural `plain`/`recomb_only`
    configs; with augmentation on it matches `augmented`/`full`."""
    ds = WindowDataset(frame, dropout_p=dropout_p, jitter_sigma=jitter_sigma,
                       seed=seed, extra=extra)
    n = len(ds)
    X = np.empty((n * n_passes, 28), np.float32)
    y = np.empty(n * n_passes, np.int64)
    w = 0
    for _ in range(n_passes):
        for i in range(n):
            xi, obsi, yi = ds[i]
            X[w, :24] = xi.numpy()
            X[w, 24:] = obsi.numpy()
            y[w] = yi
            w += 1
    return X, y


def train_gbt(splits, seed=0, dropout_p=0.0, jitter_sigma=0.0, extra=None,
              n_passes=2, n_estimators=600):
    """Fit LightGBM on train(+extra) with early stopping on val. Returns the
    fitted classifier; score arbitrary frames with `gbt_predict`."""
    Xtr, ytr = _materialize(splits["train"], dropout_p, jitter_sigma, extra,
                            seed, n_passes if (dropout_p or jitter_sigma) else 1)
    Xva, yva = common.xy(splits["val"])

    clf = LGBMClassifier(
        objective="multiclass", num_class=10, n_estimators=n_estimators,
        learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
        n_jobs=-1, verbosity=-1)
    clf.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="multi_logloss",
            callbacks=[early_stopping(40, verbose=False), log_evaluation(0)])
    return clf


def gbt_predict(clf, frame) -> np.ndarray:
    """Class-index predictions on a common-schema frame (24 probs + 4 obs)."""
    X, _ = common.xy(frame)
    return clf.predict(X).astype(np.int64)
