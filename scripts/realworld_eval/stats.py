"""Paired significance tests for model-vs-model comparisons on a shared test
set — factored out of `scripts/49_significance.py` so later experiments
(`scripts/51_recombination_balance.py`, ...) reuse one implementation rather
than re-deriving it, the same pattern `merged_gap.py` follows for the gap
decomposition.

Why these tests and not a t-test on two accuracy numbers: both systems are
scored on the SAME clips, so the comparison is paired. Only clips where exactly
one system is correct carry information about which is better; a two-sample
test ignores that pairing and is badly under-powered here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, t as student_t

N_BOOT = 10000


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes.

    n10 = a right / b wrong; n01 = b right / a wrong. Under H0 each discordant
    clip is a fair coin, so n10 ~ Binomial(n01+n10, 0.5). Uses the exact
    binomial test rather than the chi-square approximation, which is unreliable
    at small discordant counts.
    """
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n_disc = n10 + n01
    if n_disc == 0:
        return {"n10": 0, "n01": 0, "n_discordant": 0, "p_value": 1.0,
                "favours": "tie"}
    p = binomtest(n10, n_disc, 0.5, alternative="two-sided").pvalue
    return {"n10": n10, "n01": n01, "n_discordant": n_disc, "p_value": float(p),
            "favours": "a" if n10 > n01 else ("b" if n01 > n10 else "tie")}


def bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray,
                   n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Percentile CI for acc(a) - acc(b), resampling CLIPS with replacement.
    Clips are resampled as pairs (one index selects both systems' outcome), so
    the within-clip pairing is preserved."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = correct_a[idx].mean(1) - correct_b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci95_low": float(lo),
            "ci95_high": float(hi),
            "frac_boot_favouring_a": float((diffs > 0).mean())}


def ci95_over_seeds(values) -> dict:
    """t-based 95% CI over per-seed metrics (small n, so t rather than z)."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = float(a.std(ddof=1))
    half = student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"mean": mean, "std": sd, "ci95_low": mean - half,
            "ci95_high": mean + half}


def majority_vote(preds: np.ndarray) -> np.ndarray:
    """preds [n_seeds, n_clips] -> [n_clips] modal prediction per clip."""
    out = np.empty(preds.shape[1], dtype=preds.dtype)
    for i in range(preds.shape[1]):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out
