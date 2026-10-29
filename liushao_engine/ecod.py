"""ECOD: unsupervised outlier score from empirical cumulative distribution tails.

Li et al., "ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
Distribution Functions", IEEE TKDE 2022.  Fitted on training-split normal
traffic only, so the score never sees attack labels.
"""
import numpy as np


def fit_ecod(x_normal, max_rows=20000, seed=42):
    x = np.asarray(x_normal, dtype=np.float64)
    if len(x) > max_rows:
        x = x[np.random.default_rng(seed).choice(len(x), max_rows, replace=False)]
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isfinite(x), x, median)
    # Skewness picks the "natural" tail per feature (ECOD's automatic variant).
    centered = x - x.mean(axis=0)
    std = x.std(axis=0)
    skew = np.divide((centered ** 3).mean(axis=0), std ** 3, out=np.zeros_like(std), where=std > 0)
    return {'sorted': np.sort(x, axis=0), 'median': median, 'skew_sign': np.sign(skew)}


def tail_logs(model, x):
    """Per-feature -log left tail and -log right tail, with 1/(n+1) smoothing."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, model['median'])
    ref = model['sorted']
    n = len(ref)
    left = np.empty_like(x)
    right = np.empty_like(x)
    for j in range(x.shape[1]):
        left[:, j] = np.searchsorted(ref[:, j], x[:, j], side='right')
        right[:, j] = n - np.searchsorted(ref[:, j], x[:, j], side='left')
    return -np.log((left + 1) / (n + 1)), -np.log((right + 1) / (n + 1))


def score_ecod(model, x):
    """O(x) = max(sum left, sum right, sum auto); larger means more outlying."""
    left, right = tail_logs(model, x)
    auto = np.where(model['skew_sign'] < 0, left, right)
    return np.maximum.reduce([left.sum(axis=1), right.sum(axis=1), auto.sum(axis=1)])
