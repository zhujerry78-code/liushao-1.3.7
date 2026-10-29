"""Kernel two-sample test with maximum mean discrepancy (Gretton et al. 2012)."""
import numpy as np


def _sq_dists(z):
    sq = (z * z).sum(axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0.0)


def mmd_test(x, y, n_perm=200, max_rows=2000, seed=42):
    """RBF kernel, median-heuristic bandwidth, permutation p-value of the biased MMD^2."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) > max_rows:
        x = x[rng.choice(len(x), max_rows, replace=False)]
    if len(y) > max_rows:
        y = y[rng.choice(len(y), max_rows, replace=False)]
    z = np.vstack([x, y])
    d = _sq_dists(z)
    upper = d[np.triu_indices(len(z), k=1)]
    bandwidth = float(np.sqrt(np.median(upper[upper > 0]) / 2)) if (upper > 0).any() else 1.0
    k = np.exp(-d / (2 * bandwidth ** 2))
    # MMD^2 = w' K w with w = +1/n on the first sample and -1/m on the second.
    base = np.concatenate([np.full(len(x), 1 / len(x)), np.full(len(y), -1 / len(y))])

    def stat(w):
        return float(w @ k @ w)

    observed = stat(base)
    null = np.array([stat(rng.permutation(base)) for _ in range(n_perm)])
    return {'mmd2': float(observed), 'p_value': float(((null >= observed).sum() + 1) / (n_perm + 1)),
            'bandwidth': bandwidth, 'rows': [int(len(x)), int(len(y))], 'n_perm': int(n_perm)}
