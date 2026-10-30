"""Exact Venn-Abers endpoints from the unchanged frozen B calibration sample.

The stored conformal arrays retain class-conditional B nonconformity scores.
For normal records alpha=BetP; for attacks alpha=1-BetP.  Recovering those
labelled scores permits exact Venn-Abers inference without retraining, choosing
a new split, or changing any calibration parameter.
"""
from functools import lru_cache
import hashlib

import numpy as np
from sklearn.isotonic import IsotonicRegression

ENGINE_INFERENCE_VERSION = 'r13-exact-va-v1'


class ExactVennAbers:
    def __init__(self, conformal):
        normal = np.asarray(conformal['normal'], dtype=np.float64)
        attack = 1 - np.asarray(conformal['attack'], dtype=np.float64)
        scores = np.r_[normal, attack]
        labels = np.r_[np.zeros(len(normal)), np.ones(len(attack))]
        if not len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Invalid frozen calibration scores')
        self.scores, inverse, self.weights = np.unique(scores, return_inverse=True, return_counts=True)
        self.labels = np.bincount(inverse, weights=labels) / self.weights
        self.endpoints = lru_cache(maxsize=8192)(self._endpoints)

    def _endpoints(self, score):
        # Grouping identical X and using counts as sample_weight preserves the
        # two original isotonic regression problems, including exact ties.
        x = np.r_[self.scores, score]
        weights = np.r_[self.weights, 1]
        values = []
        for label in (0., 1.):
            iso = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')
            iso.fit(x, np.r_[self.labels, label], sample_weight=weights)
            values.append(float(iso.predict([score])[0]))
        return min(values), max(values)

    def predict(self, scores):
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Venn-Abers query scores must be finite probabilities')
        unique, inverse = np.unique(values, return_inverse=True)
        if not len(unique):
            return values.copy(), values.copy()
        pairs = np.asarray([self.endpoints(float(score)) for score in unique])
        return pairs[inverse, 0], pairs[inverse, 1]


_VA_CACHE = {}


def exact_va(engine, scores):
    conformal = engine['conformal']
    digest = hashlib.sha256()
    for name in ('normal', 'attack'):
        values = np.asarray(conformal[name], dtype='<f8')
        digest.update(name.encode()); digest.update(str(values.shape).encode()); digest.update(values.tobytes())
    key = digest.hexdigest()
    if key not in _VA_CACHE:
        if len(_VA_CACHE) >= 2:
            _VA_CACHE.pop(next(iter(_VA_CACHE)))
        _VA_CACHE[key] = ExactVennAbers(conformal)
    return _VA_CACHE[key].predict(scores)


def analyze_current_engine(pred, features, bundle, engine):
    import liushao_engine
    result = liushao_engine.analyze(pred, features, bundle, engine)
    low, high = exact_va(engine, result.ds_betp.to_numpy(dtype=float))
    result['va_low'], result['va_high'] = low, high
    return result
