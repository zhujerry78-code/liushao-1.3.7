"""Inductive Venn-Abers predictor (Vovk & Petej 2014): a calibrated probability interval.

For a test score s, isotonic regression is refitted with (s, 0) and with (s, 1)
added to the calibration set, giving [p0, p1].  The pair is precomputed on a grid
of calibration-score quantiles and interpolated at inference time.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_va(cal_scores, cal_y, grid_size=256):
    s, y = np.asarray(cal_scores, dtype=np.float64), np.asarray(cal_y, dtype=np.float64)
    grid = np.unique(np.concatenate([[0.0, 1.0], np.quantile(s, np.linspace(0, 1, grid_size))]))
    p0, p1 = np.empty_like(grid), np.empty_like(grid)
    for i, g in enumerate(grid):
        for label, out in ((0.0, p0), (1.0, p1)):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            iso.fit(np.append(s, g), np.append(y, label))
            out[i] = iso.predict([g])[0]
    return {'grid': grid, 'p0': p0, 'p1': p1}


def predict_va(model, scores):
    scores = np.asarray(scores, dtype=np.float64)
    p0 = np.interp(scores, model['grid'], model['p0'])
    p1 = np.interp(scores, model['grid'], model['p1'])
    return np.minimum(p0, p1), np.maximum(p0, p1)
