"""Dempster-Shafer fusion of heterogeneous detectors on the frame {attack, normal}.

Each source score is mapped to a calibrated probability p (isotonic regression on
calibration data), then discounted by a reliability r (Shafer 1976):
m({attack}) = r p, m({normal}) = r (1 - p), m(Theta) = 1 - r.
Sources are combined with Dempster's rule; K is the mass of total conflict.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def fit_calibrator(score, y):
    # increasing='auto' lets calibration learn a detector's polarity (e.g. hyper-regular bot flows).
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing='auto', out_of_bounds='clip').fit(score, y)
    return {'x': iso.X_thresholds_.astype(np.float64), 'y': iso.y_thresholds_.astype(np.float64)}


def calibrate(cal, score):
    return np.clip(np.interp(np.asarray(score, dtype=np.float64), cal['x'], cal['y']), EPS, 1 - EPS)


def reliability(score, y, cap=0.99):
    """Gini coefficient 2*AUC-1 as discount factor; a coin-flip source carries no mass."""
    return float(np.clip(2 * roc_auc_score(y, score) - 1, 0.0, cap))


def bpa(p, r):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([r * p, r * (1 - p), np.full_like(p, 1 - r)], axis=1)


def combine(masses):
    """Dempster's rule over any number of BPAs (columns: attack, normal, theta).

    Returns normalised (attack, normal, theta) and the total conflict K, i.e. the
    unnormalised conjunctive mass on the empty set.
    """
    a, n, u = masses[0][:, 0], masses[0][:, 1], masses[0][:, 2]
    empty = np.zeros_like(a)
    for m in masses[1:]:
        a2, n2, u2 = m[:, 0], m[:, 1], m[:, 2]
        empty = empty + a * n2 + n * a2
        a, n, u = a * a2 + a * u2 + u * a2, n * n2 + n * u2 + u * n2, u * u2
    total = np.maximum(1 - empty, EPS)
    return a / total, n / total, u / total, empty


def fuse(scores, params):
    masses = [bpa(calibrate(params['calibrators'][name], scores[name]), params['reliability'][name])
              for name in params['sources']]
    belief, _, ignorance, conflict = combine(masses)
    return {'ds_belief': belief, 'ds_plausibility': belief + ignorance,
            'ds_conflict': conflict, 'ds_betp': belief + ignorance / 2}
