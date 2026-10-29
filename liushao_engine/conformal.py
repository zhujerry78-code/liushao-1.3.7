"""Label-conditional (Mondrian) split conformal prediction (Vovk et al. 2005).

Nonconformity of label "attack" is 1 - BetP(attack); of label "normal" it is BetP(attack).
p_y(x) = (#{calibration of class y with alpha_i >= alpha_y(x)} + 1) / (n_y + 1),
so for each class P(y in set) >= 1 - epsilon under exchangeability.
"""
import numpy as np

NORMAL, ATTACK, UNSURE = '正常', '攻击', '不确定'


def fit_conformal(betp, y, epsilon=0.05):
    betp, y = np.asarray(betp, dtype=np.float64), np.asarray(y, dtype=int)
    return {'attack': np.sort(1 - betp[y == 1]), 'normal': np.sort(betp[y == 0]), 'epsilon': float(epsilon)}


def _p_values(sorted_cal, alpha):
    greater_equal = len(sorted_cal) - np.searchsorted(sorted_cal, alpha, side='left')
    return (greater_equal + 1) / (len(sorted_cal) + 1)


def predict_conformal(params, betp, epsilon=None):
    betp = np.asarray(betp, dtype=np.float64)
    eps = params['epsilon'] if epsilon is None else epsilon
    p_attack = _p_values(params['attack'], 1 - betp)
    p_normal = _p_values(params['normal'], betp)
    has_attack, has_normal = p_attack > eps, p_normal > eps
    label = np.where(has_attack & ~has_normal, ATTACK, np.where(has_normal & ~has_attack, NORMAL, UNSURE))
    return {'cp_p_attack': p_attack, 'cp_p_normal': p_normal, 'cp_set': label,
            'cp_set_size': has_attack.astype(int) + has_normal.astype(int)}
