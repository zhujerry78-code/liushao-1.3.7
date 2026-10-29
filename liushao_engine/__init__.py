"""Trusted triage engine (1.3.7): evidence fusion and uncertainty on top of the frozen base learners.

Chain: heterogeneous detectors (RF, LR, ECOD) -> Dempster-Shafer fusion ->
Mondrian conformal sets -> Venn-Abers interval -> tree-path attribution -> MMD drift.
The frozen RF/LR bundle and its hash-locked sources are never modified; this layer
only appends columns.  Every caller falls back to the 1.3.6 behaviour on failure.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from liushao.model import ranked, review_ranked

from .attribution import contributions, tree_paths
from .conformal import ATTACK, NORMAL, UNSURE, fit_conformal, predict_conformal
from .ecod import fit_ecod, score_ecod
from .evidence import calibrate, fit_calibrator, fuse, reliability
from .mmd import mmd_test
from .venn_abers import fit_va, predict_va

ENGINE_FORMAT = 1
SOURCES = ('rf', 'lr', 'ecod')
COLUMNS = ['ecod_score', 'ds_belief', 'ds_plausibility', 'ds_conflict', 'ds_betp', 'cp_set', 'cp_set_size',
           'cp_p_attack', 'cp_p_normal', 'va_low', 'va_high', 'engine_review', 'engine_version']


def fit_engine(bundle, x_train, y_train, x_cal, y_cal, *, epsilon=0.05, conflict_quantile=0.99, seed=42):
    """Fit on training normals (ECOD) and two disjoint halves of validation.

    Half A fits the isotonic calibrators, source reliabilities and conflict cut;
    half B is held for conformal and Venn-Abers calibration, so the conformal
    calibration scores are exchangeable with new data.
    """
    y_train, y_cal = np.asarray(y_train, dtype=int), np.asarray(y_cal, dtype=int)
    ecod = fit_ecod(np.asarray(x_train)[y_train == 0], seed=seed)
    rng = np.random.default_rng(seed)
    half_a = np.zeros(len(y_cal), dtype=bool)
    for label in (0, 1):  # stratified 50/50 split of validation
        idx = np.flatnonzero(y_cal == label)
        half_a[rng.choice(idx, len(idx) // 2, replace=False)] = True
    scores = base_scores(bundle, ecod, x_cal)
    calibrators = {s: fit_calibrator(scores[s][half_a], y_cal[half_a]) for s in SOURCES}
    fusion = {'sources': list(SOURCES), 'calibrators': calibrators,
              'reliability': {s: reliability(calibrate(calibrators[s], scores[s][half_a]), y_cal[half_a])
                              for s in SOURCES}}
    fused_a = fuse({s: v[half_a] for s, v in scores.items()}, fusion)
    fused_b = fuse({s: v[~half_a] for s, v in scores.items()}, fusion)
    conflict_cut = float(np.quantile(fused_a['ds_conflict'], conflict_quantile))
    return {'format': ENGINE_FORMAT, 'model_version': bundle['model_version'], 'ecod': ecod, 'fusion': fusion,
            'conformal': fit_conformal(fused_b['ds_betp'], y_cal[~half_a], epsilon),
            'venn_abers': fit_va(fused_b['ds_betp'], y_cal[~half_a]),
            'conflict_cut': conflict_cut, 'conflict_quantile': conflict_quantile,
            'drift_reference': drift_space(bundle, x_train.sample(min(2000, len(x_train)), random_state=seed)),
            'split': {'ecod_rows': int(min(len(ecod['sorted']), (y_train == 0).sum())),
                      'calibration_a': int(half_a.sum()), 'calibration_b': int((~half_a).sum()),
                      'calibration_b_attacks': int(y_cal[~half_a].sum())}}


def base_scores(bundle, ecod, x):
    return {'rf': bundle['rf'].predict_proba(x)[:, 1], 'lr': bundle['lr'].predict_proba(x)[:, 1],
            'ecod': score_ecod(ecod, x)}


def engine_version(engine):
    digest = hashlib.sha256(repr((engine['model_version'], engine['fusion']['reliability'],
                                  engine['conflict_cut'], engine['conformal']['epsilon'])).encode())
    return 'engine-' + digest.hexdigest()[:12]


def analyze(pred, features, bundle, engine):
    """Append engine columns to the frozen predictions; never reorders or drops rows."""
    if engine is None or engine['model_version'] != bundle['model_version']:
        raise ValueError('研判引擎与底座模型版本不一致')
    out = pred.copy()
    if not len(features):
        for col in COLUMNS:
            out[col] = pd.Series(dtype=object if col in ('cp_set', 'engine_version') else float)
        return out
    scores = {'rf': pred.rf_score.to_numpy(), 'lr': pred.lr_score.to_numpy(),
              'ecod': score_ecod(engine['ecod'], features)}
    fused = fuse(scores, engine['fusion'])
    sets = predict_conformal(engine['conformal'], fused['ds_betp'])
    va_low, va_high = predict_va(engine['venn_abers'], fused['ds_betp'])
    out['ecod_score'] = scores['ecod']
    for col, value in {**fused, **sets}.items():
        out[col] = value
    out['va_low'], out['va_high'] = va_low, va_high
    out['engine_review'] = (out.cp_set == UNSURE) | (out.ds_conflict >= engine['conflict_cut'])
    out['engine_version'] = engine_version(engine)
    return out


def has_engine(frame):
    return all(col in frame for col in COLUMNS)


def engine_ranked(frame):
    """Main queue: fused belief first, frozen RF score as tie-break."""
    if not has_engine(frame):
        return ranked(frame)
    return frame.sort_values(['ds_belief', 'rf_score', 'record_id'], ascending=[False, False, True], kind='stable')


def engine_review_ranked(frame):
    """Review queue: conformal "uncertain" or high-conflict records.

    Ordered by pignistic BetP closest to 0.5, then by conflict K (rule chosen on validation).
    """
    if not has_engine(frame):
        return review_ranked(frame)
    subset = frame[frame.engine_review].copy()
    subset['threshold_distance'] = (subset.rf_score - subset.rf_threshold).abs()
    subset['betp_distance'] = (subset.ds_betp - .5).abs()
    return subset.sort_values(['betp_distance', 'ds_conflict', 'record_id'], ascending=[True, False, True],
                              kind='stable')


@lru_cache(maxsize=2)
def _paths(rf, features):
    return tree_paths(rf, list(features))


def explain(features_row, bundle, top_k=8):
    """Top-k tree-path contributions for one record (frame with a single row)."""
    rf, names = bundle['rf'], tuple(bundle['config']['features'])
    paths = _paths(rf, names)
    contrib, bias = contributions(paths, rf, features_row[list(names)])
    row = contrib[0]
    order = np.argsort(-np.abs(row))[:top_k]
    table = pd.DataFrame({'字段': [names[i] for i in order], '贡献': row[order]})
    table['方向'] = np.where(table['贡献'] >= 0, '推高攻击得分', '拉低攻击得分')
    return table, float(bias), float(bias + row.sum())


def source_masses(record, engine):
    """Per-source basic probability assignment for one prediction row (a Series)."""
    from .evidence import bpa
    rows = []
    for name, label in (('rf', '随机森林 RF'), ('lr', '逻辑回归 LR'), ('ecod', '无监督 ECOD')):
        score = record['ecod_score' if name == 'ecod' else name + '_score']
        p = calibrate(engine['fusion']['calibrators'][name], [score])
        m = bpa(p, engine['fusion']['reliability'][name])[0]
        rows.append({'证据源': label, '原始分数': float(score), '可靠度 r': engine['fusion']['reliability'][name],
                     'm(攻击)': m[0], 'm(正常)': m[1], 'm(未知Θ)': m[2]})
    return pd.DataFrame(rows)


def ecod_explain(features_row, engine, top_k=6):
    """Features with the largest ECOD tail contribution -log F for one record."""
    from .ecod import tail_logs
    left, right = tail_logs(engine['ecod'], features_row.to_numpy(dtype=float))
    tail = np.maximum(left[0], right[0])
    order = np.argsort(-tail)[:top_k]
    side = np.where(left[0] >= right[0], '偏低（左尾）', '偏高（右尾）')
    return pd.DataFrame({'字段': [features_row.columns[i] for i in order], '尾部贡献': tail[order],
                         '方向': side[order]})


def drift_space(bundle, x):
    """Imputed, signed-log, standardised space of the frozen LR pipeline."""
    return np.asarray(bundle['lr'][:-1].transform(x), dtype=np.float32)


def drift_check(engine, bundle, features, max_rows=1000, n_perm=200):
    return mmd_test(engine['drift_reference'], drift_space(bundle, features), n_perm=n_perm, max_rows=max_rows)


def load_engine(path, bundle):
    """Load artifacts/engine.joblib; None when absent or bound to another base model."""
    path = Path(path)
    if not path.exists():
        return None
    engine = joblib.load(path)
    if engine.get('format') != ENGINE_FORMAT or engine.get('model_version') != bundle['model_version']:
        return None
    return engine


__all__ = ['ATTACK', 'NORMAL', 'UNSURE', 'COLUMNS', 'analyze', 'drift_check', 'engine_ranked',
           'engine_review_ranked', 'engine_version', 'explain', 'fit_engine', 'has_engine', 'load_engine']
