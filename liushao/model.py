"""Training, frozen thresholds, evidence and label-independent inference."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import clean


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def choose_threshold(y, score):
    grid = np.linspace(.05, .95, 91)
    values = [(f1_score(y, score >= threshold, zero_division=0), threshold) for threshold in grid]
    return float(max(values)[1])


def fit_bundle(x, y, xv, yv, data_version, n_estimators=160):
    started = time.perf_counter()
    rf = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                   ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=18,
                    min_samples_leaf=2, class_weight='balanced_subsample', random_state=42, n_jobs=-1))])
    lr = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                   ('log', FunctionTransformer(signed_log)), ('scale', StandardScaler()),
                   ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))])
    rf.fit(x, y)
    rf_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    lr.fit(x, y)
    lr_seconds = time.perf_counter() - tick
    rf_val, lr_val = rf.predict_proba(xv)[:, 1], lr.predict_proba(xv)[:, 1]
    rt, lt = choose_threshold(yv, rf_val), choose_threshold(yv, lr_val)
    margin = float(np.clip(np.quantile(np.abs(rf_val - rt), .1), .01, .1))
    normal = x[np.asarray(y) == 0]
    reference = normal.quantile([.01, .5, .99]).T.replace([np.inf, -np.inf], np.nan)
    reference.columns = ['q01', 'median', 'q99']
    reference = reference.astype(object).where(pd.notna(reference), None).to_dict('index')
    config = {'data_version': data_version, 'features': x.columns.tolist(), 'rf_threshold': rt,
              'lr_threshold': lt, 'near_margin': margin, 'sklearn': sklearn.__version__,
              'algorithm_version': 'liushao-mvp-1', 'n_estimators': n_estimators,
              'max_depth': 18, 'min_samples_leaf': 2, 'seed': 42,
              'threshold_rule': 'validation F1 maximum over 0.05..0.95 step0.01; ties higher threshold',
              'margin_rule': 'validation absolute distance 10th percentile, clipped to [0.01,0.10]'}
    version = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return {'rf': rf, 'lr': lr, 'config': config, 'model_version': version, 'reference': reference,
            'fit_seconds': {'rf': rf_seconds, 'lr': lr_seconds},
            'training_rows': len(x), 'normal_reference_rows': len(normal)}


def infer(raw, bundle, source_hash, source_name):
    config = bundle['config']
    x, meta, rejected = clean(raw, config['features'], source_hash, source_name)
    # Dataset ground truth never reaches ranking, evidence, reviews, or UI.
    meta = meta.drop(columns=['label'], errors='ignore')
    rejected = rejected.drop(columns=['label'], errors='ignore')
    if len(x):
        rf = bundle['rf'].predict_proba(x)[:, 1]
        lr = bundle['lr'].predict_proba(x)[:, 1]
    else:
        rf = lr = np.array([], dtype=float)
    meta['rf_score'] = rf
    meta['lr_score'] = lr
    meta['rf_anomaly'] = rf >= config['rf_threshold']
    meta['lr_anomaly'] = lr >= config['lr_threshold']
    meta['disagreement'] = meta.rf_anomaly != meta.lr_anomaly
    meta['near_threshold'] = np.abs(rf - config['rf_threshold']) <= config['near_margin']
    meta['model_version'] = bundle['model_version']
    meta['rf_threshold'] = config['rf_threshold']
    meta['lr_threshold'] = config['lr_threshold']
    meta['near_margin'] = config['near_margin']
    return meta, x, rejected


def ranked(frame):
    return frame.sort_values(['rf_score', 'record_id'], ascending=[False, True], kind='stable')


def review_ranked(frame):
    subset = frame[frame.disagreement | frame.near_threshold].copy()
    subset['threshold_distance'] = (subset.rf_score - subset.rf_threshold).abs()
    return subset.sort_values(['disagreement', 'threshold_distance', 'record_id'], ascending=[False, True, True])


def evidence_for(row, bundle):
    from .reference import reference_position
    from .feature_metadata import feature_definition
    rows = []
    for field in bundle['config']['features']:
        value = row[field]
        reference = bundle['reference'][field]
        low, high = reference['q01'], reference['q99']
        flag = reference_position(field, value, reference)['status']
        unit = feature_definition(field)[1]
        rows.append({'字段': field, '原值': value, '单位': unit, '正常P01': low,
                     '正常中位数': reference['median'], '正常P99': high, '统计提示': flag})
    return pd.DataFrame(rows)


def classification_metrics(y, scores, threshold):
    y = np.asarray(y)
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'fpr': float(fp / (tn + fp)) if tn + fp else None,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'normal_support': int((y == 0).sum()), 'attack_support': int((y == 1).sum()),
            'threshold': threshold}


def ranking_metrics(y, order, budgets=(20, 50, 100)):
    y = np.asarray(y)
    order = np.asarray(order, dtype=int)
    total = int(y.sum())
    output = []
    for k in budgets:
        actual = min(k, len(order))
        found = int(y[order[:actual]].sum())
        output.append({'k_requested': k, 'k_actual': actual, 'attacks_found': found,
                       'precision_at_k': found / actual if actual else None,
                       'recall_at_k': found / total if total else None, 'total_attacks': total})
    return output
