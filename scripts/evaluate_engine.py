"""Evaluate the 1.3.7 engine on the held-out test split (and optionally a cross-day file).

Nothing is fitted here; the engine and base model are loaded frozen.
Usage:
    python -m scripts.evaluate_engine --data <processed dir> [--cross-day <csv>] --output docs/results/engine_evaluation.json
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from liushao.data import clean, sha256, write_json
from liushao.model import classification_metrics, infer, ranked, review_ranked
from liushao_engine import ATTACK, NORMAL, UNSURE, analyze, drift_check, engine_ranked, engine_review_ranked, \
    engine_version, explain
from liushao_engine.attribution import contributions, tree_paths
from liushao_engine.conformal import predict_conformal
from scripts.fit_engine import load_split

BUDGETS = (20, 50, 100, 200, 500, 1000, 2000)


def top_k(y, order):
    total = int(y.sum())
    out = []
    for k in BUDGETS:
        n = min(k, len(order))
        found = int(y[order[:n]].sum())
        out.append({'k': n, 'attacks_found': found, 'precision': found / n if n else None,
                    'recall': found / total if total else None})
    return out


def ece(p, y, bins=10):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def decisions(frame):
    """Engine point decision: conformal singleton when available, else BetP >= 0.5."""
    betp_call = frame.ds_betp.to_numpy() >= .5
    return np.where(frame.cp_set == ATTACK, True, np.where(frame.cp_set == NORMAL, False, betp_call))


def queue_report(y, decision, queue):
    y = np.asarray(y, dtype=bool)
    err = decision != y
    return {'records': int(len(y)), 'decision_errors': int(err.sum()), 'fp': int((decision & ~y).sum()),
            'fn': int((~decision & y).sum()), 'queue_records': int(queue.sum()),
            'queue_share': float(queue.mean()), 'errors_in_queue': int((err & queue).sum()),
            'error_capture_rate': float((err & queue).sum() / err.sum()) if err.any() else None,
            'unreviewed_errors': int((err & ~queue).sum()),
            'queue_precision_for_errors': float(err[queue].mean()) if queue.any() else None}


def conformal_report(frame, y, engine):
    out = {}
    betp = frame.ds_betp.to_numpy()
    for eps in (0.01, 0.05, 0.10):
        sets = predict_conformal(engine['conformal'], betp, epsilon=eps)
        has_attack, has_normal = sets['cp_p_attack'] > eps, sets['cp_p_normal'] > eps
        covered = np.where(y == 1, has_attack, has_normal)
        single = sets['cp_set_size'] == 1
        out[str(eps)] = {
            'coverage': float(covered.mean()),
            'coverage_attack': float(covered[y == 1].mean()) if (y == 1).any() else None,
            'coverage_normal': float(covered[y == 0].mean()) if (y == 0).any() else None,
            'guarantee': 1 - eps, 'singleton_share': float(single.mean()),
            'singleton_accuracy': float(covered[single].mean()) if single.any() else None,
            'uncertain': int((sets['cp_set'] == UNSURE).sum()),
            'both_labels': int((sets['cp_set_size'] == 2).sum()), 'empty': int((sets['cp_set_size'] == 0).sum())}
    return out


def evaluate_split(x, y, meta, bundle, engine, tag):
    tick = time.perf_counter()
    pred, features, rejected = infer(x, bundle, tag, tag)
    base_seconds = time.perf_counter() - tick
    if len(rejected) or len(pred) != len(y):
        raise ValueError(f'{tag}: 推理未一一对应')
    pred['record_id'] = meta.record_id.to_numpy() if 'record_id' in meta else pred.record_id
    tick = time.perf_counter()
    frame = analyze(pred, features, bundle, engine)
    engine_seconds = time.perf_counter() - tick
    y = np.asarray(y, dtype=int)
    rf_decision = frame.rf_anomaly.to_numpy()
    rule_queue = (frame.disagreement | frame.near_threshold).to_numpy()
    eng_decision = decisions(frame)
    eng_queue = frame.engine_review.to_numpy()
    by_position = frame.reset_index(drop=True)
    report = {
        'records': int(len(y)), 'attacks': int(y.sum()),
        'auc': {'rf_score': float(roc_auc_score(y, frame.rf_score)), 'lr_score': float(roc_auc_score(y, frame.lr_score)),
                'ecod_score_raw': float(roc_auc_score(y, frame.ecod_score)),
                'ecod_score_polarity_calibrated': float(max(roc_auc_score(y, frame.ecod_score),
                                                            1 - roc_auc_score(y, frame.ecod_score))),
                'ds_belief': float(roc_auc_score(y, frame.ds_belief)),
                'ds_betp': float(roc_auc_score(y, frame.ds_betp))},
        'base_rf_at_frozen_threshold': classification_metrics(y, frame.rf_score, bundle['config']['rf_threshold']),
        'engine_point_decision': classification_metrics(y, eng_decision.astype(float), .5),
        'review_queue': {'rule_1_3_6_rf_decision': queue_report(y, rf_decision, rule_queue),
                         'engine_1_3_7': queue_report(y, eng_decision, eng_queue),
                         'engine_queue_on_rf_decision': queue_report(y, rf_decision, eng_queue)},
        'top_k': {'rf_score_1_3_6': top_k(y, ranked(by_position).index.to_numpy()),
                  'ds_belief_1_3_7': top_k(y, engine_ranked(by_position).index.to_numpy())},
        'review_order_first_k_errors': {},
        'conformal': conformal_report(frame, y, engine),
        'conflict': {'mean': float(frame.ds_conflict.mean()),
                     'mean_on_rf_errors': float(frame.ds_conflict[rf_decision != y].mean()) if (rf_decision != y).any() else None,
                     'mean_on_rf_correct': float(frame.ds_conflict[rf_decision == y].mean()),
                     'auc_conflict_detects_rf_error': float(roc_auc_score(rf_decision != y, frame.ds_conflict))
                     if 0 < (rf_decision != y).sum() < len(y) else None},
        'venn_abers': {'mean_width': float((frame.va_high - frame.va_low).mean()),
                       'ece_va_mid': ece((frame.va_low + frame.va_high) / 2, y),
                       'ece_rf_score': ece(frame.rf_score, y), 'ece_ds_betp': ece(frame.ds_betp, y),
                       'brier_va_mid': float((((frame.va_low + frame.va_high) / 2 - y) ** 2).mean()),
                       'brier_rf_score': float(((frame.rf_score - y) ** 2).mean())},
        'seconds': {'base_inference': base_seconds, 'engine_layer': engine_seconds,
                    'engine_per_1000_rows': engine_seconds / len(y) * 1000},
        'cp_set_counts': frame.cp_set.value_counts().to_dict()}
    err_rf, err_eng = rf_decision != y, eng_decision != y
    rule_order, eng_order = review_ranked(by_position).index.to_numpy(), engine_review_ranked(by_position).index.to_numpy()
    for k in (20, 50, 100):
        report['review_order_first_k_errors'][str(k)] = {
            'rule_1_3_6': int(err_rf[rule_order[:k]].sum()), 'engine_1_3_7': int(err_eng[eng_order[:k]].sum())}
    return report, frame, features


def attribution_check(bundle, features, rows=500):
    sample = features.iloc[:rows]
    tick = time.perf_counter()
    paths = tree_paths(bundle['rf'], bundle['config']['features'])
    build = time.perf_counter() - tick
    tick = time.perf_counter()
    contrib, bias = contributions(paths, bundle['rf'], sample)
    per_row = (time.perf_counter() - tick) / len(sample)
    rf = bundle['rf'].predict_proba(sample)[:, 1]
    table, _, _ = explain(sample.iloc[[0]], bundle)
    return {'rows': int(len(sample)), 'max_additivity_error': float(np.abs(bias + contrib.sum(axis=1) - rf).max()),
            'bias': bias, 'path_index_build_seconds': build, 'seconds_per_row': per_row,
            'example_top_features': table['字段'].tolist()}


def load_cross_day(path, bundle, rows, seed=42):
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw['Label'].astype(str).str.strip() != 'Label']  # repeated header lines in CIC files
    raw = raw.sample(min(rows, len(raw)), random_state=seed).reset_index(drop=True)
    x, meta, rejected = clean(raw, bundle['config']['features'], 'cross-day', Path(path).name)
    return x, (meta.label != 'Benign').astype(int).to_numpy(), meta, int(len(rejected))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed')
    p.add_argument('--artifacts', default='artifacts')
    p.add_argument('--cross-day', default=None)
    p.add_argument('--cross-day-rows', type=int, default=50000)
    p.add_argument('--output', default='docs/results/engine_evaluation.json')
    args = p.parse_args()
    artifacts = Path(args.artifacts)
    bundle = joblib.load(artifacts / 'model.joblib')
    engine = joblib.load(artifacts / 'engine.joblib')
    descriptor = json.loads((artifacts / 'engine.json').read_text(encoding='utf-8'))
    if sha256(artifacts / 'engine.joblib') != descriptor['engine_sha256']:
        raise ValueError('引擎参数哈希不匹配')
    splits, summary = load_split(Path(args.data), bundle, ['train', 'test'])
    xt, yt, mt = splits['test']
    test, _, test_features = evaluate_split(xt, yt, mt, bundle, engine, 'test')
    test['label_support'] = mt.label.value_counts().to_dict()
    result = {'format': 1, 'engine_version': engine_version(engine), 'engine_sha256': descriptor['engine_sha256'],
              'base_model_version': bundle['model_version'], 'data_version': summary['data_version'],
              'test': test, 'attribution': attribution_check(bundle, test_features)}
    tick = time.perf_counter()
    result['drift'] = {'test_vs_train_reference': drift_check(engine, bundle, test_features)}
    result['drift']['seconds_per_check'] = time.perf_counter() - tick
    if args.cross_day:
        xc, yc, mc, rejected = load_cross_day(args.cross_day, bundle, args.cross_day_rows)
        cross, _, cross_features = evaluate_split(xc, yc, mc, bundle, engine, 'cross-day')
        cross.update(source=Path(args.cross_day).name, source_sha256=sha256(args.cross_day),
                     sampled_rows=args.cross_day_rows, rejected_rows=rejected,
                     label_support=mc.label.value_counts().to_dict())
        result['cross_day'] = cross
        result['drift']['cross_day_vs_train_reference'] = drift_check(engine, bundle, cross_features)
    result['machine'] = {'platform': platform.platform(), 'python': platform.python_version()}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str)[:20000])


if __name__ == '__main__':
    main()
