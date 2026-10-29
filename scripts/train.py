"""Frozen training and independent test evaluation; no test-driven tuning."""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning

from liushao.data import sha256, write_json
from liushao.model import classification_metrics, fit_bundle, infer, ranked, ranking_metrics, review_ranked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed')
    p.add_argument('--output', default='artifacts')
    args = p.parse_args()
    start = time.perf_counter()
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / 'summary.json').read_text(encoding='utf-8'))
    for name in ['features.parquet', 'manifest.parquet']:
        if sha256(data / name) != summary[name + '_sha256']:
            raise ValueError('数据哈希不匹配: ' + name)
    x, manifest = pd.read_parquet(data / 'features.parquet'), pd.read_parquet(data / 'manifest.parquet')
    y = (manifest.label != 'Benign').astype(int).to_numpy()
    train, val, test = [manifest.split.eq(name).to_numpy() for name in ['train', 'validation', 'test']]
    print('Training', int(train.sum()), 'validation', int(val.sum()), 'test', int(test.sum()), flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        bundle = fit_bundle(x[train], y[train], x[val], y[val], summary['data_version'])
    joblib.dump(bundle, out / 'model.joblib', compress=3)
    descriptor = {k: v for k, v in bundle.items() if k not in ['rf', 'lr']}
    descriptor['model_sha256'] = sha256(out / 'model.joblib')
    descriptor['source_code_sha256'] = {str(path): sha256(path) for path in sorted(Path('liushao').glob('*.py'))}
    write_json(out / 'model.json', descriptor)
    print('Models fitted', bundle['fit_seconds'], flush=True)
    xt = x[test].reset_index(drop=True)
    mt = manifest[test].reset_index(drop=True)
    tick = time.perf_counter()
    pred, _, rejected = infer(xt, bundle, 'evaluation', 'test_features')
    prediction_seconds = time.perf_counter() - tick
    pred['record_id'] = mt.record_id
    pred['source_file'] = mt.source_file
    pred['source_row'] = mt.source_row
    pred['timestamp'] = mt.timestamp
    pred['source_sha256'] = mt.source_sha256
    yt = y[test]
    errors = pred.rf_anomaly.to_numpy() != yt
    flagged = (pred.disagreement | pred.near_threshold).to_numpy()
    review_order = review_ranked(pred).index.to_numpy()
    score_order = ranked(pred).index.to_numpy()
    original_order = np.arange(len(pred))
    def error_budget(order, k):
        n = min(k, len(order))
        return {'k_actual': n, 'errors_found': int(errors[order[:n]].sum())}
    # Equal actual budgets across all three lists; total candidate population is unchanged.
    budgets = []
    for k in [20, 50, 100]:
        common = min(k, len(review_order), len(pred))
        budgets.append({'k_requested': k, 'common_budget': common,
                        'original': error_budget(original_order, common),
                        'score': error_budget(score_order, common),
                        'review': error_budget(review_order, common)})
    per_class = {}
    for label in sorted(mt.label.unique()):
        take = mt.label.eq(label).to_numpy()
        per_class[label] = {'support': int(take.sum()), 'predicted_anomalies': int(pred.rf_anomaly[take].sum())}
    results = {'model_version': bundle['model_version'], 'data_version': summary['data_version'],
               'rf': classification_metrics(yt, pred.rf_score, bundle['config']['rf_threshold']),
               'lr': classification_metrics(yt, pred.lr_score, bundle['config']['lr_threshold']),
               'test_per_class': per_class, 'unjudgeable_test_rows': len(rejected),
               'ranking_original': ranking_metrics(yt, original_order),
               'ranking_score': ranking_metrics(yt, score_order),
               'review': {'flagged': int(flagged.sum()), 'errors_total': int(errors.sum()),
                          'errors_flagged': int((errors & flagged).sum()),
                          'error_rate_in_flags': float(errors[flagged].mean()) if flagged.any() else None,
                          'missed_attacks_flagged': int((flagged & ~pred.rf_anomaly.to_numpy() & (yt == 1)).sum()),
                          'false_alarms_flagged': int((flagged & pred.rf_anomaly.to_numpy() & (yt == 0)).sum()),
                          'same_budget_error_discovery': budgets},
               'timing': {'fit_seconds': bundle['fit_seconds'], 'test_predict_seconds': prediction_seconds,
                          'total_seconds_including_reads_fit_save_eval': time.perf_counter() - start},
               'machine': {'platform': platform.platform(), 'python': platform.python_version(),
                           'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
                           'ram_bytes': psutil.virtual_memory().total,
                           'process_rss_at_end_bytes': psutil.Process().memory_info().rss},
               'limitations': ['same source and capture days; no deployment or cross-environment claim',
                               'Hulk has no test support; SlowHTTPTest has only one test record',
                               'review does not change classifier or automatically retrain',
                               'near-duplicate audit limited to six significant digit equality']}
    pred['ground_truth_evaluation_only'] = yt
    pred['attack_type_evaluation_only'] = mt.label
    pred.to_csv(out / 'test_predictions.csv', index=False)
    write_json(out / 'metrics.json', results)
    write_json('docs/results/metrics.json', results)
    # A fixed uniform sample from test for the product demo, never used for performance claims.
    rng = np.random.default_rng(42)
    demo_indices = np.sort(rng.choice(len(xt), size=min(1000, len(xt)), replace=False))
    demo = xt.iloc[demo_indices].reset_index(drop=True)
    demo_meta = mt.iloc[demo_indices].reset_index(drop=True)
    demo['Timestamp'] = demo_meta.timestamp.dt.strftime('%d/%m/%Y %H:%M:%S')
    demo.to_csv(out / 'demo.csv', index=False)
    demo_meta.drop(columns=['label', 'exact_fingerprint', 'near_fingerprint', 'split', 'purge_reason', 'block_id']).to_parquet(out / 'demo_origins.parquet', index=False)
    write_json(out / 'demo.json', {'csv_sha256': sha256(out / 'demo.csv'), 'origins_sha256': sha256(out / 'demo_origins.parquet'),
                                 'model_version': bundle['model_version'], 'selection': 'seed42 uniform1000 from held-out test', 'rows': len(demo)})
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
