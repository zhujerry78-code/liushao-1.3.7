"""Replay the original holdout and block-bootstrap unchanged frozen estimators; never fit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from liushao.data import sha256
from liushao.model import infer, classification_metrics
from app_model import validate_bundle, verified_model

MODEL_SHA = '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7'
DATA_VERSION = 'e946961bf545243c081c8995ae582b56c968bf35094d12c2e0a17d7bfea04b87'
DATA_HASHES = {'features.parquet': 'fa0b563c6ed52d7bbd16858ef90c906d4b776df161287176f651dbe119f87069',
               'manifest.parquet': '1c86e8b99a8f1b3502fde8c34f881ab94733bfc911d6514b9c8919fa19271b9b',
               'rejected.csv': 'e92bfc3faa22a3a439a56df8de5645380413dca8c4f8d7555d06dbb7223cbf6a'}
EXPECTED = {'rf': [13476, 56, 0, 1987], 'lr': [13396, 136, 7, 1980]}
SEED, REPETITIONS = 20261017, 2000


def metrics_from_counts(counts):
    tn, fp, fn, tp = np.asarray(counts, dtype=float)
    def ratio(a, b):
        return float(a / b) if b else None
    return {'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn), 'fpr': ratio(fp, fp + tn)}


def bootstrap(block_counts, *, seed=SEED, repetitions=REPETITIONS):
    models = list(block_counts)
    arrays = {name: np.asarray(block_counts[name], dtype=np.int64) for name in models}
    count = len(arrays[models[0]])
    if count < 2 or any(a.shape != (count, 4) or (a < 0).any() for a in arrays.values()):
        raise ValueError('块混淆矩阵形状或计数无效')
    draws = np.random.default_rng(seed).integers(0, count, size=(repetitions, count))
    result = {}
    for name, array in arrays.items():
        samples = [metrics_from_counts(row) for row in array[draws].sum(axis=1)]
        result[name] = {}
        for metric in ('precision', 'recall', 'f1', 'fpr'):
            values = [sample[metric] for sample in samples if sample[metric] is not None]
            bounds = np.quantile(values, [.025, .5, .975]).tolist() if values else [None] * 3
            result[name][metric] = dict(zip(('lower', 'median', 'upper'), bounds),
                                        valid_replicates=len(values), undefined_replicates=repetitions-len(values))
    return {'seed': seed, 'repetitions': repetitions, 'blocks': count, 'confidence': .95,
            'method': 'paired source-file/15-minute block percentile bootstrap with replacement',
            'metrics': result}


def reevaluate(artifacts, processed, output):
    artifacts, processed, output = map(Path, (artifacts, processed, output))
    for name, expected in DATA_HASHES.items():
        if sha256(processed / name) != expected:
            raise ValueError('恢复数据哈希不匹配: ' + name)
    summary = json.loads((processed / 'summary.json').read_bytes())
    if summary['data_version'] != DATA_VERSION or summary['split_counts'] != {
            'train': 161331, 'validation': 21468, 'test': 15519, 'purged': 11175}:
        raise ValueError('数据版本或划分不匹配')
    if sha256(artifacts / 'model.joblib') != MODEL_SHA:
        raise ValueError('禁止加载另一模型')
    descriptor_bytes = (artifacts / 'model.json').read_bytes()
    descriptor = json.loads(descriptor_bytes)
    if descriptor['config']['data_version'] != DATA_VERSION or any(
            descriptor['config'][k] != v for k, v in {'rf_threshold': .12, 'lr_threshold': .84, 'near_margin': .1}.items()):
        raise ValueError('冻结配置不匹配')
    snapshot = verified_model(artifacts / 'model.joblib', artifacts / 'model.json')
    bundle = joblib.load(io.BytesIO(snapshot.content))
    validate_bundle(bundle, descriptor)
    features = pd.read_parquet(processed / 'features.parquet')
    manifest = pd.read_parquet(processed / 'manifest.parquet')
    if len(features) != len(manifest) or not manifest.record_id.is_unique:
        raise ValueError('数据与来源映射不唯一')
    test = manifest.split.eq('test')
    xt = features.loc[test].reset_index(drop=True)
    mt = manifest.loc[test].reset_index(drop=True)
    if list(xt.columns) != bundle['config']['features'] or len(mt) != 15519 or mt.block_id.nunique() != 11:
        raise ValueError('测试特征或时间块不匹配')
    rebuilt_blocks = mt.source_file + '|' + mt.timestamp.dt.floor('15min').astype(str)
    if not mt.block_id.equals(rebuilt_blocks):
        raise ValueError('时间块身份不匹配')
    # Fix accumulation order without changing estimator parameters or model bytes.
    with joblib.parallel_backend('sequential'):
        pred, accepted, rejected = infer(xt, bundle, 'frozen-test', 'frozen-test-features')
    if len(rejected) or len(pred) != len(mt) or pred.source_row.tolist() != list(range(2, len(mt)+2)):
        raise ValueError('推理结果未一一对应原测试集')
    # infer IDs describe this temporary feature frame, not the original source files.
    for key in ('record_id', 'source_file', 'source_sha256', 'source_row', 'timestamp', 'block_id', 'label'):
        pred[key] = mt[key].to_numpy()
    y = mt.label.ne('Benign').to_numpy()
    metrics = {key: classification_metrics(y, pred[key+'_score'], bundle['config'][key+'_threshold']) for key in EXPECTED}
    for key, expected in EXPECTED.items():
        if [metrics[key][c] for c in ('tn', 'fp', 'fn', 'tp')] != expected:
            raise ValueError('重新推理与可信历史混淆矩阵不一致: ' + key)
    blocks, counts = [], {'rf': [], 'lr': []}
    for block in sorted(mt.block_id.unique()):
        mask = mt.block_id.eq(block).to_numpy()
        item = {'block_id': block, 'rows': int(mask.sum()), 'attacks': int(y[mask].sum())}
        for key in EXPECTED:
            cm = classification_metrics(y[mask], pred.loc[mask, key+'_score'], bundle['config'][key+'_threshold'])
            values = [cm[c] for c in ('tn', 'fp', 'fn', 'tp')]
            item[key] = dict(zip(('tn', 'fp', 'fn', 'tp'), values))
            counts[key].append(values)
        blocks.append(item)
    interval = bootstrap(counts)
    report = {'format': 1, 'status': 'passed', 'training_executed': False,
              'evaluated_at': datetime.now(timezone.utc).isoformat(),
              'identity': {'model_version': bundle['model_version'], 'model_sha256': MODEL_SHA,
                           'descriptor_sha256': hashlib.sha256(descriptor_bytes).hexdigest(),
                           'training_descriptor_sha256': descriptor.get('training_descriptor_sha256', hashlib.sha256(descriptor_bytes).hexdigest()),
                           'data_version': DATA_VERSION},
              'data_hashes': DATA_HASHES, 'records': len(pred), 'rejected': len(rejected),
              'class_support': {str(k): int(v) for k, v in mt.label.value_counts().items()},
              'metrics': metrics, 'intervals': interval,
              'attack_blocks': sum(b['attacks'] > 0 for b in blocks),
              'historical_comparison': 'aggregate confusion matrices; matching historical per-record predictions unavailable',
              'limits': ['基于11个观察时间块的条件95%重采样区间，不是独立部署保证。',
                         '跨块相关性仍可能存在；零漏报导致的退化区间不保证未来100%召回。'],
              'script_sha256': sha256(Path(__file__)), 'python': platform.python_version(),
              'inference_parallel_backend': 'sequential (fixed accumulation order; estimator parameters unchanged)',
              'dependencies': {key: importlib.metadata.version(key) for key in ('numpy', 'pandas', 'scikit-learn', 'pyarrow', 'joblib')}}
    output.mkdir(parents=True, exist_ok=False)
    pred.to_csv(output / 'test_predictions.csv', index=False, float_format='%.17g')
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save('block_counts.json', blocks)
    save('intervals.json', interval)
    report['outputs'] = {name: sha256(output / name) for name in ('test_predictions.csv', 'block_counts.json', 'intervals.json')}
    save('reevaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--processed-data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = reevaluate(args.artifacts, args.processed_data, args.output)
    print(json.dumps({'status': report['status'], 'metrics': report['metrics'], 'intervals': report['intervals']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
