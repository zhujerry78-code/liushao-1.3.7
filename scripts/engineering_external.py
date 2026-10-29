"""Evaluate a frozen bundle on one explicit official CSV; never train or change splits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import time
from urllib.parse import unquote, urlparse

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn

from app_model import CheckedPredictor
from liushao.data import fingerprints, sha256, validate_csv_layout, version_id, write_json
from liushao.model import infer


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SUMMARY = ROOT / 'docs/results/data_summary.json'
BASELINE_METRICS = ROOT / 'docs/results/metrics.json'
OUTPUT_ROOT = ROOT / 'test-results/engineering/external'
SPLITS = ('train', 'validation', 'test', 'purged')
STRATA = ('all', 'exact_overlap', 'near_overlap', 'any_overlap', 'no_overlap')
OFFICIAL_HOST = 'cse-cic-ids2018.s3.ca-central-1.amazonaws.com'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_file(path, expected, expected_bytes=None):
    path = Path(path)
    require(path.is_file(), f'必要文件不存在: {path}')
    require(re.fullmatch(r'[0-9a-f]{64}', expected) is not None, '必须提供完整小写 SHA-256')
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, f'文件大小不匹配: {path}')
    actual = sha256(path)
    require(actual == expected, f'文件 SHA-256 不匹配: {path}')
    return actual


def verify_source_module(path, expected):
    """Accept only LF/CRLF variants of the same registered inference source bytes."""
    content = Path(path).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    lf_content = content.replace(b'\r\n', b'\n')
    normalized = hashlib.sha256(lf_content).hexdigest()
    crlf = hashlib.sha256(lf_content.replace(b'\n', b'\r\n')).hexdigest()
    require(expected in (actual, normalized, crlf), f'模型核心源码内容不匹配: {path}')
    return {'expected_sha256': expected, 'working_tree_sha256': actual,
            'lf_sha256': normalized,
            'match': ('raw' if actual == expected else
                      'crlf_to_lf' if normalized == expected else 'lf_to_crlf')}


def verify_source_registration(descriptor):
    """Require the existing inference-source registration before loading a bundle."""
    sources = descriptor.get('source_code_sha256')
    require(isinstance(sources, dict), '缺少有效的 source_code_sha256 源码登记')
    require(all(isinstance(name, str) for name in sources), '源码路径必须为字符串')
    normalized = {name.replace('\\', '/') for name in sources}
    require(len(normalized) == len(sources), '源码登记路径重复')
    required_sources = {'liushao/__init__.py', 'liushao/model.py', 'liushao/data.py'}
    require(required_sources <= normalized, '缺少模型/适配源码哈希')
    source_verification = {}
    for relative, expected in sources.items():
        source_path = (ROOT / relative.replace('\\', '/')).resolve()
        require(source_path.is_relative_to(ROOT / 'liushao') and source_path.suffix == '.py',
                'descriptor 源码路径超出模型模块或不是 Python 文件')
        require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected) is not None,
                f'源码哈希无效: {relative}')
        source_verification[relative] = verify_source_module(source_path, expected)
    return source_verification


def verify_source_inputs(args, summary):
    """Verify the explicit source independently of any model package."""
    verify_file(args.source, args.source_sha256, args.source_bytes)
    parsed = urlparse(args.source_url)
    require(parsed.scheme == 'https' and parsed.hostname == OFFICIAL_HOST,
            '来源必须是明确的 CSE-CIC-IDS2018 官方 HTTPS 对象')
    require(unquote(parsed.path).startswith('/Processed Traffic Data for ML Algorithms/'),
            '来源必须指向官方特征 CSV 目录')
    source_name = Path(unquote(parsed.path)).name
    require(source_name == Path(args.source).name, '来源 URL 与文件名不一致')
    require(args.source_sha256 not in {s['sha256'] for s in summary['sources']},
            '新场景文件与原来源相同，不能称为独立采集日验证')
    return {'url': args.source_url, 'name': source_name, 'sha256': args.source_sha256,
            'bytes': args.source_bytes}


def load_reference_data(reference_dir):
    """Verify the registered original features and splits without loading a model."""
    baseline = read_json(BASELINE_SUMMARY)
    reference_dir = Path(reference_dir)
    summary = read_json(reference_dir / 'summary.json')
    for key in ('data_version', 'features', 'config', 'sources', 'split_counts'):
        require(summary[key] == baseline[key], f'原数据登记信息不匹配: {key}')
    for name in ('features.parquet', 'manifest.parquet'):
        expected = baseline[name + '_sha256']
        require(summary[name + '_sha256'] == expected, f'原数据哈希声明不匹配: {name}')
        verify_file(reference_dir / name, expected)
    require(version_id(summary['config'], summary['sources'],
                       summary['manifest.parquet_sha256']) == summary['data_version'],
            '原数据版本无法按原规则重算')
    x = pd.read_parquet(reference_dir / 'features.parquet')
    manifest = pd.read_parquet(reference_dir / 'manifest.parquet')
    require(len(x) == len(manifest), '原特征与 manifest 行数不匹配')
    require(x.columns.tolist() == summary['features'], '原特征文件列顺序不匹配')
    require(manifest.record_id.is_unique, '原 manifest 记录 ID 不唯一')
    require(set(manifest.split.unique()) <= set(SPLITS), '原 manifest 含未知划分')
    require(manifest.split.value_counts().to_dict() == summary['split_counts'], '原划分数量不匹配')
    for name, near in (('exact_fingerprint', False), ('near_fingerprint', True)):
        recomputed = fingerprints(x, near=near)
        require(np.array_equal(recomputed.astype(str), manifest[name].astype(str)),
                f'原特征与 manifest 指纹不匹配: {name}')
    references = {
        kind: {split: set(manifest.loc[manifest.split.eq(split), column].astype(str))
               for split in SPLITS}
        for kind, column in (('exact', 'exact_fingerprint'), ('near', 'near_fingerprint'),
                             ('ids_by_split', 'record_id'))
    }
    references['ids'] = set(manifest.record_id)
    provenance = {'data_version': summary['data_version'],
                  'features_sha256': summary['features.parquet_sha256'],
                  'manifest_sha256': summary['manifest.parquet_sha256'],
                  'summary_sha256': sha256(reference_dir / 'summary.json'),
                  'baseline_summary_sha256': sha256(BASELINE_SUMMARY)}
    return summary, references, provenance


def load_frozen_inputs(args):
    """Match the real artifact bytes, baseline data and descriptor before deserializing."""
    summary, references, reference_provenance = load_reference_data(args.reference_data)
    source_provenance = verify_source_inputs(args, summary)
    model_bytes = Path(args.model).read_bytes()
    actual_model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    require(actual_model_sha256 == args.model_sha256, '模型文件哈希不匹配')
    descriptor_bytes = Path(args.descriptor).read_bytes()
    descriptor = json.loads(descriptor_bytes.decode('utf-8'))
    require(descriptor.get('model_sha256') == args.model_sha256, '模型 descriptor 哈希不匹配')
    source_verification = verify_source_registration(descriptor)
    baseline_metrics = read_json(BASELINE_METRICS)
    bundle = joblib.load(io.BytesIO(model_bytes))
    config = bundle['config']
    require(config == descriptor['config'], '冻结配置与 descriptor 不一致')
    require(bundle['model_version'] == descriptor['model_version'], '模型版本不一致')
    require(bundle['model_version'] == baseline_metrics['model_version'],
            '模型版本与原冻结实验不同；停止独立验证，不重新选择阈值')
    for model in ('rf', 'lr'):
        require(config[model + '_threshold'] == baseline_metrics[model]['threshold'],
                f'{model} 阈值与原冻结实验不同')
    require(bundle['reference'] == descriptor['reference'], '正常参考与 descriptor 不一致')
    require(bundle['training_rows'] == descriptor['training_rows'], '训练条数与 descriptor 不一致')
    computed_version = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    require(computed_version == bundle['model_version'], '模型配置版本校验失败')
    require(config['data_version'] == summary['data_version'], '模型与原数据版本不匹配')
    require(config['features'] == summary['features'], '模型与原特征顺序不匹配')
    require(config['sklearn'] == sklearn.__version__, 'scikit-learn 版本与冻结模型不匹配')
    for key in ('rf_threshold', 'lr_threshold', 'near_margin'):
        require(np.isfinite(config[key]) and 0 <= config[key] <= 1,
                f'冻结参数无效: {key}')
    for name in ('rf', 'lr'):
        require(list(bundle[name].feature_names_in_) == config['features'],
                f'{name} 内部特征顺序不匹配')
        require(np.array_equal(bundle[name].classes_, [0, 1]), f'{name} 类别顺序不匹配')
    rf_parameters = bundle['rf'].named_steps['classifier'].get_params()
    for parameter, key in (('n_estimators', 'n_estimators'), ('max_depth', 'max_depth'),
                           ('min_samples_leaf', 'min_samples_leaf'), ('random_state', 'seed')):
        require(rf_parameters[parameter] == config[key], f'RF 实际参数与冻结配置不匹配: {key}')
    require(set(bundle['reference']) == set(config['features']), '正常参考字段不匹配')
    require(bundle['training_rows'] == summary['split_counts'].get('train', 0), '训练样本数不匹配')
    provenance = {
        'source': source_provenance,
        'model': {'sha256': actual_model_sha256,
                  'descriptor_sha256': hashlib.sha256(descriptor_bytes).hexdigest(),
                  'version': bundle['model_version'], 'config': config,
                  'source_verification': source_verification,
                  'non_inference_source_entries': sorted(
                      name.replace('\\', '/') for name in descriptor['source_code_sha256']
                      if name.replace('\\', '/') not in
                      {'liushao/__init__.py', 'liushao/model.py', 'liushao/data.py'})},
        'reference': {**reference_provenance,
                      'baseline_metrics_sha256': sha256(BASELINE_METRICS)},
    }
    return bundle, references, provenance


def csv_chunks(path, header, chunk_rows):
    offset = 0
    for chunk in pd.read_csv(path, header=0, names=header, dtype=str,
                             keep_default_na=False, chunksize=chunk_rows):
        chunk.attrs['source_rows'] = np.arange(offset + 2, offset + len(chunk) + 2)
        yield chunk
        offset += len(chunk)


def preflight_csv(args, features):
    """Check the entire layout and label vocabulary before any prediction."""
    header = validate_csv_layout(args.source)
    require(set(features) <= set(header), '新场景缺少必需特征')
    require('Label' in header, '独立评价需要单独的 Label 列')
    allowed = {'Benign', *args.attack_label}
    require(bool(args.attack_label) and 'Benign' not in args.attack_label,
            '必须明确指定独立场景攻击标签，且不能包含 Benign')
    labels = Counter()
    timestamp_dates = Counter()
    for chunk in csv_chunks(args.source, header, args.chunk_rows):
        values = chunk.Label.str.strip()
        bad = ~values.isin(allowed)
        if bad.any():
            index = int(np.flatnonzero(bad.to_numpy())[0])
            line = int(chunk.attrs['source_rows'][index])
            raise ValueError(f'来源第 {line} 行存在未授权评价标签: {values.iloc[index]!r}')
        labels.update(values)
        if 'Timestamp' in chunk:
            parsed = pd.to_datetime(chunk.Timestamp, format='%d/%m/%Y %H:%M:%S', errors='coerce')
            timestamp_dates.update(parsed.dt.strftime('%Y-%m-%d').fillna('<invalid>'))
        else:
            timestamp_dates['<missing_column>'] += len(chunk)
    require(bool(labels), '独立场景文件没有记录')
    return header, labels, timestamp_dates


def confusion_counts(y, predicted):
    y, predicted = np.asarray(y, dtype=bool), np.asarray(predicted, dtype=bool)
    return Counter(tn=int((~y & ~predicted).sum()), fp=int((~y & predicted).sum()),
                   fn=int((y & ~predicted).sum()), tp=int((y & predicted).sum()))


def finish_counts(counts):
    tn, fp, fn, tp = (counts[k] for k in ('tn', 'fp', 'fn', 'tp'))
    return {**{k: counts[k] for k in ('tn', 'fp', 'fn', 'tp')},
            'records': tn + fp + fn + tp, 'normal_support': tn + fp,
            'attack_support': tp + fn,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'fpr': fp / (tn + fp) if tn + fp else None,
            'undefined': [name for name, denominator in
                          [('precision', tp + fp), ('recall', tp + fn),
                           ('f1', 2 * tp + fp + fn), ('fpr', tn + fp)] if not denominator]}


def evaluate(args, bundle, references, header, labels, out):
    # Reject invalid output before the unchanged infer converts scores to labels.
    checked_bundle = {**bundle, **{
        name: CheckedPredictor(bundle[name], name) for name in ('rf', 'lr')
    }}
    counts = {stratum: {model: Counter() for model in ('rf', 'lr')} for stratum in STRATA}
    class_counts = {stratum: Counter({label: 0 for label in labels}) for stratum in STRATA}
    class_detected = {model: Counter({label: 0 for label in labels}) for model in ('rf', 'lr')}
    overlaps = {kind: Counter({split: 0 for split in SPLITS}) for kind in ('exact', 'near')}
    seen = {'exact': set(), 'near': set()}
    duplicates = Counter(exact=0, near=0)
    quality = Counter(raw_rows=0, valid_rows=0, unjudgeable_rows=0, missing_cells=0,
                      rows_with_missing_features=0, record_id_overlap=0, invalid_timestamp_rows=0)
    rejected_labels = Counter()
    rejected_reasons = Counter()
    wrote_predictions = False
    for chunk in csv_chunks(args.source, header, args.chunk_rows):
        source_rows = chunk.attrs['source_rows']
        label_by_row = dict(zip(source_rows, chunk.Label.str.strip()))
        raw = chunk.drop(columns='Label')
        raw.attrs['source_rows'] = source_rows
        pred, features, rejected = infer(raw, checked_bundle, args.source_sha256,
                                         Path(args.source).name)
        quality.update(raw_rows=len(chunk), valid_rows=len(pred), unjudgeable_rows=len(rejected),
                       missing_cells=int(features.isna().sum().sum()),
                       rows_with_missing_features=int(features.isna().any(axis=1).sum()),
                       invalid_timestamp_rows=int(pred.timestamp.isna().sum()
                                                  + rejected.timestamp.isna().sum()))
        rejected_labels.update(label_by_row[int(row)] for row in rejected.source_row)
        rejected_reasons.update(rejected.reason)
        if not len(pred):
            continue
        truth_labels = np.array([label_by_row[int(row)] for row in pred.source_row])
        y = truth_labels != 'Benign'
        masks = {}
        for kind, near in (('exact', False), ('near', True)):
            fingerprint = fingerprints(features, near=near).astype(str)
            union = np.zeros(len(pred), dtype=bool)
            for split in SPLITS:
                mask = fingerprint.isin(references[kind][split]).to_numpy()
                overlaps[kind][split] += int(mask.sum())
                union |= mask
            masks[kind + '_overlap'] = union
            before = len(seen[kind])
            seen[kind].update(fingerprint)
            duplicates[kind] += len(fingerprint) - (len(seen[kind]) - before)
        masks['any_overlap'] = masks['exact_overlap'] | masks['near_overlap']
        masks['no_overlap'] = ~masks['any_overlap']
        masks['all'] = np.ones(len(pred), dtype=bool)
        quality['record_id_overlap'] += int(pred.record_id.isin(references['ids']).sum())
        for stratum, take in masks.items():
            class_counts[stratum].update(truth_labels[take])
            for model in ('rf', 'lr'):
                prediction = pred[model + '_anomaly'].to_numpy()
                counts[stratum][model].update(confusion_counts(y[take], prediction[take]))
        for model in ('rf', 'lr'):
            class_detected[model].update(truth_labels[pred[model + '_anomaly'].to_numpy()])
        if args.write_predictions:
            pred['ground_truth_evaluation_only'] = truth_labels
            for name in ('exact_overlap', 'near_overlap', 'any_overlap'):
                pred[name] = masks[name]
            pred.to_csv(out / 'predictions.csv', mode='a', header=not wrote_predictions, index=False)
            wrote_predictions = True
        print(f'已推理 {quality["raw_rows"]:,} 行；可判断 {quality["valid_rows"]:,} 行', flush=True)
    require(quality['raw_rows'] == sum(labels.values()), '实际推理行数与预检不一致')
    require(quality['valid_rows'] + quality['unjudgeable_rows'] == quality['raw_rows'], '行数不守恒')
    return {'quality': dict(quality), 'source_label_support': dict(labels),
            'unjudgeable_label_support': dict(rejected_labels),
            'unjudgeable_reasons': dict(rejected_reasons),
            'valid_label_predicted_anomalies': {k: dict(v) for k, v in class_detected.items()},
            'reference_overlap_rows': {k: dict(v) for k, v in overlaps.items()},
            'within_source_duplicate_extra_rows': dict(duplicates),
            'strata': {s: {'label_support': dict(class_counts[s]),
                          **{m: finish_counts(counts[s][m]) for m in ('rf', 'lr')}} for s in STRATA},
            'metric_rule': 'Matches classification_metrics: zero_division=0; undefined named explicitly.',
            'scope': 'One unseen capture day in the same benchmark network; not future-time or deployment validation.',
            'selection': 'All judgeable rows retained. Overlap strata are descriptive, never used to change original splits.'}


def run(args):
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) is not None, 'run-id 只能含字母数字下划线横线')
    require(args.chunk_rows > 0 and args.source_bytes > 0, '块大小和源文件大小必须为正数')
    out = OUTPUT_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'script_sha256': sha256(__file__),
                'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
                                'ram_bytes': psutil.virtual_memory().total,
                                'packages': {n: version(n) for n in
                                             ('numpy', 'pandas', 'scikit-learn', 'joblib', 'pyarrow')}},
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(out / 'manifest.json', manifest)
    try:
        bundle, references, provenance = load_frozen_inputs(args)
        manifest.update(provenance)
        header, labels, timestamp_dates = preflight_csv(args, bundle['config']['features'])
        manifest['source_timestamp_dates'] = dict(timestamp_dates)
        from scripts.engineering_external_preflight import (
            audit_source, verify_evaluation_matches_audit,
        )
        audit = audit_source(args, bundle['config']['features'], references, header, labels, out)
        manifest['data_audit_sha256'] = sha256(out / 'audit.json')
        manifest['data_audit_completed_before_inference'] = True
        write_json(out / 'manifest.json', manifest)
        manifest['preflight_seconds'] = time.perf_counter() - started
        tick = time.perf_counter()
        result = evaluate(args, bundle, references, header, labels, out)
        verify_evaluation_matches_audit(result, audit)
        result['streaming_inference_and_metrics_seconds'] = time.perf_counter() - tick
        verify_file(args.source, args.source_sha256, args.source_bytes)
        verify_file(args.model, args.model_sha256)
        manifest['status'] = 'completed'
        manifest['total_seconds'] = time.perf_counter() - started
        manifest['process_rss_at_end_bytes'] = psutil.Process().memory_info().rss
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'metrics.json', result)
        manifest['metrics_sha256'] = sha256(out / 'metrics.json')
        if (out / 'predictions.csv').exists():
            manifest['predictions_sha256'] = sha256(out / 'predictions.csv')
        write_json(out / 'manifest.json', manifest)
        return out
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        total_seconds=time.perf_counter() - started)
        write_json(out / 'manifest.json', manifest)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'descriptor', 'reference-data', 'source'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('model-sha256', 'source-url', 'source-sha256', 'run-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-bytes', required=True, type=int)
    parser.add_argument('--attack-label', required=True, action='append')
    parser.add_argument('--chunk-rows', type=int, default=20000)
    parser.add_argument('--write-predictions', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(run(parse_args()))
