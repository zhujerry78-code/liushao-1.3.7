"""Read the trusted exported ZIP and replay the frozen test set without fitting.

Only a new aggregate report is written. Historical artifacts and datasets are read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from liushao.data import sha256
from liushao.model import classification_metrics, infer

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SHA = 'c161dd8161321a4cc4bafb9df6dd70ae121f1c2452b85b206922e672e301998b'
INNER_SHA = 'b50e021be5475fb9e8f87116b425fa270cf5969f897d28c16855028ed2dafd32'
MODEL_SHA = '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7'
DESCRIPTOR_SHA = 'e280995a00084234c538720541db6475bb51382c4e2ffc9c0e4db8cbde933880'


def checked_zip_member(content, expected_sha256, name):
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('ZIP SHA-256 不匹配，禁止加载其中模型')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('ZIP成员重复')
        return archive.read(name)


def compare_metrics(expected, actual):
    if set(expected) != set(actual):
        return False
    return all(math.isclose(value, actual[key], rel_tol=0, abs_tol=1e-12)
               if isinstance(value, float) else value == actual[key]
               for key, value in expected.items())


def verify(package):
    original = json.loads((ROOT / 'docs/phase2/audit.json').read_text(encoding='utf-8'))
    protected = {name: sha256(ROOT / name) for name in original['original_file_hashes']
                 if name.replace('\\', '/') not in {'app.py', 'liushao/reviews.py'}}
    # Application changes do not change the frozen inference/data/training artifacts.
    for name, actual in protected.items():
        expected = original['original_file_hashes'][name]
        path = ROOT / name
        normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected and normalized != expected:
            raise ValueError('原受保护文件不匹配: ' + name)
    content = Path(package).read_bytes()
    inner = checked_zip_member(content, PACKAGE_SHA,
                               '模型文件/liushao-frozen-c7a028b4e4cafedd.zip')
    model_bytes = checked_zip_member(inner, INNER_SHA, 'artifacts/model.joblib')
    descriptor_bytes = checked_zip_member(inner, INNER_SHA, 'artifacts/model.json')
    if hashlib.sha256(model_bytes).hexdigest() != MODEL_SHA:
        raise ValueError('冻结模型SHA-256不一致')
    if hashlib.sha256(descriptor_bytes).hexdigest() != DESCRIPTOR_SHA:
        raise ValueError('冻结描述SHA-256不一致')
    descriptor = json.loads(descriptor_bytes)
    config = descriptor['config']
    computed = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    if computed != descriptor['model_version'] or computed != 'c7a028b4e4cafedd':
        raise ValueError('模型配置版本不一致')
    if (config['rf_threshold'], config['lr_threshold'], config['near_margin']) != (.12, .84, .1):
        raise ValueError('冻结阈值不一致')
    for name in ('liushao/__init__.py', 'liushao/data.py', 'liushao/model.py'):
        expected = {k.replace('\\', '/'): v for k, v in descriptor['source_code_sha256'].items()}[name]
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError('冻结推理源码不匹配: ' + name)
    installed = {}
    for line in (ROOT / 'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        installed[name] = version(name)
        if installed[name] != expected:
            raise ValueError('依赖版本不一致: ' + name)
    if platform.python_version_tuple()[:2] != ('3', '12'):
        raise ValueError('需要Python3.12')
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        bundle = joblib.load(io.BytesIO(model_bytes))
    if bundle['config'] != config or bundle['model_version'] != computed:
        raise ValueError('原包内部配置不匹配')
    pipeline_checks = {}
    for name, steps in [('rf', ['impute', 'classifier']), ('lr', ['impute', 'log', 'scale', 'classifier'])]:
        pipe = bundle[name]
        assert list(pipe.named_steps) == steps
        assert pipe.feature_names_in_.tolist() == config['features']
        assert pipe.classes_.tolist() == [0, 1]
        assert pipe.named_steps['impute'].n_features_in_ == 77
        assert pipe.named_steps['classifier'].n_features_in_ == 79
        pipeline_checks[name] = {'steps': steps, 'classes': [0, 1], 'input_features': 77,
                                 'classifier_features': 79}
    data = ROOT / 'data/processed'
    summary = json.loads((data / 'summary.json').read_text(encoding='utf-8'))
    for name in ('features.parquet', 'manifest.parquet'):
        if sha256(data / name) != summary[name + '_sha256']:
            raise ValueError('原数据/划分哈希不一致: ' + name)
    if config['data_version'] != summary['data_version']:
        raise ValueError('原数据版本不一致')
    x = pd.read_parquet(data / 'features.parquet')
    manifest = pd.read_parquet(data / 'manifest.parquet')
    assert x.columns.tolist() == config['features']
    test = manifest.split.eq('test').to_numpy()
    xt, mt = x[test].reset_index(drop=True), manifest[test].reset_index(drop=True)
    y = mt.label.ne('Benign').astype(int).to_numpy()
    pred, values, rejected = infer(xt, bundle, 'frozen-test', 'frozen-test-features')
    assert len(pred) == 15519 and len(rejected) == 0 and len(values) == len(pred)
    baseline = json.loads((ROOT / 'docs/results/metrics.json').read_text(encoding='utf-8'))
    metrics = {name: classification_metrics(y, pred[name + '_score'], config[name + '_threshold'])
               for name in ('rf', 'lr')}
    matches = {name: compare_metrics(baseline[name], metrics[name]) for name in ('rf', 'lr')}
    historical = pd.read_csv(ROOT / 'artifacts/test_predictions.csv', float_precision='round_trip')
    assert mt.record_id.tolist() == historical.record_id.tolist()
    differences = {name: float(np.max(np.abs(pred[name] - historical[name])))
                   for name in ('rf_score', 'lr_score')}
    flags = {name: bool(np.array_equal(pred[name], historical[name]))
             for name in ('rf_anomaly', 'lr_anomaly', 'disagreement', 'near_threshold')}
    assert all(matches.values()) and all(flags.values()) and all(v <= 1e-12 for v in differences.values())
    assert all(sha256(ROOT / name) == before for name, before in protected.items())
    return {'model_version': computed, 'model_sha256': MODEL_SHA, 'descriptor_sha256': DESCRIPTOR_SHA,
            'package_sha256': PACKAGE_SHA, 'inner_package_sha256': INNER_SHA,
            'package_members': zipfile.ZipFile(io.BytesIO(content)).namelist(),
            'inner_members': zipfile.ZipFile(io.BytesIO(inner)).namelist(),
            'data_version': config['data_version'], 'data_hashes': {n: sha256(data / n) for n in
                ('features.parquet', 'manifest.parquet')}, 'thresholds': {k: config[k] for k in
                ('rf_threshold', 'lr_threshold', 'near_margin')}, 'pipelines': pipeline_checks,
            'test_records': len(pred), 'unjudgeable_records': len(rejected),
            'class_support': {str(k): int(v) for k, v in mt.label.value_counts().items()},
            'metrics': metrics, 'historical_metric_match': matches,
            'historical_flags_exact': flags, 'max_absolute_score_difference': differences,
            'python': platform.python_version(), 'installed_packages': installed,
            'protected_files_unchanged': len(protected), 'training_executed': False,
            'scope': 'Original held-out test only; same source/days, mainly Bot, no deployment or independent-scene claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from scripts.evaluation_preflight import unavailable
    if unavailable(ROOT, ['docs/phase2/audit.json', 'docs/results/metrics.json',
                           'data/processed/features.parquet', 'data/processed/manifest.parquet',
                           'data/processed/summary.json', 'artifacts/test_predictions.csv'],
                   git_ref='HEAD', output=args.output):
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        result = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
                  'git_head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                      cwd=ROOT, encoding='utf-8').strip(), 'script_sha256': sha256(__file__)}
        try:
            result.update(verify(args.package), passed=True)
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result.get(k) for k in ('passed', 'model_version', 'test_records', 'metrics', 'error')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
