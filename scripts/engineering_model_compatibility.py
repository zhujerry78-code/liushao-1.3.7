"""Check trusted existing MVP bundles without training or changing the product loader.

Compatibility with the legacy interface, exact frozen-artifact identity, and missing
handoff contracts are deliberately reported separately. Joblib is not a safe format
for untrusted input: a matching sidecar hash proves consistency, not provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from app_model import CompatibilityError, require, validate_bundle, validate_descriptor

from liushao.data import read_csv_checked, sha256
from liushao.model import infer, ranked
from scripts.engineering_ci import locked_environment
from scripts.engineering_external import verify_source_module
from scripts.engineering_run import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def check_sample(path: Path, bundle: dict, expected_sha256: str | None = None) -> dict:
    """Exercise existing cleaning/inference and label isolation on an explicit CSV."""
    require(path.stat().st_size <= 50 * 1024 * 1024, '兼容检查 CSV 最多 50 MB')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    require(expected_sha256 is None or digest == expected_sha256, '样例与可信交接 SHA-256 不一致')
    raw = read_csv_checked(io.BytesIO(content))
    predictions, values, rejected = infer(raw, bundle, digest, path.name)
    require(len(predictions) + len(rejected) == len(raw), '推理与拒绝条数不守恒')
    require(bool(len(predictions)), '样例没有可判断记录；未验证真实推理')
    require(values.columns.tolist() == bundle['config']['features'], '推理特征顺序不一致')
    ids = predictions.record_id.tolist() + rejected.record_id.tolist()
    require(len(ids) == len(set(ids)), '推理记录身份重复')
    require(set(ids) == {f'{digest}:{row}' for row in raw.attrs['source_rows']}, '来源行身份不一致')
    require(not {'Label', 'label', 'ground_truth'} & set(predictions.columns), '标签泄漏到预测结果')
    for name in ('rf', 'lr'):
        probabilities = np.asarray(bundle[name].predict_proba(values))
        require(probabilities.shape == (len(values), 2) and np.isrealobj(probabilities)
                and np.isfinite(probabilities).all()
                and ((probabilities >= 0) & (probabilities <= 1)).all()
                and np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12),
                f'{name} predict_proba 输出无效')
        np.testing.assert_allclose(predictions[name + '_score'], probabilities[:, 1],
                                   rtol=0, atol=1e-12)
        require(np.array_equal(predictions[name + '_anomaly'],
                               predictions[name + '_score'] >= bundle['config'][name + '_threshold']),
                f'{name} 阈值判断不一致')
    require(np.array_equal(predictions.disagreement, predictions.rf_anomaly != predictions.lr_anomaly),
            '分歧判断不一致')
    require(np.array_equal(predictions.near_threshold,
                           (predictions.rf_score - predictions.rf_threshold).abs() <= predictions.near_margin),
            '近阈值判断不一致')
    changed_labels = raw.copy(deep=True)
    changed_labels['Label'] = 'engineering-label-isolation-only'
    repeated, repeated_values, repeated_rejected = infer(changed_labels, bundle, digest, path.name)
    pd.testing.assert_frame_equal(values, repeated_values)
    pd.testing.assert_frame_equal(rejected, repeated_rejected)
    pd.testing.assert_frame_equal(predictions, repeated, check_exact=False, rtol=0, atol=1e-12)
    require(ranked(predictions).record_id.tolist() == ranked(repeated).record_id.tolist(),
            '重复推理排序不一致')
    return {'sha256': digest, 'rows': len(raw), 'predicted_rows': len(predictions),
            'rejected_rows': len(rejected), 'features': len(values.columns),
            'label_isolation': True, 'repeat_ranking': True,
            'scope': 'Existing CSV inference only; no training, quality metric or browser E2E claim.'}


def inspect_model(model: Path, descriptor: Path, *, sample: Path | None = None,
                  baseline_metrics: Path | None = None, expected_sha256: str | None = None,
                  frozen_sha256: str | None = None,
                  expected_descriptor_sha256: str | None = None,
                  expected_sample_sha256: str | None = None) -> dict:
    """Return a fail-closed report; do not write to model/data/review paths."""
    started = time.perf_counter()
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'compatible': False, 'checks': [], 'sample': None, 'frozen_baseline': None,
              'contract_gaps': [], 'interface': 'legacy liushao-mvp-1',
              'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__},
              'model': str(model), 'descriptor': str(descriptor),
              'limits': ['This independent checker does not change the Streamlit loading path.',
                         'Matching hashes do not make untrusted joblib files safe to load.',
                         'Compatibility does not certify training, performance or a new schema.']}
    stage = 'descriptor'
    try:
        descriptor = Path(descriptor)
        descriptor_bytes = descriptor.read_bytes()
        descriptor_sha256 = hashlib.sha256(descriptor_bytes).hexdigest()
        require(expected_descriptor_sha256 is None
                or descriptor_sha256 == expected_descriptor_sha256,
                '模型描述与可信交接 SHA-256 不一致')
        metadata = json.loads(descriptor_bytes.decode('utf-8'))
        validate_descriptor(metadata)
        report['model_version'] = metadata['model_version']
        report['descriptor_sha256'] = descriptor_sha256
        report['thresholds'] = {key: metadata['config'][key] for key in
                                ('rf_threshold', 'lr_threshold', 'near_margin')}
        report['contract_gaps'] = [
            'No agreed schema_version contract is checked; legacy compatibility only.',
            'Class semantics 0=normal/1=anomaly are inferred from existing train.py, not a package mapping.',
            'The bundle does not attest the training dependency lock; only the current repository lock is checked.',
        ]
        report['checks'].append({'name': stage, 'passed': True})
        stage = 'environment'
        environment = locked_environment()
        require(environment['passed'], f'当前 Python/依赖不符合仓库锁: {environment}')
        report['checks'].append({'name': stage, 'passed': True, 'details': environment})
        stage = 'source_code'
        sources = metadata.get('source_code_sha256')
        require(isinstance(sources, dict), '缺少 source_code_sha256')
        normalized = {name.replace('\\', '/'): digest for name, digest in sources.items()}
        required_sources = {'liushao/__init__.py', 'liushao/model.py', 'liushao/data.py'}
        require(required_sources <= set(normalized), '缺少模型/适配源码哈希')
        evidence = {}
        for name, digest in normalized.items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT / 'liushao') and path.suffix == '.py', '源码路径超出模型模块')
            require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None,
                    f'源码哈希无效: {name}')
            if name in required_sources:
                evidence[name] = verify_source_module(path, digest)
        report['non_inference_source_entries'] = sorted(set(normalized) - required_sources)
        report['checks'].append({'name': stage, 'passed': True, 'details': evidence})
        stage = 'model_hash'
        declared = metadata.get('model_sha256')
        require(isinstance(declared, str) and re.fullmatch(r'[0-9a-f]{64}', declared) is not None,
                '模型 SHA-256 无效')
        model_bytes = Path(model).read_bytes()
        actual = hashlib.sha256(model_bytes).hexdigest()
        report['model_sha256'] = actual
        require(actual == declared, '模型文件与 descriptor SHA-256 不一致')
        require(expected_sha256 is None or actual == expected_sha256, '模型与指定 SHA-256 不一致')
        report['checks'].append({'name': stage, 'passed': True, 'details': actual})
        stage = 'deserialize'
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            bundle = joblib.load(io.BytesIO(model_bytes))
        report['checks'].append({'name': stage, 'passed': True})
        stage = 'fitted_bundle'
        validate_bundle(bundle, metadata)
        report['checks'].append({'name': stage, 'passed': True, 'details': {
            'input_features': len(metadata['config']['features']),
            'classifier_features': {name: int(bundle[name].named_steps['classifier'].n_features_in_)
                                    for name in ('rf', 'lr')},
            'rf_trees': len(bundle['rf'].named_steps['classifier'].estimators_),
        }})
        if sample is not None:
            stage = 'sample_inference'
            report['sample'] = check_sample(Path(sample), bundle, expected_sample_sha256)
            report['checks'].append({'name': stage, 'passed': True, 'details': report['sample']})
        if baseline_metrics is not None:
            stage = 'frozen_baseline'
            baseline = json.loads(Path(baseline_metrics).read_text(encoding='utf-8'))
            comparisons = {
                'model_version': (baseline['model_version'], metadata['model_version']),
                'data_version': (baseline['data_version'], metadata['config']['data_version']),
                **{name + '_threshold': (baseline[name]['threshold'],
                                        metadata['config'][name + '_threshold']) for name in ('rf', 'lr')},
            }
            differences = {key: {'expected': expected, 'actual': actual_value}
                           for key, (expected, actual_value) in comparisons.items() if expected != actual_value}
            report['frozen_baseline'] = {
                'matched': not differences and actual == frozen_sha256,
                'metadata_match': not differences, 'artifact_match': actual == frozen_sha256,
                'expected_sha256': frozen_sha256, 'differences': differences,
                'metrics_sha256': sha256(baseline_metrics),
                'scope': 'Exact frozen artifact and recorded metadata, not an algorithm evaluation.',
            }
        report['compatible'] = True
    except Exception as error:
        report['checks'].append({'name': stage, 'passed': False,
                                 'error': f'{type(error).__name__}: {error}'})
    report['seconds'] = round(time.perf_counter() - started, 3)
    return report


def exit_code(report: dict, require_frozen: bool = False) -> int:
    if not report['compatible']:
        return 1
    if require_frozen and not (report.get('frozen_baseline') or {}).get('matched'):
        return 2
    return 0


def verify_frozen_handoff(model: Path, descriptor: Path, sample: Path, manifest: dict) -> dict:
    """Verify all five original handoff files before trusting a frozen-package load."""
    paths = {'model.joblib': model, 'model.json': descriptor, 'demo.csv': sample,
             'demo.json': sample.parent / 'demo.json',
             'demo_origins.parquet': sample.parent / 'demo_origins.parquet'}
    expected = {Path(item['path']).name: item for item in manifest['files']}
    require(set(paths) <= set(expected), '可信交接缺少原冻结文件身份')
    evidence = {}
    for name, path in paths.items():
        content = Path(path).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(digest == expected[name]['sha256'] and len(content) == expected[name]['bytes'],
                f'原冻结可信交接文件不匹配: {name}')
        evidence[name] = {'sha256': digest, 'bytes': len(content)}
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--require-frozen', action='store_true')
    parser.add_argument('--output', type=Path, help='New JSON report path; existing files are never overwritten')
    args = parser.parse_args(argv)
    # Reserve output before loading: a typo must not overwrite a model or an older report.
    output = args.output
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, {'status': 'unfinished', 'compatible': False}, exclusive=True)
    report = {'compatible': False, 'checks': [], 'sample': None, 'frozen_baseline': None}
    try:
        audit = json.loads((ROOT / 'docs/phase2/audit.json').read_text(encoding='utf-8'))
        frozen_hash = {key.replace('\\', '/'): value for key, value in
                       audit['original_file_hashes'].items()}['artifacts/model.joblib']
        trusted = {}
        handoff = None
        if args.require_frozen:
            manifest = json.loads((ROOT / 'docs/model-handoff-20261016/'
                                   'FROZEN_MODEL_MANIFEST.json').read_text(encoding='utf-8'))
            trusted = {Path(item['path']).name: item['sha256'] for item in manifest['files']}
            require(trusted['model.joblib'] == frozen_hash, '可信交接与历史冻结模型 SHA 不一致')
            require(args.expected_sha256 in (None, frozen_hash), '指定 SHA 与原冻结交接不一致')
            args.sample = args.sample or args.model.parent / 'demo.csv'
            handoff = verify_frozen_handoff(args.model, args.descriptor, args.sample, manifest)
        report = inspect_model(args.model, args.descriptor, sample=args.sample,
                               expected_sha256=frozen_hash if args.require_frozen else args.expected_sha256,
                               baseline_metrics=ROOT / 'docs/results/metrics.json',
                               frozen_sha256=frozen_hash,
                               expected_descriptor_sha256=trusted.get('model.json'),
                               expected_sample_sha256=trusted.get('demo.csv'))
        report['trusted_handoff'] = handoff
    except Exception as error:
        report['checks'].append({'name': 'trusted_handoff', 'passed': False,
                                 'error': f'{type(error).__name__}: {error}'})
    report['require_frozen'] = args.require_frozen
    report['exit_code'] = exit_code(report, args.require_frozen)
    report['status'] = 'completed' if report['exit_code'] == 0 else 'failed'
    if output is not None:
        atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
