"""Synthetic fixtures validate evaluator integrity; these are not attack experiments."""
from argparse import Namespace
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from liushao.data import fingerprints, sha256, version_id, write_json
from liushao.model import classification_metrics, fit_bundle, infer
from scripts import engineering_external as external


ATTACKS = ['DoS attacks-GoldenEye', 'DoS attacks-Slowloris']


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """Small, explicitly synthetic artifacts with a fully linked provenance chain."""
    reference = tmp_path / 'reference'
    reference.mkdir()
    x = pd.DataFrame({'a': [0., 2., 3., 4., 5., 6., 7., 9.], 'b': [1.] * 8})
    manifest = pd.DataFrame({'record_id': [f'original:{n}' for n in range(8)],
                             'label': ['Benign', 'Benign', 'Bot', 'Bot'] * 2,
                             'split': ['train'] * 4 + ['validation'] * 2 + ['test', 'purged'],
                             'exact_fingerprint': fingerprints(x),
                             'near_fingerprint': fingerprints(x, near=True)})
    x.to_parquet(reference / 'features.parquet', index=False)
    manifest.to_parquet(reference / 'manifest.parquet', index=False)
    config = {'seed': 42, 'sample_modulus': 10, 'block_minutes': 15, 'files': []}
    sources = [{'sha256': '1' * 64, 'name': 'original.csv'}]
    manifest_hash = sha256(reference / 'manifest.parquet')
    data_version = version_id(config, sources, manifest_hash)
    summary = {'config': config, 'sources': sources, 'features': list(x.columns),
               'data_version': data_version, 'split_counts': manifest.split.value_counts().to_dict(),
               'features.parquet_sha256': sha256(reference / 'features.parquet'),
               'manifest.parquet_sha256': manifest_hash}
    write_json(reference / 'summary.json', summary)
    baseline = tmp_path / 'baseline.json'
    write_json(baseline, summary)
    monkeypatch.setattr(external, 'BASELINE_SUMMARY', baseline)
    monkeypatch.setattr(external, 'OUTPUT_ROOT', tmp_path / 'runs')
    bundle = fit_bundle(x.iloc[:4], [0, 0, 1, 1], x.iloc[4:6], [0, 1],
                        data_version, n_estimators=8)
    baseline_metrics = tmp_path / 'baseline_metrics.json'
    write_json(baseline_metrics, {'model_version': bundle['model_version'],
                                  **{model: {'threshold': bundle['config'][model + '_threshold']}
                                     for model in ('rf', 'lr')}})
    monkeypatch.setattr(external, 'BASELINE_METRICS', baseline_metrics)
    model = tmp_path / 'model.joblib'
    joblib.dump(bundle, model)
    descriptor = tmp_path / 'model.json'
    write_json(descriptor, {**{k: v for k, v in bundle.items() if k not in ('rf', 'lr')},
                            'model_sha256': sha256(model),
                            'source_code_sha256': {
                                name: sha256(external.ROOT / name) for name in
                                ('liushao/__init__.py', 'liushao/model.py', 'liushao/data.py')
                            }})
    source = tmp_path / 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'
    pd.DataFrame({'a': ['0', '2.0000001', '8', 'invalid', '8'],
                  'b': ['1', '1', '1', 'invalid', '1'],
                  'Label': ['Benign', ATTACKS[0], ATTACKS[1], ATTACKS[0], ATTACKS[1]]
                 }).to_csv(source, index=False)
    args = Namespace(model=model, descriptor=descriptor, reference_data=reference,
                     source=source, model_sha256=sha256(model), source_sha256=sha256(source),
                     source_bytes=source.stat().st_size,
                     source_url='https://' + external.OFFICIAL_HOST
                                + '/Processed%20Traffic%20Data%20for%20ML%20Algorithms/'
                                + source.name,
                     attack_label=ATTACKS.copy(), run_id='synthetic-test', chunk_rows=2,
                     write_predictions=True)
    return args, bundle


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_streamed_metrics_overlap_and_ids_match_batch(inputs):
    args, bundle = inputs
    before = {p: sha256(p) for p in (args.model, args.source,
              args.reference_data / 'features.parquet', args.reference_data / 'manifest.parquet')}
    out = external.run(args)
    result = load(out / 'metrics.json')
    assert load(out / 'manifest.json')['status'] == 'completed'
    assert load(out / 'manifest.json')['source_timestamp_dates'] == {'<missing_column>': 5}
    assert result['quality']['raw_rows'] == 5
    assert result['quality']['valid_rows'] == 4
    assert result['quality']['unjudgeable_rows'] == 1
    assert result['reference_overlap_rows']['exact']['train'] == 1
    assert result['reference_overlap_rows']['near']['train'] == 2
    assert result['strata']['any_overlap']['rf']['records'] == 2
    assert result['strata']['no_overlap']['rf']['records'] == 2
    assert result['within_source_duplicate_extra_rows'] == {'exact': 1, 'near': 1}
    pred = pd.read_csv(out / 'predictions.csv')
    assert pred.source_row.tolist() == [2, 3, 4, 6]
    assert pred.record_id.tolist() == [f'{args.source_sha256}:{r}' for r in [2, 3, 4, 6]]
    raw = pd.read_csv(args.source, dtype=str, keep_default_na=False)
    batch, _, _ = infer(raw, bundle, args.source_sha256, args.source.name)
    y = np.array([False, True, True, True])
    for model in ('rf', 'lr'):
        expected = classification_metrics(y, batch[model + '_score'],
                                          bundle['config'][model + '_threshold'])
        for key in ('precision', 'recall', 'f1', 'fpr', 'tp', 'fp', 'tn', 'fn'):
            assert result['strata']['all'][model][key] == expected[key]
    assert before == {p: sha256(p) for p in before}
    with pytest.raises(FileExistsError):
        external.run(args)


@pytest.mark.parametrize('fault', ['source_hash', 'model_hash', 'missing_manifest',
                                  'reference_features', 'descriptor_config', 'baseline_model_version',
                                  'baseline_rf_threshold', 'baseline_lr_threshold'])
def test_provenance_failure_never_emits_success_metrics(inputs, fault):
    args, _ = inputs
    if fault == 'source_hash':
        args.source_sha256 = '0' * 64
    elif fault == 'model_hash':
        args.model_sha256 = '0' * 64
    elif fault == 'missing_manifest':
        (args.reference_data / 'manifest.parquet').unlink()
    elif fault == 'reference_features':
        with (args.reference_data / 'features.parquet').open('ab') as stream:
            stream.write(b'changed')
    elif fault == 'descriptor_config':
        descriptor = load(args.descriptor)
        descriptor['config']['rf_threshold'] = .123
        write_json(args.descriptor, descriptor)
    else:
        baseline = load(external.BASELINE_METRICS)
        if fault == 'baseline_model_version':
            baseline['model_version'] = '0' * 16
        else:
            model = fault.split('_')[1]
            baseline[model]['threshold'] += .01
        write_json(external.BASELINE_METRICS, baseline)
    with pytest.raises((ValueError, FileNotFoundError)):
        external.run(args)
    out = external.OUTPUT_ROOT / args.run_id
    assert load(out / 'manifest.json')['status'] == 'failed'
    assert not (out / 'metrics.json').exists()
    assert not (out / 'predictions.csv').exists()


@pytest.mark.parametrize('content', [
    'a,b,Label\n1,1,Benign\n2,1,Unknown\n',
    'a,b,Label\n1,1,Benign\n2,1,\n',
    'a,b,Label\n1,1,Benign\n2,1\n',
    'a,a,Label\n1,1,Benign\n',
    'b,Label\n1,Benign\n',
])
def test_all_labels_and_layout_are_checked_before_inference(inputs, monkeypatch, content):
    args, _ = inputs
    args.source.write_text(content, encoding='utf-8')
    args.source_sha256 = sha256(args.source)
    args.source_bytes = args.source.stat().st_size
    def never_infer(*_args, **_kwargs):
        pytest.fail('Malformed or unsupported evaluation input must fail before inference')
    monkeypatch.setattr(external, 'infer', never_infer)
    with pytest.raises(ValueError):
        external.run(args)
    out = external.OUTPUT_ROOT / args.run_id
    assert not (out / 'metrics.json').exists()
    assert not (out / 'predictions.csv').exists()


def test_undefined_rates_are_named_and_no_output_can_escape(inputs):
    args, _ = inputs
    result = external.finish_counts(external.confusion_counts([], []))
    assert result['fpr'] is None
    assert result['precision'] == result['recall'] == result['f1'] == 0
    assert set(result['undefined']) == {'precision', 'recall', 'f1', 'fpr'}
    args.run_id = '../../escape'
    with pytest.raises(ValueError, match='run-id'):
        external.run(args)
    assert not external.OUTPUT_ROOT.exists()


def test_source_verification_accepts_only_git_line_endings(tmp_path):
    source = tmp_path / 'source.py'
    expected = hashlib.sha256(b'x = 1\n').hexdigest()
    source.write_bytes(b'x = 1\r\n')
    result = external.verify_source_module(source, expected)
    assert result['match'] == 'crlf_to_lf'
    assert result['working_tree_sha256'] != result['expected_sha256']
    source.write_bytes(b'x = 2\r\n')
    with pytest.raises(ValueError, match='源码内容不匹配'):
        external.verify_source_module(source, expected)
