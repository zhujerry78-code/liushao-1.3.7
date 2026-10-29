"""Data-only auditing must precede prediction and cannot certify model performance."""
from argparse import Namespace
import copy

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from liushao import model
from liushao.data import sha256
from scripts import engineering_external as external
from scripts import engineering_external_preflight as preflight
from test_engineering_external import ATTACKS, inputs, load  # noqa: F401


@pytest.fixture
def audit_inputs(inputs, monkeypatch, tmp_path):
    args, _ = inputs
    monkeypatch.setattr(preflight, 'OUTPUT_ROOT', tmp_path / 'audits')
    return Namespace(**{key: value for key, value in vars(args).items() if key not in
                        ('model', 'descriptor', 'model_sha256', 'write_predictions')})


def change_source(args, content):
    args.source.write_text(content, encoding='utf-8')
    args.source_sha256 = sha256(args.source)
    args.source_bytes = args.source.stat().st_size


def forbid_model_use(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Data audit must not deserialize, fit or predict with any model')
    for target, name in ((joblib, 'load'), (model, 'fit_bundle'), (model, 'infer'),
                         (external, 'infer'), (Pipeline, 'fit'), (Pipeline, 'predict_proba')):
        monkeypatch.setattr(target, name, forbidden)


def test_data_only_audit_covers_quality_rejected_ids_and_split_overlaps(audit_inputs, monkeypatch):
    args = audit_inputs
    change_source(args, 'a,b,Label\n'
                  f'0,1,Benign\n2.0000001,1,{ATTACKS[0]}\n8,1,{ATTACKS[1]}\n'
                  f'bad,,{ATTACKS[0]}\nInfinity,1e40,{ATTACKS[1]}\nNaN,1,Benign\n'
                  f'8,1,{ATTACKS[1]}\n')
    forbid_model_use(monkeypatch)
    before = {path: sha256(path) for path in (args.source,
              args.reference_data / 'features.parquet', args.reference_data / 'manifest.parquet')}
    out = preflight.run(args)
    audit, manifest = load(out / 'audit.json'), load(out / 'manifest.json')
    assert manifest['status'] == 'completed'
    assert manifest['model_loaded'] is manifest['inference_performed'] is False
    assert audit['inference_performed'] is False
    assert audit['quality'] == {
        'raw_rows': 7, 'valid_rows': 5, 'unjudgeable_rows': 2, 'missing_cells': 1,
        'rows_with_missing_features': 1, 'record_id_overlap': 0, 'invalid_timestamp_rows': 7,
    }
    assert audit['numeric_quality_by_feature']['a'] == {
        'empty_cells': 0, 'nonnumeric_cells': 2, 'infinite_cells': 1,
        'float32_overflow_cells': 0, 'cleaned_missing_cells': 3,
    }
    assert audit['numeric_quality_by_feature']['b']['empty_cells'] == 1
    assert audit['numeric_quality_by_feature']['b']['float32_overflow_cells'] == 1
    assert audit['reference_overlap_rows']['exact']['train'] == 1
    assert audit['reference_overlap_rows']['near']['train'] == 2
    assert audit['within_source_duplicate_extra_rows'] == {'exact': 1, 'near': 1}
    assert sum(audit['strata_label_support']['all'].values()) == 5
    assert sum(audit['strata_label_support']['any_overlap'].values()) == 2
    assert sum(audit['strata_label_support']['no_overlap'].values()) == 3
    rejected = pd.read_csv(out / 'rejected.csv')
    assert rejected.source_row.tolist() == [5, 6]
    assert rejected.record_id.tolist() == [f'{args.source_sha256}:{row}' for row in (5, 6)]
    assert rejected.reason.tolist() == ['全部必需特征无效，无法判断'] * 2
    assert rejected.ground_truth_evaluation_only.tolist() == ATTACKS
    assert not (out / 'metrics.json').exists()
    assert not (out / 'predictions.csv').exists()
    assert before == {path: sha256(path) for path in before}
    with pytest.raises(FileExistsError):
        preflight.run(args)


def test_batch_size_does_not_change_audit_population_or_counts(audit_inputs, monkeypatch):
    forbid_model_use(monkeypatch)
    audits = []
    for chunk_rows in (1, 2, 100):
        args = copy.copy(audit_inputs)
        args.chunk_rows = chunk_rows
        args.run_id = f'batch-{chunk_rows}'
        audits.append(load(preflight.run(args) / 'audit.json'))
    assert audits[0] == audits[1] == audits[2]


@pytest.mark.parametrize('fault', ['source_hash', 'source_bytes', 'source_url', 'reference_hash'])
def test_unverified_inputs_never_publish_audit(audit_inputs, monkeypatch, fault):
    args = audit_inputs
    forbid_model_use(monkeypatch)
    if fault == 'source_hash':
        args.source_sha256 = '0' * 64
    elif fault == 'source_bytes':
        args.source_bytes += 1
    elif fault == 'source_url':
        args.source_url = 'https://example.com/' + args.source.name
    else:
        with (args.reference_data / 'features.parquet').open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        preflight.run(args)
    out = preflight.OUTPUT_ROOT / args.run_id
    assert load(out / 'manifest.json')['status'] == 'failed'
    assert not (out / 'audit.json').exists()


def test_later_audit_failure_keeps_only_failed_manifest_and_partial_diagnostics(
    audit_inputs, monkeypatch,
):
    args = audit_inputs
    change_source(args, f'a,b,Label\nbad,bad,{ATTACKS[0]}\n1,1,Benign\n2,1,Benign\n')
    args.chunk_rows = 1
    original = preflight.clean
    calls = []
    def fail_later(*values, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('later audit chunk failure')
        return original(*values, **kwargs)
    monkeypatch.setattr(preflight, 'clean', fail_later)
    forbid_model_use(monkeypatch)
    with pytest.raises(ValueError, match='later audit'):
        preflight.run(args)
    out = preflight.OUTPUT_ROOT / args.run_id
    assert load(out / 'manifest.json')['status'] == 'failed'
    assert not (out / 'audit.json').exists()
    assert pd.read_csv(out / 'rejected.csv').source_row.tolist() == [2]


def test_all_rejected_batch_and_file_are_auditable_without_prediction(audit_inputs, monkeypatch):
    args = audit_inputs
    change_source(args, f'a,b,Label\nbad,bad,{ATTACKS[0]}\n,,Benign\n')
    args.chunk_rows = 1
    forbid_model_use(monkeypatch)
    audit = load(preflight.run(args) / 'audit.json')
    assert audit['quality']['valid_rows'] == 0
    assert audit['quality']['unjudgeable_rows'] == 2
    assert all(sum(counts.values()) == 0 for counts in audit['strata_label_support'].values())


def test_whole_audit_is_saved_before_first_prediction(inputs, monkeypatch):
    args, _ = inputs
    original = external.infer
    calls = []
    def checked_infer(*values, **kwargs):
        out = external.OUTPUT_ROOT / args.run_id
        audit = load(out / 'audit.json')
        manifest = load(out / 'manifest.json')
        assert audit['quality']['raw_rows'] == 5
        assert audit['status'] == 'completed'
        assert manifest['data_audit_completed_before_inference'] is True
        assert manifest['data_audit_sha256'] == sha256(out / 'audit.json')
        calls.append(1)
        return original(*values, **kwargs)
    monkeypatch.setattr(external, 'infer', checked_infer)
    out = external.run(args)
    assert calls
    assert load(out / 'manifest.json')['status'] == 'completed'


def test_audit_and_evaluation_population_mismatch_prevents_success_metrics(inputs, monkeypatch):
    args, _ = inputs
    original = external.evaluate
    def inconsistent(*values, **kwargs):
        result = original(*values, **kwargs)
        result['quality']['valid_rows'] += 1
        return result
    monkeypatch.setattr(external, 'evaluate', inconsistent)
    with pytest.raises(ValueError, match='预测前数据审计不一致'):
        external.run(args)
    out = external.OUTPUT_ROOT / args.run_id
    assert load(out / 'manifest.json')['status'] == 'failed'
    assert not (out / 'metrics.json').exists()


def test_cli_requires_no_model_or_descriptor(audit_inputs):
    args = audit_inputs
    parsed = preflight.parse_args([
        '--reference-data', str(args.reference_data), '--source', str(args.source),
        '--source-url', args.source_url, '--source-sha256', args.source_sha256,
        '--source-bytes', str(args.source_bytes), '--attack-label', ATTACKS[0],
        '--attack-label', ATTACKS[1], '--run-id', 'data-only',
    ])
    assert not {'model', 'descriptor', 'model_sha256'}.intersection(vars(parsed))
    assert parsed.attack_label == ATTACKS
