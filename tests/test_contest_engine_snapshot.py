"""Real frozen parameters plus synthetic branch cases; no human-study claim."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from app_engine_evidence import (build_engine_evidence, ecod_contributions, engine_identity,
                                 historical_engine_label, require_complete_engine_evidence)
from app_evidence import build_record_evidence
from app_service import load_trusted_engine, with_engine
from app_ui_state import export_snapshot
from liushao import reviews
from liushao.model import infer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def frozen_engine_case():
    from app_model import verified_model, load_model_bundle
    artifacts = ROOT / 'artifacts'
    if not (artifacts / 'engine.joblib').exists():
        pytest.skip('Source-only distribution excludes frozen binary assets')
    verified = verified_model(artifacts / 'model.joblib', artifacts / 'model.json')
    bundle = load_model_bundle(verified.content, verified.descriptor)
    raw = pd.read_csv(artifacts / 'demo.csv').head(20)
    pred, features, _ = infer(raw, bundle, 'a'*64, 'engineering-frozen-demo.csv')
    pred = with_engine(pred, features, bundle, artifacts)
    engine = load_trusted_engine(artifacts, bundle)
    assert engine is not None and 'engine_sha256' in pred
    return bundle, pred, features, engine, verified


def payload_for(case):
    bundle, pred, features, engine, verified = case
    record = pred.iloc[0]
    payload = build_record_evidence(record, features, bundle, meta={
        'input_sha256':'a'*64, 'model_sha256':verified.model_sha256,
        'descriptor_sha256':verified.descriptor_sha256, 'input_csv_row':2})
    payload['engine'] = build_engine_evidence(record, features.iloc[[0]], bundle, engine)
    return payload


def test_full_engine_snapshot_saved_reopened_and_exported_without_recalculation(tmp_path, frozen_engine_case, monkeypatch):
    bundle, pred, features, engine, verified = frozen_engine_case
    payload = payload_for(frozen_engine_case)
    expected = copy.deepcopy(payload)
    db = tmp_path / 'reviews.sqlite'
    record = pred.iloc[0]
    reviews.save_review(db, str(record.record_id), bundle['model_version'], '待补证',
                        '自动化工程保存，非真人试用', evidence_snapshot=payload)
    payload['engine']['results']['cp_set'] = 'mutated-after-save'
    saved = reviews.history(db)[0]
    assert saved['evidence_snapshot'] == expected
    assert reviews.evidence_for_review(db, saved['id'])['evidence_snapshot'] == expected
    import liushao_engine
    assert set(liushao_engine.COLUMNS) == set(expected['engine']['results'])
    block = expected['engine']
    assert len(block['ecod_decomposition']['fields']) == len(features.columns)
    assert len(block['source_masses']) == 3
    assert block['ecod_decomposition']['contribution_sum'] == pytest.approx(record.ecod_score, abs=1e-10)
    monkeypatch.setattr(liushao_engine, 'analyze', lambda *args, **kwargs: pytest.fail('history export recomputed engine'))
    exported = export_snapshot({'name':'x','digest':'a'*64,'key':'key'}, pred, [saved],
                               {'model_version':bundle['model_version']}, True)
    assert json.loads(exported.history)[0]['evidence_snapshot'] == expected


@pytest.mark.parametrize('column', ['engine_sha256','engine_descriptor_sha256','engine_version',
                                  'cp_set','cp_p_attack','ds_conflict','va_high','engine_review'])
def test_stale_identity_or_modified_engine_result_is_rejected(frozen_engine_case, column):
    bundle, pred, features, engine, _ = frozen_engine_case
    record = pred.iloc[0].copy()
    if column in ('engine_sha256','engine_descriptor_sha256'): record[column] = 'f'*64
    elif column in ('engine_version','cp_set'): record[column] = 'wrong'
    elif column == 'engine_review': record[column] = not bool(record[column])
    else: record[column] = float(record[column]) + .03
    with pytest.raises(ValueError):
        build_engine_evidence(record, features.iloc[[0]], bundle, engine)


def test_descriptor_change_changes_cache_identity_without_changing_parameters(tmp_path, frozen_engine_case):
    bundle, pred, features, engine, _ = frozen_engine_case
    for name in ('engine.joblib','engine.json'):
        shutil.copyfile(ROOT / 'artifacts' / name, tmp_path / name)
    descriptor = json.loads((tmp_path / 'engine.json').read_bytes())
    descriptor['review_note'] = 'synthetic descriptor identity regression'
    (tmp_path / 'engine.json').write_text(json.dumps(descriptor), encoding='utf-8')
    changed = load_trusted_engine(tmp_path, bundle)
    assert changed is not engine
    assert changed['_artifact_identity']['engine_sha256'] == engine['_artifact_identity']['engine_sha256']
    assert changed['_artifact_identity']['engine_descriptor_sha256'] != engine['_artifact_identity']['engine_descriptor_sha256']
    with pytest.raises(ValueError, match='身份不一致'):
        build_engine_evidence(pred.iloc[0], features.iloc[[0]], bundle, changed)


def test_missing_and_corrupt_engine_do_not_fabricate_evidence(tmp_path, frozen_engine_case):
    bundle, pred, features, engine, _ = frozen_engine_case
    assert load_trusted_engine(tmp_path, bundle) is None
    shutil.copyfile(ROOT / 'artifacts' / 'engine.json', tmp_path / 'engine.json')
    (tmp_path / 'engine.joblib').write_bytes(b'corrupt synthetic bytes')
    assert load_trusted_engine(tmp_path, bundle) is None
    with pytest.raises(ValueError, match='无法校验'):
        build_engine_evidence(pred.iloc[0], features.iloc[[0]], bundle, None)
    with pytest.raises(ValueError, match='engine absent'):
        require_complete_engine_evidence({'engine':{'status':'unavailable','reason':'engine absent'}})


def test_historical_identity_is_distinct_and_legacy_does_not_get_reconstructed(frozen_engine_case):
    payload = payload_for(frozen_engine_case)
    assert '与当前分析引擎不同' in historical_engine_label(payload, ('f'*64,'e'*64))
    assert '与当前分析引擎不同' not in historical_engine_label(payload, engine_identity(frozen_engine_case[1].iloc[0]))
    assert '旧快照未包含' in historical_engine_label({'schema_version':2})
    base = build_engine_evidence(pd.Series({'rf_score':.5}), None, None, None)
    assert base['status'] == 'not_used'
    assert '仅使用 RF/LR' in historical_engine_label({'engine':base})


@pytest.mark.parametrize('row,sign', [([-10,-10],[-1,-1]),([10,10],[1,1]),([-10,10],[-1,1])])
def test_ecod_additive_breakdown_uses_winning_aggregate_not_per_field_maximum(row, sign):
    from liushao_engine.ecod import score_ecod
    ref = np.column_stack([np.arange(10.), np.arange(10.)])
    model = {'sorted':ref,'median':np.array([4.5,4.5]),'skew_sign':np.array(sign)}
    frame = pd.DataFrame([row],columns=['a','b'])
    result = ecod_contributions(frame, {'ecod':model})
    assert result['contribution_sum'] == pytest.approx(score_ecod(model, frame.to_numpy())[0], abs=1e-12)
    assert abs(result['residual']) < 1e-12
    assert len(result['fields']) == 2


def test_ecod_opposite_tails_cannot_be_added_as_independent_maxima():
    from liushao_engine.ecod import score_ecod
    model = {'sorted':np.column_stack([np.arange(10.),np.arange(10.)]),
             'median':np.array([4.5,4.5]),'skew_sign':np.array([1,-1])}
    frame = pd.DataFrame([[-10,20]], columns=['a','b'])
    result = ecod_contributions(frame, {'ecod':model})
    assert result['contribution_sum'] == pytest.approx(np.log(11))
    assert result['contribution_sum'] == pytest.approx(score_ecod(model, frame.to_numpy())[0])
    assert result['contribution_sum'] < 2*np.log(11)
