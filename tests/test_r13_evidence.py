"""Evidence mechanics on synthetic models; these are not human/model efficacy tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app_evidence import build_record_evidence, feature_definition, neighbor_summary


FIELDS = ['Fwd Pkt Len Max', 'PSH Flag Cnt', 'Flow Duration', 'Flow Pkts/s',
          'TotLen Fwd Pkts', 'Pkt Len Var']


class LinearFixture:
    """Deliberately transparent scoring double with independent imputer values."""
    def __init__(self, weights, intercept=.5):
        self.weights = np.asarray(weights)
        self.intercept = intercept
        self.named_steps = {'impute': SimpleNamespace(statistics_=np.full(6, .25))}

    def predict_proba(self, frame):
        values = frame.to_numpy(dtype=float)
        values = np.where(np.isnan(values), self.named_steps['impute'].statistics_, values)
        score = np.clip(self.intercept + values @ self.weights, 0., 1.)
        return np.column_stack([1-score, score])


@pytest.fixture
def local_case():
    features = pd.DataFrame([[1.] * 6, [2.] * 6, [1.] * 6, [0.] * 6], columns=FIELDS)
    bundle = {'rf': LinearFixture([.25, -.2, .1, 0, 0, 0]),
              'lr': LinearFixture([.25, -.2, .1, 0, 0, 0]),
              'config': {'features': FIELDS, 'rf_threshold': .5, 'lr_threshold': .5, 'near_margin': .1},
              'model_version': 'synthetic-engineering-only',
              'reference': {name: {'q01': -1., 'median': 0., 'q99': 1.} for name in FIELDS}}
    record = pd.Series({'record_id': 'record-1', 'source_sha256': 'a' * 64,
                        'source_file': 'synthetic.csv', 'source_row': 2,
                        'model_version': bundle['model_version'], 'rf_score': .65, 'lr_score': .65}, name=0)
    return record, features, bundle


def test_independent_signed_effects_joint_replacement_and_input_preservation(local_case):
    record, features, bundle = local_case
    original = features.copy(deep=True)
    before_record = record.copy(deep=True)
    snapshot = build_record_evidence(record, features, bundle, meta={
        'input_sha256': 'b' * 64, 'model_sha256': 'c' * 64, 'descriptor_sha256': 'd' * 64,
        'input_csv_row': 2, 'input_name': 'test.csv'})
    cards = {item['field']: item for item in snapshot['cards']}
    assert cards['Fwd Pkt Len Max']['rf']['delta'] == pytest.approx(-.25)
    assert cards['PSH Flag Cnt']['rf']['delta'] == pytest.approx(.2)
    assert snapshot['joint_test']['fields'] == ['Fwd Pkt Len Max', 'Flow Duration']
    assert snapshot['joint_test']['rf']['after'] == pytest.approx(.3)
    assert not snapshot['assessment']['sufficient']  # Practical conflict effects are uncalibrated.
    assert cards['Fwd Pkt Len Max']['role']=='支持当前模型提示'
    assert cards['PSH Flag Cnt']['role']=='反向线索'
    assert all(item['rf']['before'] == pytest.approx(.65) for item in snapshot['cards'])
    assert snapshot['input_sha256'] == 'b' * 64
    assert type(snapshot['input_csv_row']) is int and snapshot['input_csv_row'] == 2
    assert type(snapshot['identity']['source_row']) is int
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    pd.testing.assert_frame_equal(original, features)
    pd.testing.assert_series_equal(before_record, record)


def test_normal_prediction_prioritizes_suppressing_features(local_case):
    record, features, bundle = local_case
    bundle['rf'] = LinearFixture([.1, -.25, -.2, 0, 0, 0])
    bundle['lr'] = LinearFixture([.1, -.25, -.2, 0, 0, 0])
    record[['rf_score', 'lr_score']] = .15
    payload = build_record_evidence(record, features, bundle)
    cards={c['field']:c for c in payload['cards']}
    assert cards['PSH Flag Cnt']['rf']['delta']==pytest.approx(.25)
    assert cards['PSH Flag Cnt']['role']=='支持当前模型提示'
    assert cards['Fwd Pkt Len Max']['role']=='反向线索'
    assert payload['joint_test']['rf']['after'] == pytest.approx(.6)


def test_opposed_models_trigger_inadequate_explanation(local_case):
    record, features, bundle = local_case
    bundle['lr'] = LinearFixture([-.25, .2, -.1, 0, 0, 0])
    record['lr_score'] = .35
    payload = build_record_evidence(record, features, bundle)
    assert not payload['assessment']['sufficient']
    assert 'Fwd Pkt Len Max' in payload['stability']['contradictory_fields']
    assert payload['selection']['direction_differences']
    assert payload['selection']['parameters']['practical_conflict']['status']=='uncalibrated'
    assert '待补证' in payload['assessment']['status']


def test_flat_model_and_missing_references_never_claim_explanation(local_case):
    record, features, bundle = local_case
    bundle['rf'] = LinearFixture([0.] * 6)
    bundle['lr'] = LinearFixture([0.] * 6)
    record[['rf_score', 'lr_score']] = .5
    for reference in bundle['reference'].values():
        reference['median'] = None
    payload = build_record_evidence(record, features, bundle)
    assert payload['tested_fields'] == 0
    assert not payload['assessment']['sufficient']
    assert payload['joint_test']['count'] == 0
    assert all(card['rf']['after'] is None for card in payload['cards'])
    json.dumps(payload, allow_nan=False)


def test_saturated_lr_small_opposite_effect_still_blocks_single_explanation(local_case):
    record, features, bundle = local_case
    bundle['lr'] = LinearFixture([-.00025, .0002, -.0001, 0, 0, 0], intercept=.999)
    record['lr_score'] = .99885
    payload = build_record_evidence(record, features, bundle)
    assert not payload['assessment']['sufficient']
    assert payload['stability']['joint_direction_conflict']
    assert payload['stability']['contradictory_fields']


def test_missing_fill_is_not_confused_with_normal_reference_median(local_case):
    record, features, bundle = local_case
    features.loc[0, FIELDS[0]] = np.nan
    record[['rf_score', 'lr_score']] = .4625
    payload = build_record_evidence(record, features, bundle)
    card = next(item for item in payload['cards'] if item['field'] == FIELDS[0])
    assert card['missing'] and card['value'] is None
    assert card['imputed_rf'] == .25 and card['reference']['p50'] == 0
    assert card['batch_percentile'] is None
    assert card['batch_observed_count'] == 3
    assert not payload['assessment']['sufficient']
    json.dumps(payload, allow_nan=False)


def test_percentile_uses_full_batch_midrank(local_case):
    record, features, bundle = local_case
    payload = build_record_evidence(record, features, bundle)
    # One smaller observation, two equal observations; (1 + 2/2) / 4.
    assert payload['cards'][0]['batch_percentile'] == 50.


@pytest.mark.parametrize('failure', ['score', 'version', 'index', 'feature_order'])
def test_stale_or_misaligned_record_is_rejected(local_case, failure):
    record, features, bundle = local_case
    if failure == 'score': record['rf_score'] = .99
    if failure == 'version': record['model_version'] = 'different'
    if failure == 'index': record.name = 99
    if failure == 'feature_order': features = features.iloc[:, ::-1]
    with pytest.raises(ValueError):
        build_record_evidence(record, features, bundle)


def test_unstable_alternate_reference_is_reported(local_case):
    record, features, bundle = local_case
    # The alternate reference crosses the current observation: ranking/direction
    # no longer survives, although the P50-only joint check was strong.
    for reference in bundle['reference'].values():
        reference['q99'] = 30.
    payload = build_record_evidence(record, features, bundle)
    assert payload['joint_test']['rf']['delta'] == pytest.approx(-.35)
    assert not payload['stability']['passed']
    assert payload['stability']['direction_reversals'][1]
    assert not payload['assessment']['sufficient']


def test_neighbor_counts_exclude_self_and_keep_batch_scope():
    context = pd.DataFrame({
        'Src IP': ['10.0.0.1', '10.0.0.1', '10.0.0.1', '10.0.0.2', 'bad'],
        'Dst IP': ['10.0.0.8', '10.0.0.8', '10.0.0.9', '10.0.0.8', '10.0.0.8'],
        'Src Port': [100, 100, 100, 200, 100], 'Dst Port': [80] * 5, 'Protocol': [6] * 5,
        'Timestamp': ['23/10/2026 12:00:00', '23/10/2026 12:00:30',
                      '23/10/2026 12:02:00', '23/10/2026 11:59:00', 'invalid']})
    result = neighbor_summary(context, 0)
    assert result['same_source'] == 2 and result['same_destination'] == 3
    assert result['source_fanout'] == 2
    assert result['repeat_five_tuple'] == 1
    assert result['time_window']['all'] == 2
    assert result['time_window']['same_source'] == 1
    assert result['time_window']['same_destination'] == 2
    json.dumps(result, allow_nan=False)


def test_absent_ip_and_undeclared_numeric_time_stay_unavailable():
    result = neighbor_summary(pd.DataFrame({'Flow Duration': [3, 3], 'Timestamp': [123, 124]}), 0)
    assert not result['available']
    assert result['same_source'] is None and result['same_destination'] is None
    assert result['repeat_five_tuple'] is None and result['time_window'] is None


def test_mixed_timezone_rows_are_excluded_instead_of_guessed():
    result = neighbor_summary(pd.DataFrame({'Src IP': ['10.0.0.1'] * 3,
        'Timestamp': ['2026-10-23T12:00:00', '2026-10-23T12:00:01+00:00', '2026-10-23T12:00:30']}), 0)
    assert result['time_window']['all'] == 1
    assert result['time_window']['valid_timestamps'] == 2
    assert '未声明时区' in result['time_window']['timezone']


def test_units_are_specific_for_rates_variance_and_time():
    assert feature_definition('Flow Duration')[1] == '微秒 (μs)'
    assert feature_definition('Flow Pkts/s')[1] == '包/秒'
    assert feature_definition('Flow Byts/s')[1] == '字节/秒 (B/s)'
    assert feature_definition('Pkt Len Var')[1] == '字节² (B²)'
    assert feature_definition('Tot Fwd Pkts')[1] == '包'
    assert '未确认' in feature_definition('unknown feature')[1]


def test_frozen_pdf_problem_record_exposes_model_conflict():
    """Version-bound r11 packaged row, not a synthetic replacement of the model."""
    from app_model import load_model_bundle
    from liushao.data import read_csv_checked
    from liushao.model import infer
    import os
    artifacts = Path(os.environ.get('LIUSHAO_ARTIFACTS_DIR','artifacts'))
    if not (artifacts / 'model.joblib').is_file():
        pytest.skip('The supplied r11 frozen artifact is not present in this checkout.')
    bundle = load_model_bundle((artifacts / 'model.joblib').read_bytes(),
                               (artifacts / 'model.json').read_text(encoding='utf-8'))
    with (artifacts / 'demo.csv').open('rb') as stream:
        raw = read_csv_checked(stream)
    predictions, features, _ = infer(raw, bundle, 'engineering-demo', 'demo.csv')
    origins = pd.read_parquet(artifacts / 'demo_origins.parquet')
    index = origins.index[origins.source_row.eq(217809)][0]
    payload = build_record_evidence(predictions.loc[index], features, bundle, context=raw)
    assert payload['scores']['rf'] == pytest.approx(1.)
    assert len(payload['cards']) == 6
    assert all(not card['reference_position']['outside'] for card in payload['cards'])
    assert payload['joint_test']['rf']['delta'] < -.1
    assert payload['joint_test']['lr']['delta'] > 0
    assert payload['stability']['joint_direction_conflict']
    assert not payload['assessment']['sufficient']
    assert not payload['neighbors']['available']
    assert all('未确认' not in feature_definition(field)[1] for field in features)
    json.dumps(payload, allow_nan=False)
