"""Regression cases for evidence summaries; all records here are synthetic."""
import copy

import pytest

from app_evidence_rules import classify_attribution_archetype, reference_position
from app_review_guidance import build_audit_note_draft, review_check_contributions
from test_r12_interval_attribution import _sample_card, _sample_payload


def _many_cards(count):
    return [_sample_card(f'Synthetic Field {index}', 50.) for index in range(count)]


def test_sixth_positive_driver_outside_reference_prevents_all_in_range_claim():
    cards = _many_cards(6)
    cards[-1] = _sample_card('Flow Duration', 250.)
    result = classify_attribution_archetype(_sample_payload(cards, [.12, .11, .10, .09, .08, .07]))
    assert result['code'] == 'outlier_feature_driven'
    assert '均在' not in result['headline']


def test_sixth_driver_with_unknown_reference_prevents_all_in_range_claim():
    cards = _many_cards(6)
    cards[-1] = _sample_card('Flow Duration', 50., p01=None, p99=None)
    result = classify_attribution_archetype(_sample_payload(cards, [.12, .11, .10, .09, .08, .07]))
    assert result['code'] == 'unverified_multi_driver_push'
    assert '其中 1 项' in result['headline']


def test_small_omitted_contributions_do_not_satisfy_selected_driver_sum():
    result = classify_attribution_archetype(_sample_payload(_many_cards(8), [.03, .03] + [.009] * 6))
    assert result['positive_sum'] > .10
    assert result['positive_driver_sum'] == pytest.approx(.06)
    assert result['code'] == 'unverified_multi_driver_push'


def test_missing_score_is_unknown_not_a_normal_or_disagreement_claim():
    payload = _sample_payload([_sample_card()])
    del payload['scores']['rf']
    result = classify_attribution_archetype(payload)
    assert result['code'] == 'score_unavailable'
    assert '两模型均未提示异常' not in result['headline']
    assert '0.0000' not in result['headline']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 'not a number', None])
def test_invalid_observation_is_never_presented_as_in_range(value):
    position = reference_position('Flow Duration', value, dict(q01=0, median=50, q99=100))
    assert position['status'] == '原始缺失'
    assert position['severity_band'] == '原始缺失'
    assert not position['near'] and not position['outside']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None])
def test_invalid_contribution_makes_classification_unavailable(bad):
    payload = _sample_payload([_sample_card()])
    payload['decomposition']['rf']['fields'][0]['contribution'] = bad
    assert classify_attribution_archetype(payload)['code'] == 'explanation_unavailable'


def test_empty_evidence_draft_never_claims_completed_review_or_no_deviation():
    text = build_audit_note_draft({})
    assert text.endswith('【复核结论】\n')
    assert '尚未人工核查' not in text
    assert '评分或分界值不完整' in text
    assert '未获得可展示的核查明细' in text
    assert '模型评分已核对' not in text
    assert '无显著单字段偏离' not in text


def test_duplicate_display_names_keep_their_own_contribution_in_draft():
    first = _sample_card('Flow Duration', 50.)
    second = _sample_card('Tot Fwd Pkts', 60.)
    first['name'] = second['name'] = '同名字段'
    payload = _sample_payload([first, second], [.24, .12])
    before = copy.deepcopy(payload)
    text = build_audit_note_draft(payload)
    assert text.count('RF路径推高 +0.2400') == 1
    assert text.count('RF路径推高 +0.1200') == 1
    assert payload == before


def test_interval_distance_is_not_labelled_as_attack_risk():
    card = _sample_card('Flow Duration', 250.)
    payload = _sample_payload([card])
    assert '显著偏离参考' in card['reference_position']['severity_band']
    assert '高风险' not in build_audit_note_draft(payload)
    assert '高风险' not in review_check_contributions(payload)[0]['区间分级']
