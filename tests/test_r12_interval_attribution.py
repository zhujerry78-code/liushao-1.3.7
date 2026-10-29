"""Unit tests for the r12-v2 White-Box Interval & Attribution Upgrade."""
import copy
import numpy as np
import pandas as pd
import pytest

from app_evidence import feature_definition
from app_evidence_rules import (
    reference_position,
    summarize_interval_bands,
    classify_attribution_archetype,
)
from app_review_guidance import (
    review_check_items,
    review_check_contributions,
    unified_interval_contribution_rows,
    build_audit_note_draft,
)
from app_triage import batch_interval_profile


def _sample_card(field='Flow Duration', value=50.0, *, p01=0.0, p50=50.0, p99=100.0, rf_delta=-0.25):
    name, unit, meaning = feature_definition(field)
    pos = reference_position(field, value, {'q01': p01, 'median': p50, 'q99': p99})
    return {
        'field': field,
        'name': name,
        'unit': unit,
        'meaning': meaning,
        'value': value,
        'reference': {'p01': p01, 'p50': p50, 'p99': p99},
        'reference_status': pos['status'],
        'reference_position': pos,
        'batch_percentile': 50.0,
        'batch_observed_count': 1000,
        'missing': value is None,
        'imputed_rf': p50 if value is None else None,
        'imputed_lr': p50 if value is None else None,
        'role': '支持当前模型提示',
        'action': {'next': '核对同目的端口的短连接与包速率，查询防火墙/服务日志和压测工单。'},
        'selection_reasons': ['显著替换响应'],
        'direction_difference': False,
        'direction_label': '未达到方向差异展示条件',
        'rf': {'before': 0.90, 'after': 0.90 + (rf_delta or 0.0), 'delta': rf_delta, 'weak': False, 'direction': '原值相对参考值推高异常分数'},
        'lr': {'before': 0.88, 'after': 0.70, 'delta': -0.18, 'weak': False, 'direction': '原值相对参考值推高异常分数'},
    }


def _sample_payload(cards, contributions=None):
    if contributions is None:
        contributions = [0.24 if i == 0 else 0.12 for i in range(len(cards))]
    rf_fields = [
        {'field': c['field'], 'contribution': float(contrib), 'parts': [{'kind': 'value', 'contribution': float(contrib)}]}
        for c, contrib in zip(cards, contributions)
    ]
    lr_fields = [
        {'field': c['field'], 'term': 0.95 if i == 0 else 0.20, 'parts': [{'kind': 'value', 'term': 0.95 if i == 0 else 0.20}]}
        for i, c in enumerate(cards)
    ]
    rf_sum = sum(contributions)
    return {
        'cards': cards,
        'all_fields': copy.deepcopy(cards),
        'scores': {'rf': 0.90, 'lr': 0.88, 'rf_threshold': 0.12, 'lr_threshold': 0.84, 'rf_anomaly': True, 'lr_anomaly': True},
        'selection': {'eligible': len(cards), 'missing_fields': [], 'insufficient_references': [], 'direction_differences': [], 'counterevidence': []},
        'decomposition': {
            'status': 'available',
            'default_fields': [c['field'] for c in cards],
            'errors': {'rf_score': 1e-12, 'lr_logit': 1e-12, 'lr_score': 1e-12},
            'rf': {'base': 0.90 - rf_sum, 'sum': rf_sum, 'reconstructed': 0.90, 'score': 0.90, 'tree_count': 160, 'top_fields': [c['field'] for c in cards], 'fields': rf_fields},
            'lr': {'intercept': -1.0, 'logit': 1.99, 'reconstructed': 0.88, 'score': 0.88, 'fields': lr_fields},
        },
    }


def test_unknown_reference_is_never_counted_as_normal_core():
    p_unknown = reference_position('Flow Duration', 50.0, {'q01': None, 'median': 50.0, 'q99': None})
    assert not p_unknown['valid'] and '参考不足' in p_unknown['severity_band']

    cards = [
        _sample_card('Flow Duration', 250.0),
        _sample_card('Tot Fwd Pkts', 130.0),
        _sample_card('Tot Bwd Pkts', 98.0),
        _sample_card('Fwd Pkt Len Max', 50.0),
        _sample_card('Flow Byts/s', 20.0, p01=None, p50=10.0, p99=None),
        _sample_card('Protocol', 6.0, p01=6.0, p50=6.0, p99=17.0),
        _sample_card('Bwd Pkt Len Max', None),
    ]
    bands = summarize_interval_bands(cards)
    assert bands['high_risk'] == 1
    assert bands['outside_moderate'] == 1
    assert bands['near_boundary'] == 1
    assert bands['core_normal'] == 1
    assert bands['insufficient_reference'] == 1
    assert bands['discrete_or_fixed'] == 1
    assert bands['missing'] == 1
    assert (bands['high_risk'] + bands['outside_moderate'] + bands['near_boundary']
            + bands['core_normal'] + bands['insufficient_reference']
            + bands['discrete_or_fixed'] + bands['missing']) == bands['total']


def test_archetype_never_overclaims_synergy_when_unavailable_or_single_field():
    p_unavail = _sample_payload([_sample_card('Flow Duration', 50.0)])
    p_unavail['decomposition'] = {'status': 'unavailable', 'reason': '计算未完成'}
    arch_unavail = classify_attribution_archetype(p_unavail)
    assert arch_unavail['code'] == 'explanation_unavailable'
    assert '协同' not in arch_unavail['label'] and '协同' not in arch_unavail['headline']

    p_zero = _sample_payload([_sample_card('Flow Duration', 50.0)], contributions=[0.002])
    arch_zero = classify_attribution_archetype(p_zero)
    assert arch_zero['code'] == 'insufficient_driver_evidence'
    assert '协同' not in arch_zero['label']

    p_single = _sample_payload([_sample_card('Flow Duration', 50.0)], contributions=[0.24])
    arch_single = classify_attribution_archetype(p_single)
    assert arch_single['code'] == 'single_feature_push'
    assert '非多维协同' in arch_single['label']

    p_multi = _sample_payload(
        [_sample_card('Flow Duration', 50.0), _sample_card('Tot Fwd Pkts', 45.0)],
        contributions=[0.24, 0.15],
    )
    arch_multi = classify_attribution_archetype(p_multi)
    assert arch_multi['code'] == 'multi_in_range_joint_push'

    p_mixed = _sample_payload(
        [_sample_card('Flow Duration', 50.0), _sample_card('Flow Byts/s', 20.0, p01=None, p50=10.0, p99=None)],
        contributions=[0.24, 0.15],
    )
    arch_mixed = classify_attribution_archetype(p_mixed)
    assert arch_mixed['code'] == 'unverified_multi_driver_push'


def test_unified_table_and_draft_note_preserve_five_key_checklist_contract():
    payload = _sample_payload([_sample_card('Flow Duration', 250.0)], contributions=[0.24])
    before = copy.deepcopy(payload)
    rows = review_check_items(payload)
    assert all(set(r) == {'数据项', '当前值', '训练正常参考', '当前状态', '具体核查'} for r in rows)
    contribs = review_check_contributions(payload)
    assert len(contribs) == len(rows) and 'RF路径推高' in contribs[0]['对异常分的影响']
    u_rows = unified_interval_contribution_rows(payload, top_only=True)
    assert len(u_rows) == 1 and u_rows[0]['RF路径贡献 (概率点)'] == pytest.approx(0.24)
    draft = build_audit_note_draft(payload)
    assert '【评分线索】\n' in draft and '越界指标推高提示' in draft
    assert draft.endswith('【复核结论】\n')
    assert payload == before


def test_batch_interval_profile_separates_unknown_reference_from_normal_core():
    feat_df = pd.DataFrame({
        'Flow Duration': [10.0, 50.0, 98.0, 150.0, 350.0, np.nan],
        'Flow Byts/s': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
        'Protocol': [6.0, 6.0, 17.0, 6.0, 6.0, 6.0],
    })
    bundle = {
        'model_version': 'c7a028b4e4cafedd',
        'config': {'features': ['Flow Duration', 'Flow Byts/s', 'Protocol'], 'rf_threshold': 0.12, 'lr_threshold': 0.84, 'near_margin': 0.10},
        'reference': {
            'Flow Duration': {'q01': 0.0, 'median': 50.0, 'q99': 100.0},
            'Flow Byts/s': {'q01': None, 'median': 2.0, 'q99': None},
            'Protocol': {'q01': 6.0, 'median': 6.0, 'q99': 17.0},
        },
    }
    prof = batch_interval_profile(feat_df, bundle)
    dur_row = prof[prof['字段'] == 'Flow Duration'].iloc[0]
    assert (dur_row['高风险极值条数'], dur_row['一般超限条数'], dur_row['接近边界条数'], dur_row['区间核心内条数'], dur_row['参考不足/未比较条数'], dur_row['原始缺失条数']) == (1, 1, 1, 2, 0, 1)

    unknown_row = prof[prof['字段'] == 'Flow Byts/s'].iloc[0]
    assert unknown_row['区间核心内条数'] == 0
    assert unknown_row['参考不足/未比较条数'] == 5
    assert unknown_row['原始缺失条数'] == 1
    assert pd.isna(unknown_row['本批越界率 (%)'])
    assert '参考不足' in unknown_row['区间取值态势提示']

    proto_row = prof[prof['字段'] == 'Protocol'].iloc[0]
    assert proto_row['区间核心内条数'] == 0
    assert proto_row['参考不足/未比较条数'] == 6
