"""Plain review language must preserve the evidence's numerical limits."""
import copy
import json

import pytest

from app_evidence import build_record_evidence, feature_definition
from app_evidence_rules import reference_position
from app_review_guidance import build_audit_note_draft, review_check_items, review_guidance
from liushao.model import infer


def card(field='Flow Duration', value=50., *, reference=None, role='支持当前模型提示', **changes):
    reference = dict(p01=0., p50=50., p99=100.) if reference is None else reference
    name, unit, meaning = feature_definition(field)
    result = dict(field=field, name=name, unit=unit, meaning=meaning, value=value,
                  reference=reference, missing=value is None, role=role,
                  reference_position=reference_position(field, value, dict(q01=reference.get('p01'),
                      median=reference.get('p50'), q99=reference.get('p99'))),
                  rf=dict(before=.9, after=.6, delta=-.3),
                  lr=dict(before=.1, after=.2, delta=.1), direction_difference=False)
    result.update(changes)
    return result


def payload(cards=None, **changes):
    cards = [card()] if cards is None else cards
    result = dict(cards=cards, all_fields=copy.deepcopy(cards),
                  scores=dict(rf=.9, lr=.1, rf_threshold=.5, lr_threshold=.5,
                              rf_anomaly=True, lr_anomaly=False),
                  selection=dict(eligible=len(cards), missing_fields=[], insufficient_references=[],
                                 direction_differences=[], counterevidence=[]))
    result.update(changes)
    return result


@pytest.mark.parametrize('value,status', [
    (0., '在参考范围内，但靠近低端'), (2., '在参考范围内，但靠近低端'),
    (4., '在参考范围内'), (50., '在参考范围内'), (96., '在参考范围内'),
    (98., '在参考范围内，但靠近高端'),
    (100., '在参考范围内，但靠近高端'), (-.1, '低于参考范围'), (100.1, '高于参考范围'),
])
def test_inside_edges_and_actual_outliers_are_distinguished(value, status):
    data = payload([card(value=value)])
    row = review_check_items(data)[0]
    assert row['当前状态'] == status
    assert f'{value:,.6g} 微秒 (μs)' == row['当前值']
    assert '0～100 微秒 (μs)' in row['训练正常参考']
    assert '中间值 50 微秒 (μs)' in row['训练正常参考']
    assert status in review_guidance(data)['observed']
    if 0 <= value <= 100:
        assert '越界' not in row['当前状态']
        assert '高于参考范围' not in row['当前状态']
        assert '低于参考范围' not in row['当前状态']


def test_missing_and_insufficient_reference_are_visible_first_without_using_imputed_values():
    normal = card('Flow Duration')
    missing = card('Tot Fwd Pkts', None, imputed_rf=1234., imputed_lr=5678.)
    no_reference = card('Flow Byts/s', 2., reference=dict(p01=None, p50=4., p99=None))
    data = payload([normal, no_reference, missing])
    rows = review_check_items(data)
    assert rows[0]['数据项'] == '发出数据包数'
    assert rows[0]['当前值'] == '原始缺失'
    assert '原始文件/采集日志中的缺失值' in rows[0]['具体核查']
    assert rows[1]['数据项'] == '每秒传输字节数'
    assert '参考不足' in rows[1]['当前状态']
    assert '参考不足' in rows[1]['训练正常参考'] and '中间值 4' in rows[1]['训练正常参考']
    assert '1234' not in json.dumps(rows, ensure_ascii=False)
    assert '5678' not in json.dumps(rows, ensure_ascii=False)
    summary = review_guidance(data)
    assert summary['next'].startswith('先核对原始缺失值')
    assert '原始缺失' in summary['observed'] and '参考不足' in summary['observed']


def test_alerts_use_chinese_for_all_fields_outside_the_six_main_cards():
    extras = [card('Flow Pkts/s', None),
              card('Init Fwd Win Byts', 2., reference={}),
              card('Bwd IAT Mean', 3., direction_difference=True),
              card('TotLen Bwd Pkts', 4., role='反向线索')]
    data = payload([card('Flow Duration')])
    data['all_fields'] += extras
    data['selection'].update(missing_fields=['Flow Pkts/s'],
        insufficient_references=['Init Fwd Win Byts'],
        direction_differences=['Bwd IAT Mean'], counterevidence=['TotLen Bwd Pkts'])
    data['stability'] = dict(joint_direction_conflict=True)
    summary = review_guidance(data)
    alerts = '；'.join(summary['alerts'])
    for name in ('每秒数据包数', '发出方向起始窗口大小', '平均返回包间隔', '返回数据总量'):
        assert name in alerts
    for field in ('Flow Pkts/s', 'Init Fwd Win Byts', 'Bwd IAT Mean', 'TotLen Bwd Pkts'):
        assert field not in alerts
    assert 'RF 与 LR 提示不同' in alerts
    assert '反应方向不同' in alerts and '同时对照主要支持项' in alerts
    assert '与当前提示相反' in alerts
    shown_rows = {row['数据项']: row for row in review_check_items(payload(extras))}
    assert '反应方向不同' in shown_rows['平均返回包间隔']['具体核查']
    assert '与当前模型提示相反' in shown_rows['返回数据总量']['具体核查']


def test_alert_counts_do_not_shrink_when_names_are_shortened():
    data = payload()
    extra = [card('Flow Pkts/s', None), card('Tot Fwd Pkts', None), card('Tot Bwd Pkts', None)]
    data['all_fields'] += extra
    summary = review_guidance(data)
    missing_alert = next(text for text in summary['alerts'] if text.startswith('原始缺失'))
    assert '3 项' in missing_alert and '等3项' in missing_alert
    assert '每秒数据包数' in missing_alert and '发出数据包数' in missing_alert
    assert '返回数据包数' not in missing_alert


def test_unknown_fields_never_leak_raw_english_names_to_main_review():
    unknown = card('private_unknown_english', 5., name='private_unknown_english')
    unknown['direction_difference'] = True
    data = payload([unknown])
    text = json.dumps(dict(summary=review_guidance(data), rows=review_check_items(data)), ensure_ascii=False)
    assert 'private_unknown_english' not in text and '其他数据项' in text


def test_summary_limits_facts_and_cards_without_modifying_frozen_evidence():
    cards = [card(field) for field in ('Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Tot Fwd Pkts',
                                      'Tot Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean')]
    data = payload(cards)
    before = copy.deepcopy(data)
    summary, rows = review_guidance(data), review_check_items(data)
    assert len(summary['facts']) == 3
    assert len(rows) == 8  # Six selected data rows and two clearly named follow-up rows.
    assert '平均返回包长度' not in [row['数据项'] for row in rows]
    assert data == before
    json.dumps(summary, ensure_ascii=False, allow_nan=False)
    json.dumps(rows, ensure_ascii=False, allow_nan=False)


def test_explanation_is_model_response_not_attack_cause_or_probability():
    data = payload()
    summary = review_guidance(data)
    assert '主要模型（RF）' in summary['reason']
    assert '常见值作对照' in summary['reason'] and '下降' in summary['reason']
    assert '不能确定攻击原因' not in summary['reason']
    assert '中位数' not in summary['reason'] and '改回' not in summary['reason']
    assert '攻击概率' not in summary['reason']
    data['scores']['rf_anomaly'] = False
    summary = review_guidance(data)
    assert '上升' in summary['reason'] and '未提示异常' in summary['reason']
    assert '不等于已确认正常' not in summary['reason']


def test_empty_or_failed_explanation_never_manufactures_main_support():
    data = payload([])
    summary = review_guidance(data)
    assert summary['facts'] == [] and '没有筛出合格' in summary['observed']
    assert '尚缺主要支持项' in summary['reason']
    data['scores']['rf_anomaly'] = False
    assert '未提示异常或信息不足' in review_guidance(data)['reason']
    data['scores']['rf_anomaly'] = True
    data['selection'] = dict(status='unavailable', reason='模型身份变化')
    summary = review_guidance(data)
    assert summary['unavailable'] == '模型身份变化'
    assert summary['facts'] == [] and '计算未完成' in summary['reason']
    assert 'RF 与 LR 提示不同' in '；'.join(summary['alerts'])
    assert review_check_items(data) == []


def test_discrete_and_fixed_reference_do_not_imply_numeric_risk():
    rows = review_check_items(payload([
        card('Protocol', 6.), card('Flow Duration', 4., reference=dict(p01=3., p50=3., p99=3.))]))
    assert '编号或标记' in rows[0]['当前状态']
    assert '编号/计数参考' in rows[0]['训练正常参考']
    assert '编号大小不代表风险' not in rows[0]['训练正常参考']
    assert rows[1]['当前状态'] == '与固定参考值不同'
    assert '固定为 3 微秒' in rows[1]['训练正常参考']


def test_supplied_neighbor_counts_are_displayed_without_inventing_log_evidence():
    data = payload(neighbors=dict(available=True, same_source=0, same_destination=7,
        source_fanout=2, repeat_five_tuple=3, window_seconds=60,
        time_window=dict(all=5, same_source=0, same_destination=4)))
    rows = review_check_items(data)
    context, logs = rows[-2:]
    for text in ('同源记录 0 条', '同目的记录 7 条', '同源涉及的目的地址 2 个',
                 '同五元组重复记录 3 条', '前后各 60 秒内其他记录 5 条', '其中同目的 4 条'):
        assert text in context['当前值']
    assert context['当前状态'] == '仅本批统计，不含当前记录'
    assert context['具体核查'] == '核对地址、时间与相邻记录是否属于同一业务。'
    assert logs['当前值'] == '文件未提供，需查日志/工单'
    assert '同端口连接数' in logs['数据项'] and '授权任务' in logs['具体核查']


def test_missing_context_never_becomes_zero_or_a_fake_connection_count():
    context, logs = review_check_items(payload())[-2:]
    assert '文件未提供可用地址/时间信息' in context['当前值']
    assert context['当前状态'] == '缺少核查上下文'
    assert '0 条' not in context['当前值']
    assert logs['当前值'] == '文件未提供，需查日志/工单'


def test_large_microsecond_references_use_grouped_numbers_without_scientific_notation():
    data = payload([card('Flow Duration', 117_306_000.,
                         reference=dict(p01=0., p50=1_234_567., p99=117_306_000.))])
    row = review_check_items(data)[0]
    assert row['当前值'] == '117,306,000 微秒 (μs)'
    assert '0～117,306,000 微秒 (μs)' in row['训练正常参考']
    assert '1,234,570 微秒 (μs)' in row['训练正常参考']
    assert 'e+' not in row['当前值'] + row['训练正常参考']
    assert '核对原始记录' in review_guidance(data)['next']


def test_tiny_nonzero_values_keep_significant_digits_without_becoming_zero():
    data = payload([card('Flow Byts/s', .000000001234567,
                         reference=dict(p01=.000000000001, p50=.00000000001234567, p99=1.))])
    row = review_check_items(data)[0]
    assert row['当前值'] == '0.00000000123457 字节/秒 (B/s)'
    assert '0.000000000001～1' in row['训练正常参考']
    assert '0.0000000000123457 字节/秒 (B/s)' in row['训练正常参考']
    assert 'e-' not in row['当前值'] + row['训练正常参考']


def test_real_built_payload_retains_all_model_scores_and_evidence(engineering_bundle):
    bundle, raw = engineering_bundle
    predictions, features, _ = infer(raw, bundle, 'a' * 64, 'engineering.csv')
    data = build_record_evidence(predictions.iloc[-1], features, bundle, context=raw)
    before = copy.deepcopy(data)
    summary, rows = review_guidance(data), review_check_items(data)
    assert len(summary['facts']) <= 3 and len(rows) <= 8
    assert data == before
    assert all(set(row) == {'数据项', '当前值', '训练正常参考', '当前状态', '具体核查'} for row in rows)
    assert '模型（RF）' in summary['reason']
    json.dumps(summary, ensure_ascii=False, allow_nan=False)
    json.dumps(rows, ensure_ascii=False, allow_nan=False)


def test_note_draft_separates_facts_and_leaves_verdict_empty_without_long_record_id():
    data = payload([card('Flow Duration', 250.), card('Tot Fwd Pkts', 60.)],
                   record_id='a' * 64 + ':444229')
    before = copy.deepcopy(data)
    text = build_audit_note_draft(data)
    for heading in ('评分线索', '区间统计', '关键数据', '后续核查', '复核结论'):
        assert f'【{heading}】\n' in text
    assert '\n\n【' in text
    assert '\n1. ' in text and '\n2. ' in text
    assert 'RF 0.9（分界 0.5）；LR 0.1（分界 0.5）' in text
    assert '250 微秒 (μs)' in text and '60 包' in text
    assert text.endswith('【复核结论】\n')
    assert data['record_id'] not in text
    assert all(phrase not in text for phrase in ('自动整理草稿', '尚未人工核查', '不能确定攻击原因'))
    assert data == before
