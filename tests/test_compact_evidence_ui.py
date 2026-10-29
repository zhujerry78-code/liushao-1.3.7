"""Presentation checks on engineering fixtures; no human-usability claim."""
import copy
import math

import pytest
from streamlit.testing.v1 import AppTest

from app_evidence import build_record_evidence
from app_evidence_ui import compact_evidence_summary, rf_contribution_rows, _contribution_spec
from liushao.model import infer


def card(index=0, **changes):
    value = dict(field=f'field{index}', name=f'指标{index}', value=0., unit='字节',
                 reference_status='接近参考边界（区间内）', missing=False,
                 role='支持当前模型提示', action=dict(next='核对同目的端口的短连接与包速率，查询防火墙/服务日志和压测工单。'))
    value.update(changes)
    return value


def summary_payload():
    return dict(scores=dict(rf=.9, lr=.1, rf_threshold=.5, lr_threshold=.5,
                            rf_anomaly=True, lr_anomaly=False),
                selection=dict(missing_fields=['missing1', 'missing2', 'missing3'],
                               insufficient_references=['ref1'], direction_differences=['diff1'],
                               counterevidence=['counter1']), cards=[card(i) for i in range(6)])


def test_compact_summary_has_three_facts_and_retains_every_warning_category():
    payload = summary_payload(); before = copy.deepcopy(payload)
    summary = compact_evidence_summary(payload)
    assert len(summary['facts']) == 3 and '指标3' not in summary['observed']
    alerts = '；'.join(summary['alerts'])
    for text in ('阈值判断不同', '原始缺失3项', '参考不足1项', '替换方向不同1项', '反向线索1项'):
        assert text in alerts
    assert 'missing1、missing2等3项' in alerts
    assert '下降' in summary['reason'] and '日志核对' in summary['reason']
    assert summary['next'].count('同端口连接与包速率') == 1
    assert payload == before


def test_near_boundary_zero_is_not_reworded_as_missing_or_outside():
    payload = summary_payload(); payload['cards'] = [card()]
    summary = compact_evidence_summary(payload)
    assert '0 字节' in summary['observed'] and '接近参考边界（区间内）' in summary['observed']
    assert '原始缺失' not in summary['observed'] and '越界' not in summary['observed']


def test_low_score_support_is_upward_response_and_no_candidate_does_not_claim_normal():
    payload = summary_payload(); payload['scores']['rf_anomaly'] = False
    assert '上升' in compact_evidence_summary(payload)['reason']
    assert '当前输入使分数更低' in compact_evidence_summary(payload)['reason']
    assert '反向线索也需一并核对' in compact_evidence_summary(payload)['reason']
    assert '对照业务与授权记录后判断' in compact_evidence_summary(payload)['next']
    payload['cards'] = []
    summary = compact_evidence_summary(payload)
    assert not summary['facts'] and '未提示异常仍需' in summary['reason']
    payload['scores']['rf_anomaly'] = True
    assert '高分尚缺' in compact_evidence_summary(payload)['reason']


def test_unavailable_does_not_manufacture_facts_or_hide_model_disagreement():
    payload = summary_payload(); payload['selection'] = dict(status='unavailable', reason='模型身份变化')
    summary = compact_evidence_summary(payload)
    assert not summary['facts'] and summary['unavailable'] == '模型身份变化'
    assert '阈值判断不同' in '；'.join(summary['alerts'])


def test_contribution_chart_has_at_most_twelve_fields_and_exact_omitted_sum():
    fields = [dict(field=str(i), contribution=(-1)**i*(i+1)/100) for i in range(20)]
    payload = dict(all_fields=[dict(field=str(i), name=f'指标{i}') for i in range(20)],
                   decomposition=dict(rf=dict(fields=fields, top_fields=['0']), default_fields=[str(i) for i in range(15)]))
    before = copy.deepcopy(payload); rows = rf_contribution_rows(payload)
    assert len([r for r in rows if r['kind'] == '字段']) == 12
    assert rows[-1]['name'] == '其余8项合计'
    assert math.fsum(r['contribution'] for r in rows) == pytest.approx(math.fsum(r['contribution'] for r in fields), abs=1e-15)
    spec = _contribution_spec(rows)
    assert {'抬高分数', '降低分数'} <= {r['direction'] for r in spec['data']['values']}
    assert spec['layer'][0]['encoding']['x']['scale']['zero'] is True
    assert payload == before


def test_main_render_keeps_long_details_off_review_page():
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import render_record_evidence\nrender_record_evidence(st.session_state.payload)')
    app.session_state.payload = summary_payload(); app.run()
    assert not app.exception and not app.dataframe and not app.expander
    assert len(app.markdown) == 1
    assert '建议核查' in app.markdown[0].value
    assert '为什么值得核查' not in app.markdown[0].value
    assert '核对' in app.markdown[0].value
    assert all('不是攻击概率' not in item.value for item in app.caption)


def test_detail_tabs_preserve_full_accounting_and_selected_evidence(engineering_bundle):
    bundle, raw = engineering_bundle
    pred, features, _ = infer(raw, bundle, 'a'*64, 'engineering.csv')
    payload = build_record_evidence(pred.iloc[-1], features, bundle)
    before = copy.deepcopy(payload)
    assert payload['decomposition']['status'] == 'available'
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import render_evidence_detail\ntabs = render_evidence_detail(st.session_state.payload)\nwith tabs[2]: st.caption("existing raw tools")')
    app.session_state.payload = payload; app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['评分图解', '证据明细', '原始数据与方法']
    assert any(exp.label == 'LR 完整计算' for exp in app.expander)
    if payload['cards']:
        choice = app.selectbox[0]
        for item in payload['cards']:
            choice.select(item['field']).run()
            assert not app.exception
            assert any(item['name'] in element.value and item['role'] in element.value for element in app.markdown)
    assert app.session_state.payload == before


def test_detail_unavailable_keeps_failure_visible_without_fake_chart():
    payload = summary_payload()
    payload.update(all_fields=[], cards=[], selection=dict(status='unavailable', reason='evidence failed'),
                   decomposition=dict(status='unavailable', reason='score failed'))
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import render_evidence_detail\nrender_evidence_detail(st.session_state.payload)')
    app.session_state.payload = payload; app.run()
    assert not app.exception
    assert any('score failed' in element.value for element in app.info)
    assert any('evidence failed' in element.value for element in app.warning)



def test_same_chinese_labels_keep_distinct_contribution_bars():
    rows = [dict(field='Fwd Seg Size Avg', name='同名数据项', contribution=.2, kind='字段'),
            dict(field='Bwd Seg Size Avg', name='同名数据项', contribution=-.1, kind='字段'),
            dict(field='__remaining__', name='其余75项合计', contribution=.3, kind='其余合计')]
    spec = _contribution_spec(rows)
    values = spec['data']['values']
    assert len({row['label'] for row in values}) == 3
    assert spec['layer'][0]['encoding']['y']['field'] == 'label'
    assert values[0]['label'] == '同名数据项（1）'
    assert values[1]['label'] == '同名数据项（2）'
    assert values[-1]['label'] == '其余75项合计'


def test_low_score_with_missing_data_does_not_call_imputed_values_original_values():
    payload = summary_payload()
    payload['scores']['rf_anomaly'] = False
    payload['cards'] = [card(missing=True, value=None, reference_status='原始缺失')]
    summary = compact_evidence_summary(payload)
    assert '原始缺失' in summary['observed'] and '当前输入使分数更低' in summary['reason']
    assert '原值使模型' not in summary['reason'] and '不等于已确认正常' not in summary['reason']
    assert summary['next'].startswith('先核对缺失原值')



def test_contribution_legend_stacks_all_labels_for_narrow_screens():
    rows = [dict(field='x', name='指标', contribution=0., kind='字段')]
    spec = _contribution_spec(rows)
    legend = spec['layer'][0]['encoding']['color']['legend']
    assert legend == {'orient': 'top', 'direction': 'vertical', 'columns': 1}
    assert spec['height'] >= 130
    assert spec['layer'][0]['encoding']['color']['scale']['domain'] == ['抬高分数', '降低分数', '无加减']


def test_evidence_table_and_selected_card_spell_out_missing_without_payload_changes(engineering_bundle):
    bundle, raw = engineering_bundle
    raw = raw.copy(); raw.loc[raw.index[-1], 'a'] = float('nan')
    pred, features, _ = infer(raw, bundle, 'a'*64, 'engineering-missing.csv')
    payload = build_record_evidence(pred.iloc[-1], features, bundle)
    before = copy.deepcopy(payload)
    assert any(c['missing'] for c in payload['cards'])
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import _render_evidence_cards\n_render_evidence_cards(st.session_state.payload)')
    app.session_state.payload = payload; app.run()
    assert not app.exception
    table = app.dataframe[0].value
    assert '缺失' in table['原值'].tolist()
    assert all(isinstance(value, str) for value in table['原值'])
    missing_card = next(c for c in payload['cards'] if c['missing'])
    app.selectbox[0].select(missing_card['field']).run()
    assert not app.exception
    assert any('原值：原始缺失' in element.value for element in app.markdown)
    assert payload == before and app.session_state.payload == before


def test_separate_checklist_has_readable_values_reference_checks_and_escaped_html(engineering_bundle):
    bundle, raw = engineering_bundle
    pred, features, _ = infer(raw, bundle, 'a'*64, 'engineering.csv')
    payload = build_record_evidence(pred.iloc[-1], features, bundle)
    if payload['cards']:
        payload['cards'][0]['name'] = '数据项<img src=x onerror=alert(1)>'
    before = copy.deepcopy(payload)
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import render_review_checklist\nrender_review_checklist(st.session_state.payload)')
    app.session_state.payload = payload; app.run()
    assert not app.exception
    html = '\n'.join(item.value for item in app.markdown)
    assert '核查重点' in html
    assert '当前值' in html and '训练参考' in html
    assert 'review-check-actions' in html and '核对地址、时间' in html
    assert '<img src=x' not in html
    if payload['cards']:
        assert '&lt;img src=x' in html
    assert payload == before and app.session_state.payload == before


def test_detail_removes_repeated_declarations_and_keeps_calculations_available(engineering_bundle):
    bundle, raw = engineering_bundle
    pred, features, _ = infer(raw, bundle, 'a'*64, 'engineering.csv')
    payload = build_record_evidence(pred.iloc[-1], features, bundle)
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import render_evidence_detail\nrender_evidence_detail(st.session_state.payload)')
    app.session_state.payload = payload; app.run()
    assert not app.exception
    text='\n'.join(x.value for group in (app.info,app.caption,app.markdown) for x in group)
    assert 'RF 评分构成' in text and '辅助模型（LR）' in text
    assert all(phrase not in text for phrase in ('不是攻击概率', '不代表攻击风险', '不能确定攻击原因',
        '不等于同时在线的连接数', '模型补用的数值不是实际采集到的数据', '不会凑满六项'))
    assert app.json and any(exp.label == '完整计算快照' for exp in app.expander)
