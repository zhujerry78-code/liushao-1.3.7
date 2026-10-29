"""Product regression checks using synthetic data, not participant records."""
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import app_evidence_ui as ui


def state(note='', **overrides):
    editor = SimpleNamespace(note=note, input_key='input-a', record_id='record-a', status='待补证')
    values = dict(editor=editor, exports=None, pending=None)
    values.update(overrides)
    return {'ui': SimpleNamespace(**values), '_note': note}


def test_draft_append_preserves_manual_note_and_does_not_change_verdict(monkeypatch):
    values = state('人工填写：需要查服务日志。')
    monkeypatch.setattr(ui, 'st', SimpleNamespace(session_state=values))
    ui._apply_draft_note('自动整理的评分摘要', 'input-a', 'record-a')
    expected = '人工填写：需要查服务日志。\n\n自动整理的评分摘要'
    assert values['_note'] == expected == values['ui'].editor.note
    assert values['ui'].editor.status == '待补证'
    ui._apply_draft_note('自动整理的评分摘要', 'input-a', 'record-a')
    assert values['_note'] == expected


def test_stale_or_locked_draft_callback_cannot_change_editor(monkeypatch):
    for overrides, input_key, record_id in (
        ({'exports': object()}, 'input-a', 'record-a'),
        ({'pending': object()}, 'input-a', 'record-a'),
        ({}, 'old-input', 'record-a'),
        ({}, 'input-a', 'old-record'),
    ):
        values = state('保留已输入内容', **overrides)
        monkeypatch.setattr(ui, 'st', SimpleNamespace(session_state=values))
        ui._apply_draft_note('自动摘要', input_key, record_id)
        assert values['_note'] == values['ui'].editor.note == '保留已输入内容'


def test_long_existing_note_is_never_truncated_or_replaced(monkeypatch):
    values = state('字' * 3998)
    monkeypatch.setattr(ui, 'st', SimpleNamespace(session_state=values))
    ui._apply_draft_note('摘要', 'input-a', 'record-a')
    assert values['_note'] == '字' * 3998
    assert ui._merged_note('字' * 3996, '摘要') == '字' * 3996 + '\n\n摘要'


def test_named_fields_keep_separate_contributions_and_escape_user_text():
    rows = [dict(数据项='同名指标<img src=x>', 当前值='1 字节', 训练正常参考='0～5 字节',
                 当前状态='在范围内', 具体核查='核查日志。',
                 _contribution=dict(rf_path_contribution=value, rf_replace_delta=-value))
            for value in (.2, -.1)]
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import _check_table\n_check_table(st.session_state.rows)')
    app.session_state.rows = rows
    app.run()
    assert not app.exception
    html = app.markdown[0].value
    assert '<img src=x>' not in html and html.count('&lt;img src=x&gt;') == 2
    body_rows = html.split('<tbody>')[1].split('</tbody>')[0].split('</tr>')
    assert '+0.2000' in body_rows[0] and '-0.2000' in body_rows[0]
    assert '-0.1000' in body_rows[1] and '+0.1000' in body_rows[1]


def test_grouped_checks_preserve_fields_and_escape_content_without_repeating_actions():
    rows = [dict(数据项=name, 具体核查='核对连接日志。') for name in ('指标一', '<script>指标二</script>')]
    app = AppTest.from_string('import streamlit as st\nfrom app_evidence_ui import _render_check_actions\n_render_check_actions(st.session_state.rows, [])')
    app.session_state.rows = rows
    app.run()
    assert not app.exception
    html = app.markdown[0].value
    assert html.count('核对连接日志') == 1
    assert '指标一' in html and '&lt;script&gt;指标二&lt;/script&gt;' in html
    assert '<script>' not in html


def test_draft_preview_adds_once_and_requires_explicit_save(tmp_path):
    from app_types import Draft, SessionState
    script = tmp_path / 'draft_preview_app.py'
    script.write_text('''
import streamlit as st
from app_evidence_ui import render_review_checklist
render_review_checklist(st.session_state.payload)
st.text_area('备注', key='_note', max_chars=4000)
''', encoding='utf-8')
    app = AppTest.from_file(str(script))
    session = SessionState()
    session.editor = Draft(input_key='input-a', record_id='record-a', status='待复核', note='手写核查计划',
                           original=('待复核', ''), baseline=0)
    app.session_state.ui = session
    app.session_state['_note'] = '手写核查计划'
    app.session_state.payload = dict(record_id='record-a', cards=[], all_fields=[], scores={},
                                    decomposition={'status': 'unavailable'})
    app.run()
    assert not app.exception
    preview = '\n'.join(element.value for element in app.markdown)
    assert 'review-note-preview' in preview and preview.count('<section>') == 5
    assert '尚未人工核查' not in preview and '自动整理当前数据' not in preview
    draft_button = app.button(key='fill_evidence_draft_record-a')
    assert draft_button.label == '追加到备注'
    draft_button.click().run()
    assert not app.exception
    assert app.text_area[0].value.startswith('手写核查计划\n\n')
    assert app.session_state.ui.editor.status == '待复核'
    assert app.button(key='fill_evidence_draft_record-a').disabled
    assert app.button(key='fill_evidence_draft_record-a').label == '已追加'
