"""Regression checks for the diagnostic report's search and navigation failures.

These are engineering fixtures, not human usability measurements.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_navigation import PAGES, page_id
from app_types import Draft, Filters, PendingAction, SessionState
from app_ui_helpers import messages, navigate, start
from app_ui_state import dirty, filtered


def callbacks(state, session=None):
    """Execute the shipped callbacks without running Streamlit's page body."""
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
    names = {'request', 'apply_action', 'row_selected', 'page_to'}
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    env = {'u': state, 'st': SimpleNamespace(session_state=session or {}),
           'PendingAction': PendingAction, 'Filters': Filters, 'dirty': dirty}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'shipped-callbacks', 'exec'), env)
    return env


@pytest.mark.parametrize('dirty_draft', [False, True])
@pytest.mark.parametrize('selection', ['row', 'cell'])
def test_selecting_search_result_preserves_table_identity_and_page(dirty_draft, selection):
    state = SessionState(selected='old', page_index=1, table_epoch=7)
    state.editor = Draft('old', 'input', '待补证' if dirty_draft else '待复核',
                         '未保存依据' if dirty_draft else '', ('待复核', ''), 0)
    key = '_records_7'
    selected = SimpleNamespace(rows=[1] if selection == 'row' else [],
                               cells=[(1, '记录')] if selection == 'cell' else [])
    env = callbacks(state, {key: SimpleNamespace(selection=selected)})
    env['row_selected'](key, ['other', 'new'])
    assert state.table_epoch == 7
    assert state.page_index == 1
    if dirty_draft:
        assert state.selected == 'old' and state.editor.note == '未保存依据'
        assert state.pending == PendingAction('record', 'new')
    else:
        assert state.selected == 'new' and state.editor is None and state.pending is None


def test_filter_and_page_changes_still_reset_the_table_selection():
    state = SessionState(selected='old', page_index=1, table_epoch=7)
    env = callbacks(state)
    env['request']('filters', Filters(line='113'))
    assert state.filters.line == '' and state.page_index == 0
    assert state.selected == 'old' and state.table_epoch == 8
    env['page_to'](1)
    assert state.page_index == 1 and state.table_epoch == 9


def test_stale_row_selection_cannot_override_current_filter_or_pending_action():
    state = SessionState(selected='old', table_epoch=8)
    session = {'_records_7': SimpleNamespace(selection=SimpleNamespace(rows=[0], cells=[]))}
    env = callbacks(state, session)
    env['row_selected']('_records_7', ['stale'])
    assert state.selected == 'old' and state.table_epoch == 8
    state.pending = PendingAction('filters', Filters(line='113'))
    env['request']('record', 'new')
    assert state.selected == 'old' and state.pending.kind == 'filters'
    assert state.table_epoch == 8


def test_full_search_finds_a_row_outside_current_fifty_row_page():
    pred = pd.DataFrame({'record_id': [f'fixture:{n}' for n in range(1, 151)],
                         'source_row': range(1, 151), 'rf_score': [.5] * 150,
                         'rf_anomaly': [True] * 150, 'review_status': ['待复核'] * 150,
                         'disagreement': [False] * 150, 'near_threshold': [False] * 150})
    assert 113 not in set(pred.iloc[50:100].source_row)
    result = filtered(pred, Filters(queue='原始顺序', line='113'), {})
    assert result.source_row.tolist() == [113]
    assert result.record_id.tolist() == ['fixture:113']
    # Global search is still combined with an explicit user filter.
    result = filtered(pred, Filters(queue='原始顺序', line='113', minimum=.6), {})
    assert result.empty


def test_model_evaluation_moves_to_help_and_legacy_routes_remain_valid(r9_app):
    assert all(page.id != 'evaluation' for page in PAGES)
    assert page_id('evaluation') == page_id('模型与评测') == 'help'
    app = AppTest.from_file(str(r9_app[0]), default_timeout=45).run()
    assert not app.exception
    assert 'nav_evaluation' not in [button.key for button in app.button]
    navigate(app, '操作说明')
    assert app.session_state.ui.result is None
    assert any('模型与评测' in expander.label for expander in app.expander)
    assert '未附评测资料' in messages(app)


def test_full_search_is_named_and_keeps_row_filter_when_opening_help(r9_app):
    app = start(r9_app)
    navigate(app, '检查记录')
    selected=app.session_state.ui.selected
    app.button(key='quick_search_toggle').click().run()
    app.text_input(key='_quick_search_input').input('103').run()
    assert not app.exception
    assert app.session_state.ui.page_index==0 and app.session_state.ui.selected==selected
    assert len(app.dataframe[0].value)==8 and app.dataframe[0].value.iloc[0]['记录']=='第 103 行'
    navigate(app,'操作说明');navigate(app,'检查记录')
    assert app.text_input(key='_quick_search_input').value=='103'
    assert app.session_state.ui.selected==selected


def test_three_downloads_have_individual_visible_explanations(r9_app):
    app = start(r9_app)
    navigate(app, '下载结果')
    labels = ('检查结果 CSV', '操作记录 JSON', '运行信息 JSON')
    for label in labels:
        assert any(label in item.value and not any(other in item.value for other in labels if other != label)
                   for item in app.markdown)
    assert not any('其他文件' in item.label for item in app.expander)
    r9_app[3].clear()
    app.button(key='prepare_exports').click().run()
    assert not app.exception
    assert set(r9_app[3]) == {'export_predictions', 'export_history', 'export_manifest','export_confirmed'}
    assert not any('其他文件' in item.label for item in app.expander)
    for label in labels:
        assert any(label in item.value for item in app.markdown)
