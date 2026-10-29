"""Export preparation is bound to a saved snapshot, through current UI controls."""
import io
import json
import sqlite3
import pandas as pd
import pytest
from app_ui_helpers import start,navigate,prepare,edit,upload,messages
from liushao import reviews as store


def test_empty_history_requires_export_preparation_and_explains_saved_counts(r9_app):
    app=start(r9_app);navigate(app,'下载结果')
    assert '操作记录为空' in messages(app) and 'export_predictions' not in r9_app[3]
    assert json.loads(prepare(app,r9_app[3])['export_history'])==[]


def test_edit_without_save_blocks_export_and_explicit_reload_discards_it(r9_app):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','未保存内容')
    navigate(app,'下载结果');assert app.button(key='prepare_exports').disabled
    app.button(key='return_editor').click().run()
    app.button(key='reload_review').click().run()
    assert app.text_area(key='_note').value=='' and store.history(r9_app[2])==[]
    assert 'export_predictions' in prepare(app,r9_app[3])


def test_preparation_catches_a_note_submitted_with_the_same_click(r9_app):
    app=start(r9_app);navigate(app,'检查记录')
    app.text_area(key='_note').input('同次提交尚未保存')
    app.button(key='nav_下载结果').click().run()
    assert app.button(key='prepare_exports').disabled
    assert app.session_state.ui.editor.note=='同次提交尚未保存' and store.history(r9_app[2])==[]


def test_save_preparation_and_return_to_editor_keep_snapshot_counts_consistent(r9_app):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'确认正常','核对依据')
    app.button(key='save_review').click().run();values=prepare(app,r9_app[3]);meta=json.loads(values['export_manifest'])
    assert meta['saved_review_records']==meta['review_history_entries']==1
    app.button(key='resume_review_edit').click().run()
    assert not app.selectbox(key='_status').disabled and app.session_state.ui.exports is None


def test_failed_save_keeps_draft_and_blocks_preparation(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','保留')
    def fail(*a,**kw):raise sqlite3.OperationalError('injected write failure')
    monkeypatch.setattr(store,'save_review',fail)
    app.button(key='save_review').click().run();navigate(app,'下载结果')
    assert app.button(key='prepare_exports').disabled and app.session_state.ui.editor.note=='保留'


def test_navigation_keeps_unsaved_changes_until_explicit_discard(r9_app):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','保留草稿')
    navigate(app,'操作说明');navigate(app,'检查记录')
    assert app.text_area(key='_note').value=='保留草稿'
    app.selectbox(key='_filter_scope').select('异常提示').run()
    assert app.session_state.ui.pending is not None
    app.button(key='pending_discard').click().run()
    assert store.history(r9_app[2])==[] and app.session_state.ui.filters.scope=='异常提示'


def test_source_switch_exits_prepared_export_and_counts_only_that_input(r9_app,monkeypatch):
    app=start(r9_app);prepare(app,r9_app[3]);app.button(key='resume_review_edit').click().run()
    app=upload(r9_app,monkeypatch,b'a,b\n1,2\n',app=app)
    assert app.session_state.ui.exports is None
    values=prepare(app,r9_app[3]);assert json.loads(values['export_manifest'])['valid_rows']==1
    assert json.loads(values['export_history'])==[]


def test_prepared_files_remain_same_snapshot_when_another_window_saves(r9_app):
    app=start(r9_app);before=dict(prepare(app,r9_app[3]));rid=app.session_state.ui.result.predictions.record_id.iloc[0]
    store.save_review(r9_app[2],rid,r9_app[4]['model_version'],'待补证','另一个窗口')
    r9_app[3].clear();app.run()
    for key in ('export_predictions','export_history','export_manifest'):assert r9_app[3][key]==before[key]


def test_storage_failure_revokes_downloads_until_explicit_repreparation(r9_app,monkeypatch):
    app=start(r9_app);prepare(app,r9_app[3]);original=store.history
    def fail(*a,**kw):raise sqlite3.OperationalError('injected read failure')
    monkeypatch.setattr(store,'history',fail);r9_app[3].clear();app.run()
    assert app.session_state.ui.exports is None and 'export_predictions' not in r9_app[3]
    app.button(key='prepare_exports').click().run()
    assert app.session_state.ui.exports.predictions is None and 'export_manifest' in r9_app[3]
    monkeypatch.setattr(store,'history',original);app.button(key='retry_reviews').click().run()
    assert app.session_state.ui.exports is None and app.button(key='prepare_exports')


def test_clear_filters_restores_all_records_and_explains_export_scope(r9_app):
    app=start(r9_app);navigate(app,'检查记录')
    app.selectbox(key='_filter_status').select('确认异常').run()
    assert len(app.dataframe[0].value)==0
    app.button(key='clear_filters').click().run()
    assert len(app.dataframe[0].value)==8
    assert len(pd.read_csv(io.BytesIO(prepare(app,r9_app[3])['export_predictions'])))==8
