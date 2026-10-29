"""Behavior regressions for r9; synthetic fixtures are not model acceptance."""
import hashlib
import io
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app_types import Draft, SessionState, invalidate_model, line_number, validate_history
from app_ui_state import csv_bytes, export_snapshot, filtered, default_filters
from app_fixtures import isolated_app, engineering_bundle, r9_app
from liushao.reviews import history, save_review



def start(fixture):
    app = AppTest.from_file(str(fixture[0]), default_timeout=45).run()
    assert not app.exception
    app.button(key='run').click().run()
    assert not app.exception
    assert app.session_state.ui.result is not None
    return app


def navigate(app, page):
    app.button(key='nav_'+page).click().run()
    assert not app.exception


@pytest.mark.parametrize('value', ['', '9'*5000, '-1', 'x', '１２３x'])
def test_line_number_invalid_never_crashes(value):
    assert line_number(value) is None


def test_model_invalidation_cannot_resurrect_old_results():
    draft = Draft('r', 'input|model', '确认正常', '依据', ('待复核',''), 7)
    state = SessionState(editor=draft, result=object(), exports=object())
    invalidate_model(state)
    assert state.result is state.active is state.editor is state.exports is None
    assert state.suspended_draft is draft
    assert state.suspended_draft.baseline == 7
    invalidate_model(state)
    assert state.suspended_draft is draft


def test_csv_escape_changes_only_export_copy():
    source = pd.DataFrame({'note':['=1+1', ' \n@SUM(A1)', '\ttext', 'ordinary', '-1'], 'score':[.1,.2,.3,.4,.5]})
    original = source.copy(deep=True)
    exported = pd.read_csv(io.BytesIO(csv_bytes(source)), dtype={'note':str})
    assert exported.note.tolist() == ["'=1+1", "' \n@SUM(A1)", "'\ttext", 'ordinary', "'-1"]
    pd.testing.assert_frame_equal(source, original)
    assert exported.score.tolist() == source.score.tolist()


def test_bad_history_is_not_silently_dropped():
    with pytest.raises(ValueError):
        validate_history([dict(id=1,record_id='r',model_version='m',status='unknown',note='',created_at='bad')], 'm')


def test_download_descriptions_and_saved_snapshot(r9_app):
    app = start(r9_app)
    navigate(app, '下载结果')
    text = '\n'.join(element.value for element in app.markdown)
    assert '检查结果 CSV' in text and '操作记录 JSON' in text and '运行信息 JSON' in text
    assert 'UTC' in text and '历次已保存' in text
    app.button(key='prepare_exports').click().run()
    assert not app.exception
    downloads = r9_app[3]
    result = pd.read_csv(io.BytesIO(downloads['export_predictions']))
    assert len(result) == 8
    assert json.loads(downloads['export_history']) == []
    fixed = app.session_state.ui.exports
    app.run()
    assert app.session_state.ui.exports.predictions == fixed.predictions


def test_storage_failure_still_downloads_manifest(r9_app):
    app = start(r9_app)
    r9_app[2].unlink()
    r9_app[2].mkdir()
    navigate(app, '下载结果')
    app.button(key='prepare_exports').click().run()
    assert not app.exception
    state = app.session_state.ui
    assert state.exports is not None and state.exports.predictions is None
    assert 'export_manifest' in r9_app[3]
    assert json.loads(r9_app[3]['export_manifest'])['review_storage_available'] is False
    app.run()
    assert app.session_state.ui.exports is not None


def test_dirty_review_pending_cancel_and_save(r9_app):
    app = start(r9_app)
    navigate(app, '检查记录')
    app.selectbox(key='_status').select('确认正常').run()
    app.text_area(key='_note').input('依据：已核对来源').run()
    current = app.session_state.ui.selected
    app.selectbox(key='_filter_scope').select('异常提示').run()
    assert app.session_state.ui.pending is not None
    assert app.button(key='save_review').disabled
    assert all(b.key != 'unlock_editor' for b in app.button)
    app.button(key='pending_cancel').click().run()
    assert app.session_state.ui.filters.scope == '全部记录'
    assert app.session_state.ui.selected == current
    app.button(key='save_review').click().run()
    assert not app.exception
    records = history(r9_app[2], model_version=r9_app[4]['model_version'])
    assert len(records) == 1 and records[0]['record_id'] == current
    assert records[0]['note'] == '依据：已核对来源'


def test_tamper_then_restore_requires_new_analysis(r9_app):
    app = start(r9_app)
    model = r9_app[1]/'model.joblib'
    original = model.read_bytes()
    model.write_bytes(b'bad model')
    app.run()
    assert not app.exception and app.session_state.ui.result is None
    model.write_bytes(original)
    app.run()
    assert app.session_state.ui.result is None
    app.button(key='run').click().run()
    assert app.session_state.ui.result is not None


def test_stale_revision_keeps_draft(r9_app):
    app = start(r9_app)
    navigate(app, '检查记录')
    rid = app.session_state.ui.selected
    app.text_area(key='_note').input('本窗口草稿').run()
    save_review(r9_app[2], rid, r9_app[4]['model_version'], '待补证', '另一个窗口', expected_review_id=0)
    app.button(key='save_review').click().run()
    assert app.session_state.ui.editor.note == '本窗口草稿'
    assert app.session_state.ui.editor.baseline == 0
    assert len(history(r9_app[2])) == 1


@pytest.mark.parametrize('queue', ['高异常得分', '待核实', '原始顺序'])
def test_save_next_recomputes_after_concurrent_status_change(r9_app, queue):
    app = start(r9_app)
    navigate(app, '检查记录')
    app.selectbox(key='_filter_queue').select(queue).run()
    app.selectbox(key='_filter_status').select('待复核').run()
    state = app.session_state.ui
    table = state.result.predictions.assign(review_status='待复核')
    order = filtered(table, state.filters, r9_app[4]).record_id.tolist()
    assert order
    if len(order) == 1:
        assert app.button(key='save_review_next').disabled
        return
    from app_ui_helpers import select_record_id
    select_record_id(app,order[0])
    app.text_area(key='_note').input('当前记录依据').run()
    save_review(r9_app[2], order[1], r9_app[4]['model_version'], '确认正常', '并发完成下一条')
    app.selectbox(key='_status').select('待补证').run()
    app.button(key='save_review_next').click().run()
    assert not app.exception
    assert app.session_state.ui.selected == (order[2] if len(order) > 2 else order[0])
    entries = history(r9_app[2])
    assert [r['record_id'] for r in entries] == [order[1], order[0]]


@pytest.mark.parametrize('status', ['确认正常', '确认异常'])
def test_blank_confirmation_does_not_write_and_can_be_corrected(r9_app, status):
    app = start(r9_app); navigate(app, '检查记录')
    app.selectbox(key='_status').select(status).run()
    app.text_area(key='_note').input('   ').run()
    app.button(key='save_review').click().run()
    assert not app.exception and history(r9_app[2]) == []
    assert app.session_state.ui.editor.status == status
    app.text_area(key='_note').input('核对依据').run()
    app.button(key='save_review').click().run()
    assert len(history(r9_app[2])) == 1


def test_history_read_outage_keeps_draft_and_conflict_baseline(r9_app, monkeypatch):
    from liushao import reviews as store
    import sqlite3
    app = start(r9_app); navigate(app, '检查记录')
    app.text_area(key='_note').input('断开前填写').run()
    rid = app.session_state.ui.selected
    original = store.history
    def unavailable(*args, **kwargs): raise sqlite3.OperationalError('injected outage')
    monkeypatch.setattr(store, 'history', unavailable)
    app.run()
    assert app.session_state.ui.editor.note == '断开前填写'
    assert app.button(key='save_review').disabled
    store.save_review(r9_app[2], rid, r9_app[4]['model_version'], '待补证', '其他窗口')
    monkeypatch.setattr(store, 'history', original)
    app.button(key='retry_reviews').click().run()
    assert app.session_state.ui.editor.baseline == 0
    app.button(key='save_review').click().run()
    assert app.session_state.ui.editor.conflict
    assert len(store.history(r9_app[2])) == 1


def test_model_fault_preserves_explicitly_restorable_draft(r9_app):
    app = start(r9_app); navigate(app, '检查记录')
    app.text_area(key='_note').input('故障前草稿').run()
    draft = app.session_state.ui.editor
    model = r9_app[1]/'model.joblib'; original = model.read_bytes()
    model.write_bytes(b'fault')
    app.run()
    assert app.session_state.ui.suspended_draft.note == '故障前草稿'
    assert app.button(key='restore_suspended').disabled
    model.write_bytes(original); navigate(app, '查看结果')
    app.button(key='run').click().run()
    app.button(key='restore_suspended').click().run()
    assert app.session_state.ui.editor.baseline == draft.baseline
    assert app.session_state.ui.editor.record_id == draft.record_id
    assert app.text_area(key='_note').value == '故障前草稿'


def test_rollback_failure_keeps_primary_error(monkeypatch):
    from liushao import reviews as store
    class Broken:
        def execute(self, *args): raise RuntimeError('primary execute failure')
        def rollback(self): raise RuntimeError('secondary rollback failure')
        def close(self): raise RuntimeError('secondary close failure')
    monkeypatch.setattr(store, 'connect', lambda path: Broken())
    with pytest.raises(RuntimeError, match='primary execute failure') as error:
        store.save_review('unused', 'record', 'model', '待补证', '')
    assert any('rollback' in note for note in error.value.__notes__)
    assert any('close' in note for note in error.value.__notes__)


@pytest.mark.parametrize('failure',['duplicate','mismatched_row','missing_column'])
def test_invalid_demo_mapping_cannot_be_treated_as_optional_quality_error(r9_app,failure):
    origins=r9_app[1]/'demo_origins.parquet';frame=pd.read_parquet(origins)
    if failure=='duplicate':frame.loc[1,'record_id']=frame.loc[0,'record_id']
    elif failure=='mismatched_row':frame.loc[0,'source_row']=9999
    else:frame=frame.drop(columns='source_sha256')
    frame.to_parquet(origins,index=False)
    path=r9_app[1]/'demo.json';metadata=json.loads(path.read_bytes())
    metadata['origins_sha256']=hashlib.sha256(origins.read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata),encoding='utf-8')
    app=AppTest.from_file(str(r9_app[0])).run();app.button(key='run').click().run()
    assert not app.exception and app.session_state.ui.result is None
    assert app.session_state.ui.run_error and r9_app[3].keys().isdisjoint({'export_predictions','export_history'})


def test_resource_cache_retains_at_most_two_model_identities(r9_app, monkeypatch):
    import copy
    import joblib
    from app_fixtures import write_bundle
    from app_ui_helpers import trust
    original_load = joblib.load
    calls = []
    def load(*args, **kwargs):
        calls.append(True)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(joblib, 'load', load)
    baseline = {n:(r9_app[1]/n).read_bytes() for n in ('model.joblib','model.json')}
    app = start(r9_app)
    for name in ('second-test-identity', 'third-test-identity'):
        bundle = copy.deepcopy(r9_app[4])
        bundle['config']['data_version'] = name
        bundle['model_version'] = hashlib.sha256(json.dumps(bundle['config'],sort_keys=True).encode()).hexdigest()[:16]
        write_bundle(r9_app[1],bundle); trust(r9_app[1],monkeypatch)
        app.run(); assert not app.exception
    for name, content in baseline.items(): (r9_app[1]/name).write_bytes(content)
    trust(r9_app[1],monkeypatch); app.run()
    assert not app.exception and len(calls) == 4


@pytest.mark.parametrize('rows',[50000,50001])
def test_input_row_limit_uses_actual_csv_records(r9_app, rows):
    from app_service import analyze_input
    content=b'a,b\n'+b'1,2\n'*rows
    if rows==50001:
        with pytest.raises(ValueError):
            analyze_input(content,'limit.csv','上传 CSV',r9_app[4],'test-model',r9_app[1])
    else:
        _,result,*_=analyze_input(content,'limit.csv','上传 CSV',r9_app[4],'test-model',r9_app[1])
        assert result.total==len(result.predictions)==50000 and result.predictions.record_id.is_unique


@pytest.mark.parametrize('extra',[0,1])
def test_byte_limit_rejects_before_parsing_only_above_50_mib(r9_app,monkeypatch,extra):
    import app_service
    class ReachedParser(Exception): pass
    def parser(*args,**kwargs): raise ReachedParser()
    monkeypatch.setattr(app_service,'read_csv_checked',parser)
    with pytest.raises(ReachedParser if extra==0 else ValueError):
        app_service.analyze_input(b'x'*(app_service.MAX_BYTES+extra),'limit.csv','上传 CSV',
                                  r9_app[4],'test-model',r9_app[1])


def test_quality_pagination_and_physical_line_locator_keep_selection(r9_app,monkeypatch):
    from app_ui_helpers import upload
    app=upload(r9_app,monkeypatch,b'a,b\n'+b',1\n'*120)
    navigate(app,'文件问题')
    assert len(app.selectbox(key='_quality_record').options)==50
    next(button for button in app.button if button.label=='下一页问题').click().run()
    assert app.session_state.ui.quality_page==1
    assert len(app.selectbox(key='_quality_record').options)==50
    app.text_input(key='_quality_line').input('121').run()
    assert len(app.selectbox(key='_quality_record').options)==1
    state=app.session_state.ui
    assert state.quality.records.loc[state.quality_selected].input_row==121
