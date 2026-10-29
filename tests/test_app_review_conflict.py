"""One history snapshot, conflict baseline, explicit reload and pending actions."""
import io
import json
import pandas as pd
import pytest
from app_ui_helpers import start,navigate,prepare,edit,messages
from liushao import reviews as store


def test_one_history_snapshot_drives_current_csv_json_and_progress(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    store.save_review(r9_app[2],rid,r9_app[4]['model_version'],'确认正常','依据')
    original=store.history;calls=[]
    def read(*a,**kw):calls.append(True);return original(*a,**kw)
    monkeypatch.setattr(store,'history',read);app.run();assert len(calls)==1
    values=prepare(app,r9_app[3]);audit=json.loads(values['export_history']);meta=json.loads(values['export_manifest'])
    frame=pd.read_csv(io.BytesIO(values['export_predictions']),keep_default_na=False)
    assert meta['review_history_entries']==len(audit)==1
    assert frame.set_index('record_id').loc[rid].review_note==audit[0]['note']=='依据'
    app.button(key='resume_review_edit').click().run();navigate(app,'查看结果')
    assert '已确认 1 / 8' in app.get('progress')[0].proto.text


@pytest.mark.parametrize('trigger',['save_review','save_review_next'])
@pytest.mark.parametrize('dirty_first',[False,True])
def test_stale_editor_preserves_draft_until_explicit_reload_and_can_save_again(r9_app,trigger,dirty_first):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    if dirty_first:edit(app,'待补证','本窗口草稿')
    store.save_review(r9_app[2],rid,r9_app[4]['model_version'],'确认正常','另一个窗口依据')
    if not dirty_first:edit(app,'待补证','本窗口草稿')
    app.button(key=trigger).click().run()
    assert app.session_state.ui.editor.note=='本窗口草稿' and app.session_state.ui.editor.baseline==0
    assert len(store.history(r9_app[2]))==1 and app.session_state.ui.selected==rid
    app.button(key='reload_review').click().run()
    assert app.text_area(key='_note').value=='另一个窗口依据'
    edit(app,'待补证','读取后修订');app.button(key='save_review').click().run()
    assert len(store.history(r9_app[2]))==2 and store.history(r9_app[2])[-1]['record_id']==rid


@pytest.mark.parametrize('failure',['demo.json','demo_origins.parquet','hash'])
def test_bad_demo_handoff_guides_restore_without_retraining(r9_app,failure):
    if failure=='hash':(r9_app[1]/'demo.csv').write_bytes(b'a,b\n1,2\n')
    else:(r9_app[1]/failure).write_bytes(b'broken')
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(r9_app[0])).run();app.button(key='run').click().run()
    assert not app.exception and app.session_state.ui.result is None
    assert app.session_state.ui.run_error and 'export_predictions' not in r9_app[3]


def test_conflicting_filtered_target_does_not_save_another_record_or_drop_draft(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'待补证','草稿');app.selectbox(key='_filter_scope').select('异常提示').run()
    pending=app.session_state.ui.pending
    store.save_review(r9_app[2],rid,r9_app[4]['model_version'],'待补证','外部写入')
    app.button(key='pending_save').click().run()
    assert app.session_state.ui.pending==pending and app.session_state.ui.editor.record_id==rid
    assert len(store.history(r9_app[2]))==1


def test_pending_cancel_and_discard_do_not_write_unrelated_records(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'待补证','草稿');app.selectbox(key='_filter_status').select('确认异常').run()
    app.button(key='pending_cancel').click().run()
    assert app.session_state.ui.selected==rid and app.session_state.ui.editor.note=='草稿'
    app.selectbox(key='_filter_status').select('确认异常').run();app.button(key='pending_discard').click().run()
    assert app.session_state.ui.selected==rid and store.history(r9_app[2])==[]
