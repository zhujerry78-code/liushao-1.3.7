"""Current UI failures preserve successful data and saved review identity."""
import io
import sqlite3
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from app_ui_helpers import start,navigate,upload,edit,prepare,messages
from liushao import reviews as store


def test_missing_demo_has_clear_error_and_keeps_source_choice(r9_app):
    (r9_app[1]/'demo.csv').unlink();app=AppTest.from_file(str(r9_app[0])).run()
    assert not app.exception and app.session_state.ui.result is None
    assert '演示文件暂不可用' in messages(app) and not app.radio(key='_source').disabled


def test_missing_demo_does_not_prevent_uploaded_csv_prediction(r9_app,monkeypatch):
    content=(r9_app[1]/'demo.csv').read_bytes();(r9_app[1]/'demo.csv').unlink()
    app=upload(r9_app,monkeypatch,content)
    frame=app.session_state.ui.result.predictions
    assert len(frame)==8 and frame.source_row.tolist()==list(range(2,10))
    assert frame.source_file.eq('uploaded.csv').all()


def test_review_read_failure_preserves_prediction_and_recovers_saved_conclusion(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'确认正常','已有依据');app.button(key='save_review').click().run()
    previous=app.session_state.ui.result.predictions.copy();original=store.history
    def fail(*a,**kw):raise sqlite3.OperationalError('injected read failure')
    monkeypatch.setattr(store,'history',fail);app.run()
    pd.testing.assert_frame_equal(previous,app.session_state.ui.result.predictions)
    assert app.button(key='save_review').disabled
    monkeypatch.setattr(store,'history',original);app.button(key='retry_reviews').click().run()
    assert app.session_state.ui.selected==rid and app.text_area(key='_note').value=='已有依据'
    exports=prepare(app,r9_app[3]);frame=pd.read_csv(io.BytesIO(exports['export_predictions']),keep_default_na=False)
    assert frame.set_index('record_id').loc[rid].review_status=='确认正常'


@pytest.mark.parametrize('view',['all','pending','last_pending'])
def test_save_immediately_refreshes_queue_once_and_keeps_exports_consistent(r9_app,view):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    if view=='last_pending':
        for other in app.session_state.ui.result.predictions.record_id:
            if other!=rid:store.save_review(r9_app[2],other,r9_app[4]['model_version'],'确认正常','先前核对')
    if view!='all':app.selectbox(key='_filter_status').select('待复核').run()
    before=store.history(r9_app[2]);edit(app,'确认正常','新核对依据');app.button(key='save_review').click().run()
    assert len(store.history(r9_app[2]))==len(before)+1
    assert app.session_state.ui.selected==rid
    if view!='all':assert f'第 {rid.rsplit(":",1)[1]} 行' not in app.dataframe[0].value['记录'].tolist()
    values=prepare(app,r9_app[3]);frame=pd.read_csv(io.BytesIO(values['export_predictions']),keep_default_na=False)
    assert len(frame)==8 and frame.set_index('record_id').loc[rid].review_note=='新核对依据'
