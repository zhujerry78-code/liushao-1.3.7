"""Current application integration with isolated synthetic inputs."""
import io
import json
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from app_fixtures import engineering_bundle,isolated_app,write_bundle,flagged_app
from app_ui_helpers import start,navigate,prepare,edit,upload,messages
from app_ui_state import filtered
from liushao.reviews import history


@pytest.mark.parametrize('failure',['descriptor_missing','model_missing','bad_hash','corrupt'])
def test_app_stops_safely_when_model_artifacts_are_unusable(r9_app,failure):
    path=r9_app[1]/('model.json' if failure=='descriptor_missing' else 'model.joblib')
    if failure.endswith('missing'):path.unlink()
    else:path.write_bytes(b'not a model')
    app=AppTest.from_file(str(r9_app[0])).run()
    assert not app.exception and app.session_state.ui.result is None
    assert app.button(key='run').disabled and 'export_predictions' not in r9_app[3]


def test_app_reports_prediction_incompatibility_without_normal_results(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,b'b,Label\n2,Bot\n')
    assert app.session_state.ui.result is None
    assert '缺少必需特征' in app.session_state.ui.run_error
    assert not history(r9_app[2])


def test_app_review_filter_and_actual_export_payloads_keep_stable_associations(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'确认正常','核对来源行');app.button(key='save_review').click().run()
    values=prepare(app,r9_app[3]);frame=pd.read_csv(io.BytesIO(values['export_predictions']),keep_default_na=False)
    assert len(frame)==8 and frame.record_id.is_unique
    assert 'label' not in {s.casefold() for s in frame.columns}
    assert frame.set_index('record_id').loc[rid].review_note=='核对来源行'
    assert json.loads(values['export_history'])==history(r9_app[2])


@pytest.mark.parametrize('queue',['高异常得分','待核实','原始顺序'])
@pytest.mark.parametrize('disagreement,near',[(False,False),(True,False),(False,True),(True,True)])
def test_app_independent_flags_intersect_and_preserve_queue_order(engineered_app,queue,disagreement,near):
    app=start(engineered_app);navigate(app,'检查记录')
    app.selectbox(key='_filter_queue').select(queue).run()
    app.checkbox(key='_filter_disagreement').set_value(disagreement).run()
    app.checkbox(key='_filter_near').set_value(near).run()
    state=app.session_state.ui
    expected=filtered(state.result.predictions.assign(review_status='待复核'),state.filters,engineered_app[4])
    displayed=app.dataframe[0].value
    assert displayed['记录'].tolist()==expected.source_row.map(lambda n:f'第 {n} 行').tolist()
    assert state.filters.disagreement is disagreement and state.filters.near is near


def test_app_flag_status_score_filters_keep_reviews_and_full_exports_associated(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'确认正常','依据');app.button(key='save_review').click().run()
    app.selectbox(key='_filter_status').select('确认正常').run()
    app.slider(key='_filter_minimum').set_value(.5).run()
    assert len(app.dataframe[0].value)<=1
    frame=pd.read_csv(io.BytesIO(prepare(app,r9_app[3])['export_predictions']),keep_default_na=False)
    assert len(frame)==8 and frame.set_index('record_id').loc[rid].review_status=='确认正常'


def test_app_restores_same_records_review_after_empty_filter(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'待补证','连续草稿')
    app.selectbox(key='_filter_status').select('确认异常').run()
    app.button(key='pending_cancel').click().run()
    assert app.session_state.ui.selected==rid and app.text_area(key='_note').value=='连续草稿'


@pytest.mark.parametrize('status',['待复核','待补证'])
def test_app_nonconfirmation_status_can_be_saved_without_basis(r9_app,status):
    app=start(r9_app);navigate(app,'检查记录');edit(app,status,'')
    app.button(key='save_review').click().run()
    assert not app.exception and history(r9_app[2])[-1]['status']==status
