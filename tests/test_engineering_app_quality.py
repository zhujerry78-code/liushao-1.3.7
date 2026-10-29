"""Quality page association and selection regressions for the current UI."""
import hashlib
import io
import json
import pandas as pd
import pytest
from app_ui_helpers import start,navigate,upload,prepare

MIXED=b'a,b,Label\n1,2,PRIVATE_GROUND_TRUTH\n,3,Bot\nbad,,Bot\ninf,4,Benign\n'


def test_mixed_quality_feedback_preserves_fields_text_and_physical_rows(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,MIXED);quality=app.session_state.ui.quality
    assert quality.affected_rows==3 and quality.rejected_rows==1
    assert sorted(quality.records.input_row.tolist())==[3,4,5]
    assert app.session_state.ui.result.rejected.source_row.tolist()==[4]
    navigate(app,'文件问题');assert not app.exception
    assert 'PRIVATE_GROUND_TRUTH' not in str([d.value for d in app.dataframe])


def test_quality_details_keep_literal_tokens_without_new_reason_classification(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,b'a,b,Label\nN/A,3,Bot\nbad,,Bot\n')
    quality=app.session_state.ui.quality
    values=[]
    for rid in quality.records.index:values+=quality.details(rid)['CSV读取值'].tolist()
    assert 'N/A' in values and 'bad' in values and '' in values


def test_clean_input_has_zero_counts_and_no_problem_record_selector(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,b'a,b\n1,2\n3,4\n')
    assert app.session_state.ui.quality.affected_rows==0
    navigate(app,'文件问题')
    assert not app.selectbox and not app.exception


def test_all_rejected_quality_feedback_is_visible_before_empty_result_stop(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,b'a,b\nbad,inf\n,\n')
    assert len(app.session_state.ui.result.predictions)==0
    assert len(app.session_state.ui.result.rejected)==2
    navigate(app,'文件问题')
    assert app.selectbox(key='_quality_record')


def test_demo_quality_keeps_input_row_separate_from_original_source(r9_app):
    csv=r9_app[1]/'demo.csv';frame=pd.read_csv(csv);frame.loc[0,'a']=float('nan');frame.to_csv(csv,index=False)
    metadata=json.loads((r9_app[1]/'demo.json').read_bytes());metadata['csv_sha256']=hashlib.sha256(csv.read_bytes()).hexdigest()
    (r9_app[1]/'demo.json').write_text(json.dumps(metadata),encoding='utf-8')
    app=start(r9_app);record=app.session_state.ui.quality.records.iloc[0]
    assert record.input_row==2 and record.source_row==101
    assert app.session_state.ui.physical_rows.iloc[0]==2


def test_quality_selection_sort_filter_review_and_exports_keep_identity(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,MIXED);navigate(app,'文件问题')
    ids=app.session_state.ui.quality.records.index.tolist();chosen=ids[1]
    app.selectbox(key='_quality_record').select(chosen).run();app.run()
    assert app.session_state.ui.quality_selected==chosen
    row=app.session_state.ui.quality.records.loc[chosen]
    assert row.input_row==4
    exports=prepare(app,r9_app[3]);result=pd.read_csv(io.BytesIO(exports['export_predictions']))
    assert result.source_row.tolist()==[2,3,5]
    assert result.record_id.is_unique and set(result.record_id).isdisjoint({chosen})


def test_input_failure_retains_identified_old_quality_and_clean_input_recovers(r9_app,monkeypatch):
    app=upload(r9_app,monkeypatch,MIXED);old=app.session_state.ui.active.key
    app=upload(r9_app,monkeypatch,b'wrong\n1\n',app=app)
    assert app.session_state.ui.active.key==old and app.session_state.ui.run_error
    app=upload(r9_app,monkeypatch,b'a,b\n1,2\n',app=app)
    assert app.session_state.ui.quality.affected_rows==0 and not app.session_state.ui.run_error


@pytest.mark.parametrize('damage',['missing','replacement'])
def test_model_change_clears_old_quality_before_new_prediction(r9_app,monkeypatch,damage):
    app=upload(r9_app,monkeypatch,MIXED)
    model=r9_app[1]/'model.joblib'
    if damage=='missing':model.unlink()
    else:model.write_bytes(b'changed')
    app.run();assert app.session_state.ui.result is None and app.session_state.ui.quality is None


def test_quality_association_failure_keeps_predictions_and_exports(r9_app,monkeypatch):
    import app_service
    def fail(*a,**kw):raise ValueError('injected presentation failure')
    monkeypatch.setattr(app_service,'build_quality_view',fail)
    app=upload(r9_app,monkeypatch,MIXED)
    assert app.session_state.ui.quality is None and len(app.session_state.ui.result.predictions)==3
    assert len(pd.read_csv(io.BytesIO(prepare(app,r9_app[3])['export_predictions'])))==3
