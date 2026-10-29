"""Experimental tools: statistics, identity and read-only behavior, not efficacy."""
import io
import json
import numpy as np
import pandas as pd
import pytest
from app_ui_helpers import start,navigate
from app_triage import diagnostics,diagnostic_view,batch_reference,feature_clues
from liushao.data import read_csv_checked
from liushao.model import infer
from liushao.reviews import history


@pytest.fixture
def tool_input(r9_app):
    raw=read_csv_checked(io.BytesIO((r9_app[1]/'demo.csv').read_bytes()))
    predictions,features,_=infer(raw,r9_app[4],'tool-fixture','synthetic.csv')
    return predictions,features,r9_app[4]


@pytest.mark.parametrize('chunk_size',[1,3,512])
def test_tree_dispersion_matches_population_standard_deviation(tool_input,chunk_size):
    p,x,bundle=tool_input;before=p.copy(deep=True);original=x.copy(deep=True)
    result=diagnostics(p,x,bundle,chunk_size=chunk_size)
    rf=bundle['rf'];values=rf.named_steps['impute'].transform(x)
    probabilities=np.array([tree.predict_proba(values)[:,1] for tree in rf.named_steps['classifier'].estimators_])
    np.testing.assert_allclose(result.rf_tree_probability_std,np.std(probabilities,axis=0,ddof=0),atol=1e-14,rtol=0)
    assert result.record_id.tolist()==p.record_id.tolist()
    pd.testing.assert_frame_equal(before,p);pd.testing.assert_frame_equal(original,x)


def test_diagnostic_view_rejects_mismatched_identity(tool_input):
    p,x,bundle=tool_input;extra=diagnostics(p,x,bundle)
    extra.loc[0,'record_id']='another input'
    with pytest.raises(ValueError):diagnostic_view(p,extra,'lr_tie_break')


def test_batch_groups_ignore_manual_reviews_and_keep_counts(tool_input):
    p,x,bundle=tool_input;original=p.copy(deep=True)
    first=batch_reference(p,x,bundle)
    changed=p.assign(review_status='确认正常',review_note='should not affect grouping')
    second=batch_reference(changed,x,bundle)
    pd.testing.assert_frame_equal(first['groups'],second['groups'])
    assert sum(first['summary']['triage_counts'].values())==len(p)
    assert first['groups'].record_id.tolist()==p.record_id.tolist()
    pd.testing.assert_frame_equal(original,p)


def test_feature_trials_are_independent_and_leave_input_unchanged(tool_input):
    p,x,bundle=tool_input;before=x.copy(deep=True)
    result=feature_clues(x.iloc[0],bundle)
    assert len(result['top3'])==min(3,len(x.columns))
    for item in result['perturbations'].itertuples():
        if item.status.startswith('已试算'):
            assert item.rf_delta==pytest.approx(item.rf_after-item.rf_before)
            assert item.lr_delta==pytest.approx(item.lr_after-item.lr_before)
    assert abs(result['summary']['lr_reconstruction_error']) < 1e-10
    pd.testing.assert_frame_equal(before,x)


def test_tool_generation_and_downloads_do_not_modify_predictions_or_reviews(r9_app):
    app=start(r9_app);before=app.session_state.ui.result.predictions.copy(deep=True)
    navigate(app,'辅助检查');app.button(key='build_diagnostics').click().run()
    assert not app.exception and 'export_diagnostics' in r9_app[3]
    payload=r9_app[3]['export_diagnostics'];app.run();assert r9_app[3]['export_diagnostics']==payload
    navigate(app,'整批数据分析');app.button(key='batch_reference').click().run()
    assert not app.exception and 'export_batch_groups' in r9_app[3]
    navigate(app,'检查记录')
    app.button(key='open_evidence_detail').click().run()
    assert not app.exception and app.session_state.ui.page == 'evidence_detail'
    app.button(key='feature_clues').click().run()
    assert not app.exception and 'export_feature_clues' in r9_app[3]
    json.loads(r9_app[3]['export_feature_clues'])
    pd.testing.assert_frame_equal(before,app.session_state.ui.result.predictions)
    assert history(r9_app[2])==[]
