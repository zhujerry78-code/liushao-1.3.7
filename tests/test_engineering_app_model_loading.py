"""Current UI model boundaries; test identities are injected explicitly."""
import hashlib
import json
import numpy as np
import pytest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest
import app_model
from app_fixtures import write_bundle
from app_ui_helpers import start, trust, navigate, messages


def _descriptor(artifacts):
    return json.loads((artifacts/'model.json').read_bytes())


def _assert_stopped(app, downloads):
    assert not app.exception
    assert app.session_state.ui.result is None
    assert app.session_state.ui.exports is None
    assert 'export_predictions' not in downloads
    navigate(app,'关于流哨')
    assert '分析程序未就绪' in messages(app)


@pytest.mark.parametrize('failure',['sklearn','algorithm_version','schema_version'])
def test_descriptor_preflight_rejects_before_deserializing(r9_app,monkeypatch,failure):
    descriptor=_descriptor(r9_app[1])
    if failure=='schema_version': descriptor[failure]='unagreed'
    else:
        descriptor['config'][failure]='unagreed'
        descriptor['model_version']=hashlib.sha256(json.dumps(descriptor['config'],sort_keys=True).encode()).hexdigest()[:16]
    (r9_app[1]/'model.json').write_text(json.dumps(descriptor),encoding='utf-8')
    trust(r9_app[1],monkeypatch)
    called=[]
    monkeypatch.setattr(joblib,'load',lambda *a,**kw:called.append(True))
    app=AppTest.from_file(str(r9_app[0])).run()
    _assert_stopped(app,r9_app[3]);assert not called


@pytest.mark.parametrize('model_name',['rf','lr'])
@pytest.mark.parametrize('failure',['feature_order','classifier_width'])
def test_actual_pipeline_incompatibility_stops_before_prediction(r9_app,monkeypatch,model_name,failure):
    pipeline=r9_app[4][model_name]
    if failure=='feature_order': pipeline.named_steps['impute'].feature_names_in_=pipeline.feature_names_in_[::-1]
    else: pipeline.named_steps['classifier'].n_features_in_=999
    write_bundle(r9_app[1],r9_app[4]);trust(r9_app[1],monkeypatch)
    app=AppTest.from_file(str(r9_app[0])).run();_assert_stopped(app,r9_app[3])


@pytest.mark.parametrize('failure',['model_bytes','descriptor','source_hash','both_files'])
def test_hot_cache_rechecks_files_and_discards_previous_results(r9_app,monkeypatch,failure):
    app=start(r9_app);r9_app[3].clear()
    if failure in ('model_bytes','both_files'):
        path=r9_app[1]/'model.joblib';content=bytearray(path.read_bytes());content[len(content)//2]^=1
        path.write_bytes(content)
    if failure in ('descriptor','source_hash','both_files'):
        descriptor=_descriptor(r9_app[1])
        if failure=='descriptor':descriptor['handoff_note']='changed'
        elif failure=='source_hash':descriptor['source_code_sha256']['liushao/model.py']='0'*64
        else:descriptor['model_sha256']=hashlib.sha256((r9_app[1]/'model.joblib').read_bytes()).hexdigest()
        (r9_app[1]/'model.json').write_text(json.dumps(descriptor),encoding='utf-8')
        if failure=='source_hash':trust(r9_app[1],monkeypatch)
    called=[];monkeypatch.setattr(joblib,'load',lambda *a,**kw:called.append(True))
    app.run();_assert_stopped(app,r9_app[3]);assert not called


def test_same_version_changed_model_bytes_require_new_prediction(r9_app,monkeypatch):
    app=start(r9_app);before=app.session_state.ui.result.predictions.copy()
    for tree in r9_app[4]['rf'].named_steps['classifier'].estimators_:
        tree.tree_.value[:]=[[[.25,.75]]]
    write_bundle(r9_app[1],r9_app[4]);trust(r9_app[1],monkeypatch)
    app.run();assert app.session_state.ui.result is None
    app.button(key='run').click().run()
    after=app.session_state.ui.result.predictions
    assert after.model_version.tolist()==before.model_version.tolist()
    assert np.allclose(after.rf_score,.75) and not np.allclose(before.rf_score,after.rf_score)


def test_hot_cache_rechecks_inference_dependency_versions(r9_app,monkeypatch):
    app=start(r9_app);r9_app[3].clear();original=app_model._dependency_version
    monkeypatch.setattr(app_model,'_dependency_version',lambda name,scope:'0.0' if name=='scipy' else original(name,scope))
    app.run();_assert_stopped(app,r9_app[3]);assert 'scipy' in messages(app)


def test_changed_valid_descriptor_discards_existing_results(r9_app,monkeypatch):
    app=start(r9_app);descriptor=_descriptor(r9_app[1]);descriptor['handoff_note']='trusted test revision'
    (r9_app[1]/'model.json').write_text(json.dumps(descriptor),encoding='utf-8');trust(r9_app[1],monkeypatch)
    app.run();assert app.session_state.ui.result is None


@pytest.mark.parametrize('model_name',['rf','lr'])
@pytest.mark.parametrize('failure',['nan','inf','negative','over_one','row_sum','one_column','wrong_rows','one_dimension','nonnumeric','complex'])
def test_prediction_output_failure_never_becomes_normal_result(r9_app,monkeypatch,model_name,failure):
    app=start(r9_app);previous=app.session_state.ui.result.predictions.copy();original=Pipeline.predict_proba
    def invalid(self,values,*a,**kw):
        name='rf' if isinstance(self.named_steps['classifier'],RandomForestClassifier) else 'lr'
        if name!=model_name:return original(self,values,*a,**kw)
        rows=len(values)
        if failure=='one_column':return np.ones((rows,1))
        if failure=='wrong_rows':return np.ones((rows-1,2))*.5
        if failure=='one_dimension':return np.ones(rows)*.5
        if failure=='nonnumeric':return np.full((rows,2),'invalid')
        pair={'nan':[np.nan,np.nan],'inf':[0,np.inf],'negative':[-.1,1.1],
              'over_one':[1.1,-.1],'row_sum':[.1,.2],'complex':[.5+.1j,.5-.1j]}[failure]
        return np.tile(pair,(rows,1))
    monkeypatch.setattr(Pipeline,'predict_proba',invalid)
    app.button(key='change_file').click().run();app.button(key='run').click().run()
    assert not app.exception and app.session_state.ui.run_error
    assert '输出' in app.session_state.ui.run_error
    # Failed new analysis retains the explicitly identified previous successful result.
    import pandas as pd
    pd.testing.assert_frame_equal(previous,app.session_state.ui.result.predictions)
