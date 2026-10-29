"""Separate model versions restore only their own persisted reviews."""
import copy
import hashlib
import json
from app_fixtures import write_bundle
from app_ui_helpers import start,navigate,edit,trust
from liushao.reviews import history


def test_direct_prediction_after_model_switch_reloads_matching_review(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'确认正常','模型 A 的依据');app.button(key='save_review').click().run()
    original={n:(r9_app[1]/n).read_bytes() for n in ('model.joblib','model.json')}
    other=copy.deepcopy(r9_app[4]);other['config']['near_margin']+=.01
    other['model_version']=hashlib.sha256(json.dumps(other['config'],sort_keys=True).encode()).hexdigest()[:16]
    write_bundle(r9_app[1],other);trust(r9_app[1],monkeypatch);app.run()
    assert app.session_state.ui.result is None
    navigate(app,'查看结果');app.button(key='run').click().run();navigate(app,'检查记录')
    assert app.text_area(key='_note').value==''
    edit(app,'待补证','模型 B 的依据');app.button(key='save_review').click().run()
    for name,content in original.items():(r9_app[1]/name).write_bytes(content)
    trust(r9_app[1],monkeypatch);app.run();navigate(app,'查看结果');app.button(key='run').click().run();navigate(app,'检查记录')
    assert app.text_area(key='_note').value=='模型 A 的依据'
    assert len(history(r9_app[2]))==2
