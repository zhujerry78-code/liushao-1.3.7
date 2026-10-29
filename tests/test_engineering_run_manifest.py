"""Export provenance follows exact verified bytes and explicit snapshot preparation."""
import hashlib
import json
import shutil
import pytest
from app_fixtures import write_bundle
from app_ui_helpers import start,navigate,prepare,trust


def assert_hashes(downloads,artifacts):
    result=json.loads(downloads['export_manifest'])
    for key,name in [('model_sha256','model.joblib'),('descriptor_sha256','model.json')]:
        assert result[key]==hashlib.sha256((artifacts/name).read_bytes()).hexdigest()
    return result


@pytest.mark.parametrize('ending',['\n','\r\n'])
def test_manifest_hashes_exact_files_and_survives_filter_rerun(r9_app,monkeypatch,ending):
    path=r9_app[1]/'model.json';value=json.dumps(json.loads(path.read_bytes()),indent=2)
    path.write_bytes(value.replace('\n',ending).encode());trust(r9_app[1],monkeypatch)
    app=start(r9_app);prepare(app,r9_app[3]);before=assert_hashes(r9_app[3],r9_app[1])
    app.run();assert assert_hashes(r9_app[3],r9_app[1])==before


def test_hashes_follow_selected_model_directory_and_invalidate_previous_result(r9_app,monkeypatch):
    app=start(r9_app);first=assert_hashes(prepare(app,r9_app[3]),r9_app[1])
    external=r9_app[0].parent/'external';shutil.copytree(r9_app[1],external)
    descriptor=json.loads((external/'model.json').read_bytes());descriptor['note']='explicit test identity'
    (external/'model.json').write_text(json.dumps(descriptor),encoding='utf-8')
    monkeypatch.setenv('LIUSHAO_ARTIFACTS_DIR',str(external));trust(external,monkeypatch)
    app.run();assert app.session_state.ui.result is None and app.session_state.ui.exports is None
    navigate(app,'查看结果');app.button(key='run').click().run()
    second=assert_hashes(prepare(app,r9_app[3]),external)
    assert first['descriptor_sha256']!=second['descriptor_sha256']


def test_unready_model_cannot_export_previous_manifest(r9_app):
    app=start(r9_app);prepare(app,r9_app[3]);r9_app[3].clear()
    (r9_app[1]/'model.joblib').write_bytes(b'bad');app.run()
    assert app.session_state.ui.exports is None and 'export_manifest' not in r9_app[3]
