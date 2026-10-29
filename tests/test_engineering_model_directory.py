"""Configured source-model directory never silently falls back to another model."""
import shutil
from app_ui_helpers import start,navigate,trust


def test_external_model_directory_and_explicit_return_to_default(r9_app,monkeypatch):
    external=r9_app[0].parent/'selected model';shutil.copytree(r9_app[1],external)
    monkeypatch.setenv('LIUSHAO_ARTIFACTS_DIR',str(external));trust(external,monkeypatch)
    app=start(r9_app);assert len(app.session_state.ui.result.predictions)==8
    monkeypatch.delenv('LIUSHAO_ARTIFACTS_DIR');app.run()
    assert len(app.session_state.ui.result.predictions)==8


def test_missing_selected_model_stops_even_when_default_is_usable(r9_app,monkeypatch):
    app=start(r9_app);monkeypatch.setenv('LIUSHAO_ARTIFACTS_DIR',str(r9_app[0].parent/'missing'))
    app.run();assert app.session_state.ui.result is None
