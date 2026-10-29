"""Dependency discovery must survive an overlapping Streamlit runner exiting."""
import importlib.metadata as metadata
from pathlib import Path
import sys
import threading

import pytest
import sklearn
from streamlit.runtime.scriptrunner.script_runner import modified_sys_path
from streamlit.testing.v1 import AppTest

import app_model
from app_fixtures import engineering_bundle, isolated_app
from test_engineering_app_model_loading import _assert_stopped, _descriptor


@pytest.mark.parametrize("dependency", ["numpy", "scikit-learn"])
def test_runtime_path_snapshot_survives_old_runner_exit(
    isolated_app, tmp_path, monkeypatch, dependency,
):
    _, artifacts, _, _, _ = isolated_app
    descriptor = _descriptor(artifacts)
    site = str(Path(sklearn.__file__).resolve().parent.parent)
    sentinel = str(tmp_path / "metadata-search-gate")
    reached = threading.Event()
    released = threading.Event()
    original_search = metadata.FastPath.search
    context = modified_sys_path(str(tmp_path / "old-runner.py"))
    contexts, failures = [], []

    def gated_search(path, prepared):
        if path.root == sentinel and prepared.normalized == metadata.Prepared(dependency).normalized:
            reached.set()
            if not released.wait(10):
                raise AssertionError("old runner did not exit before metadata lookup resumed")
        return original_search(path, prepared)

    def check_runtime():
        try:
            contexts.append(app_model._runtime_context(descriptor))
        except BaseException as error:
            failures.append(error)

    # Metadata bytes and installed packages are real and never changed. The two
    # paths make the iteration race deterministic without a second fallback site.
    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", [sentinel, site])
        patch.setattr(metadata.FastPath, "search", gated_search)
        context.__enter__()
        worker = threading.Thread(target=check_runtime, daemon=True)
        worker.start()
        try:
            assert reached.wait(10), "runtime did not reach dependency discovery"
            context.__exit__(None, None, None)
            released.set()
            worker.join(10)
            assert not worker.is_alive(), "runtime dependency check did not finish"
        finally:
            released.set()
            worker.join(10)
            context.__exit__(None, None, None)
            assert not worker.is_alive(), "metadata test worker was not cleaned up"
    assert not failures, failures
    assert len(contexts) == 1
    assert contexts[0]["dependencies"]["scikit-learn"] == sklearn.__version__


def test_dependency_metadata_search_does_not_fall_back_to_global_path(tmp_path):
    # sklearn is installed globally in this process, but absent in these scopes.
    for search_path in ((), (str(tmp_path),)):
        with pytest.raises(app_model.CompatibilityError, match="元数据: scikit-learn"):
            app_model._dependency_version("scikit-learn", search_path)


@pytest.mark.parametrize("damage", ["missing", "missing-version", "incompatible-version"])
def test_cached_model_rejects_changed_metadata_and_recovers(
    r9_app, tmp_path, monkeypatch, damage,
):
    from app_ui_helpers import start, navigate, prepare, messages
    app_path, _, database, downloads, _ = r9_app
    site = tmp_path / "isolated-metadata"
    installed_version = metadata.version("scipy")
    info = site / f"scipy-{installed_version}.dist-info"
    info.mkdir(parents=True)
    metadata_file = info / "METADATA"
    valid = f"Metadata-Version: 2.1\nName: scipy\nVersion: {installed_version}\n"
    metadata_file.write_text(valid, encoding="utf-8")
    original = app_model._dependency_version

    def scoped_version(name, search_path):
        return original(name, (str(site),) if name == "scipy" else search_path)

    monkeypatch.setattr(app_model, "_dependency_version", scoped_version)
    app = start(r9_app)
    database_before = database.read_bytes()
    downloads.clear()
    if damage == 'missing':
        metadata_file.unlink()
    else:
        damaged = 'Metadata-Version: 2.1\nName: scipy\n'
        if damage == 'incompatible-version': damaged += 'Version: 0.0.0-incompatible\n'
        metadata_file.write_text(damaged,encoding='utf-8')
    app.run(); _assert_stopped(app, downloads)
    assert 'scipy' in messages(app)
    assert database.read_bytes() == database_before
    metadata_file.write_text(valid,encoding='utf-8')
    navigate(app,'查看结果')
    assert app.session_state.ui.result is None
    app.button(key='run').click().run(); prepare(app,downloads)
    assert 'export_predictions' in downloads
    assert database.read_bytes() == database_before
