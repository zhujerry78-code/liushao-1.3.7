"""AppTest process restart regression with synthetic data, not real-model/browser E2E."""
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from liushao.data import sha256
from liushao.model import fit_bundle
from liushao import reviews


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATIONS = ("确认异常", "确认正常")
FIRST_BASIS = '依据：核对来源行、原始特征与实验日志。\n保留中文、逗号,和"引号"。'
SECOND_BASIS = '修订依据：补充上下文后重新确认，保留第一次判断及其依据。'


@pytest.fixture(scope="module")
def restart_bundle():
    features = pd.DataFrame({"a": [0., .1, .2, .3, 5., 6., 7., 8.], "b": [1.] * 8})
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bundle = fit_bundle(
        features, labels, features, labels, "review-restart-fixture-not-an-experiment",
        n_estimators=8,
    )
    return bundle, features


def _prepare_workspace(workspace, restart_bundle):
    """Copy current app/package so subprocesses cannot fall back to another checkout."""
    workspace.mkdir()
    shutil.copyfile(ROOT / "app.py", workspace / "app.py")
    shutil.copyfile(ROOT / "app_model.py", workspace / "app_model.py")
    shutil.copyfile(ROOT / "app_quality.py", workspace / "app_quality.py")
    shutil.copyfile(ROOT / "requirements-lock.txt", workspace / "requirements-lock.txt")
    shutil.copytree(ROOT / "liushao", workspace / "liushao", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc",
    ))
    for name in ('app_types.py','app_service.py','app_ui_state.py','app_ui_tools.py','app_triage.py',
                 'app_record_tools.py','app_input.py','app_session.py','app_desktop_ready.py','VERSION',
                 'app_evidence.py','app_evidence_ui.py','app_review_guidance.py','app_evidence_rules.py','app_decomposition.py','app_quick_search.py',
                 'liushao_icon.svg','app_navigation.py','app_navigation_scroll.py','app_score_charts.py','app_evaluation.py',
                 'delivery/safe_files.py','delivery/evaluation_resources.py'):
        (workspace/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name,workspace/name)
    artifacts = workspace / "artifacts"
    artifacts.mkdir()
    bundle, features = restart_bundle
    joblib.dump(bundle, artifacts / "model.joblib")
    descriptor = {key: value for key, value in bundle.items() if key not in {"rf", "lr"}}
    descriptor["model_sha256"] = sha256(artifacts / "model.joblib")
    descriptor["source_code_sha256"] = {
        path.relative_to(workspace).as_posix(): sha256(path)
        for path in sorted((workspace / "liushao").glob("*.py"))
    }
    (artifacts / "model.json").write_text(json.dumps(descriptor), encoding="utf-8")
    features.assign(Label=["Benign"] * 4 + ["Bot"] * 4).to_csv(
        artifacts / "demo.csv", index=False,
    )
    pd.DataFrame({
        "record_id": [f"restart-source:{row}" for row in range(101, 109)],
        "source_file": ["synthetic-restart.csv"] * len(features),
        "source_sha256": ["restart-source"] * len(features),
        "source_row": range(101, 109),
        "timestamp": [pd.NaT] * len(features),
    }).to_parquet(artifacts / "demo_origins.parquet", index=False)
    (artifacts / "demo.json").write_text(json.dumps({
        "csv_sha256": sha256(artifacts / "demo.csv"),
        "origins_sha256": sha256(artifacts / "demo_origins.parquet"),
    }), encoding="utf-8")


def _save(app, status, note):
    app.selectbox(key="_status").select(status)
    app.text_area(key="_note").input(note)
    app.button(key="save_review").click().run()
    assert not app.exception


def _assert_exports(app, downloads, database, chosen, model_version, status, basis):
    _prepare_exports(app, downloads)
    """Inspect the actual bytes supplied to the application's download buttons."""
    csv_payload = downloads["export_predictions"]
    json_payload = downloads["export_history"]
    assert isinstance(csv_payload, bytes) and isinstance(json_payload, bytes)
    assert csv_payload.startswith(b"\xef\xbb\xbf")
    exported = pd.read_csv(io.BytesIO(csv_payload), keep_default_na=False)
    audit = json.loads(json_payload)
    assert len(exported) == 8 and exported.record_id.is_unique
    assert not {"Label", "label", "ground_truth"} & set(exported.columns)
    assert exported.model_version.eq(model_version).all()
    selected = exported.set_index("record_id").loc[chosen]
    assert selected.review_status == status
    assert selected.review_note == basis
    assert exported.loc[exported.record_id.ne(chosen), "review_status"].eq("待复核").all()
    assert audit == reviews.history(database, chosen, model_version)
    assert {row["record_id"] for row in audit} == {chosen}
    assert {row["model_version"] for row in audit} == {model_version}
    assert audit[-1]["status"] == status and audit[-1]["note"] == basis
    assert reviews.current_reviews(database, model_version)[chosen] == audit[-1]
    return exported, audit


def _prepare_exports(app, downloads):
    app.button(key='nav_下载结果').click().run()
    app.button(key='prepare_exports').click().run()
    assert not app.exception and 'export_predictions' in downloads
    app.button(key='resume_review_edit').click().run()


def _reject_without_changes(app, downloads, database, status):
    old_history = reviews.history(database)
    old_csv, old_json = downloads['export_predictions'], downloads['export_history']
    _save(app, status, ' \t\n\u3000 ')
    assert any('必须填写判断依据' in error.value for error in app.error)
    assert reviews.history(database) == old_history
    assert app.session_state.ui.exports is None
    app.button(key='nav_下载结果').click().run()
    assert app.button(key='prepare_exports').disabled
    app.button(key='return_editor').click().run()
    app.button(key='reload_review').click().run()
    _prepare_exports(app, downloads)
    assert downloads['export_predictions'] == old_csv
    assert downloads['export_history'] == old_json


def _run_child(workspace, phase, first_status):
    assert Path(reviews.__file__).resolve().parent == workspace / "liushao"
    assert Path(os.environ["LIUSHAO_REVIEW_DB"]) == workspace / "reviews.sqlite"
    database = workspace / "reviews.sqlite"
    downloads = {}
    original_download = st.download_button

    def capture_download(label, data, *args, **kwargs):
        downloads[kwargs.get("key", label)] = data
        return original_download(label, data, *args, **kwargs)

    os.environ['LIUSHAO_EXPECTED_MODEL_SHA256'] = sha256(workspace/'artifacts/model.joblib')
    os.environ['LIUSHAO_EXPECTED_DESCRIPTOR_SHA256'] = sha256(workspace/'artifacts/model.json')
    other_status = next(value for value in CONFIRMATIONS if value != first_status)
    state_path = workspace / "first-state.json"
    with patch.object(st, "download_button", capture_download):
        app = AppTest.from_file(str(workspace / "app.py"), default_timeout=30).run()
        app.button(key="run").click().run()
        assert not app.exception
        model_version = joblib.load(workspace / "artifacts/model.joblib")["model_version"]
        app.button(key='nav_检查记录').click().run()
        _prepare_exports(app, downloads)
        initial_export = pd.read_csv(io.BytesIO(downloads["export_predictions"]),
                                     keep_default_na=False)
        if phase == "write":
            chosen = "restart-source:102"
            app.session_state.ui.selected=chosen
            app.session_state.ui.editor=None
            app.run()
            assert not reviews.history(database)
            _save(app, first_status, FIRST_BASIS)
            assert not app.error
            _, audit = _assert_exports(
                app, downloads, database, chosen, model_version, first_status, FIRST_BASIS,
            )
            assert len(audit) == 1
            _reject_without_changes(app, downloads, database, other_status)
            (workspace / "first-predictions.csv").write_bytes(downloads["export_predictions"])
            (workspace / "first-history.json").write_bytes(downloads["export_history"])
            state_path.write_text(json.dumps({
                "record_id": chosen, "model_version": model_version, "pid": os.getpid(),
            }), encoding="utf-8")
        else:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            chosen = state["record_id"]
            assert model_version == state["model_version"]
            app.session_state.ui.selected=chosen
            app.session_state.ui.editor=None
            app.run()
            assert app.selectbox(key="_status").value == first_status
            assert app.text_area(key="_note").value == FIRST_BASIS
            assert downloads["export_predictions"] == (workspace / "first-predictions.csv").read_bytes()
            assert downloads["export_history"] == (workspace / "first-history.json").read_bytes()
            _, first_audit = _assert_exports(
                app, downloads, database, chosen, model_version, first_status, FIRST_BASIS,
            )
            _save(app, other_status, SECOND_BASIS)
            assert not app.error
            _, audit = _assert_exports(
                app, downloads, database, chosen, model_version, other_status, SECOND_BASIS,
            )
            assert len(audit) == 2 and audit[:-1] == first_audit
            assert audit[-1]["id"] > first_audit[-1]["id"]
            _reject_without_changes(app, downloads, database, first_status)
            _assert_exports(app, downloads, database, chosen, model_version, other_status, SECOND_BASIS)
            (workspace / "final-predictions.csv").write_bytes(downloads["export_predictions"])
            (workspace / "final-history.json").write_bytes(downloads["export_history"])

        exported = pd.read_csv(io.BytesIO(downloads["export_predictions"]), keep_default_na=False)
        prediction_columns = [column for column in initial_export if not column.startswith("review_")]
        pd.testing.assert_frame_equal(initial_export[prediction_columns], exported[prediction_columns])
        print(json.dumps({
            "phase": phase, "pid": os.getpid(), "record_id": chosen,
            "history_rows": len(reviews.history(database)), "model_version": model_version,
            "app_sha256": sha256(workspace / "app.py"),
            "reviews_sha256": sha256(workspace / "liushao/reviews.py"),
        }))


@pytest.mark.parametrize("first_status", CONFIRMATIONS)
def test_confirmation_rejection_exports_and_history_survive_app_process_restart(
    tmp_path, restart_bundle, first_status,
):
    workspace = tmp_path / "app-process-workspace"
    _prepare_workspace(workspace, restart_bundle)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(workspace),
           "LIUSHAO_ARTIFACTS_DIR": str(workspace / "artifacts"),
           "LIUSHAO_REVIEW_DB": str(workspace / "reviews.sqlite")}
    results = []
    for phase in ("write", "restart"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(workspace),
             phase, first_status],
            cwd=workspace, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90,
        )
        (workspace / f"{phase}-stdout.log").write_text(completed.stdout, encoding="utf-8")
        (workspace / f"{phase}-stderr.log").write_text(completed.stderr, encoding="utf-8")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results.append(json.loads(completed.stdout))
    assert results[0]["phase"] == "write" and results[0]["history_rows"] == 1
    assert results[1]["phase"] == "restart" and results[1]["history_rows"] == 2
    for key in ("record_id", "model_version", "app_sha256", "reviews_sha256"):
        assert results[0][key] == results[1][key]


if __name__ == "__main__":
    assert len(sys.argv) == 5 and sys.argv[1] == "--child"
    _run_child(Path(sys.argv[2]).resolve(), sys.argv[3], sys.argv[4])
