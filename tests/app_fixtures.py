"""Original app in an isolated directory, with small labelled engineering fixtures.

These are application integration checks, not external-data/model validation or browser E2E.
"""
import copy
import hashlib
import io
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from liushao.data import sha256
from liushao.model import fit_bundle
from liushao.reviews import current_reviews, history, save_review


@pytest.fixture(scope="module")
def engineering_bundle():
    features = pd.DataFrame({"a": [0., .1, .2, .3, 5., 6., 7., 8.], "b": [1.] * 8})
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bundle = fit_bundle(
        features, labels, features, labels, "engineering-fixture-not-an-experiment",
        n_estimators=8,
    )
    return bundle, features


def write_bundle(artifacts, bundle):
    joblib.dump(bundle, artifacts / "model.joblib")
    root = Path(__file__).resolve().parents[1]
    descriptor = {key: value for key, value in bundle.items() if key not in {"rf", "lr"}}
    descriptor["model_sha256"] = sha256(artifacts / "model.joblib")
    descriptor["source_code_sha256"] = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted((root / "liushao").glob("*.py"))
    }
    (artifacts / "model.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )


@pytest.fixture
def isolated_app(tmp_path, monkeypatch, engineering_bundle):
    """Copy app byte for byte; never use or overwrite the working app's artifacts/DB."""
    monkeypatch.delenv("LIUSHAO_ARTIFACTS_DIR", raising=False)
    source = Path(__file__).resolve().parents[1] / "app.py"
    app_path = tmp_path / "app.py"
    shutil.copyfile(source, app_path)
    assert source.read_bytes() == app_path.read_bytes()
    helper = source.parent / "app_model.py"
    shutil.copyfile(helper, tmp_path / helper.name)
    shutil.copyfile(source.parent / "app_quality.py", tmp_path / "app_quality.py")
    shutil.copyfile(source.parent / "requirements-lock.txt", tmp_path / "requirements-lock.txt")
    shutil.copyfile(source.parent / "VERSION", tmp_path / "VERSION")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle, features = engineering_bundle
    bundle = copy.deepcopy(bundle)
    write_bundle(artifacts, bundle)
    demo = features.assign(Label=["Benign"] * 4 + ["Bot"] * 4)
    demo.to_csv(artifacts / "demo.csv", index=False)
    origins = pd.DataFrame({
        "record_id": [f"engineering-source:{i}" for i in range(101, 109)],
        "source_file": ["engineering-fixture.csv"] * len(demo),
        "source_sha256": ["engineering-source"] * len(demo),
        "source_row": range(101, 109),
        "timestamp": [pd.NaT] * len(demo),
    })
    origins.to_parquet(artifacts / "demo_origins.parquet", index=False)
    (artifacts / "demo.json").write_text(json.dumps({
        "csv_sha256": sha256(artifacts / "demo.csv"),
        "origins_sha256": sha256(artifacts / "demo_origins.parquet"),
    }), encoding="utf-8")
    for name in ('app_record_tools.py','app_input.py','app_session.py','liushao_icon.svg','app_score_charts.py','app_evaluation.py','app_navigation.py','app_navigation_scroll.py','app_types.py','app_service.py','app_ui_state.py','app_ui_tools.py','app_triage.py'):
        shutil.copyfile(source.parent/name, tmp_path/name)
    monkeypatch.setenv('LIUSHAO_EXPECTED_MODEL_SHA256', sha256(artifacts/'model.joblib'))
    monkeypatch.setenv('LIUSHAO_EXPECTED_DESCRIPTOR_SHA256', sha256(artifacts/'model.json'))
    database = tmp_path / "isolated-reviews.sqlite"
    monkeypatch.setenv("LIUSHAO_REVIEW_DB", str(database))
    downloads = {}
    original_download = st.download_button
    def capture_download(label, data, *args, **kwargs):
        downloads[kwargs.get('key', label)] = data
        return original_download(label, data, *args, **kwargs)
    monkeypatch.setattr(st, 'download_button', capture_download)
    st.cache_resource.clear()
    yield app_path, artifacts, database, downloads, bundle
    st.cache_resource.clear()



@pytest.fixture
def r9_app(isolated_app):
    return isolated_app


@pytest.fixture
def engineered_app(r9_app,monkeypatch):
    """Engineered flag combinations test UI wiring, never model accuracy."""
    import app_service
    original=app_service.infer
    def flags(*args,**kwargs):
        predictions,features,rejected=original(*args,**kwargs)
        n=len(predictions)
        predictions['disagreement']=[bool(i%2) for i in range(n)]
        predictions['near_threshold']=[bool((i//2)%2) for i in range(n)]
        return predictions,features,rejected
    monkeypatch.setattr(app_service,'infer',flags)
    return r9_app


@pytest.fixture
def flagged_app(engineered_app):
    from app_ui_helpers import start
    return start(engineered_app),engineered_app[2],engineered_app[3],engineered_app[4]
