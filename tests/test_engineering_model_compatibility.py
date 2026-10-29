"""Model-package compatibility regressions; only tiny synthetic fitted fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from liushao.data import sha256
from liushao.model import fit_bundle
from scripts import engineering_model_compatibility as compatibility


@pytest.fixture(scope="module")
def synthetic_bundle():
    """Fit once, never read or train on the project's actual experiment data."""
    values = pd.DataFrame({
        "Flow Duration": [0., 0.1, 0.2, 0.3, 5., 6., 7., 8.],
        "Tot Fwd Pkts": [1., 2., 1., 2., 4., 5., 6., 7.],
    })
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return fit_bundle(values, labels, values, labels, "a" * 64, n_estimators=8)


def descriptor_for(bundle, model_sha256="b" * 64):
    descriptor = copy.deepcopy({key: value for key, value in bundle.items()
                                if key not in {"rf", "lr"}})
    descriptor["model_sha256"] = model_sha256
    root = Path(__file__).resolve().parents[1]
    descriptor["source_code_sha256"] = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted((root / "liushao").glob("*.py"))
    }
    return descriptor


def refresh_version(descriptor):
    descriptor["model_version"] = hashlib.sha256(
        json.dumps(descriptor["config"], sort_keys=True).encode()
    ).hexdigest()[:16]


def write_package(tmp_path, synthetic_bundle):
    model = tmp_path / "model.joblib"
    descriptor_path = tmp_path / "model.json"
    joblib.dump(synthetic_bundle, model, compress=3)
    descriptor = descriptor_for(synthetic_bundle, sha256(model))
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    return model, descriptor_path, descriptor


def test_accepts_existing_train_descriptor_and_fitted_pipeline_format(synthetic_bundle):
    descriptor = descriptor_for(synthetic_bundle)
    compatibility.validate_descriptor(descriptor)
    compatibility.validate_bundle(synthetic_bundle, descriptor)


@pytest.mark.parametrize("features", [
    [], ["Flow Duration", "Flow Duration"], ["Label"], ["Timestamp"],
    ["record_id"], ["ground_truth"],
])
def test_rejects_empty_duplicate_or_metadata_features(synthetic_bundle, features):
    descriptor = descriptor_for(synthetic_bundle)
    descriptor["config"]["features"] = features
    refresh_version(descriptor)
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_descriptor(descriptor)


@pytest.mark.parametrize(("field", "value"), [
    ("sklearn", "0.0.0"), ("algorithm_version", "different-algorithm"),
    ("rf_threshold", -0.01), ("rf_threshold", float("nan")),
    ("lr_threshold", 1.01), ("near_margin", float("inf")),
])
def test_rejects_incompatible_versions_and_invalid_decision_config(
    synthetic_bundle, field, value
):
    descriptor = descriptor_for(synthetic_bundle)
    descriptor["config"][field] = value
    refresh_version(descriptor)
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_descriptor(descriptor)


def test_rejects_version_that_does_not_identify_descriptor_config(synthetic_bundle):
    descriptor = descriptor_for(synthetic_bundle)
    descriptor["model_version"] = "0" * 16
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_descriptor(descriptor)


@pytest.mark.parametrize("reference", [
    {}, {"q01": 0., "median": 1.},
    {"q01": 0., "median": float("nan"), "q99": 2.},
    {"q01": 2., "median": 1., "q99": 0.},
])
def test_rejects_missing_or_invalid_feature_reference(synthetic_bundle, reference):
    descriptor = descriptor_for(synthetic_bundle)
    descriptor["reference"]["Flow Duration"] = reference
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_descriptor(descriptor)


def test_rejects_bundle_descriptor_decision_mismatch(synthetic_bundle):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle["config"]["rf_threshold"] = 0.12345
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


def test_rejects_bundle_descriptor_evidence_mismatch(synthetic_bundle):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle["reference"]["Flow Duration"] = {"q01": 0., "median": 1., "q99": 2.}
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize("model_name", ["rf", "lr"])
def test_rejects_internal_feature_order_mismatch(synthetic_bundle, model_name):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    imputer = bundle[model_name].named_steps["impute"]
    imputer.feature_names_in_ = imputer.feature_names_in_[::-1]
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize("model_name", ["rf", "lr"])
def test_rejects_class_order_that_would_invert_anomaly_scores(synthetic_bundle, model_name):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle[model_name].named_steps["classifier"].classes_ = np.array([1, 0])
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize("step", ["impute", "scale"])
def test_rejects_unfitted_preprocessing(synthetic_bundle, step):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    replacement = SimpleImputer(strategy="median", add_indicator=True,
                                keep_empty_features=True) if step == "impute" else StandardScaler()
    bundle["lr"].set_params(**{step: replacement})
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


def test_rejects_pipeline_parameter_drift(synthetic_bundle):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle["rf"].set_params(classifier__max_depth=17)
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize("failure", ["hash", "descriptor", "expected_hash"])
def test_preflight_failure_never_deserializes_model(
    tmp_path, monkeypatch, synthetic_bundle, failure
):
    model, descriptor_path, descriptor = write_package(tmp_path, synthetic_bundle)
    arguments = {}
    if failure == "hash":
        model.write_bytes(model.read_bytes() + b"tampered")
    elif failure == "descriptor":
        descriptor["config"]["sklearn"] = "0.0.0"
        refresh_version(descriptor)
        descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    else:
        arguments["expected_sha256"] = "0" * 64
    loaded = []

    def forbidden_load(*args, **kwargs):
        loaded.append(args)
        raise AssertionError("A failed preflight must not deserialize joblib")

    monkeypatch.setattr(compatibility.joblib, "load", forbidden_load)
    report = compatibility.inspect_model(model, descriptor_path, **arguments)
    assert not report["compatible"]
    assert any(not check["passed"] for check in report["checks"])
    assert loaded == []
    assert compatibility.exit_code(report) == 1


def test_missing_package_returns_actionable_failed_check(tmp_path):
    report = compatibility.inspect_model(tmp_path / "absent.joblib", tmp_path / "absent.json")
    assert not report["compatible"]
    failed = [check for check in report["checks"] if not check["passed"]]
    assert failed and all(check.get("error") or check.get("details") for check in failed)


def test_unreadable_model_with_matching_hash_is_still_incompatible(tmp_path, synthetic_bundle):
    model, descriptor_path, descriptor = write_package(tmp_path, synthetic_bundle)
    model.write_bytes(b"")
    descriptor["model_sha256"] = sha256(model)
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    report = compatibility.inspect_model(model, descriptor_path)
    assert not report["compatible"]
    assert any(not check["passed"] for check in report["checks"])


def test_missing_sample_feature_cannot_be_reported_as_success(tmp_path, synthetic_bundle):
    model, descriptor_path, _ = write_package(tmp_path, synthetic_bundle)
    sample = tmp_path / "missing-feature.csv"
    sample.write_text("Flow Duration\n1\n", encoding="utf-8")
    report = compatibility.inspect_model(model, descriptor_path, sample=sample)
    assert not report["compatible"]
    assert any(not check["passed"] for check in report["checks"])


def test_valid_package_and_partial_invalid_sample_are_accounted_for(
    tmp_path, synthetic_bundle
):
    model, descriptor_path, _ = write_package(tmp_path, synthetic_bundle)
    sample = tmp_path / "sample.csv"
    sample.write_text(
        "Flow Duration,Tot Fwd Pkts,Label\n0,1,Bot\n8,7,Benign\nbad,Infinity,Bot\n",
        encoding="utf-8",
    )
    report = compatibility.inspect_model(model, descriptor_path, sample=sample)
    assert report["compatible"], report["checks"]
    assert report["sample"]["rows"] == 3
    assert report["sample"]["predicted_rows"] == 2
    assert report["sample"]["rejected_rows"] == 1
    assert report["frozen_baseline"] is None
    assert compatibility.exit_code(report) == 0
    assert compatibility.exit_code(report, require_frozen=True) == 2
    assert report["contract_gaps"], "Legacy missing schema metadata must remain visible"


def test_descriptor_baseline_match_alone_cannot_claim_frozen_artifact_identity(
    tmp_path, synthetic_bundle
):
    model, descriptor_path, descriptor = write_package(tmp_path, synthetic_bundle)
    baseline = tmp_path / "metrics.json"
    baseline.write_text(json.dumps({
        "model_version": descriptor["model_version"],
        "data_version": descriptor["config"]["data_version"],
        "rf": {"threshold": descriptor["config"]["rf_threshold"]},
        "lr": {"threshold": descriptor["config"]["lr_threshold"]},
    }), encoding="utf-8")
    report = compatibility.inspect_model(model, descriptor_path, baseline_metrics=baseline)
    assert report["compatible"], report["checks"]
    assert not report["frozen_baseline"]["matched"]
    assert compatibility.exit_code(report, require_frozen=True) == 2
    verified = compatibility.inspect_model(
        model, descriptor_path, baseline_metrics=baseline, frozen_sha256=sha256(model)
    )
    assert verified["compatible"], verified["checks"]
    assert verified["frozen_baseline"]["matched"]
    assert compatibility.exit_code(verified, require_frozen=True) == 0


@pytest.mark.parametrize("model_name", ["rf", "lr"])
def test_review_regression_rejects_changed_missing_value_semantics(synthetic_bundle, model_name):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle[model_name].named_steps["impute"].missing_values = 0
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize("corruption", ["nan_coefficient", "wrong_shape", "infinite_intercept"])
def test_review_regression_rejects_corrupt_fitted_lr(synthetic_bundle, corruption):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    classifier = bundle["lr"].named_steps["classifier"]
    if corruption == "nan_coefficient":
        classifier.coef_[0, 0] = np.nan
    elif corruption == "wrong_shape":
        classifier.coef_ = classifier.coef_[:, :-1]
    else:
        classifier.intercept_[0] = np.inf
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize("corruption", ["missing_tree", "unfitted_tree"])
def test_review_regression_rejects_corrupt_fitted_forest(synthetic_bundle, corruption):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle["rf"].named_steps["classifier"].estimators_[0] = (
        None if corruption == "missing_tree" else DecisionTreeClassifier()
    )
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


def test_review_regression_rejects_schema_declared_only_in_bundle(synthetic_bundle):
    bundle = copy.deepcopy(synthetic_bundle)
    descriptor = descriptor_for(bundle)
    bundle["schema_version"] = "unknown-v2"
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_bundle(bundle, descriptor)


@pytest.mark.parametrize(("location", "field", "declaration"), [
    ("top", "schema_version", "unknown-v2"),
    ("config", "schema_version", "unknown-v2"),
    ("top", "class_mapping", {"0": "anomaly", "1": "normal"}),
    ("config", "class_mapping", {"0": "anomaly", "1": "normal"}),
])
def test_review_regression_rejects_unknown_declared_contract(
    synthetic_bundle, location, field, declaration
):
    descriptor = descriptor_for(synthetic_bundle)
    target = descriptor if location == "top" else descriptor["config"]
    target[field] = declaration
    refresh_version(descriptor)
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.validate_descriptor(descriptor)


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "scripts.engineering_model_compatibility",
         *map(str, arguments)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )


def test_cli_existing_output_never_overwrites_package(tmp_path, synthetic_bundle):
    model, descriptor_path, _ = write_package(tmp_path, synthetic_bundle)
    original_model = model.read_bytes()
    result = run_cli("--model", model, "--descriptor", descriptor_path, "--output", model)
    assert result.returncode != 0
    assert model.read_bytes() == original_model
    assert "FileExistsError" in result.stderr


@pytest.mark.skipif(not (compatibility.ROOT/'docs/phase2/audit.json').is_file(),reason='Historical audit and handoff are absent from supplied snapshot; frozen runtime validated separately')
def test_cli_require_frozen_distinguishes_compatible_synthetic_package(tmp_path, synthetic_bundle):
    model, descriptor_path, _ = write_package(tmp_path, synthetic_bundle)
    compatible = run_cli('--model', model, '--descriptor', descriptor_path)
    assert compatible.returncode == 0, compatible.stderr or compatible.stdout
    assert json.loads(compatible.stdout)['compatible']
    output = tmp_path / "new-report.json"
    result = run_cli(
        "--model", model, "--descriptor", descriptor_path, "--require-frozen", "--output", output
    )
    assert result.returncode == 1, result.stderr or result.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["compatible"]
    assert report['checks'][-1]['name'] == 'trusted_handoff'
    assert report["require_frozen"] and report["exit_code"] == 1
    assert json.loads(result.stdout) == report


@pytest.mark.parametrize("model_name", ["rf", "lr"])
@pytest.mark.parametrize("scores", [
    [0.5, np.nan], [0., np.inf], [0.2, 0.3, 0.5], [-0.1, 1.1], [0.4, 0.4],
], ids=["nan", "infinity", "wrong-width", "outside-range", "not-normalized"])
def test_engineering_sample_entry_rejects_invalid_model_output(
    tmp_path, monkeypatch, synthetic_bundle, model_name, scores
):
    """This guard belongs to the checker; it does not certify the core UI loader."""
    bundle = copy.deepcopy(synthetic_bundle)
    sample = tmp_path / "valid-input.csv"
    sample.write_text("Flow Duration,Tot Fwd Pkts\n0,1\n8,7\n", encoding="utf-8")
    monkeypatch.setattr(
        bundle[model_name], "predict_proba",
        lambda values: np.tile(np.asarray(scores), (len(values), 1)),
    )
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.check_sample(sample, bundle)
