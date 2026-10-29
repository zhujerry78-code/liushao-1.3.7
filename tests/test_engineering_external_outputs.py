"""Invalid predictor outputs must fail evaluation, never become normal labels."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from scripts import engineering_external as external
from test_engineering_external import inputs


def invalid_probabilities(rows, failure):
    if failure == "columns":
        return np.ones((rows, 1))
    if failure == "rows":
        return np.full((rows + 1, 2), 0.5)
    pair = {
        "nan": [np.nan, np.nan],
        "inf": [0.0, np.inf],
        "range": [-0.1, 1.1],
        "sum": [0.2, 0.2],
        "nonnumeric": ["bad", "output"],
        "complex": [0.5 + 0.1j, 0.5 - 0.1j],
    }[failure]
    return np.tile(pair, (rows, 1))


def corrupt_prediction(monkeypatch, model_name, failure, first_bad_call=1):
    original = Pipeline.predict_proba
    calls = []

    def predict(self, values, *args, **kwargs):
        name = "rf" if isinstance(
            self.named_steps["classifier"], RandomForestClassifier
        ) else "lr"
        if name == model_name:
            calls.append(len(values))
            if len(calls) >= first_bad_call:
                return invalid_probabilities(len(values), failure)
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "predict_proba", predict)
    return calls


def assert_failed_without_metrics(args, model_name):
    out = external.OUTPUT_ROOT / args.run_id
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert model_name + " 模型输出" in manifest["error"]
    assert "metrics_sha256" not in manifest
    assert not (out / "metrics.json").exists()
    return out


@pytest.mark.parametrize("model_name", ["rf", "lr"])
@pytest.mark.parametrize("failure", [
    "nan", "inf", "columns", "rows", "range", "sum", "nonnumeric", "complex",
])
def test_invalid_output_stops_before_labels_metrics_or_exports(
    inputs, monkeypatch, model_name, failure,
):
    args, _ = inputs
    calls = corrupt_prediction(monkeypatch, model_name, failure)
    with pytest.raises(ValueError, match=model_name + " 模型输出"):
        external.run(args)
    assert calls == [2]
    out = assert_failed_without_metrics(args, model_name)
    assert not (out / "predictions.csv").exists()


@pytest.mark.parametrize("model_name", ["rf", "lr"])
def test_later_chunk_failure_does_not_publish_success_metrics(
    inputs, monkeypatch, model_name,
):
    args, _ = inputs
    calls = corrupt_prediction(monkeypatch, model_name, "nan", first_bad_call=2)
    with pytest.raises(ValueError, match=model_name + " 模型输出"):
        external.run(args)
    assert calls == [2, 1]
    out = assert_failed_without_metrics(args, model_name)
    # Earlier valid chunks may remain for diagnosis, explicitly under failed status.
    partial = pd.read_csv(out / "predictions.csv")
    assert partial.source_row.tolist() == [2, 3]
    assert np.isfinite(partial[["rf_score", "lr_score"]].to_numpy()).all()
