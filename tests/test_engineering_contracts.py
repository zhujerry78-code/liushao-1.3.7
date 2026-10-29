"""Engineering fixtures: input boundaries and durable review identity, no experiment data."""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liushao.data import clean, read_csv_checked
from liushao.model import infer
from liushao.reviews import current_reviews, history, save_review


@pytest.mark.parametrize(
    ("content", "error", "message"),
    [
        (b"a,,b\n1,2,3\n", ValueError, "空字段"),
        (b"\n1,2\n", ValueError, "表头为空"),
        (b"a,b\n1\n", ValueError, "第 2 行字段数不一致"),
        (b"a,b\n\n1,2,3\n", ValueError, "第 3 行字段数不一致"),
        (b'a,b\n"1,2\n', csv.Error, "unexpected end"),
        (b"a,b\n\xff,2\n", UnicodeDecodeError, "utf-8"),
    ],
)
def test_malformed_csv_fails_with_location_or_decoder_reason(content, error, message):
    with pytest.raises(error, match=message):
        read_csv_checked(io.BytesIO(content))


def test_csv_row_limit_counts_records_and_bom_preserves_source_identity():
    content = "\ufeff a ,b\r\n1,2\r\n\r\n3,4\r\n".encode("utf-8")
    raw = read_csv_checked(io.BytesIO(content), max_rows=2)
    assert list(raw.columns) == ["a", "b"]
    assert raw.attrs["source_rows"] == [2, 4]
    _, first, _ = clean(raw, ["a", "b"], "same-content", "original.csv")
    _, renamed, _ = clean(raw, ["a", "b"], "same-content", "renamed.csv")
    assert first.record_id.tolist() == renamed.record_id.tolist()
    assert first.record_id.tolist() == ["same-content:2", "same-content:4"]
    with pytest.raises(ValueError, match="最多支持 1 条"):
        read_csv_checked(io.BytesIO(content), max_rows=1)


def test_clean_accounts_for_overflow_partial_missing_and_rejected_rows():
    content = b"a,b\nInfinity,2\n-1e40,3\ninvalid,NaN\n4,5\n"
    raw = read_csv_checked(io.BytesIO(content))
    values, meta, rejected = clean(raw, ["a", "b"], "content", "input.csv")
    assert len(values) + len(rejected) == len(raw)
    assert meta.source_row.tolist() == [2, 3, 5]
    assert meta.missing_features.tolist() == [1, 1, 0]
    assert values.a.isna().tolist() == [True, True, False]
    assert rejected.source_row.tolist() == [4]
    assert rejected.record_id.tolist() == ["content:4"]
    assert rejected.reason.str.contains("无法判断").all()
    assert not set(meta.record_id) & set(rejected.record_id)
    pd.testing.assert_frame_equal(raw, read_csv_checked(io.BytesIO(content)))


def test_clean_reports_all_missing_required_fields_before_prediction():
    with pytest.raises(ValueError, match="缺少必需特征: a, c"):
        clean(pd.DataFrame({"b": [1.0]}), ["c", "b", "a"], "hash", "input.csv")


class FixedScores:
    """Deliberately synthetic scores for exact decision-boundary checks."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, values):
        assert len(values) == len(self.scores)
        return np.column_stack([1 - self.scores, self.scores])


def test_inference_threshold_and_near_margin_boundaries_are_inclusive():
    bundle = {
        "config": {
            "features": ["a"], "rf_threshold": 0.5, "lr_threshold": 0.5,
            "near_margin": 0.125,
        },
        "rf": FixedScores([0.374, 0.375, 0.5, 0.625, 0.626]),
        "lr": FixedScores([0.25] * 5),
        "model_version": "synthetic-boundary-fixture",
    }
    raw = pd.DataFrame({"a": range(5), "Label": ["Bot"] * 5})
    predictions, values, rejected = infer(raw, bundle, "hash", "boundary.csv")
    assert predictions.rf_anomaly.tolist() == [False, False, True, True, True]
    assert predictions.near_threshold.tolist() == [False, True, True, True, False]
    assert predictions.disagreement.tolist() == [False, False, True, True, True]
    assert predictions.lr_anomaly.tolist() == [False] * 5
    assert not {"Label", "label", "ground_truth"} & set(predictions.columns)
    assert values.columns.tolist() == ["a"] and rejected.empty


@pytest.mark.parametrize(
    ("record_id", "model_version", "status", "note"),
    [
        ("record", "v1", "unknown", "reason"),
        ("", "v1", "待复核", "reason"),
        ("record", "", "待复核", "reason"),
        ("record", "v1", "待复核", "字" * 4001),
    ],
)
def test_rejected_review_write_preserves_existing_history(
    tmp_path, record_id, model_version, status, note
):
    database = tmp_path / "reviews.sqlite"
    save_review(database, "record", "v1", "待补证", "保留原记录")
    original = history(database)
    with pytest.raises(ValueError):
        save_review(database, record_id, model_version, status, note)
    assert history(database) == original


def test_review_history_and_identity_survive_independent_python_processes(tmp_path):
    database = tmp_path / "reviews.sqlite"
    commands = [
        ["source:2", "v1", "待复核", "导入"],
        ["source:2", "v1", "确认异常", '依据：字段偏离；"保留引号"\n第二行'],
        ["source:3", "v1", "确认正常", "独立记录"],
        ["source:2", "v2", "待补证", "新模型独立结论"],
        ["source:2", "v1", "待补证", "复核后撤回，等待上下文"],
    ]
    writer = (
        "import json,sys; from liushao.reviews import save_review; "
        "[save_review(sys.argv[1], *row) for row in json.loads(sys.argv[2])]"
    )
    result = subprocess.run(
        [sys.executable, "-c", writer, str(database), json.dumps(commands)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    reader = (
        "import json,sys; from liushao.reviews import history,current_reviews; "
        "print(json.dumps({'history':history(sys.argv[1]), "
        "'v1':current_reviews(sys.argv[1],'v1'), 'v2':current_reviews(sys.argv[1],'v2')}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", reader, str(database)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    reopened = json.loads(result.stdout)
    assert len(reopened["history"]) == len(commands)
    assert [row["id"] for row in reopened["history"]] == sorted(
        {row["id"] for row in reopened["history"]}
    )
    assert reopened["v1"]["source:2"]["note"] == commands[-1][-1]
    assert reopened["v1"]["source:3"]["status"] == "确认正常"
    assert reopened["v2"]["source:2"]["note"] == "新模型独立结论"
    assert history(database, "source:2", "v1")[1]["note"] == commands[1][-1]
    assert len(history(database, "source:2", "v1")) == 3
    assert all(row["created_at"].endswith("+00:00") for row in reopened["history"])
    assert current_reviews(database, "unseen-model") == {}
