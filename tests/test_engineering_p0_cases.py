"""Validate P0 fault cases and the checks used to compare browser exports."""
import io

import numpy as np
import pandas as pd
import pytest

from liushao.data import clean, read_csv_checked
from scripts.engineering_p0_cases import assert_frames, input_cases


@pytest.fixture
def content():
    rows = [" a ,b,c,Label,Timestamp"]
    rows.extend(f"{i},{i + 10},{i + 20},Benign,16/02/2018 09:00:00" for i in range(1, 9))
    return ("\ufeff" + "\n".join(rows) + "\n").encode("utf-8")


@pytest.mark.parametrize("case", [
    "partial-nan", "partial-posinf", "partial-neginf", "partial-empty", "partial-overflow",
])
def test_partial_invalid_cells_keep_records_and_other_source_values(content, case):
    generated, error = input_cases(content)[case]
    assert error is None
    raw = read_csv_checked(io.BytesIO(generated))
    values, meta, rejected = clean(raw, ["a", "b", "c"], "p0-case", case + ".csv")
    assert len(values) == 6 and rejected.empty
    assert meta.source_row.tolist() == list(range(2, 8))
    assert meta.missing_features.tolist() == [1, 0, 0, 0, 0, 0]
    assert np.isnan(values.loc[0, "a"])
    assert values.b.tolist() == list(range(11, 17))
    assert values.c.tolist() == list(range(21, 27))
    original = read_csv_checked(io.BytesIO(content)).head(6)
    pd.testing.assert_frame_equal(raw.iloc[:, 1:], original.iloc[:, 1:])
    pd.testing.assert_frame_equal(raw.iloc[1:], original.iloc[1:])


def test_mixed_case_distinguishes_blank_physical_line_from_invalid_record(content):
    generated, error = input_cases(content)["mixed-invalid"]
    assert error is None
    raw = read_csv_checked(io.BytesIO(generated))
    assert raw.attrs["source_rows"] == [2, 4, 5]
    values, meta, rejected = clean(raw, ["a", "b", "c"], "mixed", "mixed.csv")
    assert values.a.tolist() == [1, 2]
    assert meta.source_row.tolist() == [2, 5]
    assert meta.record_id.tolist() == ["mixed:2", "mixed:5"]
    assert rejected.source_row.tolist() == [4]
    assert rejected.record_id.tolist() == ["mixed:4"]
    assert rejected.reason.tolist() == ["全部必需特征无效，无法判断"]
    assert len(meta) + len(rejected) == len(raw) == 3


def test_all_invalid_reaches_clean_and_returns_only_rejected_records(content):
    generated, error = input_cases(content)["all-invalid"]
    assert error is None
    values, meta, rejected = clean(
        read_csv_checked(io.BytesIO(generated)), ["a", "b", "c"], "invalid", "invalid.csv",
    )
    assert values.empty and meta.empty
    assert rejected.source_row.tolist() == list(range(2, 8))
    assert rejected.reason.eq("全部必需特征无效，无法判断").all()


@pytest.mark.parametrize("case", [
    "missing-column", "duplicate-column", "duplicate-trimmed-column", "bad-width",
])
def test_structural_cases_fail_with_declared_error_before_any_prediction(content, case):
    generated, error = input_cases(content)[case]
    assert error
    with pytest.raises(ValueError, match=error):
        clean(read_csv_checked(io.BytesIO(generated)), ["a", "b", "c"], "bad", case + ".csv")


def test_header_only_is_distinct_from_all_invalid_and_has_app_guard_message(content):
    generated, error = input_cases(content)["header-only"]
    raw = read_csv_checked(io.BytesIO(generated))
    assert raw.empty and raw.columns.tolist() == ["a", "b", "c", "Label", "Timestamp"]
    assert error == "请输入 1 至 50,000 条记录"
    assert len(input_cases(content)) == 12
    assert len(read_csv_checked(io.BytesIO(content))) == 8


@pytest.mark.parametrize("content", [
    b"a,b\n1,2\n", b"a,Label\n1,Benign\n2,Bot\n", b"a,b\nNaN,2\n3,4\n",
])
def test_case_builder_rejects_sources_that_cannot_support_partial_or_mixed_cases(content):
    with pytest.raises(ValueError, match="至少"):
        input_cases(content)


def test_comparator_accepts_csv_numeric_dtypes_and_equivalent_missing_positions():
    expected = pd.DataFrame({
        "record_id": ["first", "second"], "source_row": [2, 5], "score": [0.2, np.nan],
        "flag": [True, False], "note": ["", "依据"],
    })
    actual = expected.copy()
    actual.index = [10, 20]
    actual["source_row"] = actual.source_row.astype(float)
    actual.loc[10, "score"] += 5e-13
    actual["score"] = actual.score.astype(object)
    actual.loc[20, "score"] = pd.NA
    assert_frames(actual, expected)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "record_id": ["first", "second"], "score": [0.2, 0.3], "feature": [11.0, 12.0],
        "flag": [True, False], "note": ["依据", ""], "nullable": [np.nan, 1.0],
    })


@pytest.mark.parametrize("change", [
    "missing-row", "reordered-row", "reordered-column", "missing-column", "flag",
    "feature", "text", "null", "empty-is-not-null", "boolean-is-not-one",
    "numeric-text-is-not-number",
])
def test_comparator_rejects_content_shape_order_or_semantics_changes(frame, change):
    changed = frame.copy()
    if change == "missing-row":
        changed = changed.iloc[:1]
    elif change == "reordered-row":
        changed = changed.iloc[::-1]
    elif change == "reordered-column":
        changed = changed[changed.columns[::-1]]
    elif change == "missing-column":
        changed = changed.drop(columns="feature")
    elif change == "flag":
        changed.loc[0, "flag"] = False
    elif change == "feature":
        changed.loc[0, "feature"] += 1e-5
    elif change == "text":
        changed.loc[0, "note"] = "不同依据"
    elif change == "null":
        changed.loc[0, "nullable"] = 0.0
    elif change == "empty-is-not-null":
        changed.loc[1, "note"] = None
    elif change == "boolean-is-not-one":
        changed["flag"] = changed.flag.astype(int)
    elif change == "numeric-text-is-not-number":
        changed["feature"] = changed.feature.astype(str)
    with pytest.raises(AssertionError):
        assert_frames(changed, frame)


def test_comparator_uses_zero_relative_tolerance():
    expected = pd.DataFrame({"feature": [1e8]})
    actual = pd.DataFrame({"feature": [1e8 + 1e-6]})
    with pytest.raises(AssertionError, match="feature"):
        assert_frames(actual, expected)


def test_comparator_does_not_round_large_integer_identity_values():
    expected = pd.DataFrame({"identity": [2 ** 60]})
    actual = pd.DataFrame({"identity": [2 ** 60 + 1]})
    with pytest.raises(AssertionError, match="identity"):
        assert_frames(actual, expected)
