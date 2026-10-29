"""Small CSV fault cases and strict output checks for P0 browser acceptance.

These helpers change test inputs only. They do not load or modify model packages.
"""
from __future__ import annotations

import csv
import io
from numbers import Integral, Real

import numpy as np
import pandas as pd

from liushao.data import EXCLUDED, read_csv_checked


def assert_review_snapshot(predictions: pd.DataFrame, audit: list[dict],
                           model_version: str) -> None:
    """Require CSV current reviews to come from the last rows of one model history.

    Complete history can include other source files. Every exported record without
    history must retain the default state, and duplicate identities are rejected.
    """
    assert predictions.record_id.is_unique, '预测导出存在重复记录身份'
    assert predictions.model_version.eq(model_version).all(), '预测导出模型身份不一致'
    previous = 0
    latest = {}
    for item in audit:
        revision = item['id']
        assert isinstance(revision, int) and not isinstance(revision, bool)
        assert revision > previous, '复核历史必须按唯一的递增修订排序'
        assert item['model_version'] == model_version, '复核历史混入其他模型'
        previous = revision
        latest[item['record_id']] = item
    for row in predictions.itertuples(index=False):
        expected = latest.get(row.record_id, {'status': '待复核', 'note': ''})
        assert row.review_status == expected['status'], f'{row.record_id}: 当前结论与历史末条不同'
        assert row.review_note == expected['note'], f'{row.record_id}: 当前备注与历史末条不同'


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def input_cases(content: bytes) -> dict[str, tuple[bytes, str | None]]:
    """Build fault cases from the first six data records of a valid source CSV.

    Headers follow the existing adapter's whitespace normalization. A case with
    ``None`` may reach ``infer``; ``all-invalid`` deliberately returns no accepted
    records. Other values are expected Chinese error fragments, including the
    app's record-count guard for ``header-only``.

    ``mixed-invalid`` has valid physical lines 2 and 5, a blank line 3, and an
    all-invalid line 4. Generated files have new identities and must never be
    mistaken for the original source file or used as model evaluation data.
    """
    raw = read_csv_checked(io.BytesIO(content)).head(6)
    header = raw.columns.tolist()
    positions = [index for index, name in enumerate(header) if name not in EXCLUDED]
    if len(raw) < 2 or len(positions) < 2:
        raise ValueError("P0 测试样本至少需要两条记录和两个特征字段")
    rows = raw.astype(str).values.tolist()
    values = pd.to_numeric(pd.Series([rows[0][index] for index in positions]), errors="coerce")
    usable = np.isfinite(values) & (values.abs() <= np.finfo(np.float32).max)
    if usable.sum() < 2:
        raise ValueError("P0 部分无效案例的首条记录至少需要两个有效数值特征")
    changed_position = positions[int(np.flatnonzero(usable)[0])]

    cases: dict[str, tuple[bytes, str | None]] = {}
    for name, value in (
        ("partial-nan", "NaN"),
        ("partial-posinf", "Infinity"),
        ("partial-neginf", "-Infinity"),
        ("partial-empty", ""),
        ("partial-overflow", "1e40"),
    ):
        changed = [row.copy() for row in rows]
        changed[0][changed_position] = value
        cases[name] = (_csv_bytes(header, changed), None)

    invalid_rows = [row.copy() for row in rows]
    for row in invalid_rows:
        for position in positions:
            row[position] = "NaN"
    cases["mixed-invalid"] = (
        _csv_bytes(header, [rows[0], [], invalid_rows[0], rows[1]]), None,
    )
    cases["all-invalid"] = (_csv_bytes(header, invalid_rows), None)
    cases["header-only"] = (_csv_bytes(header, []), "请输入 1 至 50,000 条记录")

    missing_header = header[:changed_position] + header[changed_position + 1:]
    missing_rows = [row[:changed_position] + row[changed_position + 1:] for row in rows]
    cases["missing-column"] = (
        _csv_bytes(missing_header, missing_rows), "缺少必需特征",
    )
    duplicated = header[changed_position]
    for name, field in (
        ("duplicate-column", duplicated),
        ("duplicate-trimmed-column", " " + duplicated + " "),
    ):
        cases[name] = (
            _csv_bytes(header + [field], [row + [row[changed_position]] for row in rows]),
            "CSV 表头包含重复字段",
        )
    bad_width_rows = [row.copy() for row in rows]
    bad_width_rows[0] = bad_width_rows[0][:-1]
    cases["bad-width"] = (_csv_bytes(header, bad_width_rows), "第 2 行字段数不一致")
    return cases


def assert_frames(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare ordered output cells without weakening booleans, text, or nulls.

    CSV numeric dtype differences are allowed; finite numbers use absolute
    tolerance 1e-12 and zero relative tolerance. Missing-value representations
    (None, NaN, pd.NA, NaT) are equivalent only at the same cell. Empty strings
    remain text. DataFrame index labels are ignored, but row order is mandatory.
    """
    assert actual.columns.tolist() == expected.columns.tolist(), "列名或列顺序不一致"
    assert actual.shape == expected.shape, (
        f"记录数不一致: actual={actual.shape}, expected={expected.shape}"
    )
    for position, (actual_row, expected_row) in enumerate(zip(
        actual.itertuples(index=False, name=None),
        expected.itertuples(index=False, name=None),
    )):
        for column, observed, wanted in zip(expected.columns, actual_row, expected_row):
            location = f"第 {position + 1} 条记录字段 {column!r}"
            observed_null, wanted_null = bool(pd.isna(observed)), bool(pd.isna(wanted))
            assert observed_null == wanted_null, f"{location} 空值位置不一致"
            if observed_null:
                continue
            observed_bool = isinstance(observed, (bool, np.bool_))
            wanted_bool = isinstance(wanted, (bool, np.bool_))
            if observed_bool or wanted_bool:
                same = observed_bool and wanted_bool and observed == wanted
            elif isinstance(observed, Integral) and isinstance(wanted, Integral):
                same = int(observed) == int(wanted)
            elif isinstance(observed, Real) and isinstance(wanted, Real):
                same = bool(np.isclose(observed, wanted, rtol=0, atol=1e-12))
            else:
                same = type(observed) is type(wanted) and observed == wanted
            assert same, f"{location} 不一致: actual={observed!r}, expected={wanted!r}"
