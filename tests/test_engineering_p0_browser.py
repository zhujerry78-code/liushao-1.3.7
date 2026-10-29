"""Regression for real-browser evidence readers; no fitting or model loading."""
import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import expect, sync_playwright

from scripts.engineering_p0_browser import P0Browser, assert_prediction_export, csv
from scripts.engineering_p0_cases import assert_frames


@pytest.fixture(scope='module')
def edge_browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.mark.parametrize('initially_open', [False, True])
@pytest.mark.parametrize('preceding_tables', [0, 2])
def test_evidence_download_opens_only_its_expander_and_ignores_other_tables(
    edge_browser, tmp_path, initially_open, preceding_tables,
):
    """Exercise native details visibility and keyboard CSV download in real Edge."""
    context = edge_browser.new_context(accept_downloads=True)
    try:
        page = context.new_page()
        unrelated = '<div data-testid="stDataFrame">unrelated table</div>' * preceding_tables
        opened = 'open' if initially_open else ''
        page.set_content(f'''
            {unrelated}
            <div data-testid="stExpander">
              <details id="quality"><summary>查看字段问题汇总</summary>
                <div data-testid="stDataFrame">quality table</div>
              </details>
            </div>
            <div data-testid="stExpander">
              <details id="evidence" {opened}>
                <summary>查看原始特征与正常参考区间</summary>
                <div data-testid="stDataFrame">
                  <button onclick="downloadEvidence()">Download as CSV</button>
                </div>
              </details>
            </div>
            <script>
              function downloadEvidence() {{
                const link = document.createElement('a');
                link.href = URL.createObjectURL(new Blob(
                  ['字段,原值\\n参考特征,42\\n'], {{type: 'text/csv;charset=utf-8'}}));
                link.download = 'evidence.csv';
                document.body.appendChild(link);
                link.click();
                link.remove();
              }}
            </script>
        ''')
        # Skip model/server construction; both exercised methods use the actual browser page.
        runner = P0Browser.__new__(P0Browser)
        runner.page, runner.out, runner.counter = page, tmp_path, 0
        table = runner.evidence_table()
        expect(table).to_be_visible()
        expect(page.locator('#evidence')).to_have_attribute('open', '')
        assert page.locator('#quality').get_attribute('open') is None
        downloaded = runner.download(None, 'evidence.csv', table=table)
        assert_frames(csv(downloaded), pd.DataFrame({'字段': ['参考特征'], '原值': [42]}))
        runner.evidence_table()
        expect(page.locator('#evidence')).to_have_attribute('open', '')
        assert page.locator('#quality').get_attribute('open') is None
    finally:
        context.close()


def test_export_reader_restores_timestamps_without_losing_score_precision(tmp_path):
    frame = pd.DataFrame({'timestamp': [pd.Timestamp('2018-03-02 01:02:17'), pd.NaT],
                          'rf_score': [0.9999943456789012, 0.12000000000000001],
                          'rf_anomaly': [True, True]})
    target = tmp_path / 'actual.csv'
    frame.to_csv(target, index=False)
    restored = csv(target)
    assert_frames(restored, frame)
    assert restored.rf_score.tolist() == frame.rf_score.tolist()


def test_reader_preserves_empty_notes_and_literal_na_text_with_numeric_missing(tmp_path):
    frame = pd.DataFrame({
        'record_id': ['record:2', 'record:3', 'record:4', 'record:5'],
        'source_file': ['NA', 'null', 'NaN', 'source.csv'],
        'timestamp': [pd.Timestamp('2018-03-02 01:02:17'), pd.NaT,
                      pd.Timestamp('2018-03-02 01:03:18'), pd.NaT],
        'rf_score': [0.12000000000000001, np.nan, 0.9999943456789012, np.nan],
        'missing_features': [0, 1, 2, 3],
        'rf_anomaly': [True, False, True, False],
        'review_status': ['待复核', '待补证', '确认正常', '确认异常'],
        'review_note': ['', 'NA', 'null', 'NaN'],
    })
    target = tmp_path / 'notes-and-missing.csv'
    frame.to_csv(target, index=False)
    restored = csv(target)
    assert restored.review_note.tolist() == ['', 'NA', 'null', 'NaN']
    assert restored.source_file.tolist() == frame.source_file.tolist()
    assert restored.rf_score.isna().tolist() == [False, True, False, True]
    assert restored.timestamp.isna().tolist() == [False, True, False, True]
    assert_frames(restored, frame)


def test_evidence_reader_retains_text_markers_but_restores_missing_numeric_cells(tmp_path):
    frame = pd.DataFrame({
        '字段': ['NA', 'null', 'NaN'],
        '原值': [np.nan, 2.0, 3.0],
        '单位': ['数据源原始量纲'] * 3,
        '正常P01': [0.0, np.nan, 1.0],
        '正常中位数': [1.0, np.nan, 2.0],
        '正常P99': [4.0, np.nan, 5.0],
        '统计提示': ['缺失，预测时用训练中位数填补', '训练正常参考不足', '参考区间内'],
    })
    target = tmp_path / 'evidence-with-missing.csv'
    frame.to_csv(target, index=False)
    restored = csv(target)
    assert restored['字段'].tolist() == ['NA', 'null', 'NaN']
    assert_frames(restored, frame)


@pytest.fixture
def prediction_frames():
    expected = pd.DataFrame({
        'record_id': ['input:2', 'input:3'],
        'source_row': [2, 3],
        'rf_score': [0.2, 0.8],
        'rf_anomaly': [False, True],
        'source_file': ['source.csv', 'source.csv'],
    })
    actual = expected.assign(review_status=['待复核', '确认异常'], review_note=['', '测试依据'])
    return actual, expected


def test_prediction_export_accepts_only_the_full_ordered_prediction_and_review_schema(
    prediction_frames,
):
    actual, expected = prediction_frames
    assert actual.columns.tolist() == expected.columns.tolist() + ['review_status', 'review_note']
    assert_prediction_export(actual, expected)


@pytest.mark.parametrize('field', ['Label', 'ground_truth', 'unexpected'])
def test_prediction_export_rejects_label_leakage_or_any_additional_field(prediction_frames, field):
    actual, expected = prediction_frames
    actual[field] = ['Benign', 'Bot']
    with pytest.raises(AssertionError):
        assert_prediction_export(actual, expected)


@pytest.mark.parametrize('change', [
    'missing-review-note', 'missing-review-status', 'missing-prediction-field',
    'reordered-prediction-fields', 'reordered-review-fields',
])
def test_prediction_export_rejects_incomplete_or_reordered_schema(prediction_frames, change):
    actual, expected = prediction_frames
    if change == 'missing-review-note':
        actual = actual.drop(columns='review_note')
    elif change == 'missing-review-status':
        actual = actual.drop(columns='review_status')
    elif change == 'missing-prediction-field':
        actual = actual.drop(columns='rf_score')
    elif change == 'reordered-prediction-fields':
        actual = actual[['source_row', 'record_id', 'rf_score', 'rf_anomaly', 'source_file',
                         'review_status', 'review_note']]
    else:
        actual = actual[expected.columns.tolist() + ['review_note', 'review_status']]
    with pytest.raises(AssertionError):
        assert_prediction_export(actual, expected)


@pytest.mark.parametrize('change', ['score', 'row-order'])
def test_prediction_export_still_checks_prediction_values_and_row_order(prediction_frames, change):
    actual, expected = prediction_frames
    if change == 'score':
        actual.loc[0, 'rf_score'] += 1e-5
    else:
        actual = actual.iloc[::-1]
    with pytest.raises(AssertionError):
        assert_prediction_export(actual, expected)
