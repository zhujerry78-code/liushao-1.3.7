"""Capacity evidence cannot hide source errors, unrelated history or missing stages."""
import copy
import hashlib
import io
import json
import time
from types import SimpleNamespace

import pandas as pd
import pytest
from playwright.sync_api import (
    Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError, expect, sync_playwright,
)

from liushao.data import read_csv_checked
from liushao.reviews import save_review
from scripts.engineering_capacity import (
    CapacityBrowser, MODEL_FILES, frozen_hashes, history_fingerprint, read_history, repeat_demo_rows,
    seed_history, validate_completed_case, validate_exports,
)


def test_repeated_csv_keeps_literal_cells_labels_and_new_physical_identity():
    content = b'a,b,Label\n1,NaN,Bot\n2,,Benign\n'
    payload = repeat_demo_rows(content, 1001)
    raw = read_csv_checked(io.BytesIO(payload))
    assert len(raw) == 1001 and raw.attrs['source_rows'] == list(range(2, 1003))
    assert raw.iloc[0].tolist() == raw.iloc[-1].tolist() == ['1', 'NaN', 'Bot']
    assert raw.iloc[1].tolist() == ['2', '', 'Benign']
    assert hashlib.sha256(payload).hexdigest() != hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_repeated_csv_rejects_invalid_load_count(count):
    with pytest.raises(ValueError):
        repeat_demo_rows(b'a\n1\n', count)


def test_capacity_fixture_reaches_real_row_guard():
    payload = repeat_demo_rows(b'a\n1\n', 50001)
    with pytest.raises(ValueError, match='50,000'):
        read_csv_checked(io.BytesIO(payload))


def test_history_seed_scope_and_fingerprint_detects_only_intended_append(tmp_path):
    db = tmp_path / 'reviews.sqlite'
    summary = seed_history(db, 'model', 'source:1002', 100)
    assert summary == {'total_rows': 100, 'related_rows': 25,
                       'unrelated_same_version_rows': 50, 'other_version_rows': 25}
    related = read_history(db, version='model', record_id='source:1002')
    assert len(related) == 25 and related[-1]['status'] == '待补证'
    before = history_fingerprint(db)
    unrelated = history_fingerprint(db, exclude_identity=('model', 'source:1002'))
    save_review(db, 'source:1002', 'model', '确认异常', 'engineering test basis',
                expected_review_id=related[-1]['id'])
    assert history_fingerprint(db)['rows'] == 101
    assert history_fingerprint(db) != before
    assert history_fingerprint(db, exclude_identity=('model', 'source:1002')) == unrelated
    assert len(read_history(db, version='capacity-other-model')) == 25
    with pytest.raises(ValueError, match='new database'):
        seed_history(db, 'model', 'source:1002', 100)


def test_read_oracle_does_not_create_missing_database(tmp_path):
    db = tmp_path / 'missing.sqlite'
    with pytest.raises(Exception):
        read_history(db)
    assert not db.exists()


def test_trusted_five_file_validation_rejects_self_consistent_changed_file(tmp_path):
    files = []
    for name in MODEL_FILES:
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        files.append({'path': 'artifacts/' + name, 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()})
    manifest = tmp_path / 'trusted.json'
    manifest.write_text(json.dumps({'files': files}), encoding='utf-8')
    assert len(frozen_hashes(tmp_path, manifest)) == 5
    (tmp_path / 'model.json').write_bytes(b'changed description')
    with pytest.raises(ValueError, match='model.json'):
        frozen_hashes(tmp_path, manifest)


@pytest.fixture
def exports():
    expected = pd.DataFrame({'record_id': ['source:2', 'source:3'],
                             'source_row': [2, 3], 'rf_score': [.2, .9],
                             'model_version': ['model', 'model']})
    actual = expected.assign(review_status=['待复核', '确认正常'], review_note=['', 'basis'])
    audit = [{'id': 1, 'record_id': 'source:3', 'model_version': 'model',
              'status': '确认正常', 'note': 'basis', 'created_at': '2026-01-01'}]
    return actual, expected, audit


def test_export_validator_accepts_complete_ordered_snapshot(exports):
    actual, expected, audit = exports
    validate_exports(actual, expected, audit, copy.deepcopy(audit), 'model')


@pytest.mark.parametrize('fault', ['extra_label', 'truncated', 'reordered', 'wrong_score',
                                   'wrong_review', 'wrong_model_history', 'unrelated_history'])
def test_export_validator_rejects_changes_that_counts_alone_miss(exports, fault):
    actual, expected, audit = exports
    expected_audit = copy.deepcopy(audit)
    if fault == 'extra_label':
        actual['Label'] = ['Benign', 'Bot']
    elif fault == 'truncated':
        actual = actual.iloc[:1]
    elif fault == 'reordered':
        actual = actual.iloc[::-1]
    elif fault == 'wrong_score':
        actual.loc[0, 'rf_score'] = .3
    elif fault == 'wrong_review':
        actual.loc[0, 'review_status'] = '确认异常'
    elif fault == 'wrong_model_history':
        audit[0]['model_version'] = 'other'
    else:
        audit.append({**audit[0], 'id': 2, 'record_id': 'unrelated'})
    with pytest.raises(AssertionError):
        validate_exports(actual, expected, audit, expected_audit, 'model')


@pytest.fixture
def completed_case():
    names = ['real_upload_full_prediction_quality_and_history_scope',
             'ranked_table_first_1000_matches_complete_export',
             'last_record_identity_and_queue_end',
             'last_review_filter_full_exports_and_unrelated_history_preserved',
             'browser_refresh_reupload_preserves_history_and_all_source_ids',
             'server_restart_reupload_preserves_history_and_all_source_ids',
             'rejected_replacement_preserves_accepted_input_and_all_exports',
             'explicit_input_removal_clears_prediction_and_review_controls',
             'valid_recovery_after_limits_preserves_last_review_and_all_exports']
    checks = [{'name': name, 'passed': True} for name in names]
    checks += [{'name': 'limit_rejection_preserves_history', 'passed': True, 'case': case}
               for case in ('rows-50001', 'bytes-over-50mib')]
    metric = {'seconds': 1., 'sample_count': 2, 'peak_sampled_rss_bytes': 100,
              'measurement_errors': []}
    direct = ('read_and_csv_validation', 'adapter_and_real_inference', 'field_quality_view')
    browser = ('upload_validate_predict_and_render', 'ranked_visible_table', 'select_last_record',
               'save_last_record', 'status_filter', 'all_exports_after_filter', 'refresh_reupload',
               'exports_after_refresh', 'restart_reupload', 'exports_after_restart',
               'reject-oversized-replacement', 'exports_after_rejected_replacement',
               'remove-accepted-input',
               'reject-rows-50001', 'reject-bytes-over-50mib', 'recover-after-limits',
               'exports_after_limits')
    return {'quality': {'affected_rows': 0}, 'checks': checks, 'passed': True,
            'page_errors': [], 'cleanup_errors': [],
            'server_starts': [{'cleanup': {'passed': True, 'remaining': [], 'errors': []}}
                              for _ in range(2)],
            'database_after': {'rows': 100001}, 'history_seed': {'total_rows': 100000},
            'direct_measurements': {name: dict(metric) for name in direct},
            'measurements': {name: dict(metric) for name in browser}}


def test_complete_case_accepts_all_required_evidence(completed_case):
    validate_completed_case(completed_case, include_limits=True)


@pytest.mark.parametrize('name', [
    'rejected_replacement_preserves_accepted_input_and_all_exports',
    'explicit_input_removal_clears_prediction_and_review_controls',
])
def test_limit_success_requires_replacement_and_removal_evidence(completed_case, name):
    completed_case['checks'] = [item for item in completed_case['checks'] if item['name'] != name]
    with pytest.raises(AssertionError, match='Missing capacity stages'):
        validate_completed_case(completed_case, include_limits=True)


@pytest.mark.parametrize('name', ['reject-oversized-replacement',
                                'exports_after_rejected_replacement', 'remove-accepted-input'])
def test_limit_success_requires_each_replacement_and_removal_measurement(completed_case, name):
    del completed_case['measurements'][name]
    with pytest.raises(AssertionError, match='measurements are missing'):
        validate_completed_case(completed_case, include_limits=True)


def test_restart_claim_requires_a_second_owned_service(completed_case):
    completed_case['server_starts'].pop()
    with pytest.raises(AssertionError):
        validate_completed_case(completed_case, include_limits=True)


def test_one_generic_measurement_cannot_prove_required_capacity_phases(completed_case):
    completed_case['measurements'] = {
        'app': copy.deepcopy(completed_case['measurements']['upload_validate_predict_and_render'])}
    with pytest.raises(AssertionError):
        validate_completed_case(completed_case, include_limits=True)


@pytest.mark.parametrize('fault', ['missing_stage', 'missing_limit', 'missing_cleanup',
                                   'leftover_process', 'page_error', 'missing_metrics',
                                   'memory_error', 'missing_append', 'missing_quality'])
def test_incomplete_case_cannot_be_reported_as_passed(completed_case, fault):
    case = completed_case
    if fault == 'missing_stage':
        case['checks'].pop(0)
    elif fault == 'missing_limit':
        case['checks'].pop()
    elif fault == 'missing_cleanup':
        case['server_starts'][0].pop('cleanup')
    elif fault == 'leftover_process':
        case['server_starts'][0]['cleanup']['remaining'] = [{'pid': 123}]
    elif fault == 'page_error':
        case['page_errors'] = ['browser error']
    elif fault == 'missing_metrics':
        case['measurements'] = {}
    elif fault == 'memory_error':
        case['measurements']['upload_validate_predict_and_render']['measurement_errors'] = ['access denied']
    elif fault == 'missing_append':
        case['database_after']['rows'] = 100000
    else:
        case['quality']['affected_rows'] = 2
    with pytest.raises(AssertionError):
        validate_completed_case(case, include_limits=True)


@pytest.mark.parametrize('field,value', [('seconds', float('nan')), ('seconds', float('inf')),
                                        ('seconds', -1), ('sample_count', 0),
                                        ('sample_count', True), ('peak_sampled_rss_bytes', 0)])
def test_invalid_measurement_never_counts_as_valid_evidence(completed_case, field, value):
    completed_case['measurements']['upload_validate_predict_and_render'][field] = value
    with pytest.raises(AssertionError):
        validate_completed_case(completed_case, include_limits=True)


@pytest.fixture(scope='module')
def edge_browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        try:
            yield browser
        finally:
            browser.close()


def test_file_chooser_reselects_the_same_local_file_after_ui_removal(edge_browser, tmp_path):
    path = tmp_path / 'repeated.csv'
    path.write_bytes(b'a\n1\n')
    page = edge_browser.new_page()
    try:
        # Native inputs retain their selection when an uploader removes only its
        # displayed entry. Browsing again clears that selection without firing
        # a separate change event that could clear the accepted application input.
        page.set_content('''<input type="file" id="file" hidden
                onchange="document.querySelector('#events').textContent = ++events;
                          document.querySelector('#entry').textContent = this.files[0].name">
            <button onclick="document.querySelector('#file').value = '';
                             document.querySelector('#file').click()">Browse files</button>
            <button onclick="document.querySelector('#entry').textContent = ''">Remove entry</button>
            <p id="entry"></p><p id="events">0</p><script>let events = 0;</script>''')
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout = page, 2000
        runner.choose_local_file(path)
        assert page.locator('#entry').inner_text() == path.name
        assert page.locator('#events').inner_text() == '1'
        page.get_by_role('button', name='Remove entry').click()
        page.locator('input[type=file]').set_input_files(str(path))
        assert page.locator('#entry').inner_text() == ''
        assert page.locator('#events').inner_text() == '1'
        runner.choose_local_file(path)
        assert page.locator('#entry').inner_text() == path.name
        assert page.locator('#events').inner_text() == '2'
    finally:
        page.close()


@pytest.mark.parametrize('fault', [None, 'wrong_main_input', 'missing_accepted_file'])
def test_rejected_replacement_requires_the_previous_input_identity(edge_browser, fault):
    page = edge_browser.new_page()
    try:
        page.set_content('''<button>Remove accepted.csv</button>
            <p>File must be 50.0MB or smaller.</p>
            <main data-testid="stMain"><p>accepted.csv</p>
                <button>检查数据并预测</button><button>保存复核</button></main>''')
        if fault == 'wrong_main_input':
            page.locator('main p').evaluate("node => node.textContent = 'rejected.csv'")
        elif fault == 'missing_accepted_file':
            page.get_by_role('button', name='Remove accepted.csv').evaluate('node => node.remove()')
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout, runner.name = page, 250, 'accepted.csv'
        if fault:
            with pytest.raises(AssertionError):
                runner.assert_retained_input()
        else:
            runner.assert_retained_input()
    finally:
        page.close()


@pytest.mark.parametrize('stale_control', [None, '检查数据并预测', '导出预测与复核 CSV',
                                         '导出完整复核历史', '导出运行清单', '保存复核',
                                         '保存并查看下一条'])
def test_empty_input_cannot_keep_prediction_review_or_export_controls(edge_browser, stale_control):
    page = edge_browser.new_page()
    try:
        page.set_content('<p>请在左侧选择 CSV，或切换到内置演示数据。</p>'
                         + (f'<button>{stale_control}</button>' if stale_control else ''))
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout = page, 250
        if stale_control:
            with pytest.raises(AssertionError):
                runner.assert_empty_input()
        else:
            runner.assert_empty_input()
    finally:
        page.close()


@pytest.mark.parametrize('rerender_once', [False, True])
@pytest.mark.parametrize('backend_reset', [False, True])
def test_capacity_picker_handles_virtual_mouse_detachment_and_transient_rerender(
    edge_browser, rerender_once, backend_reset,
):
    page = edge_browser.new_page()
    try:
        page.set_content('''
            <input role="combobox" aria-label="Selected first. capacity records"
                   onclick="renderOptions(this)" oninput="renderOptions(this)"
                   onkeydown="selectOption(event, this)">
            <ul id="options"></ul>
            <script>
              let transient = RERENDER_ONCE;
              let resetAfterSelect = BACKEND_RESET;
              let selected;
              function renderOptions(input) {
                if (!input.value) return;
                const list = document.querySelector('#options');
                list.innerHTML = '<li role="option">last record</li>';
                const option = list.firstElementChild;
                option.onmousedown = () => { list.innerHTML = ''; };
                if (transient) {
                  transient = false;
                  input.value = '';
                  list.innerHTML = '';
                }
              }
              function selectOption(event, input) {
                if (event.key === 'ArrowDown') selected = document.querySelector('[role=option]');
                if (event.key === 'Enter' && selected) {
                  input.setAttribute('aria-label', 'Selected ' + selected.textContent + '. capacity records');
                  document.querySelector('#options').innerHTML = '';
                  if (resetAfterSelect) {
                    resetAfterSelect = false;
                    const running = document.createElement('div');
                    running.setAttribute('data-testid', 'stStatusWidgetRunningIcon');
                    document.body.appendChild(running);
                    setTimeout(() => {
                      input.setAttribute('aria-label', 'Selected first. capacity records');
                      input.value = '';
                      running.remove();
                    }, 200);
                  }
                }
              }
            </script>
        '''.replace('RERENDER_ONCE', 'true' if rerender_once else 'false')
           .replace('BACKEND_RESET', 'true' if backend_reset else 'false'))
        runner = CapacityBrowser.__new__(CapacityBrowser)
        # The combined fixture deliberately consumes two five-second failed
        # selection windows before recovery; this is not a product time budget.
        runner.page, runner.timeout = page, 15000
        runner.pick('capacity records', 'last record')
        assert page.get_by_role('combobox').get_attribute('aria-label') == (
            'Selected last record. capacity records')
        assert page.get_by_test_id('stStatusWidgetRunningIcon').count() == 0
    finally:
        page.close()


def test_capacity_picker_never_bypasses_a_running_service_deadline(edge_browser):
    page = edge_browser.new_page()
    try:
        page.set_content('''<div data-testid="stStatusWidgetRunningIcon">Running...</div>
            <input role="combobox" aria-label="Selected first. capacity records">''')
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout = page, 250
        with pytest.raises(AssertionError):
            runner.pick('capacity records', 'last record')
        assert page.get_by_role('combobox').input_value() == ''
    finally:
        page.close()


def test_capacity_picker_retries_only_transient_duplicate_widgets_between_keys(edge_browser):
    page = edge_browser.new_page()
    try:
        page.set_content('''
            <input role="combobox" aria-label="Selected first. capacity records"
                   oninput="showOption()" onkeydown="choose(event, this)">
            <ul id="options"></ul>
            <script>
              let duplicateOnce = true;
              function showOption() {
                document.querySelector('#options').innerHTML = '<li role="option">last record</li>';
              }
              function choose(event, input) {
                if (event.key === 'ArrowDown' && duplicateOnce) {
                  duplicateOnce = false;
                  const previous = input.cloneNode(true);
                  document.body.appendChild(previous);
                  setTimeout(() => previous.remove(), 200);
                }
                if (event.key === 'Enter') {
                  input.setAttribute('aria-label', 'Selected last record. capacity records');
                  document.querySelector('#options').innerHTML = '';
                }
              }
            </script>
        ''')
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout = page, 10000
        runner.pick('capacity records', 'last record')
        assert page.get_by_role('combobox').get_attribute('aria-label') == (
            'Selected last record. capacity records')
        assert page.get_by_role('combobox').count() == 1
    finally:
        page.close()


def test_capacity_picker_rejects_persistent_duplicate_controls_without_selecting(edge_browser):
    page = edge_browser.new_page()
    try:
        page.set_content('''<input role="combobox" aria-label="Selected first. capacity records">
            <input role="combobox" aria-label="Selected second. capacity records">''')
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout = page, 250
        with pytest.raises(AssertionError):
            runner.pick('capacity records', 'last record')
        assert page.get_by_role('combobox').count() == 2
        assert page.locator('input').evaluate_all('(inputs) => inputs.every(input => !input.value)')
    finally:
        page.close()


def test_capacity_picker_does_not_retry_other_playwright_errors(monkeypatch):
    from scripts import engineering_capacity

    calls = []

    def disconnected(_):
        calls.append(True)
        raise PlaywrightError('Browser connection closed')

    monkeypatch.setattr(engineering_capacity, 'expect', disconnected)
    runner = CapacityBrowser.__new__(CapacityBrowser)
    runner.page = SimpleNamespace(get_by_role=lambda *a, **k: object(),
                                  get_by_test_id=lambda *a: object())
    runner.timeout = 10000
    with pytest.raises(PlaywrightError, match='Browser connection closed'):
        runner.pick('capacity records', 'last record')
    assert calls == [True]


@pytest.fixture
def record_picker(edge_browser):
    pages = []
    expect.set_options(timeout=1000)

    def create(mode):
        page = edge_browser.new_page()
        pages.append(page)
        target = 'a' * 64 + ':10001'
        option = f'行 10001 ｜ 得分 0.0000 ｜ {target[:12]}'
        page.set_content('''
            <input role="combobox" aria-label="Selected first. 选择记录查看详情"
                   oninput="showOption()" onkeydown="choose(event, this)">
            <ul id="options"></ul>
            <div data-testid="stCode"><code id="detail">first:2</code></div>
            <script>
              const mode = MODE, target = TARGET, option = OPTION;
              window.selections = 0;
              function showOption() {
                const item = document.createElement('li');
                item.setAttribute('role', 'option');
                item.textContent = option;
                document.querySelector('#options').replaceChildren(item);
              }
              function choose(event, input) {
                if (event.key !== 'Enter') return;
                window.selections += 1;
                input.setAttribute('aria-label', 'Selected ' + option + '. 选择记录查看详情');
                document.querySelector('#options').replaceChildren();
                if (mode === 'correct' || (mode === 'late_reset' && window.selections > 1)) {
                  setTimeout(() => document.querySelector('#detail').textContent = target, 80);
                } else if (mode === 'prefix_detail') {
                  document.querySelector('#detail').textContent = target + '0';
                } else if (mode === 'late_reset' || mode === 'permanent_reset') {
                  // The first label and Running=0 can both pass before the
                  // previous backend rerender begins and replaces the selection.
                  setTimeout(() => {
                    const running = document.createElement('div');
                    running.setAttribute('data-testid', 'stStatusWidgetRunningIcon');
                    document.body.appendChild(running);
                  }, 60);
                  setTimeout(() => {
                    input.setAttribute('aria-label', 'Selected first. 选择记录查看详情');
                    input.value = '';
                    document.querySelector('#detail').textContent = 'first:2';
                    document.querySelector('[data-testid=stStatusWidgetRunningIcon]').remove();
                  }, 220);
                }
              }
            </script>
        '''.replace('MODE', json.dumps(mode)).replace('TARGET', json.dumps(target))
           .replace('OPTION', json.dumps(option)))
        runner = CapacityBrowser.__new__(CapacityBrowser)
        runner.page, runner.timeout = page, 1200
        predictions = pd.DataFrame({'record_id': [target], 'source_row': [10001],
                                    'rf_score': [0.], 'source_file': ['fixture.csv']})
        return runner, predictions, target

    try:
        yield create
    finally:
        for page in pages:
            page.close()
        expect.set_options(timeout=5000)


@pytest.mark.parametrize('mode,selections', [('correct', 1), ('late_reset', 2)])
def test_capacity_record_selection_waits_for_detail_after_late_rerender(record_picker, mode, selections):
    runner, predictions, target = record_picker(mode)
    runner.select_record(target, predictions=predictions)
    assert runner.page.locator('#detail').inner_text() == target
    assert runner.page.evaluate('window.selections') == selections


@pytest.mark.parametrize('mode', ['wrong_detail', 'prefix_detail'])
def test_capacity_record_selection_never_retries_stable_wrong_detail(record_picker, mode):
    runner, predictions, target = record_picker(mode)
    started = time.monotonic()
    with pytest.raises(AssertionError):
        runner.select_record(target, predictions=predictions)
    assert runner.page.evaluate('window.selections') == 1
    assert time.monotonic() - started < 2.5


def test_capacity_record_selection_permanent_reset_shares_one_deadline(record_picker):
    runner, predictions, target = record_picker('permanent_reset')
    started = time.monotonic()
    # The shared deadline can expire in either the detail assertion or a picker action.
    with pytest.raises((AssertionError, PlaywrightTimeoutError)):
        runner.select_record(target, predictions=predictions)
    assert runner.page.evaluate('window.selections') >= 2
    assert time.monotonic() - started < 2.5
