"""Acceptance assertions must detect stale conclusions and preserve raw CSV text."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pandas as pd
import pytest
from playwright.sync_api import expect

from scripts.engineering_p0_browser import csv
from scripts.engineering_p0_cases import assert_review_snapshot
from scripts import engineering_browser, engineering_p0_browser
from scripts.engineering_run import start_owned_process
from test_engineering_p0_browser import edge_browser  # noqa: F401


def test_quality_download_keeps_original_nonfinite_and_empty_cell_text(tmp_path):
    target = tmp_path / 'quality.csv'
    expected = pd.DataFrame({
        '字段': ['one', 'two', 'three', 'four'],
        'CSV读取值': ['NaN', 'Infinity', '-Infinity', ''],
        '适配结果': ['适配后缺失'] * 4,
        '处理状态': ['可判断，使用既有训练填补器'] * 4,
    })
    expected.to_csv(target, index=False)
    pd.testing.assert_frame_equal(csv(target), expected)


@pytest.fixture
def snapshots():
    records = pd.DataFrame({
        'record_id': ['first:2', 'first:3', 'first:4'],
        'model_version': ['frozen'] * 3,
        'review_status': ['确认正常', '待补证', '待复核'],
        'review_note': ['latest', 'context', ''],
    })
    audit = [
        {'id': 1, 'record_id': 'first:2', 'model_version': 'frozen',
         'status': '确认异常', 'note': 'old'},
        {'id': 3, 'record_id': 'first:3', 'model_version': 'frozen',
         'status': '待补证', 'note': 'context'},
        {'id': 5, 'record_id': 'first:2', 'model_version': 'frozen',
         'status': '确认正常', 'note': 'latest'},
        # Other uploads belong in the complete model history, not the current CSV.
        {'id': 8, 'record_id': 'second:2', 'model_version': 'frozen',
         'status': '确认异常', 'note': 'other input'},
    ]
    return records, audit


def test_same_model_snapshot_accepts_last_history_conclusion_and_untouched_rows(snapshots):
    records, audit = snapshots
    assert_review_snapshot(records, audit, 'frozen')


@pytest.mark.parametrize('fault', ['stale-status', 'stale-note', 'foreign-version',
                                  'duplicate-history-id', 'unordered-history',
                                  'missing-history', 'duplicate-record'])
def test_snapshot_assertion_rejects_inconsistent_csv_or_history(snapshots, fault):
    records, original = snapshots
    audit = json.loads(json.dumps(original))
    if fault == 'stale-status':
        records.loc[0, 'review_status'] = '确认异常'
    elif fault == 'stale-note':
        records.loc[0, 'review_note'] = 'old'
    elif fault == 'foreign-version':
        audit[-1]['model_version'] = 'rebuilt'
    elif fault == 'duplicate-history-id':
        audit[-1]['id'] = audit[-2]['id']
    elif fault == 'unordered-history':
        audit.reverse()
    elif fault == 'missing-history':
        audit = [row for row in audit if row['record_id'] != 'first:2']
    elif fault == 'duplicate-record':
        records = pd.concat([records, records.iloc[:1]], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_review_snapshot(records, audit, 'frozen')


@pytest.mark.parametrize('runner', ['standard', 'p0'])
def test_browser_initialization_failure_preserves_atomic_failed_report(
    tmp_path, monkeypatch, runner,
):
    output = tmp_path / 'new-evidence'
    missing = tmp_path / 'missing-model'
    if runner == 'standard':
        module = engineering_browser
        argv = ['browser', '--workspace', str(module.ROOT), '--artifacts', str(missing),
                '--expected-model-sha256', '0' * 64, '--output', str(output)]
    else:
        module = engineering_p0_browser
        argv = ['p0', '--frozen-artifacts', str(missing), '--rebuilt-artifacts', str(missing),
                '--output', str(output)]
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(FileNotFoundError):
        module.main()
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed' and report['passed'] is False
    assert 'FileNotFoundError' in report['failure']
    assert report['source_before']['head'] == report['source_after']['head']
    assert report['source_integrity']['passed'] is True
    assert list(output.iterdir()) == [output / 'report.json']


@pytest.mark.parametrize('runner', ['standard', 'p0'])
def test_browser_refuses_old_output_without_replacing_old_evidence(tmp_path, monkeypatch, runner):
    output = tmp_path / 'old-evidence'
    output.mkdir()
    report = output / 'report.json'
    report.write_text('previous report', encoding='utf-8')
    if runner == 'standard':
        module = engineering_browser
        argv = ['browser', '--workspace', str(module.ROOT), '--output', str(output)]
    else:
        module = engineering_p0_browser
        argv = ['p0', '--frozen-artifacts', str(tmp_path), '--rebuilt-artifacts', str(tmp_path),
                '--output', str(output)]
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(FileExistsError):
        module.main()
    assert report.read_text(encoding='utf-8') == 'previous report'


@pytest.fixture
def select_server(tmp_path):
    application = tmp_path / 'virtualized_select.py'
    application.write_text(
        'import streamlit as st\n'
        'selected = st.selectbox("Virtual record", [f"row-{i:04d}" for i in range(1000)])\n'
        'st.text_input("Draft", key="draft")\n'
        'records = {f"current:{i}": (i, i / 10) for i in (2, 3)}\n'
        'chosen = st.selectbox("选择记录查看详情", list(records), format_func=lambda rid: '
        'f"行 {records[rid][0]} ｜ 得分 {records[rid][1]:.4f} ｜ {rid[:12]}")\n'
        'st.code(chosen)\n'
        'st.write("Chosen " + selected)\n', encoding='utf-8')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    process = owner = None
    with (tmp_path / 'virtualized-select.log').open('w', encoding='utf-8') as log:
        try:
            process, owner = start_owned_process(
                [sys.executable, '-m', 'streamlit', 'run', str(application),
                 '--server.address=127.0.0.1', f'--server.port={port}', '--server.headless=true',
                 '--server.fileWatcherType=none', '--browser.gatherUsageStats=false'],
                cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONUTF8': '1'},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                assert process.poll() is None
                try:
                    with opener.open(url + '/_stcore/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.05)
            else:
                pytest.fail('isolated selectbox fixture did not become healthy')
            yield url
        finally:
            if owner is not None:
                result = owner.cleanup()
                assert result['passed'], result


def test_record_picker_searches_real_streamlit_virtualized_options(edge_browser, select_server):
    """A distant row is absent from the dropdown DOM until its search input is used."""
    page = edge_browser.new_page()
    page.set_default_timeout(2000)
    try:
        page.goto(select_server)
        control = page.get_by_role('combobox', name='Virtual record', exact=False)
        control.click()
        assert page.get_by_role('option', name='row-0999', exact=True).count() == 0
        control.press('Escape')
        runner = engineering_p0_browser.P0Browser.__new__(engineering_p0_browser.P0Browser)
        runner.page = page
        runner.pick('Virtual record', 'row-0999')
        assert control.get_attribute('aria-label') == 'Selected row-0999. Virtual record'
        expect(page.get_by_text('Chosen row-0999', exact=True)).to_be_visible()
    finally:
        page.close()


def test_second_review_window_opens_app_with_an_independent_session(edge_browser, select_server):
    context = edge_browser.new_context()
    try:
        first = context.new_page()
        first.goto(select_server)
        first.get_by_role('textbox', name='Draft', exact=True).fill('first window draft')
        first.get_by_role('textbox', name='Draft', exact=True).press('Enter')
        runner = engineering_p0_browser.P0Browser.__new__(engineering_p0_browser.P0Browser)
        runner.context, runner.errors, runner.url = context, [], select_server
        second = runner.new_review_window()
        assert second.url == select_server + '/'
        expect(second.get_by_role('textbox', name='Draft', exact=True)).to_have_value('')
        second.get_by_role('textbox', name='Draft', exact=True).fill('second window draft')
        expect(first.get_by_role('textbox', name='Draft', exact=True)).to_have_value('first window draft')
        assert runner.errors == []
    finally:
        context.close()


def test_unique_click_waits_for_obsolete_streamlit_widget_to_disappear(edge_browser):
    page = edge_browser.new_page()
    try:
        page.set_content('''
            <button id="old" onclick="window.clicked='obsolete'">原始顺序</button>
            <button id="current" onclick="window.clicked='current'">原始顺序</button>
            <script>setTimeout(() => document.getElementById('old').remove(), 250)</script>
        ''')
        engineering_p0_browser.click_unique(page.get_by_text('原始顺序', exact=True), timeout=2000)
        assert page.evaluate('window.clicked') == 'current'
    finally:
        page.close()


def test_unique_click_refuses_persistent_duplicate_widgets_without_clicking(edge_browser):
    page = edge_browser.new_page()
    try:
        page.set_content('''
            <button onclick="window.clicked=true">原始顺序</button>
            <button onclick="window.clicked=true">原始顺序</button>
        ''')
        with pytest.raises(AssertionError, match='count'):
            engineering_p0_browser.click_unique(page.get_by_text('原始顺序', exact=True), timeout=200)
        assert page.evaluate('window.clicked') is None
    finally:
        page.close()


def test_record_selection_uses_this_upload_identity_instead_of_previous_snapshot(
    edge_browser, select_server,
):
    page = edge_browser.new_page()
    try:
        page.goto(select_server)
        runner = engineering_p0_browser.P0Browser.__new__(engineering_p0_browser.P0Browser)
        runner.page, runner.kind = page, 'frozen'
        runner.snapshots = {'frozen': pd.DataFrame({
            'record_id': ['current:3'], 'source_row': [99], 'rf_score': [0.99],
            'source_file': ['previous.csv']})}
        current = pd.DataFrame({'record_id': ['current:3'], 'source_row': [3], 'rf_score': [0.3],
                                'source_file': ['current.csv']})
        runner.select_record('current:3', predictions=current)
        expect(page.locator('[data-testid="stCode"]').filter(has_text='current:3')).to_be_visible()
    finally:
        page.close()
