"""Real Edge capacity checks with repeated demo rows and isolated review history.

These are engineering loads, never independent algorithm evaluation samples.
No training, model changes, dependency changes, or database migration is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv as csv_module
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import sys
import threading
import time

import pandas as pd
from playwright.sync_api import (
    Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError, expect, sync_playwright,
)
import psutil

from app_model import load_model_bundle, prepare_model
from app_quality import build_quality_view
from liushao.data import read_csv_checked, sha256
from liushao.model import infer, ranked
from liushao.reviews import connect
from scripts.engineering_p0_browser import (
    DISPLAY, P0Browser, assert_prediction_export, csv,
)
from scripts.engineering_p0_cases import assert_frames, assert_review_snapshot
from scripts.engineering_run import AcceptanceRun


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (1001, 10000, 50000)
MAX_BYTES = 50 * 1024 * 1024
MODEL_FILES = ('model.joblib', 'model.json', 'demo.csv', 'demo.json', 'demo_origins.parquet')


def repeat_demo_rows(content: bytes, count: int) -> bytes:
    """Repeat parsed CSV cells without inventing independent source observations."""
    if type(count) is not int or count < 1:
        raise ValueError('The engineering row count must be a positive integer')
    source = read_csv_checked(io.BytesIO(content))
    if source.empty:
        raise ValueError('The demo must contain at least one record')
    stream = io.StringIO(newline='')
    writer = csv_module.writer(stream, lineterminator='\n')
    writer.writerow(source.columns)
    records = list(source.itertuples(index=False, name=None))
    writer.writerows(records[index % len(records)] for index in range(count))
    return stream.getvalue().encode('utf-8')


def frozen_hashes(artifacts: Path, manifest_path: Path | None = None) -> dict:
    """Require all five exact trusted files before any joblib deserialization."""
    manifest_path = manifest_path or ROOT / 'docs/model-handoff-20261016/FROZEN_MODEL_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {Path(item['path']).name: item for item in manifest['files']}
    result = {}
    for name in MODEL_FILES:
        path = artifacts / name
        item = expected[name]
        actual = sha256(path)
        if actual != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError(f'Trusted frozen handoff mismatch: {name}')
        result[name] = actual
    return result


def seed_history(path: Path, version: str, record_id: str, count: int) -> dict:
    """Populate a new test DB with related, unrelated and other-version history."""
    if path.exists() or count < 60:
        raise ValueError('History needs a new database and at least 60 fixture rows')
    db = connect(path)
    related = 25
    other_version = 25
    try:
        def records():
            for index in range(count - related - other_version):
                yield (f'capacity-unrelated:{index}', version, '待补证',
                       'CAPACITY_FIXTURE unrelated history', '2026-01-01T00:00:00+00:00')
            for index in range(other_version):
                yield (record_id, 'capacity-other-model', '确认正常',
                       f'CAPACITY_FIXTURE other model {index}', '2026-01-01T00:00:00+00:00')
            for index in range(related):
                yield (record_id, version, '待补证',
                       f'CAPACITY_FIXTURE related revision {index}', '2026-01-01T00:00:00+00:00')
        db.executemany(
            'INSERT INTO reviews (record_id, model_version, status, note, created_at) '
            'VALUES (?, ?, ?, ?, ?)', records())
        db.commit()
        actual = db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        if actual != count:
            raise AssertionError('History fixture did not write the declared number of rows')
        return {'total_rows': count, 'related_rows': related,
                'unrelated_same_version_rows': count - related - other_version,
                'other_version_rows': other_version}
    finally:
        db.close()


def read_history(path: Path, *, version: str | None = None,
                 record_id: str | None = None) -> list[dict]:
    """Read the oracle without creating or changing the test database."""
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        where, parameters = [], []
        for column, value in (('model_version', version), ('record_id', record_id)):
            if value is not None:
                where.append(column + '=?')
                parameters.append(value)
        query = 'SELECT * FROM reviews' + (' WHERE ' + ' AND '.join(where) if where else '')
        return [dict(row) for row in db.execute(query + ' ORDER BY id', parameters)]


def history_fingerprint(path: Path, *, exclude_identity: tuple[str, str] | None = None) -> dict:
    """Hash ordered logical rows, so read/limit rejection can prove no history mutation."""
    digest, count = hashlib.sha256(), 0
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        parameters = ()
        query = 'SELECT * FROM reviews'
        if exclude_identity is not None:
            query += ' WHERE NOT (model_version=? AND record_id=?)'
            parameters = exclude_identity
        for row in db.execute(query + ' ORDER BY id', parameters):
            digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            digest.update(b'\n')
            count += 1
    return {'rows': count, 'sha256': digest.hexdigest()}


def validate_exports(actual: pd.DataFrame, expected: pd.DataFrame, audit: list[dict],
                     expected_audit: list[dict], model_version: str) -> None:
    """Check full values/order, exact schema and snapshot identity, not row counts alone."""
    assert_prediction_export(actual, expected)
    assert actual.record_id.is_unique
    assert actual.source_row.tolist() == list(range(2, len(expected) + 2))
    assert audit == expected_audit, 'History export contains missing or unrelated records'
    assert_review_snapshot(actual, audit, model_version)


def validate_completed_case(case: dict, *, include_limits: bool) -> None:
    """A successful loop is insufficient when a required stage or cleanup is absent."""
    required = {
        'real_upload_full_prediction_quality_and_history_scope',
        'ranked_table_first_1000_matches_complete_export',
        'last_record_identity_and_queue_end',
        'last_review_filter_full_exports_and_unrelated_history_preserved',
        'browser_refresh_reupload_preserves_history_and_all_source_ids',
        'server_restart_reupload_preserves_history_and_all_source_ids',
    }
    direct_phases = {'read_and_csv_validation', 'adapter_and_real_inference', 'field_quality_view'}
    browser_phases = {
        'upload_validate_predict_and_render', 'ranked_visible_table', 'select_last_record',
        'save_last_record', 'status_filter', 'all_exports_after_filter', 'refresh_reupload',
        'exports_after_refresh', 'restart_reupload', 'exports_after_restart',
    }
    if case['quality']['affected_rows']:
        required.add('last_problem_field_cells_preserve_input_and_editor_identity')
        browser_phases.add('last_quality_problem_record')
    checks = case.get('checks', [])
    names = {item['name'] for item in checks if item.get('passed') is True}
    if include_limits:
        required.update({
            'rejected_replacement_preserves_accepted_input_and_all_exports',
            'explicit_input_removal_clears_prediction_and_review_controls',
            'valid_recovery_after_limits_preserves_last_review_and_all_exports',
        })
        cases = {item.get('case') for item in checks
                 if item['name'] == 'limit_rejection_preserves_history' and item.get('passed')}
        assert cases == {'rows-50001', 'bytes-over-50mib'}, 'Missing an input-limit rejection'
        browser_phases.update({'reject-rows-50001', 'reject-bytes-over-50mib',
                               'reject-oversized-replacement', 'exports_after_rejected_replacement',
                               'remove-accepted-input',
                               'recover-after-limits', 'exports_after_limits'})
    assert required <= names, f'Missing capacity stages: {sorted(required - names)}'
    assert all(item.get('passed') is True for item in checks)
    assert case.get('passed') is True and not case.get('page_errors')
    assert not case.get('cleanup_errors')
    assert len(case.get('server_starts', [])) == 2, 'Missing a real owned-service restart'
    assert all(
        item.get('cleanup', {}).get('passed') is True
        and not item['cleanup'].get('remaining') and not item['cleanup'].get('errors')
        for item in case['server_starts'])
    assert case['database_after']['rows'] == case['history_seed']['total_rows'] + 1
    for measurements, required_phases in ((case['direct_measurements'], direct_phases),
                                          (case['measurements'], browser_phases)):
        assert required_phases <= measurements.keys(), 'Required capacity measurements are missing'
        for value in measurements.values():
            assert math.isfinite(value['seconds']) and value['seconds'] >= 0
            assert type(value['sample_count']) is int and value['sample_count'] > 0
            assert value['peak_sampled_rss_bytes'] > 0 and not value['measurement_errors']


class MemorySamples:
    """Sample aggregate RSS, a lower bound on peak; never claim an exact peak or SLA."""
    def __init__(self, processes, interval: float = .05):
        self.processes = processes
        self.interval = interval
        self.stop_event = threading.Event()
        self.values = []
        self.errors = []

    def sample(self):
        try:
            rss = 0
            for process in self.processes():
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            self.values.append(rss)
        except (OSError, psutil.Error, RuntimeError) as error:
            self.errors.append(f'{type(error).__name__}: {error}')

    def loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.started = time.perf_counter()
        self.cpu_before = psutil.cpu_percent(interval=None)
        self.available_before = psutil.virtual_memory().available
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.sample()
        self.result = {
            'seconds': time.perf_counter() - self.started,
            'peak_sampled_rss_bytes': max(self.values, default=0),
            'start_rss_bytes': self.values[0] if self.values else None,
            'end_rss_bytes': self.values[-1] if self.values else None,
            'sample_count': len(self.values), 'sample_interval_seconds': self.interval,
            'measurement_errors': self.errors,
            'system_cpu_percent_before': self.cpu_before,
            'system_cpu_percent_after': psutil.cpu_percent(interval=None),
            'system_available_bytes_before': self.available_before,
            'system_available_bytes_after': psutil.virtual_memory().available,
        }
        if self.thread.is_alive() or not self.values or self.errors:
            raise RuntimeError('Capacity memory measurement did not complete reliably')


@contextmanager
def measured(target: dict, name: str, processes, scope: str):
    sample = MemorySamples(processes)
    try:
        with sample:
            yield
    finally:
        if hasattr(sample, 'result'):
            target[name] = {**sample.result, 'scope': scope}


class CapacityBrowser(P0Browser):
    """Use the established UI helpers and owned-server lifecycle for one capacity case."""
    def __init__(self, output: Path, acceptance: AcceptanceRun, model: dict,
                 payload: bytes, predictions: pd.DataFrame, quality, case: dict, timeout: int):
        self.out, self.acceptance, self.case = output, acceptance, case
        self.report = acceptance.report
        self.models = {'frozen': model}
        self.kind = 'frozen'
        self.content, self.predictions, self.quality = payload, predictions, quality
        self.digest = hashlib.sha256(payload).hexdigest()
        self.name = case['input_name']
        self.last = predictions.iloc[-1]
        self.version = model['bundle']['model_version']
        self.database = output / 'reviews.sqlite'
        self.timeout = timeout
        self.process = self.process_tree = self.log = self.page = self.context = None
        self.browser = self.playwright = None
        self.errors, self.starts = [], []
        self.counter = 0
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        self.url = f'http://127.0.0.1:{self.port}'
        case.update(checks=[], server_starts=self.starts, page_errors=self.errors,
                    measurements={}, database=str(self.database))

    def mark(self, name, **details):
        self.case['checks'].append({'name': name, 'passed': True, **details})
        self.acceptance.checkpoint()
        print(f'PASS {self.case["rows"]}: {name}', flush=True)

    def server_processes(self):
        # Query only current kernel Job members; unrelated host/Edge processes are excluded.
        if self.process_tree is None or self.process_tree.job is None:
            return []
        return [psutil.Process(pid) for pid in self.process_tree.job.pids()]

    def phase(self, name):
        return measured(self.case['measurements'], name, self.server_processes,
                        'Owned Streamlit process tree sampled RSS; excludes browser and oracle')

    def pick(self, label, option, *, deadline=None):
        """Search and select by keyboard; virtual options can unmount on mouse scroll.

        Only this idempotent selection is retried after a rerender, within the
        configured deadline. The final exact widget label and caller's full
        record_id check still have to succeed; no save action is retried.
        """
        if deadline is None:
            deadline = time.monotonic() + self.timeout / 1000

        def remaining():
            return max(1, min(5000, int((deadline - time.monotonic()) * 1000)))

        control = self.page.get_by_role('combobox', name=re.compile(label))
        while True:
            try:
                # Exports and queue changes can still be running while the old
                # widget is visible. Wait for Streamlit's real running indicator
                # to clear; otherwise a late render can undo a transient label.
                idle_timeout = max(1, int((deadline - time.monotonic()) * 1000))
                expect(self.page.get_by_test_id('stStatusWidgetRunningIcon')).to_have_count(
                    0, timeout=idle_timeout)
                expect(control).to_have_count(1, timeout=remaining())
                control.click(timeout=remaining())
                control.fill(option, timeout=remaining())
                selected = self.page.get_by_role('option', name=option, exact=True)
                expect(selected).to_have_count(1, timeout=remaining())
                expect(selected).to_be_visible(timeout=remaining())
                control.press('ArrowDown', timeout=remaining())
                control.press('Enter', timeout=remaining())
                idle_timeout = max(1, int((deadline - time.monotonic()) * 1000))
                expect(self.page.get_by_test_id('stStatusWidgetRunningIcon')).to_have_count(
                    0, timeout=idle_timeout)
                expect(control).to_have_attribute(
                    'aria-label', re.compile(re.escape('Selected ' + option + '.')),
                    timeout=remaining())
                return
            except (PlaywrightError, AssertionError) as error:
                if (isinstance(error, PlaywrightError)
                        and not isinstance(error, PlaywrightTimeoutError)
                        and 'strict mode violation' not in str(error)):
                    raise
                if time.monotonic() >= deadline:
                    raise
                self.page.keyboard.press('Escape')

    def select_record(self, rid, *, predictions=None):
        """Complete only when the selected label and exact rendered identity agree.

        A late rerender may reset a transient selection after Running=0. Retry
        only an observed control/label reset, within the original deadline; a
        stable selected label with wrong details must fail without reselecting.
        """
        frame = self.snapshots[self.kind] if predictions is None else predictions
        row = frame.set_index('record_id').loc[rid]
        option = f'行 {row.source_row} ｜ 得分 {row.rf_score:.4f} ｜ {rid[:12]}'
        label = '选择记录查看详情'
        control = self.page.get_by_role('combobox', name=re.compile(label))
        deadline = time.monotonic() + self.timeout / 1000
        while time.monotonic() < deadline:
            self.pick(label, option, deadline=deadline)
            while time.monotonic() < deadline:
                # Inspect both visible UI identities in one DOM read, so a
                # transient label cannot substitute for the backend detail.
                state = control.evaluate_all('''(controls, wanted) => {
                    if (controls.length !== 1 ||
                        !controls[0].getAttribute('aria-label')?.includes(wanted.label))
                        return 'reset';
                    const details = [...document.querySelectorAll('[data-testid="stCode"] code')]
                        .filter(node => node.textContent.trim() === wanted.rid);
                    const visible = details.length === 1 && details[0].getClientRects().length > 0
                        && getComputedStyle(details[0]).visibility !== 'hidden';
                    return visible && !document.querySelector('[data-testid="stStatusWidgetRunningIcon"]')
                        ? 'ready' : 'waiting';
                }''', {'label': 'Selected ' + option + '.', 'rid': rid})
                if state == 'ready':
                    return
                if state == 'reset':
                    break
                self.page.wait_for_timeout(min(50, max(1, (deadline - time.monotonic()) * 1000)))
        raise AssertionError(f'Record selection did not settle on exact record_id {rid} '
                             f'within {self.timeout} ms')

    def wait_result(self):
        expect(self.page.get_by_role('button', name='导出预测与复核 CSV', exact=True)).to_be_visible()
        expect(self.page.get_by_text(re.compile(f'符合条件 {self.case["rows"]:,} 条；'))).to_be_visible()
        expect(self.page.get_by_text(re.compile(
            f'涉及 {self.quality.affected_rows:,} 条记录；已定位 {self.quality.problem_cells:,}'))).to_be_visible()

    def exports(self, tag):
        actual = csv(self.download('导出预测与复核 CSV', tag + '-predictions.csv'))
        audit = json.loads(self.download('导出完整复核历史', tag + '-history.json').read_text(encoding='utf-8'))
        expected_audit = read_history(self.database, version=self.version,
                                      record_id=self.last.record_id)
        validate_exports(actual, self.predictions, audit, expected_audit, self.version)
        manifest = json.loads(self.download('导出运行清单', tag + '-manifest.json').read_text(encoding='utf-8'))
        assert manifest['input_sha256'] == self.digest
        assert manifest['valid_rows'] == len(self.predictions) and manifest['unjudgeable_rows'] == 0
        assert manifest['model_version'] == self.version
        assert manifest['model_sha256'] == self.report['model_sha256']
        assert manifest['descriptor_sha256'] == self.report['artifact_sha256']['model.json']
        for name in ('rf_threshold', 'lr_threshold', 'near_margin'):
            assert manifest[name] == self.models['frozen']['bundle']['config'][name]
        return actual

    def assert_retained_input(self):
        """A frontend-rejected replacement must not relabel the accepted input."""
        expect(self.page.get_by_role('button', name='Remove ' + self.name,
                                     exact=True)).to_be_visible(timeout=self.timeout)
        main_name = self.page.get_by_test_id('stMain').get_by_role('paragraph').filter(
            has_text=re.compile('^' + re.escape(self.name) + '$'))
        expect(main_name).to_be_visible(timeout=self.timeout)
        expect(self.page.get_by_role('button', name='检查数据并预测',
                                     exact=True)).to_be_visible(timeout=self.timeout)
        expect(self.page.get_by_role('button', name='保存复核',
                                     exact=True)).to_be_visible(timeout=self.timeout)

    def assert_no_results(self):
        for label in ('导出预测与复核 CSV', '导出完整复核历史', '导出运行清单', '保存复核',
                      '保存并查看下一条'):
            expect(self.page.get_by_role('button', name=label, exact=True)).to_have_count(
                0, timeout=self.timeout)

    def assert_empty_input(self):
        expect(self.page.get_by_text('请在左侧选择 CSV，或切换到内置演示数据。',
                                     exact=True)).to_be_visible(timeout=self.timeout)
        expect(self.page.get_by_role('button', name='检查数据并预测',
                                     exact=True)).to_have_count(0, timeout=self.timeout)
        self.assert_no_results()

    def choose_local_file(self, path: Path):
        # Use the same chooser path as a user. Its click resets the native input
        # selection, allowing the same rejected path to trigger change again.
        with self.page.expect_file_chooser(timeout=self.timeout) as chooser:
            self.click_button('Browse files')
        chooser.value.set_files(str(path), timeout=self.timeout)

    def limits(self, demo):
        before = history_fingerprint(self.database)
        # Streamlit rejects oversized replacements before they become app input
        # and retains the previously accepted file. Check that file's complete
        # identity/exports, then explicitly remove it and test empty-input refusal.
        # Playwright rejects byte buffers over 50 MiB: use a real local file path.
        oversized = self.out / 'bytes-over-50mib.csv'
        oversized.write_bytes(b'x' * (MAX_BYTES + 1))
        message = 'File must be 50.0MB or smaller.'
        with self.phase('reject-oversized-replacement'):
            self.choose_local_file(oversized)
            expect(self.page.get_by_text(message, exact=True)).to_be_visible()
            self.assert_retained_input()
        with self.phase('exports_after_rejected_replacement'):
            self.exports('rejected-replacement')
        assert history_fingerprint(self.database) == before
        self.mark('rejected_replacement_preserves_accepted_input_and_all_exports',
                  accepted_input_name=self.name, accepted_input_sha256=self.digest,
                  rejected_input_name=oversized.name, rejection_layer='streamlit_upload_limit')
        with self.phase('remove-accepted-input'):
            self.click_button('Remove ' + oversized.name)
            self.click_button('Remove ' + self.name)
            self.assert_empty_input()
        assert history_fingerprint(self.database) == before
        self.mark('explicit_input_removal_clears_prediction_and_review_controls')
        for tag, payload, message, layer in (
            ('bytes-over-50mib', oversized, 'File must be 50.0MB or smaller.',
             'streamlit_upload_limit'),
            ('rows-50001', repeat_demo_rows(demo, 50001), '最多支持 50,000 条记录',
             'application_csv_row_guard'),
        ):
            with self.phase('reject-' + tag):
                if isinstance(payload, Path):
                    self.choose_local_file(payload)
                    self.assert_empty_input()
                else:
                    self.upload(payload, tag + '.csv')
                expect(self.page.get_by_text(re.compile(re.escape(message)))).to_be_visible()
                self.assert_no_results()
            assert history_fingerprint(self.database) == before
            size = payload.stat().st_size if isinstance(payload, Path) else len(payload)
            self.mark('limit_rejection_preserves_history', case=tag, bytes=size,
                      rejection_layer=layer, observed_message=message)
        with self.phase('recover-after-limits'):
            self.upload(self.content, self.name)
            self.wait_result()
        with self.phase('exports_after_limits'):
            self.exports('after-limit-recovery')
        assert history_fingerprint(self.database) == before
        self.mark('valid_recovery_after_limits_preserves_last_review_and_all_exports')

    def run(self, *, demo: bytes, include_limits: bool):
        untouched = history_fingerprint(self.database,
                                        exclude_identity=(self.version, self.last.record_id))
        cleanup_errors = []
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(channel='msedge', headless=True)
            self.context = self.browser.new_context(viewport={'width': 1600, 'height': 1100},
                                                    accept_downloads=True)
            self.context.add_init_script('delete window.showSaveFilePicker;')
            self.context.tracing.start(screenshots=True, snapshots=True, sources=False)
            self.page = self.context.new_page()
            self.page.set_default_timeout(self.timeout)
            expect.set_options(timeout=self.timeout)
            self.page.on('pageerror', lambda error: self.errors.append(str(error)))
            self.start('frozen')
            with self.phase('upload_validate_predict_and_render'):
                self.upload(self.content, self.name)
                self.wait_result()
            actual = self.exports('initial')
            assert history_fingerprint(self.database) == self.case['database_before']
            prior_related = read_history(self.database, version=self.version,
                                         record_id=self.last.record_id)
            self.mark('real_upload_full_prediction_quality_and_history_scope')
            with self.phase('ranked_visible_table'):
                table = csv(self.download(None, 'ranked-visible-table.csv',
                                           table=self.page.locator('[data-testid="stDataFrame"]').first))
                expected = ranked(actual).head(1000)[list(DISPLAY)].rename(columns=DISPLAY)
                assert_frames(table, expected)
            self.mark('ranked_table_first_1000_matches_complete_export')
            with self.phase('select_last_record'):
                self.click_text('原始顺序')
                self.select_record(self.last.record_id, predictions=self.predictions)
                expect(self.page.get_by_text(
                    f'来源：{self.name}，第 {self.case["rows"] + 1} 行', exact=True)).to_be_visible()
                expect(self.page.get_by_role('button', name='保存并查看下一条', exact=True)).to_be_disabled()
            self.mark('last_record_identity_and_queue_end')
            if not self.quality.records.empty:
                with self.phase('last_quality_problem_record'):
                    issue = self.quality.records.iloc[-1]
                    self.pick('选择问题记录核对字段',
                              f'CSV 第 {issue.input_row} 行 ｜ {issue.record_id}')
                    details = csv(self.download(None, 'last-quality-fields.csv',
                                               table=self.page.locator('[data-testid="stDataFrame"]').last))
                    assert_frames(details, self.quality.details(issue.record_id))
                    expect(self.page.locator('[data-testid="stCode"]').filter(
                        has_text=self.last.record_id)).to_be_visible()
                self.mark('last_problem_field_cells_preserve_input_and_editor_identity')
            with self.phase('save_last_record'):
                self.pick('复核结论', '确认异常')
                note = f'CAPACITY_FIXTURE {self.case["rows"]} rows: engineering identity check only'
                self.page.get_by_role('textbox', name='复核备注', exact=True).fill(note)
                self.click_button('保存复核')
                expect(self.page.get_by_text('已保存复核，修改历史已追加。', exact=True)).to_be_visible()
            saved = read_history(self.database, version=self.version, record_id=self.last.record_id)
            assert len(saved) == self.case['history_seed']['related_rows'] + 1
            assert saved[:-1] == prior_related, 'Saving changed existing review history'
            assert saved[-1]['status'] == '确认异常' and saved[-1]['note'] == note
            assert history_fingerprint(self.database, exclude_identity=(
                self.version, self.last.record_id)) == untouched
            with self.phase('status_filter'):
                self.pick('复核状态筛选', '确认异常')
                expect(self.page.get_by_text(re.compile('符合条件 1 条；'))).to_be_visible()
                self.select_record(self.last.record_id, predictions=self.predictions)
                table = csv(self.download(None, 'filtered-last-record.csv',
                                           table=self.page.locator('[data-testid="stDataFrame"]').first))
                assert len(table) == 1 and int(table['来源行号'].iloc[0]) == self.case['rows'] + 1
            with self.phase('all_exports_after_filter'):
                self.exports('filtered')
            self.mark('last_review_filter_full_exports_and_unrelated_history_preserved')
            self.capture('last-record-filtered')
            with self.phase('refresh_reupload'):
                self.page.reload()
                self.upload(self.content, self.name)
                self.wait_result()
            with self.phase('exports_after_refresh'):
                self.exports('refreshed')
            self.mark('browser_refresh_reupload_preserves_history_and_all_source_ids')
            before_restart = history_fingerprint(self.database)
            # P0Browser.start stops and verifies the old owned Job before creating
            # a new service; a page refresh alone cannot establish persistence.
            self.start('frozen')
            with self.phase('restart_reupload'):
                self.upload(self.content, self.name)
                self.wait_result()
                self.click_text('原始顺序')
                self.select_record(self.last.record_id, predictions=self.predictions)
                expect(self.page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(note)
                expect(self.page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
                    'aria-label', re.compile('Selected 确认异常' + r'\.'))
            with self.phase('exports_after_restart'):
                self.exports('restarted')
            assert history_fingerprint(self.database) == before_restart
            self.mark('server_restart_reupload_preserves_history_and_all_source_ids')
            if include_limits:
                self.limits(demo)
            with sqlite3.connect(self.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
                assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            self.case['database_after'] = history_fingerprint(self.database)
            assert self.case['database_after']['rows'] == self.case['history_seed']['total_rows'] + 1
            self.case['passed'] = not self.errors
        except Exception as error:
            self.case['failure'] = f'{type(error).__name__}: {error}'
            if self.page is not None:
                try:
                    self.capture('failure')
                    (self.out / 'failure-ui.txt').write_text(
                        self.page.locator('body').aria_snapshot(), encoding='utf-8')
                except Exception as capture_error:
                    self.case['failure_capture_error'] = repr(capture_error)
            raise
        finally:
            for label, action in (
                ('trace', lambda: self.context.tracing.stop(path=str(self.out / 'trace.zip'))),
                ('context', lambda: self.context.close()),
                ('browser', lambda: self.browser.close()),
                ('playwright', lambda: self.playwright.stop()),
                ('server', self.stop),
            ):
                try:
                    if label == 'server' or getattr(self, label if label != 'trace' else 'context') is not None:
                        action()
                except Exception as error:
                    cleanup_errors.append(f'{label}: {type(error).__name__}: {error}')
            self.case['cleanup_errors'] = cleanup_errors
            self.case['passed'] = bool(self.case.get('passed') and not cleanup_errors
                                       and all(x.get('cleanup', {}).get('passed') for x in self.starts))
            self.acceptance.checkpoint()
        if not self.case['passed']:
            raise RuntimeError('Capacity browser checks or cleanup failed')


def run_capacity(args, acceptance: AcceptanceRun) -> None:
    report = acceptance.report
    if args.expected_head and report['source_before']['head'] != args.expected_head:
        raise ValueError('Current HEAD does not match --expected-head')
    artifacts = args.artifacts.resolve()
    hashes = frozen_hashes(artifacts)
    report.update(scope='Engineering capacity only; repeated demo rows are not independent samples',
                  git_head=report['source_before']['head'], artifact_sha256=hashes,
                  model_sha256=hashes['model.joblib'], cases=[],
                  full_matrix=set(args.rows) == set(DEFAULT_ROWS) and args.history_rows >= 100000,
                  background_note=args.background_note,
                  limits=['No new training, model/threshold changes, algorithm metrics or user benefit.',
                          'Sampled RSS is a lower bound; no arbitrary performance pass threshold.',
                          'Direct-stage oracle RSS and owned-server RSS are separate measurements.',
                          'Native browser save picker disabled; existing Streamlit CSV fallback used.',
                          'This matrix does not replace pressure/endurance or new-machine testing.'])
    acceptance.checkpoint()
    model_bytes, descriptor_json, _ = prepare_model(artifacts / 'model.joblib', artifacts / 'model.json')
    bundle = load_model_bundle(model_bytes, descriptor_json)
    report['model_version'] = bundle['model_version']
    report['environment'] = {'python': sys.version, 'executable': sys.executable,
                             'platform': sys.platform, 'cpu_count': psutil.cpu_count(),
                             'physical_memory_bytes': psutil.virtual_memory().total}
    demo = (artifacts / 'demo.csv').read_bytes()
    self_process = psutil.Process(os.getpid())
    for count in args.rows:
        output = acceptance.output / f'rows-{count}'
        output.mkdir()
        case = {'rows': count, 'input_name': f'capacity-{count}.csv', 'passed': False,
                'direct_measurements': {}, 'load_generation': 'Repeated existing demo CSV cells'}
        report['cases'].append(case)
        payload = repeat_demo_rows(demo, count)
        if len(payload) > MAX_BYTES:
            raise ValueError('Repeated load exceeds the independent 50 MiB application limit')
        (output / case['input_name']).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        case.update(input_sha256=digest, input_bytes=len(payload))
        with measured(case['direct_measurements'], 'read_and_csv_validation',
                      lambda: [self_process], 'Harness process RSS, not application server RSS'):
            raw = read_csv_checked(io.BytesIO(payload))
        with measured(case['direct_measurements'], 'adapter_and_real_inference',
                      lambda: [self_process], 'Harness process RSS, not application server RSS'):
            predictions, features, rejected = infer(raw, bundle, digest, case['input_name'])
        assert len(predictions) == count and rejected.empty
        assert predictions.source_row.tolist() == list(range(2, count + 2))
        assert predictions.record_id.tolist() == [f'{digest}:{row}' for row in range(2, count + 2)]
        with measured(case['direct_measurements'], 'field_quality_view',
                      lambda: [self_process], 'Harness process RSS, not application server RSS'):
            quality = build_quality_view(raw, predictions, features, rejected)
        case['quality'] = {'affected_rows': quality.affected_rows,
                           'problem_cells': quality.problem_cells,
                           'rejected_rows': quality.rejected_rows}
        case['history_seed'] = seed_history(output / 'reviews.sqlite', bundle['model_version'],
                                            predictions.iloc[-1].record_id, args.history_rows)
        case['database_before'] = history_fingerprint(output / 'reviews.sqlite')
        acceptance.checkpoint()
        browser = CapacityBrowser(output, acceptance, {'directory': artifacts, 'bundle': bundle},
                                  payload, predictions, quality, case, args.timeout_ms)
        browser.run(demo=demo, include_limits=count == max(args.rows))
        validate_completed_case(case, include_limits=count == max(args.rows))
    report['original_model_files_unchanged'] = frozen_hashes(artifacts) == hashes
    report['files'] = {str(path.relative_to(acceptance.output)): sha256(path)
                       for path in acceptance.output.rglob('*')
                       if path.is_file() and path != acceptance.path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=list(DEFAULT_ROWS))
    parser.add_argument('--history-rows', type=int, default=100000)
    parser.add_argument('--timeout-ms', type=int, default=180000)
    parser.add_argument('--expected-head')
    parser.add_argument('--background-note', default='Concurrent host workload not independently controlled')
    args = parser.parse_args()
    if (len(set(args.rows)) != len(args.rows) or any(not 1 <= n <= 50000 for n in args.rows)
            or args.history_rows < 100000 or args.timeout_ms < 1000):
        parser.error('Use distinct row counts in 1..50000, at least 100000 history rows and a positive timeout')
    acceptance = AcceptanceRun(args.output.resolve(), ROOT, report_name='report.json',
                               execution_files=['scripts/engineering_capacity.py',
                                                'tests/test_engineering_capacity.py',
                                                'docs/engineering/CAPACITY_ACCEPTANCE.md'])
    try:
        run_capacity(args, acceptance)
        result = acceptance.finish(True)
        if not result['passed']:
            raise RuntimeError('Capacity acceptance source integrity failed')
    except Exception as error:
        acceptance.report['failure'] = f'{type(error).__name__}: {error}'
        acceptance.finish(False)
        raise
    print(json.dumps({'passed': True, 'full_matrix': result['full_matrix'],
                      'cases': len(result['cases'])}), flush=True)


if __name__ == '__main__':
    main()
