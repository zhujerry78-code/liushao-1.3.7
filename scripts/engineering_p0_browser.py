"""P0 only: real model switching, four review states and input recovery in Edge.

No training, model configuration changes, shared protocol changes, or GitHub calls.
All databases, model fault copies and browser evidence belong to a fresh output directory.

Legacy page workflow; r9 UI acceptance uses engineering_r9_acceptance.py.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from playwright.sync_api import expect, sync_playwright

from app_model import load_model_bundle, prepare_model
from liushao.data import read_csv_checked, sha256
from liushao.model import infer
from liushao.reviews import STATUSES, history
from scripts.engineering_p0_cases import assert_frames, assert_review_snapshot, input_cases
from scripts.engineering_run import AcceptanceRun, start_owned_process

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = 'p0-same-input.csv'
DISPLAY = {'source_row': '来源行号', 'rf_score': '主模型得分', 'lr_score': '参照得分',
           'rf_anomaly': '提示异常', 'disagreement': '判断分歧', 'near_threshold': '接近阈值',
           'review_status': '复核状态', 'source_file': '来源文件'}


def click_unique(control, timeout=30000):
    """Wait out an obsolete rerender widget; persistent duplicate targets must fail."""
    expect(control).to_have_count(1, timeout=timeout)
    control.click(timeout=timeout)


def csv(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    text_columns = [name for name in (
        'record_id', 'source_file', 'source_sha256', 'dataset_id', 'model_version',
        'config_version', 'review_status', 'review_note', 'reason',
        '字段', '单位', '统计提示', '来源文件', '复核状态',
        'CSV读取值', '适配结果', '处理状态',
    ) if name in frame]
    if text_columns:
        # Preserve empty notes and literal NA/null/NaN in known text fields.
        text = pd.read_csv(path, usecols=text_columns, dtype=str, keep_default_na=False)
        frame[text_columns] = text[text_columns]
    if 'timestamp' in frame:
        frame['timestamp'] = pd.to_datetime(frame.timestamp, errors='raise')
    return frame


def assert_prediction_export(actual, expected):
    assert actual.columns.tolist() == expected.columns.tolist() + ['review_status', 'review_note'], (
        '预测导出列清单不一致；不能隐藏额外标签或丢失字段')
    assert_frames(actual[list(expected.columns)], expected)


class P0Browser:
    def __init__(self, args, *, acceptance=None):
        self.out = args.output.resolve()
        self.acceptance = acceptance or AcceptanceRun(self.out, ROOT, report_name='report.json')
        self.report = self.acceptance.report
        self.report.update(scope='P0 engineering acceptance only; fixed demo, no training',
                           checks=[])
        self.acceptance.checkpoint()
        self.models, self.artifact_hashes = {}, {}
        for kind, directory, expected_hash in (
            ('frozen', args.frozen_artifacts, args.frozen_sha256),
            ('rebuilt', args.rebuilt_artifacts, args.rebuilt_sha256),
        ):
            directory = directory.resolve()
            content, descriptor_text, _ = prepare_model(directory / 'model.joblib', directory / 'model.json')
            model_sha = hashlib.sha256(content).hexdigest()
            if model_sha != expected_hash:
                raise ValueError(f'{kind}: trusted binary hash mismatch')
            descriptor = json.loads(descriptor_text)
            bundle = load_model_bundle(content, descriptor_text)
            self.models[kind] = {'directory': directory, 'descriptor': descriptor, 'bundle': bundle}
            self.artifact_hashes[kind] = {
                'model.joblib': model_sha,
                # These are the exact descriptor bytes passed to the loader.
                'model.json': hashlib.sha256(descriptor_text.encode('utf-8')).hexdigest(),
                **{name: sha256(directory / name) for name in
                   ('demo.csv', 'demo.json', 'demo_origins.parquet')}}
        self.content = (self.models['frozen']['directory'] / 'demo.csv').read_bytes()
        assert self.content == (self.models['rebuilt']['directory'] / 'demo.csv').read_bytes()
        assert self.models['frozen']['bundle']['model_version'] != self.models['rebuilt']['bundle']['model_version']
        self.digest = hashlib.sha256(self.content).hexdigest()
        self.direct = {}
        for kind, model in self.models.items():
            pred, values, rejected = infer(read_csv_checked(io.BytesIO(self.content)), model['bundle'],
                                           self.digest, INPUT_NAME)
            assert len(pred) == 1000 and not len(rejected)
            self.direct[kind] = (pred, values)
        self.ids = self.direct['frozen'][0].record_id.iloc[:4].tolist()
        assert self.ids == self.direct['rebuilt'][0].record_id.iloc[:4].tolist()
        self.saved = {kind: {} for kind in self.models}
        self.histories = {kind: [] for kind in self.models}
        self.snapshots = {}
        self.process = self.process_tree = self.log = self.page = self.context = None
        self.browser = self.playwright = None
        self.errors, self.starts = [], []
        self.kind = None
        self.counter = 0
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        self.url = f'http://127.0.0.1:{self.port}'
        self.report.update({
            'scope': 'P0 engineering acceptance only; fixed demo, no quality experiment or training',
            'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False, 'checks': [],
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {n: sha256(ROOT / n) for n in
                              ('app.py', 'app_model.py', 'app_quality.py', 'liushao/model.py', 'liushao/data.py',
                               'liushao/reviews.py', 'scripts/engineering_p0_browser.py',
                               'scripts/engineering_p0_cases.py')},
            'input_sha256': self.digest, 'input_rows': 1000,
            'models': {kind: {'directory': str(m['directory']), 'version': m['bundle']['model_version'],
                              'sha256': self.artifact_hashes[kind]['model.joblib'],
                              'thresholds': {n: m['bundle']['config'][n] for n in
                                             ('rf_threshold', 'lr_threshold', 'near_margin')}}
                       for kind, m in self.models.items()},
            'database': str(self.out / 'reviews.sqlite'), 'server_starts': self.starts,
            'page_errors': self.errors,
            'limits': ['Review keys remain model_version + record_id; no content-key migration.',
                       'Runtime failures are explicitly injected after validated original loading.',
                       'Displayed evidence matches package references, not independently audited training.'],
        })
        self.write_report()

    def mark(self, name, **details):
        if self.process_tree is not None:
            self.process_tree.capture()
        self.report['checks'].append({'name': name, 'model': self.kind, 'passed': True, **details})
        self.write_report()
        print(f'PASS {self.kind}: {name}', flush=True)

    def write_report(self):
        self.acceptance.checkpoint()

    def stop(self):
        result = None
        if self.process is not None:
            result = self.process_tree.cleanup()
            self.starts[-1]['cleanup'] = result
            self.starts[-1]['stopped_exit_code'] = self.process.returncode
            self.process = None
            self.process_tree = None
        if self.log is not None:
            self.log.close()
            self.log = None
        if result is not None and not result['passed']:
            raise RuntimeError('Owned Streamlit process cleanup failed: ' + repr(result))

    def start(self, kind, *, directory=None, launcher=None):
        self.stop()
        self.kind = kind
        directory = directory or self.models[kind]['directory']
        self.log = (self.out / f'server-{len(self.starts) + 1:02}.log').open('w', encoding='utf-8')
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONPATH': str(ROOT),
               'LIUSHAO_ARTIFACTS_DIR': str(directory), 'LIUSHAO_REVIEW_DB': str(self.out / 'reviews.sqlite')}
        argv = [sys.executable, '-m', 'streamlit', 'run', str(launcher or ROOT / 'app.py'),
                '--server.address=127.0.0.1', f'--server.port={self.port}', '--server.headless=true',
                '--server.fileWatcherType=none', '--browser.gatherUsageStats=false']
        self.process, self.process_tree = start_owned_process(
            argv, cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.process_tree.capture()
        self.starts.append({'kind': kind, 'artifacts': str(directory), 'argv': argv,
                            'pid': self.process.pid, 'database': env['LIUSHAO_REVIEW_DB']})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(90):
            self.process_tree.capture()
            if self.process.poll() is not None:
                raise RuntimeError('Owned Streamlit service exited')
            if not self.process_tree.owns_tcp_listener('127.0.0.1', self.port):
                time.sleep(.3)
                continue
            try:
                with opener.open(self.url + '/_stcore/health', timeout=1) as response:
                    if response.status == 200 and response.read().strip() == b'ok':
                        if (self.process.poll() is not None
                                or not self.process_tree.owns_tcp_listener('127.0.0.1', self.port)):
                            raise RuntimeError('Health listener no longer belongs to owned server')
                        self.page.goto(self.url)
                        return
            except OSError:
                time.sleep(.3)
        raise TimeoutError('Streamlit startup timeout')

    def capture(self, name):
        self.page.screenshot(path=str(self.out / (name + '.png')))

    def pick(self, label, option):
        control = self.page.get_by_role('combobox', name=re.compile(label))
        # Streamlit briefly keeps the previous widget during a rerender.
        expect(control).to_have_count(1)
        click_unique(control)
        selected_option = self.page.get_by_role('option', name=option, exact=True)
        if selected_option.count() == 0:
            # Large Streamlit lists render only the visible options. Search via
            # the real combobox instead of assuming a distant row is in the DOM.
            control.fill(option)
        click_unique(selected_option)
        expect(control).to_have_count(1)
        expect(control).to_have_attribute('aria-label', re.compile(re.escape('Selected ' + option + '.')))

    def checkbox(self, label, value):
        control = self.page.get_by_role('checkbox', name=label, exact=True)
        expect(control).to_have_count(1)
        if control.is_checked() != value:
            self.click_text(label)
        expect(control).to_be_checked(checked=value)

    def upload(self, content=None, name=INPUT_NAME):
        self.click_text('上传 CSV')
        uploader = self.page.locator('input[type=file]')
        expect(uploader).to_have_count(1)
        uploader.set_input_files(
            {'name': name, 'mimeType': 'text/csv', 'buffer': self.content if content is None else content})
        expect(self.page.locator('[data-testid="stMain"]').get_by_role('paragraph').filter(
            has_text=re.compile('^' + re.escape(name) + '$'))).to_be_visible()
        self.click_button('检查数据并预测')

    def click_text(self, label):
        click_unique(self.page.get_by_text(label, exact=True))

    def click_button(self, label):
        click_unique(self.page.get_by_role('button', name=label, exact=True))

    def evidence_table(self):
        expander = self.page.locator('[data-testid="stExpander"]').filter(
            has=self.page.get_by_text('查看原始特征与正常参考区间', exact=True))
        expect(expander).to_have_count(1)
        if expander.locator('details').get_attribute('open') is None:
            expander.locator('summary').click()
        table = expander.locator('[data-testid="stDataFrame"]')
        expect(table).to_have_count(1)
        expect(table).to_be_visible()
        return table

    def download(self, button, stem, *, table=None):
        self.counter += 1
        path = self.out / f'{self.counter:03}-{stem}'
        if table is not None:
            table.hover()
            button = table.get_by_role('button', name='Download as CSV', exact=True)
        else:
            prepare = self.page.get_by_role('button', name='核对并准备导出', exact=True)
            if button != '导出无法判断记录' and prepare.count():
                click_unique(prepare)
            button = self.page.get_by_role('button', name=button, exact=True)
        expect(button).to_have_count(1)
        with self.page.expect_download() as download:
            if table is not None:
                # Small tables can place the floating toolbar under the fixed page header.
                # Activate its actual accessible button with a keyboard, without bypassing the UI.
                button.focus()
                button.press('Enter')
            else:
                click_unique(button)
        download.value.save_as(path)
        resume = self.page.get_by_role('button', name='返回复核编辑', exact=True)
        if table is None and resume.count():
            click_unique(resume)
            expect(self.page.get_by_role('button', name='核对并准备导出', exact=True)).to_be_visible()
        return path

    def actual_history(self):
        return history(self.out / 'reviews.sqlite', model_version=self.models[self.kind]['bundle']['model_version'])

    def baseline(self, tag):
        self.upload()
        version = self.models[self.kind]['bundle']['model_version']
        expect(self.page.locator('[data-testid="stCode"]').filter(has_text=version)).to_be_visible()
        actual = csv(self.download('导出预测与复核 CSV', tag + '-predictions.csv'))
        expected = self.direct[self.kind][0]
        assert_prediction_export(actual, expected)
        for rid, saved in self.saved[self.kind].items():
            row = actual.set_index('record_id').loc[rid]
            assert row.review_status == saved['status'] and row.review_note == saved['note']
        untouched = actual[~actual.record_id.isin(self.saved[self.kind])]
        assert untouched.review_status.eq('待复核').all() and untouched.review_note.eq('').all()
        exported_history = json.loads(self.download('导出完整复核历史', tag + '-history.json').read_text(encoding='utf-8'))
        assert exported_history == self.histories[self.kind] == self.actual_history()
        assert_review_snapshot(actual, exported_history, version)
        meta = json.loads(self.download('导出运行清单', tag + '-manifest.json').read_text(encoding='utf-8'))
        assert meta['model_version'] == version and meta['input_sha256'] == self.digest
        for key in ('rf_threshold', 'lr_threshold', 'near_margin'):
            assert meta[key] == self.models[self.kind]['bundle']['config'][key]
        self.snapshots[self.kind] = actual
        if self.saved[self.kind]:
            self.click_text('原始顺序')
            self.pick('复核状态筛选', '全部')
            self.checkbox('仅模型分歧', False)
            self.checkbox('仅近阈值', False)
            self.assert_editor(self.ids[0])
        self.mark(tag, rows=len(actual), history_rows=len(exported_history), actual_version=version)
        return actual

    def select_record(self, rid, *, predictions=None):
        frame = self.snapshots[self.kind] if predictions is None else predictions
        row = frame.set_index('record_id').loc[rid]
        option = f'行 {row.source_row} ｜ 得分 {row.rf_score:.4f} ｜ {rid[:12]}'
        self.pick('选择记录查看详情', option)
        expect(self.page.locator('[data-testid="stCode"]').filter(has_text=rid)).to_be_visible()

    def assert_editor(self, rid):
        self.select_record(rid)
        saved = self.saved[self.kind][rid]
        expect(self.page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(saved['note'])
        expect(self.page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
            'aria-label', re.compile('Selected ' + saved['status'] + r'\.'))

    def table(self, queue='原始顺序', *, status='全部', disagreement=False, near=False, tag):
        self.click_text(queue)
        self.pick('复核状态筛选', status)
        self.checkbox('仅模型分歧', disagreement)
        self.checkbox('仅近阈值', near)
        frame = self.snapshots[self.kind].copy()
        for rid, saved in self.saved[self.kind].items():
            take = frame.record_id.eq(rid)
            frame.loc[take, 'review_status'] = saved['status']
            frame.loc[take, 'review_note'] = saved['note']
        if queue == '高异常得分':
            frame = frame.sort_values(['rf_score', 'record_id'], ascending=[False, True], kind='stable')
        elif queue == '待核实':
            frame = frame[frame.disagreement | frame.near_threshold].copy()
            frame['distance'] = frame.rf_score.sub(frame.rf_threshold).abs()
            frame = frame.sort_values(['disagreement', 'distance', 'record_id'], ascending=[False, True, True])
        if status != '全部':
            frame = frame[frame.review_status.eq(status)]
        if disagreement:
            frame = frame[frame.disagreement]
        if near:
            frame = frame[frame.near_threshold]
        expect(self.page.get_by_text(re.compile(f'符合条件 {len(frame):,} 条；'))).to_be_visible()
        if frame.empty:
            # Streamlit intentionally omits its grid CSV toolbar for zero rows.
            expect(self.page.locator('[data-testid="stDataFrame"]').first.get_by_role(
                'button', name='Download as CSV', exact=True)).to_have_count(0)
            expect(self.page.get_by_role('combobox', name=re.compile('选择记录查看详情'))).to_have_count(0)
            return frame
        actual = csv(self.download(None, tag + '-rendered-table.csv',
                                   table=self.page.locator('[data-testid="stDataFrame"]').first))
        assert_frames(actual, frame[list(DISPLAY)].rename(columns=DISPLAY))
        return frame

    def save(self, rid, status, note):
        # Bind identity before changing a widget: selection after rerender is insufficient.
        expect(self.page.locator('[data-testid="stCode"]').filter(has_text=rid)).to_be_visible()
        previous = self.actual_history()
        self.pick('复核结论', status)
        self.page.get_by_role('textbox', name='复核备注', exact=True).fill(note)
        self.click_button('保存复核')
        expect(self.page.get_by_text('已保存复核，修改历史已追加。', exact=True)).to_be_visible()
        deadline = time.monotonic() + 30
        actual = self.actual_history()
        while len(actual) == len(self.histories[self.kind]) and time.monotonic() < deadline:
            self.page.wait_for_timeout(100)
            actual = self.actual_history()
        assert len(actual) == len(self.histories[self.kind]) + 1
        assert actual[:-1] == previous == self.histories[self.kind]
        assert {key: actual[-1][key] for key in ('record_id', 'status', 'note')} == {
            'record_id': rid, 'status': status, 'note': note}
        self.histories[self.kind] = actual
        self.saved[self.kind][rid] = {'status': status, 'note': note}

    def assert_exports(self, tag):
        actual = csv(self.download('导出预测与复核 CSV', tag + '-predictions.csv'))
        audit = json.loads(self.download('导出完整复核历史', tag + '-history.json')
                           .read_text(encoding='utf-8'))
        assert_prediction_export(actual, self.direct[self.kind][0])
        assert audit == self.actual_history() == self.histories[self.kind]
        assert_review_snapshot(actual, audit, self.models[self.kind]['bundle']['model_version'])
        counts = actual.review_status.value_counts()
        confirmed = int(counts.get('确认异常', 0) + counts.get('确认正常', 0))
        expect(self.page.get_by_text(
            f'复核进度（本次导入）：已确认 {confirmed} / {len(actual)}；'
            f'待补证 {int(counts.get("待补证", 0))}；待复核 {int(counts.get("待复核", 0))}',
            exact=True)).to_be_visible()
        return actual, audit

    def new_review_window(self):
        page = self.context.new_page()
        page.on('pageerror', lambda error: self.errors.append(str(error)))
        page.goto(self.url)
        return page

    def stale_save_conflict(self):
        """Two real browser sessions edit one revision; losing draft needs explicit reload."""
        self.table(tag='conflict-source-order')
        rid = self.ids[0]
        self.select_record(rid)
        first_page = self.page
        second_page = self.new_review_window()
        try:
            self.page = second_page
            self.upload()
            self.click_text('原始顺序')
            self.select_record(rid)
            draft = 'P0陈旧窗口草稿：冲突后须保留，不能自动覆盖。'
            self.pick('复核结论', '待补证')
            self.page.get_by_role('textbox', name='复核备注', exact=True).fill(draft)
            self.page = first_page
            winner = 'P0另一个窗口先保存：这条应成为唯一成功修订。'
            self.save(rid, '待复核', winner)
            before = self.actual_history()
            self.page = second_page
            self.click_button('保存并查看下一条')
            expect(self.page.get_by_text(re.compile(
                '复核保存失败：.*复核记录已被其他窗口修改'))).to_be_visible()
            expect(self.page.locator('[data-testid="stCode"]').filter(has_text=rid)).to_be_visible()
            expect(self.page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(draft)
            expect(self.page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
                'aria-label', re.compile(r'Selected 待补证\.'))
            assert self.actual_history() == before
            # A second submit must still use the original baseline, rather than auto-retry.
            self.click_button('保存复核')
            expect(self.page.get_by_text(re.compile('复核保存失败：.*复核记录已被其他窗口修改'))).to_be_visible()
            assert self.actual_history() == before
            expect(self.page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(draft)
            self.capture('conflict-preserved-draft')
            self.click_button('读取最新复核并替换草稿')
            expect(self.page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(winner)
            self.save(rid, '待复核', 'P0显式读取后保存：使用最新修订，旧历史保留。')
            self.assert_exports('conflict-after-explicit-reload')
            self.mark('two_sessions_same_revision_one_winner_preserved_draft_explicit_reload')
        finally:
            self.page = first_page
            second_page.close()
        # The winning window is now stale too; a fresh session must read persisted history.
        self.page.reload()
        self.baseline('conflict-first-window-restored')

    def continuous_review(self):
        frame = self.table(tag='continuous-fixed-order')
        target, next_id = frame.record_id.iloc[4:6].tolist()
        self.select_record(target)
        previous = self.actual_history()
        self.pick('复核结论', '确认正常')
        self.page.get_by_role('textbox', name='复核备注', exact=True).fill('')
        self.click_button('保存并查看下一条')
        expect(self.page.get_by_text('复核保存失败：确认异常或确认正常时必须填写判断依据',
                                     exact=True)).to_be_visible()
        expect(self.page.locator('[data-testid="stCode"]').filter(has_text=target)).to_be_visible()
        assert self.actual_history() == previous
        note = 'P0连续复核：保存前已固定本条和下一条编号，非真实安全结论。'
        self.page.get_by_role('textbox', name='复核备注', exact=True).fill(note)
        self.click_button('保存并查看下一条')
        expect(self.page.locator('[data-testid="stCode"]').filter(has_text=next_id)).to_be_visible()
        actual = self.actual_history()
        assert len(actual) == len(previous) + 1 and actual[:-1] == previous
        assert {key: actual[-1][key] for key in ('record_id', 'status', 'note')} == {
            'record_id': target, 'status': '确认正常', 'note': note}
        self.histories[self.kind] = actual
        self.saved[self.kind][target] = {'status': '确认正常', 'note': note}
        exported, _ = self.assert_exports('continuous-saved-target')
        assert exported.set_index('record_id').loc[target, 'review_note'] == note
        assert exported.set_index('record_id').loc[next_id, 'review_note'] == ''
        # The queue's final item still supports a normal save.
        last = frame.record_id.iloc[-1]
        self.select_record(last)
        expect(self.page.get_by_role('button', name='保存并查看下一条', exact=True)).to_be_disabled()
        self.save(last, '待补证', 'P0队列末条保存：补证占位，仅验证保存机制。')
        expect(self.page.locator('[data-testid="stCode"]').filter(has_text=last)).to_be_visible()
        self.assert_exports('continuous-last-item')
        self.mark('continuous_review_fixed_ids_sqlite_csv_failure_stays_and_final_item_saves',
                  target_id=target, next_id=next_id, last_id=last)

    def create_states(self):
        self.table(tag=self.kind + '-initial-raw')
        for rid, status in zip(self.ids, STATUSES, strict=True):
            self.select_record(rid)
            self.save(rid, status, f'P0自动化验证 {self.kind} {status}：核对记录身份和保存机制，非真实安全结论。')
        rid = self.ids[0]
        self.select_record(rid)
        for status in ('确认异常', '确认正常'):
            for blank in ('', ' \t\n\u3000'):
                before = self.actual_history()
                self.pick('复核结论', status)
                self.page.get_by_role('textbox', name='复核备注', exact=True).fill(blank)
                self.click_button('保存复核')
                expect(self.page.get_by_text('复核保存失败：确认异常或确认正常时必须填写判断依据', exact=True)).to_be_visible()
                assert self.actual_history() == before
        for status in ('确认异常', '确认正常', '待补证', '待复核'):
            self.save(rid, status, f'P0连续修改 {self.kind} {status}：测试依据，原预测保持不变。')
        self.mark('four_status_saves_blank_rejections_and_append_only_edits', history_rows=8)

    def verify_states(self, tag):
        self.baseline(tag)
        for rid, status in zip(self.ids, STATUSES, strict=True):
            self.table(status=status, tag=tag + '-' + status)
            self.select_record(rid)
            expect(self.page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(
                self.saved[self.kind][rid]['note'])
            expect(self.page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
                'aria-label', re.compile('Selected ' + status + r'\.'))
            self.capture(tag + '-' + status)
        self.table(tag=tag + '-all-restored')
        self.mark(tag + '-four_status_filters_and_editors')

    def all_tables_and_evidence(self):
        for queue in ('原始顺序', '高异常得分', '待核实'):
            for disagreement, near in ((False, False), (True, False), (False, True), (True, True)):
                self.table(queue, disagreement=disagreement, near=near,
                           tag=f'full-{queue}-{int(disagreement)}{int(near)}')
        self.table(tag='evidence-raw')
        descriptor = self.models[self.kind]['descriptor']
        predictions, values = self.direct[self.kind]
        cases = {'first': predictions.index[0]}
        for flag in ('disagreement', 'near_threshold'):
            flagged = predictions[predictions[flag]]
            if len(flagged):
                cases[flag] = flagged.index[0]
        for tag, position in cases.items():
            record = predictions.loc[position]
            self.select_record(record.record_id)
            actual = csv(self.download(None, tag + '-all-rendered-evidence.csv',
                                       table=self.evidence_table()))
            assert_frames(actual, self.expected_evidence(values.loc[position], descriptor))
            self.assert_evidence_card(record, actual, descriptor)
            self.capture(tag + '-evidence-card')
        self.mark('all_12_queue_filter_tables_77_evidence_fields_and_actual_evidence_card',
                  evidence_rows=len(actual),
                  card_cases={tag: predictions.loc[position, 'record_id']
                              for tag, position in cases.items()})

    @staticmethod
    def expected_evidence(values, descriptor):
        rows = []
        for field in descriptor['config']['features']:
            value, ref = values[field], descriptor['reference'][field]
            flag = ('缺失，预测时用训练中位数填补' if pd.isna(value) else
                    '训练正常参考不足' if ref['q01'] is None or ref['q99'] is None else
                    '超出正常参考区间' if value < ref['q01'] or value > ref['q99'] else '参考区间内')
            unit = '微秒' if 'Duration' in field or 'IAT' in field or field.startswith(('Active ', 'Idle ')) else '数据源原始量纲'
            rows.append({'字段': field, '原值': value, '单位': unit, '正常P01': ref['q01'],
                         '正常中位数': ref['median'], '正常P99': ref['q99'], '统计提示': flag})
        return pd.DataFrame(rows)

    def assert_evidence_card(self, record, actual, descriptor):
        expect(self.page.get_by_text(
            f'主模型判断：{"提示异常" if record.rf_anomaly else "未提示异常"}；'
            f'参照模型判断：{"提示异常" if record.lr_anomaly else "未提示异常"}。',
            exact=True)).to_be_visible()
        reasons = []
        if record.disagreement:
            reasons.append('两模型类别判断不一致')
        if record.near_threshold:
            reasons.append('主模型得分接近既定阈值'
                           f'（与阈值的距离不超过 {descriptor["config"]["near_margin"]:.6g}）')
        expect(self.page.get_by_text(
            '本条提示：' + ('；'.join(reasons) if reasons else '未触发分歧或近阈值提示'),
            exact=True)).to_be_visible()
        outside = actual['统计提示'].eq('超出正常参考区间')
        missing = actual['原值'].isna()
        expect(self.page.get_by_text(
            f'特征核对：{int(outside.sum())} 项超出训练正常参考区间；'
            f'{int(missing.sum())} 项缺失。', exact=True)).to_be_visible()
        for _, item in actual[outside | missing].head(3).iterrows():
            value = '缺失' if pd.isna(item['原值']) else f'{item["原值"]:.6g}'
            low = '未提供' if pd.isna(item['正常P01']) else f'{item["正常P01"]:.6g}'
            high = '未提供' if pd.isna(item['正常P99']) else f'{item["正常P99"]:.6g}'
            expect(self.page.get_by_text(
                f'• {item["字段"]}：{value}（{item["单位"]}）；'
                f'正常参考 {low}—{high}；{item["统计提示"]}。', exact=True)).to_be_visible()

    def quality_feedback(self, content, name, pred, values, rejected):
        """Check raw cell text, physical rows and counts against the exact uploaded CSV."""
        raw = read_csv_checked(io.BytesIO(content))
        fields = self.models[self.kind]['descriptor']['config']['features']
        missing = values.isna()
        accepted_counts = missing.sum(axis=0).astype('int64')
        problem_cells = int(accepted_counts.sum()) + len(rejected) * len(fields)
        affected = int(missing.any(axis=1).sum()) + len(rejected)
        expect(self.page.get_by_text(
            f'涉及 {affected:,} 条记录；已定位 {problem_cells:,} 个问题单元格；'
            f'整行拒绝 {len(rejected):,} 条。', exact=True)).to_be_visible()
        expander = self.page.locator('[data-testid="stExpander"]').filter(
            has=self.page.get_by_text('查看字段问题汇总', exact=True))
        expect(expander).to_have_count(1)
        if expander.locator('details').get_attribute('open') is None:
            expander.locator('summary').click()
        expected_summary = pd.DataFrame({
            '字段': fields, '可判断记录缺失数': accepted_counts.reindex(fields).to_numpy(),
            '整行拒绝涉及数': len(rejected),
            '问题单元格数': accepted_counts.reindex(fields).to_numpy() + len(rejected),
        })
        assert_frames(csv(self.download(None, name + '-quality-summary.csv',
                                       table=expander.locator('[data-testid="stDataFrame"]'))),
                      expected_summary)
        problems = []
        for position, row in enumerate(pred.itertuples(index=False)):
            if missing.iloc[position].any():
                problems.append((row.source_row, row.record_id,
                                 missing.columns[missing.iloc[position]].tolist(), False))
        problems.extend((row.source_row, row.record_id, fields, True)
                        for row in rejected.itertuples(index=False))
        raw_positions = dict(zip(raw.attrs['source_rows'], range(len(raw))))
        for physical_row, rid, bad_fields, refused in sorted(problems):
            self.pick('选择问题记录核对字段', f'CSV 第 {physical_row} 行 ｜ {rid}')
            expect(self.page.get_by_text(f'本次 CSV 第 {physical_row} 行：{name}',
                                         exact=True)).to_be_visible()
            expect(self.page.get_by_text(f'记录来源：{name}，第 {physical_row} 行',
                                         exact=True)).to_be_visible()
            details = []
            for field in bad_fields:
                value = raw.iloc[raw_positions[physical_row]][field]
                details.append({
                    '字段': field, 'CSV读取值': value,
                    '适配结果': ('读取值为空；' if value == '' else '') + (
                        '全部必需字段无效；具体原因未细分' if refused
                        else '适配后缺失；具体原因未细分'),
                    '处理状态': '整行无法判断，未进入模型' if refused else '可判断，使用既有训练填补器',
                })
            # This is the final field-detail table, below the queue/evidence/summary grids.
            actual = csv(self.download(None, f'{name}-row-{physical_row}-quality.csv',
                                       table=self.page.locator('[data-testid="stDataFrame"]').last))
            assert_frames(actual, pd.DataFrame(details))
        if not affected:
            expect(self.page.get_by_text('本次未发现适配后缺失或被拒绝的记录。正常参考区间外的数值不计为输入无效。',
                                         exact=True)).to_be_visible()

    def empty_filter_recovery(self):
        frame = self.snapshots[self.kind].set_index('record_id')
        status = next(self.saved[self.kind][rid]['status'] for rid in self.ids[1:]
                      if not (frame.loc[rid, 'disagreement'] and frame.loc[rid, 'near_threshold']))
        empty = self.table(status=status, disagreement=True, near=True, tag='explicit-empty-filter')
        assert empty.empty
        expect(self.page.get_by_text('没有符合当前筛选条件的记录，请调整筛选条件。', exact=True)).to_be_visible()
        expect(self.page.get_by_role('button', name='保存复核', exact=True)).to_have_count(0)
        assert len(csv(self.download('导出预测与复核 CSV', 'empty-filter-all-predictions.csv'))) == 1000
        self.table(tag='explicit-empty-filter-restored')
        self.assert_editor(self.ids[0])
        assert self.actual_history() == self.histories[self.kind]
        self.mark('empty_filter_restores_editor_history_and_all_rows_export')

    def anomalies(self):
        for name, (content, error) in input_cases(self.content).items():
            self.upload(content, name + '.csv')
            if error is not None:
                expect(self.page.get_by_text(re.compile(re.escape('无法判断：' + error)))).to_be_visible()
                expect(self.page.locator('[data-testid="stDataFrame"]')).to_have_count(0)
                for label in ('导出预测与复核 CSV', '保存复核', '导出运行清单'):
                    expect(self.page.get_by_role('button', name=label, exact=True)).to_have_count(0)
            else:
                pred, values, rejected = infer(read_csv_checked(io.BytesIO(content)),
                                               self.models[self.kind]['bundle'],
                                               hashlib.sha256(content).hexdigest(), name + '.csv')
                if len(pred):
                    actual = csv(self.download('导出预测与复核 CSV', name + '-predictions.csv'))
                    assert_prediction_export(actual, pred)
                    assert actual.review_status.eq('待复核').all() and actual.review_note.eq('').all()
                    if name.startswith('partial-'):
                        assert values.iloc[0].isna().sum() >= 1
                        self.click_text('原始顺序')
                        self.select_record(pred.record_id.iloc[0], predictions=pred)
                        evidence = csv(self.download(None, name + '-missing-evidence.csv',
                                                     table=self.evidence_table()))
                        for field in values.columns[values.iloc[0].isna()]:
                            row = evidence.set_index('字段').loc[field]
                            assert pd.isna(row['原值']) and row['统计提示'] == '缺失，预测时用训练中位数填补'
                else:
                    expect(self.page.get_by_text('没有可供预测的有效记录，请修正特征后重新导入。', exact=True)).to_be_visible()
                    for label in ('导出预测与复核 CSV', '保存复核'):
                        expect(self.page.get_by_role('button', name=label, exact=True)).to_have_count(0)
                if len(rejected):
                    assert_frames(csv(self.download('导出无法判断记录', name + '-rejected.csv')), rejected)
                self.quality_feedback(content, name + '.csv', pred, values, rejected)
            self.capture('input-' + name)
            assert self.actual_history() == self.histories[self.kind]
            self.mark('input_' + name + '_feedback_and_stale_result_clear')
            self.baseline('recovery-' + name)
            expect(self.page.get_by_text(re.compile('^本次 CSV 第 .*：' + re.escape(name + '.csv') + '$'))).to_have_count(0)
            expect(self.page.get_by_text('没有可供预测的有效记录，请修正特征后重新导入。', exact=True)).to_have_count(0)

    def model_faults(self):
        frozen = self.models['frozen']['directory']
        for name in ('missing', 'corrupt', 'mismatched-pair'):
            directory = self.out / 'fault-copies' / name
            directory.mkdir(parents=True)
            shutil.copyfile(frozen / 'model.json', directory / 'model.json')
            if name == 'corrupt':
                (directory / 'model.joblib').write_bytes((frozen / 'model.joblib').read_bytes()[:-32])
            elif name == 'mismatched-pair':
                shutil.copyfile(self.models['rebuilt']['directory'] / 'model.joblib', directory / 'model.joblib')
            self.start('frozen', directory=directory)
            expect(self.page.get_by_text(re.compile('模型尚未就绪，无法判断：'))).to_be_visible()
            expect(self.page.get_by_role('button', name='检查数据并预测', exact=True)).to_have_count(0)
            self.capture('model-' + name)
            assert self.actual_history() == self.histories['frozen']
            self.mark('isolated_model_' + name + '_refused_without_fallback')
            self.start('frozen')
            self.baseline('model-recovery-' + name)
        for failure in ('exception', 'nan-output'):
            launcher = self.out / ('inject-' + failure + '.py')
            body = ("raise RuntimeError('P0_TEST_ONLY_PREDICTION_FAILURE')" if failure == 'exception'
                    else "return np.full((len(values), 2), np.nan)")
            launcher.write_text(
                'import runpy\nimport numpy as np\nimport app_model\nimport streamlit as st\n'
                'if st.sidebar.button("P0_TEST_ONLY_ARM_FAILURE"):\n    app_model._p0_fault_armed = True\n'
                'if st.sidebar.button("P0_TEST_ONLY_CLEAR_FAILURE"):\n    app_model._p0_fault_armed = False\n'
                'if not hasattr(app_model, "_p0_original_load"):\n'
                '    app_model._p0_original_load = app_model.load_model_bundle\n'
                'original = app_model._p0_original_load\n'
                'class Injected:\n    def __init__(self, original):\n        self.original = original\n'
                '    def predict_proba(self, values):\n'
                '        if not getattr(app_model, "_p0_fault_armed", False):\n'
                '            return self.original.predict_proba(values)\n        ' + body + '\n'
                'def load(*args):\n    bundle = original(*args)\n'
                '    bundle["rf"] = app_model.CheckedPredictor(Injected(bundle["rf"]), "rf")\n    return bundle\n'
                'app_model.load_model_bundle = load\n'
                f'runpy.run_path({str(ROOT / "app.py")!r}, run_name="__main__")\n', encoding='utf-8')
            self.start('frozen', launcher=launcher)
            self.baseline('injected-before-' + failure)
            self.click_button('P0_TEST_ONLY_ARM_FAILURE')
            self.click_button('检查数据并预测')
            message = 'P0_TEST_ONLY_PREDICTION_FAILURE' if failure == 'exception' else 'rf 模型输出无效，无法判断'
            expect(self.page.get_by_text(re.compile(re.escape(message)))).to_be_visible()
            expect(self.page.get_by_role('button', name='导出预测与复核 CSV', exact=True)).to_have_count(0)
            expect(self.page.locator('[data-testid="stDataFrame"]')).to_have_count(0)
            expect(self.page.get_by_role('button', name='保存复核', exact=True)).to_have_count(0)
            self.capture('injected-' + failure)
            assert self.actual_history() == self.histories['frozen']
            self.mark('explicit_test_only_warm_injected_' + failure + '_clears_old_results')
            self.click_button('P0_TEST_ONLY_CLEAR_FAILURE')
            self.baseline('injected-recovery-' + failure)

    def run(self):
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(channel='msedge', headless=True, slow_mo=60)
            self.context = self.browser.new_context(viewport={'width': 1600, 'height': 1100},
                                                    accept_downloads=True, record_video_dir=str(self.out / 'video'),
                                                    record_video_size={'width': 1600, 'height': 1100})
            # Headless Edge cancels native save dialogs. Use Streamlit's shipped CSV polyfill.
            self.context.add_init_script('delete globalThis.showSaveFilePicker')
            self.report['browser_file_export_mode'] = 'Native save picker disabled; Streamlit CSV polyfill, unchanged table data'
            self.context.tracing.start(screenshots=True, snapshots=True, sources=False)
            self.page = self.context.new_page()
            self.page.set_default_timeout(30000)
            expect.set_options(timeout=30000)
            self.page.on('pageerror', lambda error: self.errors.append(str(error)))
            self.start('frozen')
            first = self.baseline('frozen-first')
            self.create_states()
            self.verify_states('frozen-saved')
            self.page.reload()
            self.verify_states('frozen-refreshed')
            self.start('rebuilt')
            self.baseline('rebuilt-first')
            self.create_states()
            self.verify_states('rebuilt-saved')
            self.start('frozen')
            self.verify_states('frozen-return')
            assert_frames(self.snapshots['frozen'][list(self.direct['frozen'][0].columns)],
                          first[list(self.direct['frozen'][0].columns)])
            self.start('rebuilt')
            self.verify_states('rebuilt-return')
            self.mark('real_bidirectional_switch_with_both_version_histories_preserved')
            self.start('frozen')
            self.verify_states('frozen-before-errors')
            self.all_tables_and_evidence()
            self.empty_filter_recovery()
            self.stale_save_conflict()
            self.continuous_review()
            self.anomalies()
            self.model_faults()
            self.verify_states('frozen-final')
            assert not self.errors
            self.report['passed'] = True
        except Exception as error:
            self.report['failure'] = f'{type(error).__name__}: {error}'
            if self.page is not None:
                with contextlib.suppress(Exception):
                    self.capture('failure')
                    (self.out / 'failure-ui.txt').write_text(self.page.locator('body').aria_snapshot(), encoding='utf-8')
            raise
        finally:
            cleanup_errors = []
            def cleanup(name, action):
                try:
                    action()
                except Exception as error:
                    cleanup_errors.append(name + ': ' + repr(error))
            if self.context is not None:
                for action in (lambda: self.context.tracing.stop(path=str(self.out / 'trace.zip')),
                               self.context.close):
                    cleanup('browser evidence', action)
            if self.browser is not None:
                cleanup('browser close', self.browser.close)
            if self.playwright is not None:
                cleanup('playwright stop', self.playwright.stop)
            cleanup('owned server stop', self.stop)
            preserved = False
            try:
                preserved = all(sha256(self.models[kind]['directory'] / name) == value
                                for kind, entries in self.artifact_hashes.items() for name, value in entries.items())
            except Exception as error:
                cleanup_errors.append('model preservation: ' + repr(error))
            self.report['original_model_files_unchanged'] = preserved
            self.report['cleanup_errors'] = cleanup_errors
            self.report['passed'] = self.report['passed'] and preserved and not cleanup_errors
            self.report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            self.report['files'] = {}
            for path in self.out.rglob('*'):
                if path.is_file() and path.name != 'report.json':
                    try:
                        self.report['files'][str(path.relative_to(self.out))] = sha256(path)
                    except Exception as error:
                        self.report['passed'] = False
                        cleanup_errors.append('evidence hash: ' + repr(error))
            self.acceptance.finish(self.report['passed'])
            print(json.dumps({'passed': self.report['passed'], 'checks': len(self.report['checks'])}), flush=True)
        if not self.report['passed']:
            raise RuntimeError('P0 evidence finalization failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-artifacts', required=True, type=Path)
    parser.add_argument('--rebuilt-artifacts', required=True, type=Path)
    parser.add_argument('--frozen-sha256', default='3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7')
    parser.add_argument('--rebuilt-sha256', default='40ee89927c853d4d0ada25f32b7ed464838562c3c0b2b52b737570d6d9d4bea6')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    acceptance = AcceptanceRun(args.output.resolve(), ROOT, report_name='report.json')
    try:
        P0Browser(args, acceptance=acceptance).run()
    except Exception as error:
        acceptance.report.setdefault('failure', f'{type(error).__name__}: {error}')
        acceptance.finish(False)
        raise


if __name__ == '__main__':
    main()
