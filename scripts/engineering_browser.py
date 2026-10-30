"""Real Edge regression with an independent model directory and dedicated review DB.
Legacy page workflow; r9 UI acceptance uses engineering_r9_acceptance.py.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import Locator, expect, sync_playwright

from app_model import load_model_bundle, prepare_model
from liushao.data import read_csv_checked, sha256
from liushao.model import infer, ranked, review_ranked
from liushao.reviews import history as review_history
from scripts.engineering_reproduce import WORK, prepare_workspace
from scripts.engineering_run import AcceptanceRun, start_owned_process

ROOT = Path(__file__).resolve().parents[1]


def scroll_to_capture_anchor(anchor: Locator, timeout: float = 45000) -> None:
    """Wait for a rerender's old heading to disappear before positioning a capture."""
    expect(anchor).to_have_count(1, timeout=timeout)
    anchor.scroll_into_view_if_needed(timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path,
                        help='Explicit independent model directory for the current engineering app')
    parser.add_argument('--expected-model-sha256', help='Trusted model hash; required with --artifacts')
    parser.add_argument('--output', type=Path, default=ROOT / 'test-results/engineering/browser')
    args = parser.parse_args()
    acceptance = AcceptanceRun(args.output.resolve(), ROOT, report_name='report.json')
    try:
        run_browser(args, acceptance)
    except Exception as error:
        acceptance.report.setdefault('failure', f'{type(error).__name__}: {error}')
        acceptance.finish(False)
        raise


def run_browser(args, acceptance):
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if args.artifacts is not None:
        if workspace != ROOT or not args.expected_model_sha256:
            raise ValueError('Explicit models require the current app workspace and trusted model SHA')
        artifacts = args.artifacts.resolve()
    else:
        if workspace != WORK.resolve():
            raise ValueError('Use the frozen workspace produced by engineering_reproduce')
        prepare_workspace(reuse=True)
        artifacts = workspace / 'artifacts'
    model_bytes, descriptor_json, _ = prepare_model(artifacts / 'model.joblib', artifacts / 'model.json')
    descriptor = json.loads(descriptor_json)
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    if args.expected_model_sha256 is not None and model_sha != args.expected_model_sha256:
        raise ValueError('Model hash mismatch')
    bundle = load_model_bundle(model_bytes, descriptor_json)
    content = (artifacts / 'demo.csv').read_bytes()
    expected, _, rejected = infer(read_csv_checked(io.BytesIO(content)), bundle,
                                 hashlib.sha256(content).hexdigest(), 'engineering-real.csv')
    assert not len(rejected) and len(expected) == 1000
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    process = None
    process_tree = None
    server_log = None
    playwright = browser = context = page = None
    checks, errors, processes = [], [], []
    report = acceptance.report
    report.update({'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'scope': 'real public held-out demo uploaded through Edge; not user research',
              'model_version': bundle['model_version'], 'model_sha256': model_sha,
              'descriptor_sha256': hashlib.sha256(descriptor_json.encode('utf-8')).hexdigest(),
              'artifact_sha256': {'model.joblib': model_sha,
                                  'model.json': hashlib.sha256(descriptor_json.encode('utf-8')).hexdigest(),
                                  'demo.csv': hashlib.sha256(content).hexdigest()},
              'input_sha256': hashlib.sha256(content).hexdigest(), 'input_rows': len(expected),
              'artifacts': str(artifacts), 'workspace': str(workspace),
              'git_head': (acceptance.report['source_before']['head'] if (ROOT/'.git').exists() else None),
              'source_identity': {'kind':'git-commit' if (ROOT/'.git').exists() else 'content-snapshot',
                                  'value':acceptance.report['source_before']['head']},
              'source_sha256': {name: sha256(workspace / name) for name in
                                ('app.py', 'app_model.py', 'app_quality.py', 'liushao/model.py',
                                 'liushao/data.py', 'liushao/reviews.py')},
              'runner_sha256': sha256(__file__),
              'checks': checks, 'page_errors': errors, 'server_pids': processes,
              'server_cleanup': []})
    acceptance.checkpoint()

    def stop():
        nonlocal process, process_tree, server_log
        cleanup = None
        if process is not None:
            cleanup = process_tree.cleanup()
            report['server_cleanup'].append(cleanup)
            process = None
            process_tree = None
        if server_log is not None:
            server_log.close()
            server_log = None
        if cleanup is not None and not cleanup['passed']:
            raise RuntimeError('Owned Streamlit process cleanup failed: ' + repr(cleanup))

    def start():
        nonlocal process, process_tree, server_log
        server_log = (output / f'server-{len(processes) + 1}.log').open('w', encoding='utf-8')
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONPATH': str(workspace),
               'LIUSHAO_REVIEW_DB': str(output / 'reviews.sqlite'),
               'LIUSHAO_ARTIFACTS_DIR': str(artifacts)}
        process, process_tree = start_owned_process(
            [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.headless=true',
             '--server.address=127.0.0.1', f'--server.port={port}',
             '--browser.gatherUsageStats=false'], cwd=workspace, env=env,
            stdout=server_log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        processes.append(process.pid)
        process_tree.capture()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(90):
            process_tree.capture()
            if process.poll() is not None:
                raise RuntimeError('Isolated Streamlit process exited before becoming healthy')
            if not process_tree.owns_tcp_listener('127.0.0.1', port):
                time.sleep(.5)
                continue
            try:
                with opener.open(url + '/_stcore/health', timeout=1) as response:
                    if response.status == 200 and response.read().strip() == b'ok':
                        if (process.poll() is not None
                                or not process_tree.owns_tcp_listener('127.0.0.1', port)):
                            raise RuntimeError('Health listener no longer belongs to owned server')
                        return
            except OSError:
                time.sleep(.5)
        raise TimeoutError('Streamlit health timeout')

    try:
        start()
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(channel='msedge', headless=True, slow_mo=120)
        context = browser.new_context(viewport={'width': 1600, 'height': 1100},
                                      record_video_size={'width': 1600, 'height': 1100},
                                      accept_downloads=True, record_video_dir=str(output / 'video'))
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        page.set_default_timeout(45000)
        expect.set_options(timeout=45000)
        page.on('pageerror', lambda error: errors.append(str(error)))

        def capture(name, anchor=None):
            process_tree.capture()
            if anchor is None:
                page.locator('[data-testid="stMain"]').evaluate('(el) => el.scrollTop = 0')
            else:
                scroll_to_capture_anchor(anchor)
            page.wait_for_timeout(1000)
            page.screenshot(path=str(output / name))
            acceptance.checkpoint()

        def checkbox(label, checked):
            # Streamlit hides the native input; operate its visible text label.
            if page.get_by_role('checkbox', name=label, exact=True).is_checked() != checked:
                page.get_by_text(label, exact=True).click()
            expect(page.get_by_role('checkbox', name=label, exact=True)).to_be_checked(checked=checked)

        def upload(payload=content, name='engineering-real.csv'):
            page.get_by_text('上传 CSV', exact=True).click()
            page.locator('input[type=file]').set_input_files(
                {'name': name, 'mimeType': 'text/csv', 'buffer': payload})
            expect(page.locator('[data-testid="stMain"]').get_by_text(name, exact=True)).to_be_visible()
            page.get_by_role('button', name='检查数据并预测', exact=True).click()

        def download(label, name):
            prepare = page.get_by_role('button', name='核对并准备导出', exact=True)
            if label != '导出无法判断记录' and prepare.count():
                prepare.click()
            with page.expect_download() as downloaded:
                page.get_by_role('button', name=label, exact=True).click()
            path = output / name
            downloaded.value.save_as(path)
            resume = page.get_by_role('button', name='返回复核编辑', exact=True)
            if resume.count():
                resume.click()
                expect(page.get_by_role('button', name='核对并准备导出', exact=True)).to_be_visible()
            return path

        def save(status, note):
            page.get_by_role('combobox', name=re.compile('复核结论')).click()
            page.get_by_role('option', name=status, exact=True).click()
            page.get_by_role('textbox', name='复核备注', exact=True).fill(note)
            page.get_by_role('button', name='保存复核', exact=True).click()
            expect(page.get_by_text('已保存复核，修改历史已追加。', exact=True)).to_be_visible()
            for _ in range(100):
                saved = review_history(output / 'reviews.sqlite', model_version=bundle['model_version'])
                if saved and saved[-1]['note'] == note and saved[-1]['status'] == status:
                    break
                page.wait_for_timeout(100)
            else:
                raise AssertionError('The newly requested review was not persisted')

        def first_record(table):
            row = table.iloc[0]
            label = f'行 {row.source_row} ｜ 得分 {row.rf_score:.4f} ｜ {row.record_id[:12]}'
            page.get_by_role('combobox', name=re.compile('选择记录查看详情')).click()
            expect(page.get_by_role('option').first).to_have_text(label)
            page.get_by_role('option').first.click()
            expect(page.locator('[data-testid="stCode"]').filter(has_text=row.record_id)).to_be_visible()
            expect(page.get_by_text(f'来源：{row.source_file}，第 {row.source_row} 行',
                                    exact=True)).to_be_visible()
            expect(page.get_by_text(
                f'主模型得分 {row.rf_score:.6f} / 阈值 {bundle["config"]["rf_threshold"]:.2f}；'
                f'参照得分 {row.lr_score:.6f} / 阈值 {bundle["config"]["lr_threshold"]:.2f}',
                exact=True)).to_be_visible()

        page.goto(url)
        upload()
        expect(page.locator('[data-testid="stCode"]').filter(
            has_text=bundle['model_version'])).to_be_visible()
        expect(page.get_by_role('combobox', name=re.compile('复核结论'))).to_be_visible()
        first_record(ranked(expected))
        checks.append('real_csv_upload_validation_prediction')
        capture('01-prediction.png')
        page.get_by_text('待核实', exact=True).click()
        hints = int((expected.disagreement | expected.near_threshold).sum())
        expect(page.get_by_text(re.compile(f'符合条件 {hints:,} 条'))).to_be_visible()
        first_record(review_ranked(expected))
        capture('02-hint-queue.png', page.get_by_text('研判队列', exact=True))
        page.get_by_text('高异常得分', exact=True).click()
        expect(page.get_by_text(re.compile('符合条件 1,000 条'))).to_be_visible()
        first_record(ranked(expected))
        checks.append('score_sort_and_hint_queue')
        for disagreement, near in ((True, False), (False, True), (True, True)):
            checkbox('仅模型分歧', disagreement)
            checkbox('仅近阈值', near)
            selected = expected[(expected.disagreement if disagreement else True)
                                & (expected.near_threshold if near else True)]
            expect(page.get_by_text(re.compile(f'符合条件 {len(selected):,} 条'))).to_be_visible()
            if len(selected):
                first_record(ranked(selected))
        capture('03-independent-filters.png', page.get_by_text('研判队列', exact=True))
        checkbox('仅模型分歧', False)
        checkbox('仅近阈值', False)
        first_record(ranked(expected))
        checks.append('independent_disagreement_near_threshold_and_intersection_filters')
        capture('04-evidence.png', page.get_by_text('记录详情与依据', exact=True))
        page.get_by_role('combobox', name=re.compile('复核结论')).click()
        page.get_by_role('option', name='确认正常', exact=True).click()
        page.get_by_role('textbox', name='复核备注', exact=True).fill('')
        page.get_by_role('button', name='保存复核', exact=True).click()
        expect(page.get_by_text('复核保存失败：确认异常或确认正常时必须填写判断依据',
                                exact=True)).to_be_visible()
        assert not review_history(output / 'reviews.sqlite', model_version=bundle['model_version'])
        checks.append('confirmation_requires_evidence_without_writing_history')
        first_note = '自动化浏览器回归：已核对来源和特征，等待背景资料。'
        second_note = '自动化验证样例结论；仅检验保存，不代表真实人工研判。'
        save('待补证', first_note)
        save('确认正常', second_note)
        page.get_by_role('combobox', name=re.compile('复核状态筛选')).click()
        page.get_by_role('option', name='确认正常', exact=True).click()
        expect(page.get_by_text(re.compile('符合条件 1 条'))).to_be_visible()
        expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(second_note)
        page.get_by_text('查看该记录的复核历史', exact=True).click()
        capture('05-review-and-filter.png', page.get_by_text('人工复核', exact=True))
        checks.append('detail_evidence_two_reviews_filter_identity')
        exported = pd.read_csv(download('导出预测与复核 CSV', 'predictions.csv'))
        history = json.loads(download('导出完整复核历史', 'history.json').read_text(encoding='utf-8'))
        manifest = json.loads(download('导出运行清单', 'manifest.json').read_text(encoding='utf-8'))
        assert len(exported) == len(expected) == manifest['valid_rows']
        assert exported.record_id.tolist() == expected.record_id.tolist()
        assert not any('label' in col.lower() or 'ground_truth' in col for col in exported)
        assert np.allclose(exported.rf_score, expected.rf_score, atol=1e-12, rtol=0)
        assert np.allclose(exported.lr_score, expected.lr_score, atol=1e-12, rtol=0)
        for column in ['rf_anomaly', 'lr_anomaly', 'disagreement', 'near_threshold', 'model_version']:
            assert exported[column].tolist() == expected[column].tolist()
        reviewed = exported[exported.review_note.eq(second_note)]
        assert len(reviewed) == 1 and reviewed.review_status.iloc[0] == '确认正常'
        rid = reviewed.record_id.iloc[0]
        assert len(history) == 2 and {item['record_id'] for item in history} == {rid}
        assert [item['note'] for item in history] == [first_note, second_note]
        assert manifest['rf_threshold'] == bundle['config']['rf_threshold']
        assert manifest['lr_threshold'] == bundle['config']['lr_threshold']
        assert manifest['model_version'] == bundle['model_version']
        assert manifest['near_margin'] == bundle['config']['near_margin']
        capture('06-export.png', page.get_by_text('导出', exact=True))
        checks.append('all_rows_export_matches_predictions_history_and_config_after_filter')
        page.reload()
        upload()
        expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(second_note)
        checks.append('refresh_and_reupload_preserve_review')
        stop()
        start()
        page.goto(url)
        upload()
        expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(second_note)
        after = json.loads(download('导出完整复核历史', 'history-after-restart.json').read_text(encoding='utf-8'))
        assert after == history and len(set(processes)) == 2
        capture('07-restart-restored.png', page.get_by_text('人工复核', exact=True))
        checks.append('real_server_process_restart_preserves_complete_history')
        raw = pd.read_csv(io.BytesIO(content)).iloc[:3].astype(object)
        raw.iloc[2, :] = 'bad'
        raw['Label'] = ['Benign', 'Bot', 'Bot']
        lines = raw.to_csv(index=False).splitlines()
        payload = ('\n'.join(lines[:2] + [''] + lines[2:]) + '\n').encode('utf-8')
        upload(payload, 'partial-invalid.csv')
        expect(page.get_by_text(re.compile('无法判断比例 33.33%'))).to_be_visible()
        partial = pd.read_csv(download('导出预测与复核 CSV', 'partial-predictions.csv'))
        unjudgeable = pd.read_csv(download('导出无法判断记录', 'unjudgeable.csv'))
        assert partial.source_row.tolist() == [2, 4] and unjudgeable.source_row.tolist() == [5]
        checks.append('invalid_values_blank_lines_and_unjudgeable_export')
        capture('08-quality-feedback.png')
        malformed = (lines[0] + ',' + lines[0].split(',')[0] + '\n' + lines[1] + ',1\n').encode()
        upload(malformed, 'duplicate-columns.csv')
        expect(page.get_by_text(re.compile('无法判断：CSV 表头包含重复字段'))).to_be_visible()
        missing = raw.drop(columns=raw.columns[0]).to_csv(index=False).encode()
        upload(missing, 'missing-column.csv')
        expect(page.get_by_text(re.compile('无法判断：缺少必需特征'))).to_be_visible()
        expect(page.get_by_role('button', name='导出预测与复核 CSV', exact=True)).to_have_count(0)
        checks.append('malformed_inputs_clear_stale_prediction_and_export')
        capture('09-input-error.png')
        upload()
        expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(second_note)
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
        recovered = json.loads(download('导出完整复核历史', 'history-after-input-recovery.json').read_text(encoding='utf-8'))
        assert recovered == history
        capture('10-recovery.png')
        checks.append('valid_reupload_recovers_after_invalid_input_without_history_loss')
        assert not errors, errors
        report.update(passed=True, reviewed_record_id=rid, browser='Microsoft Edge / Playwright')
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        if page is not None:
            with contextlib.suppress(Exception):
                page.screenshot(path=str(output / 'failure.png'), full_page=True)
                (output / 'failure-ui.txt').write_text(page.locator('body').aria_snapshot(), encoding='utf-8')
        raise
    finally:
        cleanup_errors = []
        def cleanup(name, action):
            try:
                action()
            except Exception as error:
                cleanup_errors.append(name + ': ' + repr(error))
        if context is not None:
            cleanup('trace', lambda: context.tracing.stop(path=str(output / 'trace.zip')))
            cleanup('browser context', context.close)
        if browser is not None:
            cleanup('browser', browser.close)
        if playwright is not None:
            cleanup('playwright', playwright.stop)
        cleanup('owned server', stop)
        report['cleanup_errors'] = cleanup_errors
        try:
            report['original_model_files_unchanged'] = all(
                sha256(artifacts / name) == value
                for name, value in report['artifact_sha256'].items())
            report['workspace_source_sha256_after'] = {
                name: sha256(workspace / name) for name in report['source_sha256']}
            report['workspace_source_unchanged'] = (
                report['source_sha256'] == report['workspace_source_sha256_after'])
        except Exception as error:
            report['original_model_files_unchanged'] = False
            report['workspace_source_unchanged'] = False
            cleanup_errors.append('model/source preservation: ' + repr(error))
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['files'] = {}
        for path in output.rglob('*'):
            if path.is_file() and path.name != 'report.json':
                try:
                    report['files'][str(path.relative_to(output))] = sha256(path)
                except Exception as error:
                    cleanup_errors.append('evidence hash: ' + repr(error))
        report['passed'] = (report['passed'] and not cleanup_errors
                            and report['original_model_files_unchanged']
                            and report['workspace_source_unchanged'])
        acceptance.finish(report['passed'])
        print(json.dumps({'passed': report['passed'], 'checks': checks}, ensure_ascii=False))
        if not report['passed'] and 'failure' not in report:
            raise RuntimeError('Browser evidence finalization or source integrity failed')


if __name__ == '__main__':
    main()
