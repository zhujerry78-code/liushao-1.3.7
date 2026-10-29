"""Exercise an extracted demo ZIP using its own installed Python and owned database.
Legacy page workflow; r9 UI acceptance uses engineering_r9_acceptance.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone


def load_study_answers(path: Path | None) -> dict | None:
    """Reject missing/invalid study checks instead of downgrading to demo mode."""
    if path is None:
        return None
    answers = json.loads(path.read_text(encoding='utf-8'))
    required = {
        'not_attack_ground_truth', 'input_rows', 'valid_rows', 'unjudgeable_rows',
        'partial_missing_rows', 'injected_quality_cell', 'T2_record_id', 'T2_rf_score',
        'T3_record_id', 'T3_disagreement', 'T3_near_threshold', 'T4_status', 'T4_note',
    }
    if not isinstance(answers, dict) or not required.issubset(answers):
        raise ValueError('Invalid study answer-key schema: required checks are missing')
    for key in ('input_rows', 'valid_rows', 'unjudgeable_rows', 'partial_missing_rows'):
        if type(answers[key]) is not int or answers[key] < 0:
            raise ValueError(f'Invalid study answer-key count: {key}')
    for key in ('not_attack_ground_truth', 'T3_disagreement', 'T3_near_threshold'):
        if type(answers[key]) is not bool:
            raise ValueError(f'Invalid study answer-key boolean: {key}')
    for key in ('injected_quality_cell', 'T4_status', 'T4_note'):
        if not isinstance(answers[key], str) or not answers[key].strip():
            raise ValueError(f'Invalid study answer-key text: {key}')
    for key in ('T2_record_id', 'T3_record_id'):
        if not isinstance(answers[key], str) or re.fullmatch(
                r'[0-9a-f]{64}:[1-9][0-9]*', answers[key]) is None:
            raise ValueError(f'Invalid study answer-key identity: {key}')
    score = answers['T2_rf_score']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid study answer-key score')
    return answers


def run(package: Path, output: Path, hold_seconds: float = 1.8,
        input_csv: Path | None = None, study_answer_key: Path | None = None) -> None:
    if study_answer_key is not None and input_csv is None:
        raise ValueError('--study-answer-key requires --input-csv')
    answers = load_study_answers(study_answer_key)
    output.mkdir(parents=True, exist_ok=False)
    package = package.resolve()
    if Path(sys.prefix).resolve() != (package / '.venv').resolve():
        raise ValueError('Run this check with the extracted package .venv Python')
    sys.path.insert(0, str(package))
    from app_model import prepare_model, load_model_bundle
    from app_quality import build_quality_view
    from liushao.data import read_csv_checked
    from liushao.model import infer, ranked, review_ranked
    from liushao.reviews import history
    from scripts.engineering_p0_browser import csv as read_prediction_csv, assert_prediction_export
    from scripts.engineering_p0_cases import assert_review_snapshot
    from scripts.engineering_run import start_owned_process
    import numpy as np
    import pandas as pd
    import psutil
    from playwright.sync_api import (Error as PlaywrightError,
                                     TimeoutError as PlaywrightTimeoutError,
                                     expect, sync_playwright)

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = json.loads((package / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, item in manifest['files'].items():
        assert digest(package / name) == item['sha256'], name
    model_bytes, descriptor, _ = prepare_model(package / 'artifacts/model.joblib',
                                              package / 'artifacts/model.json')
    bundle = load_model_bundle(model_bytes, descriptor)
    content = (input_csv or package / 'artifacts/demo.csv').read_bytes()
    input_name = input_csv.name if input_csv else 'judge-demo.csv'
    raw = read_csv_checked(io.BytesIO(content))
    expected, features, rejected = infer(raw, bundle,
                                        hashlib.sha256(content).hexdigest(), input_name)
    quality = build_quality_view(raw, expected, features, rejected)
    assert len(expected) == 1000 and not len(rejected) and len(features.columns) == 77
    assert bundle['model_version'] == 'c7a028b4e4cafedd'
    if answers is not None:
        assert answers['not_attack_ground_truth'] is True
        assert len(raw) == answers['input_rows']
        assert len(expected) == answers['valid_rows']
        assert len(rejected) == answers['unjudgeable_rows']
        assert quality.accepted_problem_rows == answers['partial_missing_rows']
        assert answers['injected_quality_cell'] == 'CSV physical row 2 / Flow Duration'
        assert ranked(expected).iloc[0].record_id == answers['T2_record_id']
        assert ranked(expected).iloc[0].rf_score == answers['T2_rf_score']
        target = review_ranked(expected).iloc[0]
        assert target.record_id == answers['T3_record_id']
        assert bool(target.disagreement) == answers['T3_disagreement']
        assert bool(target.near_threshold) == answers['T3_near_threshold']
    database = package / 'demo-state/reviews.sqlite'
    assert not database.exists() or not history(database), (
        'A first-run check requires a demonstration database without previous reviews')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'software_sha': manifest['software_sha'],
              'software_version': manifest['software_version'],
              'model_version': bundle['model_version'],
              'model_sha256': digest(package / 'artifacts/model.joblib'),
              'descriptor_sha256': digest(package / 'artifacts/model.json'),
              'input_sha256': hashlib.sha256(content).hexdigest(), 'input_rows': len(expected),
              'input_name': input_name,
              'study_answer_key_sha256': digest(study_answer_key) if study_answer_key else None,
              'package': str(package), 'python': sys.executable, 'prefix': sys.prefix,
              'base_prefix': sys.base_prefix, 'database': str(database),
              'runner_sha256': digest(__file__), 'checks': [], 'cleanup': [],
              'service_runs': [],
              'page_errors': [], 'study_checks': [],
              'scope': 'Fresh extracted ZIP / new venv; automated UI demonstration; no user study'}
    (output / 'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    process = owner = log = None
    pids = []

    def start():
        nonlocal process, owner, log
        log = (output / f'server-{len(pids) + 1}.log').open('w', encoding='utf-8')
        command = ['cmd.exe', '/d', '/c', 'run-demo.cmd', '--headless', '--port', str(port)]
        process, owner = start_owned_process(
            command,
            cwd=package, env={**os.environ, 'LIUSHAO_NONINTERACTIVE': '1', 'PYTHONUTF8': '1'},
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        pids.append(process.pid)
        report['service_runs'].append({
            'launcher_pid': process.pid,
            'launcher_create_time': psutil.Process(process.pid).create_time(),
            'command': command, 'cwd': str(package), 'port': port,
            'started_utc': datetime.now(timezone.utc).isoformat(),
        })
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(100):
            owner.capture()
            if process.poll() is not None:
                raise RuntimeError(f'Demo launcher exited: {process.returncode}')
            if owner.owns_tcp_listener('127.0.0.1', port):
                try:
                    with opener.open(url + '/_stcore/health', timeout=1) as response:
                        if response.read().strip() == b'ok':
                            return
                except OSError:
                    pass
            time.sleep(.3)
        raise TimeoutError('Demo launcher did not own a healthy service')

    def stop():
        nonlocal process, owner, log
        if owner is not None:
            cleanup = owner.cleanup()
            report['cleanup'].append(cleanup)
            owner = process = None
            if not cleanup['passed']:
                raise RuntimeError('Demo service cleanup failed')
        if log:
            log.close()
            log = None

    try:
        start()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=True, slow_mo=180)
            context = browser.new_context(viewport={'width': 1920, 'height': 1080},
                                          record_video_size={'width': 1920, 'height': 1080},
                                          record_video_dir=str(output / 'video'), accept_downloads=True)
            context.add_init_script('delete window.showSaveFilePicker;')
            page = context.new_page()
            page.set_default_timeout(45000)
            expect.set_options(timeout=45000)
            page.on('pageerror', lambda error: report['page_errors'].append(str(error)))

            def capture(name, heading=None):
                if heading:
                    page.get_by_text(heading, exact=True).evaluate(
                        '(element) => element.scrollIntoView({block: "start"})')
                else:
                    page.locator('[data-testid="stMain"]').evaluate('(element) => element.scrollTop = 0')
                page.wait_for_timeout(hold_seconds * 1000)
                page.screenshot(path=str(output / name))

            def upload():
                page.get_by_text('上传 CSV', exact=True).click()
                page.locator('input[type=file]').set_input_files(
                    {'name': input_name, 'mimeType': 'text/csv', 'buffer': content})
                expect(page.locator('[data-testid="stMain"]').get_by_text(input_name, exact=True)).to_be_visible()
                page.get_by_role('button', name='检查数据并预测', exact=True).click()
                expect(page.get_by_text(re.compile('符合条件 1,000 条'))).to_be_visible()

            def pick(label, option, *, deadline=None, keyboard=True):
                # Same bounded, idempotent selection as main's capacity runner.
                # No save or other state-changing operation is retried.
                if deadline is None:
                    deadline = time.monotonic() + 45

                def remaining():
                    return max(1, min(5000, int((deadline - time.monotonic()) * 1000)))

                control = page.get_by_role('combobox', name=re.compile(label))
                while True:
                    try:
                        expect(page.get_by_test_id('stStatusWidgetRunningIcon')).to_have_count(
                            0, timeout=max(1, int((deadline - time.monotonic()) * 1000)))
                        expect(control).to_have_count(1, timeout=remaining())
                        control.click(timeout=remaining())
                        if keyboard:
                            control.fill(option, timeout=remaining())
                        option_node = page.get_by_role('option', name=option, exact=True)
                        expect(option_node).to_have_count(1, timeout=remaining())
                        expect(option_node).to_be_visible(timeout=remaining())
                        if keyboard:
                            control.press('ArrowDown', timeout=remaining())
                            control.press('Enter', timeout=remaining())
                        else:
                            # The short quality list needs no search or scroll.
                            # ArrowDown can advance its already active option.
                            option_node.click(timeout=remaining())
                        expect(page.get_by_test_id('stStatusWidgetRunningIcon')).to_have_count(
                            0, timeout=max(1, int((deadline - time.monotonic()) * 1000)))
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
                        page.keyboard.press('Escape')

            def select_record(rid):
                row = expected.set_index('record_id').loc[rid]
                option = f'行 {row.source_row} ｜ 得分 {row.rf_score:.4f} ｜ {rid[:12]}'
                control = page.get_by_role('combobox', name=re.compile('选择记录查看详情'))
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    pick('选择记录查看详情', option, deadline=deadline)
                    while time.monotonic() < deadline:
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
                            return rid
                        if state == 'reset':
                            break
                        page.wait_for_timeout(min(50, max(1, (deadline - time.monotonic()) * 1000)))
                raise AssertionError(f'Record selection did not settle on exact record_id {rid}')

            def select_first(table):
                # Download the actual visible queue to prove its order before
                # selecting the exact identity. Selection alone cannot prove ranking.
                visible = pd.read_csv(download(None, f'queue-{len(list(output.glob("queue-*.csv"))) + 1}.csv',
                                               table=page.get_by_test_id('stDataFrame').first),
                                      float_precision='round_trip')
                assert visible['来源行号'].tolist() == table.head(1000).source_row.tolist()
                assert np.allclose(visible['主模型得分'], table.head(1000).rf_score,
                                   atol=1e-12, rtol=0)
                return select_record(table.iloc[0].record_id)

            def check(label, value):
                if page.get_by_role('checkbox', name=label, exact=True).is_checked() != value:
                    page.get_by_text(label, exact=True).click()
                expect(page.get_by_role('checkbox', name=label, exact=True)).to_be_checked(checked=value)

            def download(label, name, *, table=None):
                expect(page.get_by_test_id('stStatusWidgetRunningIcon')).to_have_count(0)
                if table is not None:
                    table.hover()
                    button = table.get_by_role('button', name='Download as CSV', exact=True)
                else:
                    prepare = page.get_by_role('button', name='核对并准备导出', exact=True)
                    if label != '导出无法判断记录' and prepare.count():
                        prepare.click()
                    button = page.get_by_role('button', name=label, exact=True)
                expect(button).to_have_count(1)
                with page.expect_download() as saved:
                    if table is not None:
                        button.focus()
                        button.press('Enter')
                    else:
                        button.click()
                destination = output / name
                saved.value.save_as(destination)
                resume = page.get_by_role('button', name='返回复核编辑', exact=True)
                if table is None and resume.count():
                    resume.click()
                    expect(page.get_by_role('button', name='核对并准备导出', exact=True)).to_be_visible()
                return destination

            page.goto(url)
            capture('00-start.png')
            upload()
            if answers is not None:
                expect(page.get_by_text(
                    f'涉及 {quality.affected_rows:,} 条记录；已定位 {quality.problem_cells:,} 个问题单元格；'
                    f'整行拒绝 {quality.rejected_rows:,} 条。', exact=True)).to_be_visible()
                expect(page.get_by_text(re.compile(
                    f'{answers["partial_missing_rows"]} 条含缺失特征，由训练中位数填补'))).to_be_visible()
                for label, value in [('有效记录', len(expected)), ('无法判断', len(rejected))]:
                    metric = page.get_by_test_id('stMetric').filter(
                        has=page.get_by_text(label, exact=True))
                    expect(metric.get_by_test_id('stMetricValue')).to_have_text(f'{value:,}')
                issue = quality.records.loc[quality.records.input_row.eq(2)]
                assert len(issue) == 1
                issue_id = issue.index[0]
                pick('选择问题记录核对字段', f'CSV 第 2 行 ｜ {issue_id}', keyboard=False)
                expect(page.get_by_text(f'本次 CSV 第 2 行：{input_name}', exact=True)).to_be_visible()
                actual_quality = pd.read_csv(download(None, 'study-row2-quality.csv',
                                             table=page.get_by_test_id('stDataFrame').last),
                                             dtype=str, keep_default_na=False)
                pd.testing.assert_frame_equal(actual_quality, quality.details(issue_id),
                                              check_dtype=False)
                missing = actual_quality[actual_quality['字段'].eq('Flow Duration')]
                assert len(missing) == 1 and missing.iloc[0]['CSV读取值'] == ''
                assert '读取值为空' in missing.iloc[0]['适配结果']
                report['study_checks'].append('T1_exact_quality_counts_and_physical_row2_missing_cell')
                report['quality'] = {'affected_rows': quality.affected_rows,
                                     'problem_cells': quality.problem_cells,
                                     'partial_missing_rows': quality.accepted_problem_rows,
                                     'rejected_rows': quality.rejected_rows}
            rid = select_first(ranked(expected))
            if answers is not None:
                assert rid == answers['T2_record_id']
                report['study_checks'].append('T2_actual_sorted_queue_and_exact_top_identity')
            report['checks'].append('launch_cmd_load_original_model_upload_1000')
            capture('01-prediction.png')
            page.get_by_text('待核实', exact=True).click()
            hints = review_ranked(expected)
            expect(page.get_by_text(re.compile(f'符合条件 {len(hints):,} 条'))).to_be_visible()
            hint_id = select_first(hints)
            if answers is not None:
                assert hint_id == answers['T3_record_id']
                expect(page.get_by_text(re.compile('本条提示：两模型类别判断不一致；主模型得分接近既定阈值'))).to_be_visible()
                hint_row = expected.set_index('record_id').loc[hint_id]
                expect(page.get_by_text(f'来源：{input_name}，第 {hint_row.source_row} 行',
                                        exact=True)).to_be_visible()
                expect(page.get_by_text(re.compile('正常参考：训练集中 .* 条记录的 P01—P99'))).to_be_visible()
                report['study_checks'].append('T3_exact_hint_identity_flags_source_and_training_reference')
            capture('02-hint-queue.png', '研判队列')
            page.get_by_text('高异常得分', exact=True).click()
            for disagreement, near in ((True, False), (False, True), (True, True)):
                check('仅模型分歧', disagreement)
                check('仅近阈值', near)
                selected = expected[(expected.disagreement if disagreement else True)
                                    & (expected.near_threshold if near else True)]
                expect(page.get_by_text(re.compile(f'符合条件 {len(selected):,} 条'))).to_be_visible()
                if len(selected):
                    select_first(ranked(selected))
            capture('03-filters.png', '研判队列')
            report['checks'].append('frozen_sort_hint_queue_independent_filters_and_intersection')
            check('仅模型分歧', False)
            check('仅近阈值', False)
            rid = (select_record(hint_id) if answers is not None
                   else select_first(ranked(expected)))
            expect(page.get_by_text('本条证据卡', exact=True)).to_be_visible()
            capture('04-evidence.png', '记录详情与依据')
            page.get_by_text('查看原始特征与正常参考区间', exact=True).click()
            capture('04b-feature-reference.png', '查看原始特征与正常参考区间')
            report['checks'].append('correct_record_source_model_scores_reference_and_feature_evidence')
            notes = ['演示操作：已核对来源与特征，待补充背景；不代表真实安全结论。',
                     '演示操作：确认正常仅用于验证保存与恢复，不作为真实安全结论。']
            if answers is not None:
                assert answers['T4_status'] == '待补证'
                notes[0] = answers['T4_note']
            for status, note in zip(['待补证', '确认正常'], notes):
                pick('复核结论', status)
                page.get_by_role('textbox', name='复核备注', exact=True).fill(note)
                page.get_by_role('button', name='保存复核', exact=True).click()
                expect(page.get_by_text('已保存复核，修改历史已追加。', exact=True)).to_be_visible()
                for _ in range(50):
                    audit = history(database, model_version=bundle['model_version'])
                    if audit and audit[-1]['note'] == note:
                        break
                    page.wait_for_timeout(100)
                else:
                    raise AssertionError('Review did not reach SQLite')
                assert {entry['record_id'] for entry in audit} == {rid}
                if answers is not None and status == '待补证':
                    assert len(audit) == 1 and audit[0]['status'] == answers['T4_status']
                    stage = read_prediction_csv(download('导出预测与复核 CSV',
                                                         'study-t4-predictions.csv'))
                    stage_history = json.loads(download('导出完整复核历史', 'study-t4-history.json')
                                               .read_text(encoding='utf-8'))
                    assert stage.record_id.tolist() == expected.record_id.tolist()
                    assert_prediction_export(stage, expected)
                    assert_review_snapshot(stage, stage_history, bundle['model_version'])
                    saved = stage.loc[stage.record_id.eq(rid)].iloc[0]
                    assert saved.review_status == answers['T4_status']
                    assert saved.review_note == answers['T4_note']
                    assert stage_history == audit
                    report['study_checks'].append('T4_exact_T3_pending_evidence_note_and_input_scoped_exports')
            assert len(audit) == 2 and {row['record_id'] for row in audit} == {rid}
            page.get_by_text('查看该记录的复核历史', exact=True).click()
            capture('05-review.png', '人工复核')
            report['checks'].append('two_review_revisions_saved_for_stable_record')
            page.reload()
            upload()
            select_record(rid)
            expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(notes[-1])
            expect(page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
                'aria-label', re.compile(re.escape('Selected 确认正常.')))
            assert history(database, model_version=bundle['model_version']) == audit
            # Recreate the actual Edge browser process and context: no browser
            # session state may account for restored SQLite-backed reviews.
            context.close()
            browser.close()
            browser = playwright.chromium.launch(channel='msedge', headless=True, slow_mo=180)
            context = browser.new_context(viewport={'width': 1920, 'height': 1080},
                                          record_video_size={'width': 1920, 'height': 1080},
                                          record_video_dir=str(output / 'video'), accept_downloads=True)
            context.add_init_script('delete window.showSaveFilePicker;')
            page = context.new_page()
            page.set_default_timeout(45000)
            page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
            page.goto(url)
            upload()
            select_record(rid)
            expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(notes[-1])
            expect(page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
                'aria-label', re.compile(re.escape('Selected 确认正常.')))
            assert history(database, model_version=bundle['model_version']) == audit
            report['fresh_edge_process_restores_review'] = True
            report['checks'].append('browser_refresh_and_reupload_restore_review')
            stop()
            start()
            page.goto(url)
            upload()
            select_record(rid)
            expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value(notes[-1])
            expect(page.get_by_role('combobox', name=re.compile('复核结论'))).to_have_attribute(
                'aria-label', re.compile(re.escape('Selected 确认正常.')))
            assert history(database, model_version=bundle['model_version']) == audit
            capture('06-restart-restored.png', '人工复核')
            report['checks'].append('real_service_shutdown_restart_restores_complete_history')
            # The final exports must still cover the entire input while the
            # visible queue is filtered down to disagreements.
            check('仅模型分歧', True)
            expect(page.get_by_text(re.compile(
                f'符合条件 {int(expected.disagreement.sum()):,} 条；'))).to_be_visible()
            exported = read_prediction_csv(download('导出预测与复核 CSV', 'predictions.csv'))
            saved_history = json.loads(download('导出完整复核历史', 'history.json').read_text(encoding='utf-8'))
            run_info = json.loads(download('导出运行清单', 'run-manifest.json').read_text(encoding='utf-8'))
            assert len(exported) == len(expected) == run_info['valid_rows'] == 1000
            assert exported.record_id.tolist() == expected.record_id.tolist()
            assert_prediction_export(exported, expected)
            assert_review_snapshot(exported, saved_history, bundle['model_version'])
            assert not any('label' in col.lower() or 'ground_truth' in col for col in exported)
            assert np.allclose(exported.rf_score, expected.rf_score, atol=1e-12, rtol=0)
            assert np.allclose(exported.lr_score, expected.lr_score, atol=1e-12, rtol=0)
            assert saved_history == audit
            assert exported.loc[exported.record_id.eq(rid), 'review_note'].iloc[0] == notes[-1]
            assert exported.loc[exported.record_id.eq(rid), 'review_status'].iloc[0] == '确认正常'
            untouched = exported.loc[~exported.record_id.eq(rid)]
            assert untouched.review_status.eq('待复核').all()
            assert untouched.review_note.eq('').all()
            assert run_info['model_version'] == bundle['model_version']
            assert run_info['data_version'] == bundle['config']['data_version']
            assert run_info['model_sha256'] == hashlib.sha256(model_bytes).hexdigest()
            assert run_info['descriptor_sha256'] == hashlib.sha256(
                descriptor.encode('utf-8')).hexdigest()
            assert run_info['input_sha256'] == hashlib.sha256(content).hexdigest()
            assert run_info['input_name'] == input_name
            assert run_info['unjudgeable_rows'] == len(rejected)
            assert run_info['rf_threshold'] == .12 and run_info['lr_threshold'] == .84
            assert run_info['near_margin'] == bundle['config']['near_margin']
            if answers is not None:
                assert saved_history[0]['status'] == answers['T4_status']
                assert saved_history[0]['note'] == answers['T4_note']
                report['study_checks'].extend([
                    'T5_two_revisions_exact_identity_after_refresh_fresh_edge_and_real_service_restart',
                    'T6_input_scoped_exports_preserve_original_T4_and_second_demonstration_revision'])
            capture('07-export.png', '导出')
            report['checks'].append('export_1000_scores_ids_review_history_and_frozen_thresholds_match')
            assert len(report['checks']) == 7
            assert len(report['study_checks']) == (6 if answers is not None else 0)
            assert not report['page_errors']
            context.close()
            browser.close()
        report.update(passed=True, reviewed_record_id=rid, service_pids=pids,
                      predicted_rows=1000, rejected_rows=0, review_history_rows=len(audit),
                      export_rows=len(exported), browser='Microsoft Edge', viewport=[1920, 1080])
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        if 'page' in locals():
            try:
                page.screenshot(path=str(output / 'failure.png'))
                (output / 'failure-ui.txt').write_text(page.locator('body').aria_snapshot(),
                                                      encoding='utf-8')
            except Exception:
                pass
        raise
    finally:
        try:
            stop()
        finally:
            report['immutable_files_unchanged'] = all(
                digest(package / name) == value['sha256'] for name, value in manifest['files'].items())
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            report['passed'] = (report['passed'] and report['immutable_files_unchanged']
                                and len(report['cleanup']) == 2
                                and all(item['passed'] for item in report['cleanup'])
                                and report.get('fresh_edge_process_restores_review') is True)
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'passed': report['passed'], 'checks': report['checks'],
                              'failure': report.get('failure'), 'report': str(output / 'report.json')},
                             ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError('Incomplete acceptance: inspect report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-csv', type=Path,
                        help='Optional fixed 1000-row study CSV; default is package artifacts/demo.csv')
    parser.add_argument('--study-answer-key', type=Path,
                        help='Optional host-only fixed study answer key; requires --input-csv')
    parser.add_argument('--hold-seconds', type=float, default=1.8,
                        help='Hold each real workflow screen for readable demonstration footage')
    arguments = parser.parse_args()
    if not 0 <= arguments.hold_seconds <= 30:
        parser.error('--hold-seconds must be between 0 and 30')
    if arguments.study_answer_key and not arguments.input_csv:
        parser.error('--study-answer-key requires --input-csv')
    run(arguments.package, arguments.output, arguments.hold_seconds,
        arguments.input_csv, arguments.study_answer_key)
