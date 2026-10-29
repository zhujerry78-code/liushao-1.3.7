"""Independent Edge checks against an already-started isolated r13 app server.

Never operates on a user's normal browser profile. The uploaded CSV is explicitly
an engineering fixture; timings are not human usability measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import pandas as pd
from playwright.sync_api import expect, sync_playwright


def source_hashes(root):
    paths = sorted(set(root.glob('app*.py')) | set((root / 'liushao').glob('*.py')))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def run(url, root, artifacts, output):
    if urlsplit(url).hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Acceptance requires a loopback isolated server')
    output.mkdir(parents=True, exist_ok=False)
    before = source_hashes(root)
    report = {'scope': 'local automated Edge regression; not human usability research',
              'source_files': before, 'url': url, 'checks': {}, 'page_errors': [],
              'external_requests_blocked': [], 'passed': False}
    checks = report['checks']
    with sync_playwright() as play:
        browser = play.chromium.launch(channel='msedge', headless=True,
            args=['--disable-background-networking', '--disable-component-update',
                  '--disable-sync', '--no-first-run'])
        context = browser.new_context(accept_downloads=True,
                                      viewport={'width': 1440, 'height': 1100})
        def route(request):
            address = request.request.url
            if urlsplit(address).hostname in ('127.0.0.1', 'localhost') or address.startswith(('data:', 'blob:')):
                request.continue_()
            else:
                report['external_requests_blocked'].append(address)
                request.abort()
        context.route('**/*', route)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.set_default_timeout(30000)
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        def settle():
            page.wait_for_timeout(250)
            expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
                'data-test-script-state', 'notRunning', timeout=120000)
            expect(page.locator('[data-testid="stException"]')).to_have_count(0)
            page.wait_for_timeout(150)
        def shot(name):
            settle()
            page.screenshot(path=str(output / name), full_page=True)
        def nav(label):
            sidebar = page.locator('[data-testid="stSidebar"]')
            target = sidebar.get_by_role('button', name=re.compile(re.escape(label) + '$'))
            if not target.is_visible() and label in ('关于流哨', '操作说明'):
                sidebar.get_by_text('使用帮助', exact=True).click()
            target.click()
            expect(page.get_by_role('heading', name=label, exact=True)).to_be_visible()
            settle()
        try:
            page.goto(url)
            settle()
            sidebar = page.locator('[data-testid="stSidebar"]')
            expect(sidebar.get_by_role('button', name=re.compile('模型与评测'))).to_have_count(0)
            nav('操作说明')
            evaluation = page.locator('[data-testid="stExpander"]').filter(
                has=page.get_by_text('模型与评测', exact=True))
            expect(evaluation).to_have_count(1)
            evaluation.locator('summary').click()
            expect(evaluation.get_by_text('同一冻结模型的历史评测 · 不代表当前上传文件的准确率', exact=True)).to_be_visible()
            checks['evaluation_is_in_help_only'] = True
            shot('01-help-evaluation.png')
            nav('查看结果')
            page.get_by_role('button', name='开始检查', exact=True).click()
            settle()
            nav('检查记录')
            search = page.locator('[data-testid="stTextInput"]').filter(
                has=page.get_by_text(re.compile('全量查找行号'))).locator('input')
            search.fill('217809')
            search.press('Enter')
            settle()
            expect(page.get_by_text('解释不足，待补证', exact=True)).to_be_visible()
            for label in ('模型说：为什么优先看这条', '数据说：6 项关键线索', '环境说：邻近流与核查动作'):
                expect(page.get_by_text(label, exact=True)).to_be_visible()
            checks['diagnostic_record_217809_has_honest_fallback'] = True
            shot('02-demo-217809-evidence.png')
            # Remaining canvas search and download checks are intentionally added
            # after observing the actual final UI selectors; no success is inferred.
            report['passed'] = bool(checks) and not report['page_errors']
        except Exception as error:
            report['error'] = f'{type(error).__name__}: {error}'
            try:
                page.screenshot(path=str(output / 'failure.png'), full_page=True)
            except Exception:
                pass
            raise
        finally:
            report['source_unchanged_during_check'] = source_hashes(root) == before
            report['passed'] = report['passed'] and report['source_unchanged_during_check']
            context.tracing.stop(path=str(output / 'trace.zip'))
            context.close()
            browser.close()
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.url, args.workspace.resolve(), args.artifacts.resolve(), args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
