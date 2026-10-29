"""Version-bound local r9 acceptance. All evidence and databases stay outside Git."""
from __future__ import annotations
import argparse
import ast
from contextlib import nullcontext
import hashlib
import io
import json
import os
import re
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--original-zip', required=True, type=Path)
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--capacity', action='store_true')
    args = parser.parse_args()
    root = args.package.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    import numpy as np
    import pandas as pd
    import psutil
    import app_model
    import app_service
    import app_types
    import liushao
    from liushao.data import read_csv_checked, sha256
    from liushao.model import infer
    from app_quality import build_quality_view
    for module in (app_model, app_service, app_types, liushao):
        assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
    report = {'package': str(root), 'module_paths': {m.__name__: m.__file__ for m in
              (app_model, app_service, app_types, liushao)}, 'checks': {}, 'failures': []}
    manifest = json.loads((root/'PACKAGE_MANIFEST.json').read_bytes())
    report['software_sha'] = manifest['software_sha']
    report['package_manifest_sha256'] = sha256(root/'PACKAGE_MANIFEST.json')
    artifacts = root/'artifacts'
    verified = app_model.verified_model(artifacts/'model.joblib', artifacts/'model.json')
    bundle = app_model.load_model_bundle(verified.content, verified.descriptor)
    payload = ((artifacts/'demo.csv').read_bytes(), '演示样本（测试集固定抽样）', '内置演示数据')
    active, result, physical, quality, error = app_service.analyze_input(*payload, bundle, verified.identity, artifacts)
    assert error is None and len(result.predictions) == 1000
    # Run the original supplied r8 analysis function, using its exact source and frozen bytes.
    with zipfile.ZipFile(args.original_zip) as z:
        source = z.read(next(n for n in z.namelist() if n.endswith('/app.py') and n.count('/') == 1))
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_input')
    class Spinner:
        @staticmethod
        def spinner(*args, **kwargs): return nullcontext()
    old = dict(bundle=bundle, model_identity=verified.identity, ARTIFACTS=artifacts,
               default_filters=__import__('app_ui_state').default_filters, st=Spinner(), hashlib=hashlib, time=time, io=io, json=json, pd=pd,
               read_csv_checked=read_csv_checked, infer=infer, sha256=sha256,
               build_quality_view=build_quality_view, u={'prediction_calls':0})
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'original-r8-input', 'exec'), old)
    old['run_input'](payload)
    assert not old['u'].get('run_error'), old['u'].get('run_error')
    previous = old['u']['result']
    scores = ['rf_score', 'lr_score']
    maximum = {}
    for column in scores:
        delta = np.abs(previous[0][column].to_numpy()-result.predictions[column].to_numpy())
        maximum[column] = float(delta.max())
        np.testing.assert_allclose(previous[0][column], result.predictions[column], atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(previous[0].drop(columns=scores), result.predictions.drop(columns=scores), check_exact=True)
    from liushao.model import ranked, review_ranked
    for ranker in (ranked, review_ranked):
        assert ranker(previous[0]).record_id.tolist() == ranker(result.predictions).record_id.tolist()
    pd.testing.assert_frame_equal(previous[1], result.features, check_exact=True)
    pd.testing.assert_frame_equal(previous[2], result.rejected, check_exact=True)
    report['checks']['frozen_r8_comparison'] = {'rows':1000, 'passed':True, 'max_score_absolute_difference':maximum, 'score_atol':1e-12, 'identities_flags_thresholds_order_exact':True}
    if args.capacity:
        raw = read_csv_checked(io.BytesIO(payload[0]))
        report['capacity'] = []
        process = psutil.Process()
        for rows in (1001,10000,50000):
            frame = pd.concat([raw] * ((rows+len(raw)-1)//len(raw)), ignore_index=True).iloc[:rows]
            content = frame.to_csv(index=False).encode('utf-8')
            start = time.perf_counter()
            _, measured, *_ = app_service.analyze_input(content, f'synthetic-{rows}.csv', '上传 CSV', bundle, verified.identity, artifacts)
            analysis_seconds = time.perf_counter()-start
            from app_ui_state import csv_bytes
            tick = time.perf_counter(); exported = csv_bytes(measured.predictions)
            report['capacity'].append(dict(rows=rows, input_bytes=len(content), analysis_seconds=analysis_seconds,
                export_seconds=time.perf_counter()-tick, export_bytes=len(exported),
                rss_bytes=process.memory_info().rss, process_peak_working_set=process.memory_info().peak_wset,
                synthetic=True))
        # Large unrelated history: real query, isolated SQLite, no human-record claims.
        from liushao.reviews import connect, history
        database = out/'synthetic-history.sqlite'
        db = connect(database)
        db.executemany('INSERT INTO reviews(record_id,model_version,status,note,created_at) VALUES(?,?,?,?,?)',
             ((f'unrelated:{i}', bundle['model_version'], '待复核', '', '2026-10-21T00:00:00+00:00') for i in range(100000)))
        db.commit(); db.close()
        tick = time.perf_counter(); entries = history(database, model_version=bundle['model_version'])
        report['history_capacity'] = dict(rows=len(entries), seconds=time.perf_counter()-tick, synthetic=True)
        tick=time.perf_counter()
        snapshot=app_service.read_review_snapshot(database,bundle['model_version'],set(result.predictions.record_id))
        report['history_capacity'].update(current_scope_seconds=time.perf_counter()-tick,
            retained_current_rows=len(snapshot.history))
        assert snapshot.history==[]
        del entries
    (out/'model-capacity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.browser:
        browser_flow(root, out, report, result)
    (out/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def browser_flow(root, out, report, expected):
    from playwright.sync_api import sync_playwright, expect
    import pandas as pd
    import psutil
    from liushao.model import ranked
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    env = {**os.environ, 'PYTHONUTF8':'1', 'LIUSHAO_ARTIFACTS_DIR':str(root/'artifacts'),
           'LIUSHAO_REVIEW_DB':str(out/'browser-reviews.sqlite')}
    for key in ('LIUSHAO_EXPECTED_MODEL_SHA256', 'LIUSHAO_EXPECTED_DESCRIPTOR_SHA256'):
        env.pop(key, None)
    log = (out/'streamlit.log').open('w',encoding='utf-8')
    server = subprocess.Popen([sys.executable,'-m','streamlit','run','app.py',f'--server.port={port}',
             '--server.address=127.0.0.1','--server.headless=true','--browser.gatherUsageStats=false'],
             cwd=root,env=env,stdout=log,stderr=log)
    url = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic()+60
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            try:
                with opener.open(url+'/_stcore/health',timeout=1) as response:
                    if response.status == 200: break
            except OSError:
                if time.monotonic() >= deadline or server.poll() is not None: raise
                time.sleep(.2)
        with sync_playwright() as play:
            browser = play.chromium.launch(channel='msedge',headless=True)
            context = browser.new_context(accept_downloads=True,viewport={'width':1440,'height':1100})
            context.tracing.start(screenshots=True,snapshots=True,sources=True)
            page = context.new_page(); page.set_default_timeout(20000)
            try:
                page.goto(url)
                page.get_by_role('button',name='开始检查',exact=True).click()
                expect(page.get_by_text('检查完成：',exact=False)).to_be_visible()
                page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile('检查记录$')).click()
                # Real table click, no session-state injection.
                canvas = page.locator('[data-testid="stDataFrame"]').first.locator('canvas').first
                canvas.scroll_into_view_if_needed()
                bounds = canvas.bounding_box()
                page.mouse.click(bounds['x']+65,bounds['y']+79)
                chosen = ranked(expected.predictions).record_id.iloc[1]
                page.get_by_role('tab',name='详细数据',exact=True).click()
                page.get_by_text('技术信息与完整编号',exact=True).click()
                expect(page.get_by_text(chosen,exact=True)).to_be_visible()
                page.get_by_role('tab',name='判断与处理',exact=True).click()
                page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('我的判断',exact=True)).get_by_role('combobox').click()
                page.get_by_role('option',name='确认正常',exact=True).click()
                page.get_by_role('textbox',name='备注',exact=True).fill('本机工程验证：核对来源，不是真人试用记录。')
                page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile('下载结果$')).click()
                expect(page.get_by_role('button',name='生成下载文件',exact=True)).to_be_disabled()
                page.get_by_role('button',name='返回填写位置',exact=True).click()
                expect(page.get_by_role('textbox',name='备注',exact=True)).to_have_value('本机工程验证：核对来源，不是真人试用记录。')
                page.get_by_role('button',name='保存',exact=True).click()
                expect(page.get_by_text('已保存。',exact=True)).to_be_visible()
                with sqlite3.connect(out/'browser-reviews.sqlite') as db:
                    saved = db.execute('SELECT record_id,status,note FROM reviews').fetchall()
                assert len(saved) == 1 and saved[0][:2] == (chosen,'确认正常'), saved
                page.get_by_role('button',name='下一页',exact=True).click()
                expect(page.get_by_text('第 2 / 20 页',exact=True)).to_be_visible()
                page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile('下载结果$')).click()
                expect(page.get_by_text('检查结果 CSV：',exact=False)).to_be_visible()
                expect(page.get_by_text('操作记录 JSON：',exact=False)).to_be_visible()
                expect(page.get_by_text('运行信息 JSON：',exact=False)).to_be_visible()
                page.screenshot(path=str(out/'download-descriptions.png'),full_page=True)
                page.get_by_role('button',name='生成下载文件',exact=True).click()
                downloaded = {}
                with page.expect_download() as event:
                    page.get_by_role('button',name='下载检查结果',exact=True).click()
                event.value.save_as(str(out/'liushao_predictions.csv'))
                page.get_by_text('其他文件',exact=True).click()
                for label, name in [('下载已保存的操作记录','liushao_reviews.json'),('下载本次运行信息','liushao_run.json')]:
                    with page.expect_download() as event:
                        page.get_by_role('button',name=label,exact=True).click()
                    assert event.value.suggested_filename == name
                    event.value.save_as(str(out/name))
                exported = pd.read_csv(out/'liushao_predictions.csv',keep_default_na=False)
                assert len(exported) == 1000
                row = exported.set_index('record_id').loc[chosen]
                assert row.review_status == '确认正常'
                assert len(json.loads((out/'liushao_reviews.json').read_bytes())) == 1
                page.screenshot(path=str(out/'downloads-prepared.png'),full_page=True)
                # Fresh independent browser session reads only saved values.
                second = context.new_page(); second.goto(url)
                second.get_by_role('button',name='开始检查',exact=True).click()
                expect(second.get_by_text('人工检查：已确认 1 / 1,000 条',exact=True)).to_be_visible()
                second.close()
                page.reload()
                page.get_by_role('button',name='开始检查',exact=True).click()
                expect(page.get_by_text('人工检查：已确认 1 / 1,000 条',exact=True)).to_be_visible()
                report['checks']['browser_table_pagination_save_download_refresh'] = {'passed':True,'saved_record_id':chosen,'rows':1000}
                page.get_by_role('button',name='更换文件',exact=True).click()
                page.locator('[data-testid="stRadio"]').get_by_text('上传文件',exact=True).click()
                page.locator('input[type=file]').set_input_files({'name':'invalid.csv','mimeType':'text/csv','buffer':b'wrong\n1\n'})
                expect(page.get_by_text('待检查：invalid.csv',exact=True)).to_be_visible()
                page.get_by_role('button',name='开始检查',exact=True).click()
                expect(page.get_by_text('下方仍是上一次成功检查的文件结果。',exact=True)).to_be_visible()
                page.get_by_role('button',name='Remove invalid.csv',exact=True).click()
                expect(page.get_by_role('button',name='开始检查',exact=True)).to_be_disabled()
                report['checks']['browser_failed_upload_and_remove']={'passed':True}
                if report.get('capacity'):
                    # Synthetic unrelated history also exercises the live UI read path.
                    with sqlite3.connect(out/'browser-reviews.sqlite') as db:
                        db.executemany('INSERT INTO reviews(record_id,model_version,status,note,created_at) VALUES(?,?,?,?,?)',
                            ((f'unrelated-browser:{i}', expected.predictions.model_version.iloc[0], '待复核', '',
                              '2026-10-21T00:00:00+00:00') for i in range(100000)))
                        history_rows=db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
                    assert history_rows == 100001
                    raw = pd.read_csv(root/'artifacts/demo.csv')
                    content = pd.concat([raw]*50,ignore_index=True).to_csv(index=False).encode('utf-8')
                    path = out/'synthetic-50000.csv';path.write_bytes(content)
                    page.locator('input[type=file]').set_input_files(str(path))
                    expect(page.get_by_text('待检查：synthetic-50000.csv',exact=True)).to_be_visible()
                    tick = time.perf_counter()
                    page.get_by_role('button',name='开始检查',exact=True).click()
                    expect(page.get_by_text('已检查 50,000 条',exact=False)).to_be_visible(timeout=120000)
                    analysis_seconds = time.perf_counter()-tick
                    page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile('检查记录$')).click()
                    expect(page.get_by_text('第 1 / 1000 页',exact=True)).to_be_visible()
                    tick = time.perf_counter()
                    page.get_by_role('button',name='下一页',exact=True).click()
                    expect(page.get_by_text('第 2 / 1000 页',exact=True)).to_be_visible()
                    page_seconds = time.perf_counter()-tick
                    tick = time.perf_counter()
                    page.get_by_role('textbox',name='查找行号',exact=True).fill('30001')
                    page.get_by_role('textbox',name='查找行号',exact=True).press('Enter')
                    expect(page.get_by_text('1 条符合条件 · 点击一行打开',exact=True)).to_be_visible()
                    filter_seconds = time.perf_counter()-tick
                    page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile('下载结果$')).click()
                    page.get_by_role('button',name='生成下载文件',exact=True).click()
                    expect(page.get_by_role('button',name='下载检查结果',exact=True)).to_be_visible(timeout=120000)
                    with page.expect_download() as event:
                        page.get_by_role('button',name='下载检查结果',exact=True).click()
                    event.value.save_as(str(out/'capacity-predictions.csv'))
                    exported=pd.read_csv(out/'capacity-predictions.csv',keep_default_na=False)
                    assert len(exported)==50000 and exported.review_status.eq('待复核').all()
                    family=[psutil.Process(server.pid),*psutil.Process(server.pid).children(recursive=True)]
                    memory=[p.memory_info() for p in family]
                    report['checks']['browser_50000_rows']={'passed':True,'synthetic':True,
                        'analysis_seconds':analysis_seconds,'page_seconds':page_seconds,'filter_seconds':filter_seconds,
                        'exported_rows':50000,'history_rows':history_rows,'server_family_rss_bytes':sum(m.rss for m in memory),
                        'server_peak_working_set_bytes':max(m.peak_wset for m in memory)}
                    page.screenshot(path=str(out/'capacity-download.png'),full_page=True)

            except BaseException:
                page.screenshot(path=str(out/'browser-failure.png'),full_page=True)
                (out/'browser-failure.html').write_text(page.content(),encoding='utf-8')
                raise
            finally:
                context.tracing.stop(path=str(out/'browser-trace.zip'))
                browser.close()
    finally:
        try:
            process = psutil.Process(server.pid)
            targets = process.children(recursive=True)+[process]
            for p in reversed(targets):
                try: p.terminate()
                except psutil.NoSuchProcess: pass
            _, alive = psutil.wait_procs(targets,timeout=8)
            for p in alive:
                try: p.kill()
                except psutil.NoSuchProcess: pass
        except psutil.NoSuchProcess: pass
        log.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        if '--output' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--output')+1])
            if directory.is_dir():
                (directory/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
