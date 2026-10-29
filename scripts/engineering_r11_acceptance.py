"""Verify a local package first, then exercise r11 using isolated evidence only."""
from pathlib import Path
import argparse
import gc
import hashlib
import io
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request

def verified_imports(root):
    sys.path.insert(0, str(root))
    from delivery.verify_demo import verify
    manifest = verify(root)
    import app_model, app_service, app_ui_state
    for module in (app_model, app_service, app_ui_state):
        assert Path(module.__file__).resolve().is_relative_to(root)
    return manifest, app_model, app_service, app_ui_state

def benchmark(root, output):
    manifest, model, service, ui = verified_imports(root)
    import numpy as np
    import pandas as pd
    import psutil
    artifacts = root/'artifacts'
    verified = model.verified_model(artifacts/'model.joblib', artifacts/'model.json')
    bundle = model.load_model_bundle(verified.content, verified.descriptor)
    raw = pd.read_csv(artifacts/'demo.csv', dtype=str)[bundle['config']['features']]
    content = pd.concat([raw]*50, ignore_index=True).to_csv(index=False).encode()
    result = None
    timings = {'analysis': [], 'legacy_export': [], 'history': []}
    from liushao.reviews import connect
    database = output.parent/'unrelated.sqlite'
    with connect(database) as db:
        db.executemany('INSERT INTO reviews(record_id,model_version,status,note,created_at) VALUES(?,?,?,?,?)',
            ((f'unrelated:{i}', bundle['model_version'], '待复核', '', '2026-10-21T00:00:00+00:00') for i in range(100000)))
    for iteration in range(4):
        result = None; gc.collect()
        tick = time.perf_counter()
        kwargs = {}
        if manifest['package_revision'] == 'r11-final':
            from app_input import preflight_input
            kwargs['preflight'] = preflight_input(content, bundle, verified.identity)
        active, result, physical, quality, error = service.analyze_input(content, 'synthetic-50000.csv', '上传 CSV', bundle, verified.identity, artifacts, **kwargs)
        analysis = time.perf_counter()-tick
        assert error is None and len(result.predictions) == 50000
        tick = time.perf_counter(); exported = ui.csv_bytes(result.predictions); export_time = time.perf_counter()-tick
        tick = time.perf_counter(); snapshot = service.read_review_snapshot(database, bundle['model_version'], set(result.predictions.record_id)); history_time = time.perf_counter()-tick
        assert snapshot.history == []
        if iteration:
            timings['analysis'].append(analysis); timings['legacy_export'].append(export_time); timings['history'].append(history_time)
        del exported
    report = dict(software_sha=manifest['software_sha'], synthetic=True, input_sha256=hashlib.sha256(content).hexdigest(),
        rows=50000, input_bytes=len(content), measurements=timings, medians={k:statistics.median(v) for k,v in timings.items()})
    if manifest['package_revision'] == 'r11-final':
        from app_record_tools import batch_positions
        times=[]
        for i in range(4):
            tick=time.perf_counter(); positions=batch_positions(result.features, result.features.index[i]); elapsed=time.perf_counter()-tick
            if i: times.append(elapsed)
        assert statistics.median(times) <= .5
        report['percentile_seconds'] = times
        notes = ('合成工程依据；并非真人试用。'*12)[:128]
        audit=[dict(id=i+1,record_id=rid,model_version=bundle['model_version'],status='确认正常',note=notes,created_at='2026-10-21T00:00:00+00:00') for i,rid in enumerate(result.predictions.record_id)]
        predictions=result.predictions.assign(review_status='确认正常', review_note=notes)
        metadata=ui.base_meta(active,result,bundle,verified)
        tick=time.perf_counter()
        fixed=ui.export_snapshot(active,predictions,audit,metadata,True,features=result.features,physical_rows=physical,bundle=bundle)
        report['confirmed_export_seconds']=time.perf_counter()-tick
        confirmed=pd.read_csv(io.BytesIO(fixed.confirmed),keep_default_na=False)
        assert len(confirmed)==50000 and confirmed['记录ID'].is_unique
        assert confirmed['备注'].eq(notes).all() and confirmed['输入CSV行号'].tolist()==list(range(2,50002))
        assert fixed.confirmed_count==50000 and fixed.missing_input_rows==0
        audit[0]['note']='='+'补'*3999
        single=ui.export_snapshot(active,predictions.iloc[:1],audit[:1],metadata,True,
            features=result.features.iloc[:1],physical_rows=physical.iloc[:1],bundle=bundle)
        assert pd.read_csv(io.BytesIO(single.confirmed)).iloc[0]['备注']=="'"+audit[0]['note']
        report['confirmed_rows']=50000; report['max_note_formula_escaped']=True
    wide=raw.copy()
    extra=pd.DataFrame({f'extra_{i}':['0']*len(raw) for i in range(590-len(raw.columns))})
    wide=pd.concat([wide,extra],axis=1).to_csv(index=False).encode()
    a=service.analyze_input(raw.to_csv(index=False).encode(),'narrow.csv','上传 CSV',bundle,verified.identity,artifacts)[1]
    b=service.analyze_input(wide,'wide.csv','上传 CSV',bundle,verified.identity,artifacts)[1]
    pd.testing.assert_frame_equal(a.features,b.features)
    report['wide_score_max_absolute_difference'] = compare_predictions(a.predictions,b.predictions)
    report['score_atol'] = 1e-12
    report['wide_columns']=590
    report['peak_working_set']=psutil.Process().memory_info().peak_wset
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def compare_predictions(left, right):
    import numpy as np
    maximum = {}
    for field in ['rf_score','lr_score']:
        np.testing.assert_allclose(left[field],right[field],atol=1e-12,rtol=0)
        maximum[field]=float(np.max(np.abs(np.asarray(left[field])-np.asarray(right[field]))))
    for field in ['rf_anomaly','lr_anomaly','disagreement','near_threshold']:
        np.testing.assert_array_equal(left[field],right[field])
    return maximum

def prepare_migration(root, out):
    manifest, model, service, ui = verified_imports(root)
    import pandas as pd
    from liushao.model import ranked
    from liushao.reviews import save_review, history
    artifacts=root/'artifacts'
    verified=model.verified_model(artifacts/'model.joblib',artifacts/'model.json')
    bundle=model.load_model_bundle(verified.content,verified.descriptor)
    content=pd.read_csv(artifacts/'demo.csv',dtype=str)[bundle['config']['features']].iloc[:4].to_csv(index=False).encode()
    demo=service.analyze_input((artifacts/'demo.csv').read_bytes(),'demo','内置演示数据',bundle,verified.identity,artifacts)[1]
    uploaded=service.analyze_input(content,'old-upload.csv','上传 CSV',bundle,verified.identity,artifacts)[1]
    old=out/'old-r10';old.mkdir()
    source=old/'demo-state/reviews.sqlite'
    save_review(source,ranked(demo.predictions).record_id.iloc[0],bundle['model_version'],'确认正常','合成工程迁移验证：演示。')
    save_review(source,ranked(uploaded.predictions).record_id.iloc[0],bundle['model_version'],'确认正常','合成工程迁移验证：上传。')
    before=source.read_bytes()
    env={**os.environ,'PYTHONUTF8':'1'};env.pop('LIUSHAO_REVIEW_DB',None)
    completed=subprocess.run([str(root/'runtime/python/python.exe'),'-X','utf8',str(root/'delivery/migrate_reviews.py'),
        '--source-root',str(old),'--target-root',str(root)],cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=90)
    (out/'migration.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    assert completed.returncode==0,completed.stdout+completed.stderr
    target=root/'demo-state/reviews.sqlite'
    assert history(source)==history(target) and source.read_bytes()==before
    return dict(passed=True,records=2,source_unchanged=True,software_sha=manifest['software_sha'],target=str(target))

def check_range_hover(page, out, name):
    """Assert the actual SVG tooltip, not just the chart specification."""
    from playwright.sync_api import expect
    expect(page.locator('[data-testid="stApp"]')).to_have_attribute('data-test-script-state','notRunning')
    # Vega replaces SVG nodes asynchronously after Streamlit finishes. A locator
    # action retries detachment; scroll_into_view uses a one-shot element handle.
    chart = page.locator('[data-testid="stVegaLiteChart"] svg:visible').first
    expect(chart).to_be_visible()
    chart.hover(position={'x':1,'y':1})
    box = chart.bounding_box()
    tooltip = page.locator('#vg-tooltip-element')
    for fy in (.15, .2, .25, .3, .35, .4, .5, .7):
        for fx in (.075, .10, .15, .20):
            page.mouse.move(box['x']+box['width']*fx, box['y']+box['height']*fy)
            page.wait_for_timeout(80)
            text = tooltip.inner_text() if tooltip.is_visible() else ''
            if '0.02' in text and '0.22' in text:
                assert '范围下限' in text and '范围上限' in text, text
                assert not any(line.strip() in ('lo', 'hi', 'model', '看', '嗨') for line in text.splitlines()), text
                tooltip.screenshot(path=str(out/(name+'-tooltip.png')))
                return text
    raise AssertionError('Near-boundary tooltip was not found: '+name)


def browser_flow(root, out, review_database=None):
    manifest, model, service, ui = verified_imports(root)
    import pandas as pd
    import psutil
    from playwright.sync_api import sync_playwright, expect
    from delivery.desktop_start import stop_process_tree
    from app_session import session_root, saved_upload
    artifacts=root/'artifacts'
    fields=json.loads((artifacts/'model.json').read_bytes())['config']['features']
    content=pd.read_csv(artifacts/'demo.csv',dtype=str)[fields].iloc[:4].to_csv(index=False).encode()
    database=review_database or out/'browser-reviews.sqlite'
    with socket.socket() as probe: probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    env={**os.environ,'PYTHONUTF8':'1','LIUSHAO_REVIEW_DB':str(database),'LIUSHAO_ARTIFACTS_DIR':str(artifacts),
         'LIUSHAO_EVALUATION_SUMMARY':str(root/'resources/evaluation/summary.json')}
    for key in ['LIUSHAO_EXPECTED_MODEL_SHA256','LIUSHAO_EXPECTED_DESCRIPTOR_SHA256']:env.pop(key,None)
    log=(out/'server.log').open('w',encoding='utf-8');server=None
    def stop():
        if server is not None and server.poll() is None:
            stop_process_tree(psutil.Process(server.pid));server.wait(timeout=15)
    def start():
        process=subprocess.Popen([sys.executable,'-m','streamlit','run','app.py','--server.address=127.0.0.1',f'--server.port={port}',
            '--server.headless=true','--browser.gatherUsageStats=false'],cwd=root,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+60
        while True:
            try:
                with opener.open(url+'/_stcore/health',timeout=1) as response:
                    if response.read().strip()==b'ok':return process
            except OSError:
                if time.monotonic()>deadline or process.poll() is not None: raise RuntimeError('isolated server did not start')
                time.sleep(.1)
    checks={};errors=[];external=[]
    summary=root/'resources/evaluation/summary.json';original_summary=summary.read_bytes()
    try:
        server=start()
        play = sync_playwright().start()
        try:
            browser=play.chromium.launch(channel='msedge',headless=True,args=['--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run'])
            context=browser.new_context(accept_downloads=True,viewport={'width':1440,'height':1100})
            from urllib.parse import urlsplit
            def route(request):
                if urlsplit(request.request.url).hostname in ('127.0.0.1','localhost') or request.request.url.startswith(('data:','blob:')):request.continue_()
                else:external.append(request.request.url);request.abort()
            context.route('**/*',route)
            context.tracing.start(screenshots=True,snapshots=True,sources=True)
            page=context.new_page();page.set_default_timeout(30000);page.on('pageerror',lambda error:errors.append(str(error)))
            def settle():
                # Let a widget event reach Streamlit before testing the end state.
                # Result text can be mounted while the rest of that run is pending.
                page.wait_for_timeout(250)
                expect(page.locator('[data-testid="stApp"]')).to_have_attribute('data-test-script-state','notRunning',timeout=120000)
                page.wait_for_timeout(250)
            def screenshot(name):
                settle();page.screenshot(path=str(out/name),full_page=True)
            def nav(label):
                target=page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile(re.escape(label)+'$'))
                if not target.is_visible() and label in ('关于流哨','操作说明'):
                    page.locator('[data-testid="stSidebar"]').get_by_text('使用帮助',exact=True).click()
                target.click();expect(page.get_by_role('heading',name=label,exact=True)).to_be_visible();settle()
            def choose(label,option):
                page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(label,exact=True)).get_by_role('combobox').click()
                page.get_by_role('option',name=option,exact=True).click()
            page.goto(url)
            if review_database is not None:
                page.get_by_role('button',name='开始检查',exact=True).click()
                expect(page.get_by_text('人工检查：已确认 1 / 1,000 条',exact=True)).to_be_visible()
                page.get_by_role('button',name='更换文件',exact=True).click()
                expect(page.get_by_role('button',name='清除已保留文件',exact=True)).to_be_visible()
                checks['migrated_demo_identity']=True
            nav('模型与评测')
            for label in ['概览','数据与划分','模型表现','人工复核']:
                page.get_by_role('tab',name=label,exact=True).click()
            expect(page.get_by_text('固定预算：寻找模型错误',exact=True)).to_be_visible()
            screenshot('evaluation-review.png')
            page.get_by_role('tab',name='模型表现',exact=True).click()
            expect(page.get_by_text('RF 对 Slowloris 检出',exact=False)).to_be_visible()
            evaluation_captions=page.locator('[data-testid="stCaptionContainer"]').all_inner_texts()
            visible_times=[text for text in evaluation_captions if '原评测时间' in text]
            assert len(visible_times)==2 and all('（UTC）' in text and '年' in text for text in visible_times), visible_times
            assert not any(re.search(r'\d{4}-\d{2}-\d{2}T',text) for text in visible_times)
            checks['readable_evaluation_times']=True
            screenshot('evaluation-model.png')
            checks['four_tabs_historical_evaluation']=True
            page.locator('[data-testid="stSidebarHeader"]').hover()
            page.locator('[data-testid="stSidebarCollapseButton"] button').click()
            opener=page.locator('[data-testid="stExpandSidebarButton"]')
            expect(opener).to_be_visible();opener.click();settle()
            checks['desktop_sidebar_restore']=True
            summary.write_bytes(b'{broken');nav('关于流哨');nav('模型与评测')
            expect(page.get_by_text('评测资料暂不可用',exact=False)).to_be_visible()
            expect(page.locator('[data-testid="stDataFrame"]')).to_have_count(0)
            summary.write_bytes(original_summary);nav('查看结果')
            page.locator('[data-testid="stRadio"]').get_by_text('格式练习',exact=True).click()
            expect(page.get_by_role('button',name='开始检查',exact=True)).to_be_disabled()
            choose('练习内容','缺值与无效记录')
            page.get_by_role('button',name='开始检查',exact=True).click()
            expect(page.get_by_text('已检查 4 条',exact=False)).to_be_visible()
            settle()
            expect(page.get_by_text('尚未开始检查',exact=False)).to_be_visible()
            expect(page.get_by_text('文件问题摘要',exact=True)).to_be_visible()
            expect(page.get_by_text('2 条填补后可判断',exact=True)).to_be_visible()
            expect(page.get_by_text('1 条无法判断',exact=True)).to_be_visible()
            expect(page.get_by_role('radio',name='对数数量',exact=True)).to_be_checked()
            chart_bounds=page.locator('[data-testid="stVegaLiteChart"]').first.bounding_box()
            summary_bounds=page.get_by_text('文件问题摘要',exact=True).bounding_box()
            assert chart_bounds['x']+chart_bounds['width'] <= summary_bounds['x']+4
            screenshot('overview-quality-and-log-count.png')
            page.get_by_text('线性数量',exact=True).click();settle()
            expect(page.get_by_text('数量采用线性刻度',exact=False)).to_be_visible()
            page.get_by_text('对数数量',exact=True).click();settle()
            checks['overview_two_columns_zero_progress_and_scale_toggle']=True
            nav('文件问题');expect(page.get_by_text('2 条填补后可判断',exact=False)).to_be_visible()
            with page.expect_download() as download:page.get_by_role('button',name='下载无法分析的记录',exact=True).click()
            download.value.save_as(str(out/'unjudgeable.csv'))
            assert len(pd.read_csv(out/'unjudgeable.csv'))==1
            checks['sample_preflight_quality_download']=True
            nav('查看结果');page.get_by_role('button',name='更换文件',exact=True).click()
            page.locator('[data-testid="stRadio"]').get_by_text('上传文件',exact=True).click()
            page.locator('input[type=file]').set_input_files({'name':'resume-case.csv','mimeType':'text/csv','buffer':content})
            expect(page.get_by_text('格式检查通过：4 条记录',exact=False)).to_be_visible()
            expect(page.get_by_role('checkbox',name='在本机保留此文件，便于下次继续')).not_to_be_checked()
            assert not session_root(database).exists()
            page.get_by_text('在本机保留此文件，便于下次继续',exact=True).click()
            expect(page.get_by_role('checkbox',name='在本机保留此文件，便于下次继续')).to_be_checked()
            page.get_by_role('button',name='开始检查',exact=True).click()
            expect(page.get_by_text('resume-case.csv · 已检查 4 条',exact=True)).to_be_visible()
            expect(page.get_by_text('已在本机保留此文件，下次可继续检查。',exact=True)).to_be_visible()
            checks['histogram_chinese_hover'] = check_range_hover(page,out,'histogram')
            assert saved_upload(database) is not None
            if review_database is not None:
                expect(page.get_by_text('人工检查：已确认 1 / 4 条',exact=True)).to_be_visible()
                checks['migrated_upload_identity']=True
            nav('检查记录')
            checks['ruler_chinese_hover'] = check_range_hover(page,out,'ruler')
            choose('我的判断','确认正常')
            page.get_by_role('textbox',name='备注',exact=True).fill('自动化工程验证依据，不是真人记录。')
            nav('下载结果');expect(page.get_by_role('button',name='生成下载文件',exact=True)).to_be_disabled()
            page.get_by_role('button',name='返回填写位置',exact=True).click()
            page.get_by_role('button',name='保存',exact=True).click()
            expect(page.get_by_text('已保存。',exact=True)).to_be_visible()
            settle()
            page.get_by_role('tab',name='详细数据',exact=True).click()
            expect(page.get_by_text('本批百分位只表示',exact=False)).to_be_visible()
            screenshot('detail-percentile.png')
            nav('下载结果');page.get_by_role('button',name='生成下载文件',exact=True).click()
            with page.expect_download() as download:page.get_by_role('button',name='下载复核结论单',exact=True).click()
            download.value.save_as(str(out/'liushao_confirmed.csv'))
            confirmed=pd.read_csv(out/'liushao_confirmed.csv',keep_default_na=False)
            assert len(confirmed)==1 and confirmed.iloc[0]['我的判断']=='确认正常'
            screenshot('confirmed-download.png')
            checks['review_draft_save_confirmed_download']=True
            stop();server=start();page.goto(url)
            expect(page.get_by_role('button',name='继续上次检查',exact=True)).to_be_visible()
            page.get_by_role('button',name='继续上次检查',exact=True).click()
            expect(page.get_by_text('人工检查：已确认 1 / 4 条',exact=True)).to_be_visible()
            screenshot('resumed-after-server-restart.png')
            checks['actual_server_restart_saved_reviews']=True
            nav('检查记录');page.get_by_role('textbox',name='备注',exact=True).fill('未保存填写')
            nav('查看结果');page.get_by_role('button',name='更换文件',exact=True).click()
            page.get_by_role('button',name='继续上次检查',exact=True).click()
            expect(page.get_by_role('button',name='取消切换',exact=True)).to_be_visible()
            page.get_by_role('button',name='取消切换',exact=True).click();nav('检查记录')
            expect(page.get_by_role('textbox',name='备注',exact=True)).to_have_value('未保存填写')
            page.get_by_role('button',name='放弃填写并读取已保存记录',exact=True).click()
            expect(page.get_by_role('textbox',name='备注',exact=True)).to_have_value('自动化工程验证依据，不是真人记录。')
            settle()
            nav('查看结果')
            if not page.get_by_role('button',name='清除已保留文件',exact=True).is_visible():page.get_by_role('button',name='更换文件',exact=True).click()
            page.get_by_role('button',name='清除已保留文件',exact=True).click()
            expect(page.get_by_text('已清除保留的输入文件',exact=False)).to_be_visible()
            assert saved_upload(database) is None and not list(session_root(database).glob('*.input'))
            # Synthetic large-input UI with unrelated saved history, never real traffic/study records.
            import sqlite3
            with sqlite3.connect(database) as db:
                model_version=json.loads((artifacts/'model.json').read_bytes())['model_version']
                db.executemany('INSERT INTO reviews(record_id,model_version,status,note,created_at) VALUES(?,?,?,?,?)',
                    ((f'ui-unrelated:{i}',model_version,'待复核','','2026-10-21T00:00:00+00:00') for i in range(100000)))
            raw=pd.read_csv(artifacts/'demo.csv',dtype=str)[fields]
            large=pd.concat([raw]*50,ignore_index=True).to_csv(index=False).encode()
            page.locator('[data-testid="stRadio"]').get_by_text('上传文件',exact=True).click()
            tick=time.perf_counter()
            page.locator('input[type=file]').set_input_files({'name':'synthetic-50000.csv','mimeType':'text/csv','buffer':large})
            expect(page.get_by_text('格式检查通过：50,000 条记录',exact=False)).to_be_visible(timeout=120000)
            settle()
            page.get_by_role('button',name='开始检查',exact=True).click()
            expect(page.get_by_text('synthetic-50000.csv · 已检查 50,000 条',exact=True)).to_be_visible(timeout=120000)
            expect(page.get_by_role('button',name='放弃填写并继续',exact=True)).not_to_be_visible()
            analysis_seconds=time.perf_counter()-tick
            screenshot('synthetic-50000-overview.png')
            expect(page.get_by_text('尚未开始检查',exact=False)).to_be_visible()
            page.locator('[data-testid="stSidebarHeader"]').hover()
            page.locator('[data-testid="stSidebarCollapseButton"] button').click()
            page.set_viewport_size({'width':390,'height':844});settle()
            screenshot('mobile-overview-distribution-and-quality.png')
            narrow_chart=page.locator('[data-testid="stVegaLiteChart"]').first.bounding_box()
            narrow_summary=page.get_by_text('文件问题摘要',exact=True).bounding_box()
            assert narrow_chart['x']>=0 and narrow_chart['x']+narrow_chart['width']<=391
            assert narrow_summary['y']>=narrow_chart['y']+narrow_chart['height']
            checks['overview_mobile_stacks_without_horizontal_overflow']=True
            page.set_viewport_size({'width':1440,'height':1100});settle()
            sidebar_opener=page.locator('[data-testid="stExpandSidebarButton"]')
            if sidebar_opener.is_visible():sidebar_opener.click();settle()
            nav('检查记录')
            expect(page.get_by_text('第 1 / 1000 页',exact=True)).to_be_visible()
            tick=time.perf_counter();page.get_by_role('button',name='下一页',exact=True).click()
            expect(page.get_by_text('第 2 / 1000 页',exact=True)).to_be_visible();settle()
            page_seconds=time.perf_counter()-tick
            page.get_by_role('textbox',name='查找行号',exact=True).fill('2')
            page.get_by_role('textbox',name='查找行号',exact=True).press('Enter')
            expect(page.get_by_text('1 条符合条件',exact=False)).to_be_visible();settle()
            page.get_by_role('button',name=re.compile('清除.*筛选')).click();settle()
            nav('下载结果');page.get_by_role('button',name='生成下载文件',exact=True).click()
            with page.expect_download() as download:page.get_by_role('button',name='下载检查结果',exact=True).click()
            download.value.save_as(str(out/'synthetic-50000-predictions.csv'))
            exported=pd.read_csv(out/'synthetic-50000-predictions.csv',keep_default_na=False)
            assert len(exported)==50000 and exported.review_status.eq('待复核').all()
            checks['synthetic_50000_ui_with_100000_unrelated_history']={
                'passed':True,'synthetic':True,'rows':50000,'history_rows':100000,
                'upload_preflight_analysis_seconds':analysis_seconds,'page_seconds':page_seconds,'exported_rows':len(exported)}
            nav('查看结果');page.get_by_role('button',name='更换文件',exact=True).click();settle()
            page.set_viewport_size({'width':390,'height':844});settle()
            collapse=page.locator('[data-testid="stSidebarCollapseButton"] button')
            opener=page.locator('[data-testid="stExpandSidebarButton"]')
            if opener.is_visible():opener.click();page.wait_for_timeout(400)
            page.locator('[data-testid="stSidebarHeader"]').hover()
            expect(collapse).to_be_in_viewport();collapse.click()
            page.wait_for_timeout(400)
            screenshot('mobile-import.png')
            opener=page.locator('[data-testid="stExpandSidebarButton"]')
            expect(opener).to_be_visible();opener.click();settle()
            nav('检查记录')
            page.locator('[data-testid="stSidebarHeader"]').hover();collapse.click();page.wait_for_timeout(400)
            page.get_by_role('tab',name='判断与处理',exact=True).click()
            page.get_by_role('button',name='保存',exact=True).scroll_into_view_if_needed()
            screenshot('mobile-review-save.png')
            bounds=page.get_by_role('button',name='保存',exact=True).bounding_box()
            assert bounds['x']>=0 and bounds['x']+bounds['width']<=391
            checks['mobile_sidebar_restore_and_save_accessible']=True
            checks['pending_restore_cancel_and_explicit_clear']=True
            assert not errors and not external, (errors,external)
            context.tracing.stop(path=str(out/'trace.zip'));context.close();browser.close()
        except BaseException:
            # Keep the failed page and trace before Playwright closes the browser.
            try:
                page.screenshot(path=str(out/'failure.png'),full_page=True)
                (out/'failure.txt').write_text(page.locator('[data-testid="stApp"]').inner_text(),encoding='utf-8')
                context.tracing.stop(path=str(out/'failure-trace.zip'))
            except Exception as capture_error:
                (out/'capture-error.txt').write_text(str(capture_error),encoding='utf-8')
            raise
        finally:
            play.stop()
    finally:
        summary.write_bytes(original_summary);stop();log.close()
    return dict(software_sha=manifest['software_sha'],passed=True,checks=checks,page_errors=errors,external_requests=external)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--browser',action='store_true')
    parser.add_argument('--prepare-migration',action='store_true')
    parser.add_argument('--review-db',type=Path)
    args=parser.parse_args();root=args.package.resolve();out=args.output.resolve()
    if not any((args.worker,args.prepare_migration,args.baseline,args.browser)):
        parser.error('Choose --prepare-migration, --baseline, --browser or --worker.')
    if args.worker:
        benchmark(root,out);return
    out.mkdir(parents=True,exist_ok=False)
    manifest,*_=verified_imports(root)
    report={'software_sha':manifest['software_sha'],'package_manifest_sha256':hashlib.sha256((root/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),'checks':{},'passed':False}
    def checkpoint():
        (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    checkpoint()
    try:
        if args.prepare_migration:
            report['checks']['migration']=prepare_migration(root,out)
            checkpoint()
        if args.baseline:
            for label,package in [('r10',args.baseline.resolve()),('r11',root)]:
                directory=out/label;directory.mkdir()
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--package',str(package),'--output',str(directory/'benchmark.json')]
                with (directory/'worker.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=240,env={**os.environ,'PYTHONUTF8':'1'})
            before=json.loads((out/'r10/benchmark.json').read_bytes());after=json.loads((out/'r11/benchmark.json').read_bytes())
            assert before['input_sha256']==after['input_sha256']
            for key,value in before['medians'].items():
                assert after['medians'][key] <= max(value*1.25,value+.5), (key,before['medians'],after['medians'])
            assert after['peak_working_set'] <= before['peak_working_set']+256*1024*1024, (before['peak_working_set'],after['peak_working_set'])
            report['checks']['capacity_comparison']={'passed':True,'baseline':before,'candidate':after}
            checkpoint()
        if args.browser:
            browser_out=out/'browser';browser_out.mkdir()
            report['checks']['browser']=browser_flow(root,browser_out,args.review_db)
        report['passed']=bool(report['checks']) and all(value.get('passed') is True for value in report['checks'].values())
        if not report['passed']:
            raise AssertionError('A required acceptance stage did not pass.')
    except BaseException:
        report['error']=traceback.format_exc()
        raise
    finally:
        checkpoint()
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
