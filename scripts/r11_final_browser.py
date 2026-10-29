"""Local Edge workflow regression, entirely isolated from normal user profiles."""
from pathlib import Path
import argparse,json,re,sys,time,hashlib
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright,expect


def run(url,out):
    if urlsplit(url).hostname not in ('127.0.0.1','localhost'):raise ValueError('loopback required')
    out.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'checks':{},'page_errors':[],'viewports':[],'scope':'automated local engineering fixture, not human usability evidence'}
    root=Path(__file__).resolve().parents[1]
    source_paths=sorted(list(root.glob('app*.py'))+list((root/'liushao').glob('*.py')))
    report['source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    checks=report['checks']
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run'])
        context=browser.new_context(viewport={'width':1440,'height':900},accept_downloads=True)
        context.route('**/*',lambda route:route.continue_() if urlsplit(route.request.url).hostname in ('127.0.0.1','localhost') or route.request.url.startswith(('data:','blob:')) else route.abort())
        context.tracing.start(screenshots=True,snapshots=True,sources=True)
        page=context.new_page();page.set_default_timeout(40000)
        page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
        def settle():
            page.wait_for_timeout(200)
            expect(page.locator('[data-testid="stApp"]')).to_have_attribute('data-test-script-state','notRunning',timeout=90000)
            expect(page.locator('[data-testid="stException"]')).to_have_count(0)
            page.wait_for_timeout(150)
        def nav(label):
            sidebar=page.locator('[data-testid="stSidebar"]')
            control=sidebar.get_by_role('button',name=re.compile(re.escape(label)+'$'))
            if not control.is_visible():
                if label in ('操作说明','关于流哨'):sidebar.get_by_text('使用帮助',exact=True).click()
                else:page.get_by_role('button',name='Open sidebar',exact=True).click()
            control.click();expect(page.get_by_role('heading',name=label,exact=True)).to_be_visible();settle()
        def search(value):
            control=page.get_by_role('textbox',name='检索记录、系统提示或我的判断',exact=True)
            if not control.count():page.get_by_role('button',name='快速检索',exact=True).click();settle()
            control.fill(value);control.press('Enter');settle()
        def select_first():
            grid=page.locator('.st-key-_records_main');grid.locator('.dvn-scroller').first.click(position={'x':105,'y':52});settle()
        def screenshot(name):
            settle();page.screenshot(path=str(out/name),full_page=True)
        try:
            page.goto(url);settle()
            page.get_by_role('button',name='开始检查',exact=True).click()
            expect(page.get_by_role('button',name='更换文件',exact=True)).to_be_visible();settle()
            nav('检查记录')
            search('第 ２１７８０９ 行')
            expect(page.get_by_text('当前筛选范围内匹配 1 条，已置顶；其他记录保留。',exact=True)).to_be_visible()
            expect(page.get_by_text('1,000 条符合条件 · 点击一行打开',exact=True)).to_be_visible()
            select_first()
            expect(page.get_by_text('输入 CSV 第 113 行',exact=False)).to_be_visible()
            expect(page.get_by_text('主要证据暂不可用',exact=False)).to_have_count(0)
            expect(page.get_by_text('距参考下界约 4.34%',exact=False).first).to_be_visible()
            checks['unicode_exact_cross_page_promotion_keeps_all_rows']=True
            checks['case_217809_reference_and_accounting']=True
            screenshot('01-record-217809.png')
            note='合成工程测试：仅核对界面保存与证据关联，不是真人调查记录。'+out.name
            page.get_by_role('textbox',name='备注',exact=True).fill(note);page.get_by_role('textbox',name='备注',exact=True).press('Tab');settle()
            score=page.locator('[data-testid="stExpander"]').filter(has=page.get_by_text('评分如何形成',exact=True))
            score.locator('summary').first.click();settle()
            expect(page.get_by_text('RF：基准分 + 全部字段加减 = 当前分数。',exact=False)).to_be_visible()
            expect(page.get_by_role('textbox',name='备注',exact=True)).to_have_value(note)
            screenshot('02-score-accounting.png');checks['expand_accounting_preserves_draft']=True
            search('does-not-match-[.*]')
            expect(page.get_by_text('未找到匹配，原表保留。',exact=True)).to_be_visible()
            expect(page.get_by_text('1,000 条符合条件 · 点击一行打开',exact=True)).to_be_visible()
            expect(page.get_by_role('textbox',name='备注',exact=True)).to_have_value(note)
            search('');checks['no_match_and_clear_keep_all_rows_and_draft']=True
            nav('下载结果');expect(page.get_by_role('button',name='生成下载文件',exact=True)).to_be_disabled()
            page.get_by_role('button',name='返回填写位置',exact=True).click();settle()
            control=page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('我的判断',exact=True)).get_by_role('combobox');control.click()
            page.get_by_role('option',name='需要补充信息（待补证）',exact=True).click();settle()
            page.get_by_role('button',name='保存',exact=True).click();settle()
            expect(page.get_by_text('已保存。',exact=True)).to_be_visible()
            checks['dirty_export_guard_and_atomic_save']=True
            search('609344');select_first()
            expect(page.get_by_text('输入 CSV 第 473 行',exact=False)).to_be_visible()
            expect(page.get_by_text('需先核对缺失字段：Flow Byts/s、Flow Pkts/s',exact=True)).to_be_visible()
            screenshot('03-missing-609344.png');checks['case_473_609344_full_missing_summary']=True
            for width,height in ((1440,900),(1280,800),(390,844)):
                page.set_viewport_size({'width':width,'height':height});settle()
                if width==390:
                    sidebar=page.locator('[data-testid="stSidebar"]')
                    if sidebar.is_visible():
                        close=page.get_by_role('button',name='Close sidebar',exact=True)
                        if close.count():close.click();settle()
                overflow=page.evaluate('({w:innerWidth,body:document.body.scrollWidth,doc:document.documentElement.scrollWidth})')
                assert max(overflow['body'],overflow['doc'])<=width+1,overflow
                save=page.get_by_role('button',name='保存',exact=True);save.scroll_into_view_if_needed()
                box=save.bounding_box();assert box and box['width']>30 and box['x']>=0 and box['x']+box['width']<=width+1
                screenshot(f'viewport-{width}x{height}.png')
                report['viewports'].append(dict(width=width,height=height,overflow=overflow))
            checks['desktop_and_narrow_no_page_overflow_save_reachable']=True
            page.set_viewport_size({'width':1440,'height':900});settle()
            missing_note='合成工程测试：609344 缺失记录的保存与下一条，非真人记录。'
            control=page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('我的判断',exact=True)).get_by_role('combobox')
            control.click();page.get_by_role('option',name='需要补充信息（待补证）',exact=True).click();settle()
            page.get_by_role('textbox',name='备注',exact=True).fill(missing_note)
            page.get_by_role('button',name='保存并下一条',exact=True).click();settle()
            expect(page.get_by_text('输入 CSV 第 113 行',exact=False)).to_be_visible()
            checks['save_next_follows_promoted_display_order']=True
            nav('下载结果');page.get_by_role('button',name='生成下载文件',exact=True).click();settle()
            labels={'predictions':'下载检查结果','history':'下载已保存的操作记录','manifest':'下载本次运行信息','confirmed':'下载复核结论单'}
            for key,label in labels.items():
                with page.expect_download() as event:page.get_by_role('button',name=label,exact=True).click()
                event.value.save_as(str(out/(key+('.csv' if key in ('predictions','confirmed') else '.json'))))
            import pandas as pd
            predictions=pd.read_csv(out/'predictions.csv');history=json.loads((out/'history.json').read_text(encoding='utf-8'))
            assert len(predictions)==1000 and history
            assert predictions.loc[predictions.source_row.eq(217809),'input_csv_row'].tolist()==[113]
            assert predictions.loc[predictions.source_row.eq(609344),'input_csv_row'].tolist()==[473]
            checks['predictions_csv_physical_and_source_rows_are_distinct']=True
            saved=[r for r in history if r['note']==note];assert saved
            assert all(r['evidence_snapshot']['identity']['source_row']==217809 and r['evidence_snapshot']['input_csv_row']==113 for r in saved)
            assert all(r['evidence_snapshot_status']=='saved' for r in saved)
            missing_saved=[r for r in history if r['note']==missing_note];assert missing_saved
            assert all(r['evidence_snapshot']['input_csv_row']==473 and r['evidence_snapshot']['identity']['source_row']==609344 for r in missing_saved)
            assert all(r['evidence_snapshot']['selection']['missing_fields']==['Flow Byts/s','Flow Pkts/s'] for r in missing_saved)
            checks['missing_snapshot_and_save_next_identity']=True
            assert pd.read_csv(out/'confirmed.csv').empty
            checks['all_rows_and_all_saved_history_have_frozen_evidence']=True
            screenshot('04-three-downloads.png')
            nav('操作说明')
            expect(page.locator('[data-testid="stSidebar"]').get_by_role('button',name=re.compile('模型与评测'))).to_have_count(0)
            expect(page.get_by_text('模型与评测',exact=True)).to_be_visible()
            checks['evaluation_is_in_help']=True
            assert not report['page_errors']
            assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in report['source_sha256'].items())
            report['passed']=True
        except BaseException as error:
            report['error']=repr(error)
            page.screenshot(path=str(out/'failure.png'),full_page=True)
            (out/'failure-ui.txt').write_text(page.locator('body').inner_text(),encoding='utf-8')
            raise
        finally:
            context.tracing.stop(path=str(out/'trace.zip'))
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            browser.close()
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();run(a.url,a.output)
