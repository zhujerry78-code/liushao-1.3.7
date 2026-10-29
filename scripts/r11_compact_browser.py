"""Isolated local Edge acceptance for the compact r11 review workflow.

This creates explicitly labelled synthetic engineering reviews in the service's
database. Run against an isolated acceptance database, never a user's records.
Output directories are new on every run so failed traces remain available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def run(url: str, out: Path) -> None:
    if urlsplit(url).hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('A local loopback acceptance service is required.')
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_files = sorted([*root.glob('app*.py'), *(root / 'liushao').glob('*.py')])
    report = {
        'passed': False,
        'scope': 'local automated engineering fixture; not human usability evidence',
        'checks': {}, 'viewports': [], 'page_errors': [],
        'source_sha256': {str(f.relative_to(root)).replace('\\', '/'): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
    }
    checks = report['checks']
    note = f'合成工程测试：紧凑界面往返和保存关联；非真人记录。{out.name}'
    external_note = f'合成工程测试：另一窗口并发写入；非真人记录。{out.name}'
    missing_note = f'合成工程测试：609344 缺失和下一条；非真人记录。{out.name}'
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True, args=[
            '--disable-background-networking', '--disable-component-update',
            '--disable-sync', '--no-first-run',
        ])

        def new_context():
            context = browser.new_context(viewport={'width': 1440, 'height': 900}, accept_downloads=True)
            context.route('**/*', lambda route: route.continue_() if urlsplit(route.request.url).hostname in ('127.0.0.1', 'localhost') or route.request.url.startswith(('data:', 'blob:')) else route.abort())
            return context

        context = new_context()
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.set_default_timeout(40000)
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))

        def settle(target=page):
            target.wait_for_timeout(200)
            expect(target.locator('[data-testid="stApp"]')).to_have_attribute('data-test-script-state', 'notRunning', timeout=90000)
            expect(target.locator('[data-testid="stException"]')).to_have_count(0)
            target.wait_for_timeout(150)

        def nav(label, target=page):
            sidebar = target.locator('[data-testid="stSidebar"]')
            button = sidebar.get_by_role('button', name=re.compile(re.escape(label) + '$'))
            if not button.is_visible():
                open_button = target.get_by_role('button', name='Open sidebar', exact=True)
                if open_button.count():
                    open_button.click()
            button.click()
            expect(target.get_by_role('heading', name=label, exact=True)).to_be_visible()
            settle(target)

        def search(value, target=page):
            control = target.get_by_role('textbox', name='检索记录、系统提示或我的判断', exact=True)
            if not control.count():
                target.get_by_role('button', name='快速检索', exact=True).click()
                settle(target)
            control.fill(value)
            control.press('Enter')
            settle(target)

        def select_first(target=page):
            target.locator('.st-key-_records_main .dvn-scroller').first.click(position={'x': 105, 'y': 52})
            settle(target)

        def status(target=page):
            return target.locator('[data-testid="stSelectbox"]').filter(has=target.get_by_text('我的判断', exact=True)).get_by_role('combobox')

        def set_status(label='需要补充信息（待补证）', target=page):
            status(target).click()
            target.get_by_role('option', name=label, exact=True).click()
            settle(target)

        def fill_note(value, target=page):
            control = target.get_by_role('textbox', name='备注', exact=True)
            control.fill(value)
            control.press('Tab')
            settle(target)

        def screenshot(name):
            settle()
            page.screenshot(path=str(out / name), full_page=True)

        def start(target=page):
            target.goto(url)
            settle(target)
            target.get_by_role('button', name='开始检查', exact=True).click()
            expect(target.get_by_role('button', name='更换文件', exact=True)).to_be_visible()
            settle(target)
            nav('检查记录', target)

        def detail():
            page.get_by_role('button', name=re.compile('查看完整评分依据$')).click()
            settle()
            expect(page.get_by_role('heading', name='评分依据', exact=True)).to_be_visible()
            expect(page.get_by_role('button', name=re.compile('返回复核$'))).to_be_visible()

        def back():
            page.get_by_role('button', name=re.compile('返回复核$')).click()
            settle()
            expect(page.get_by_role('heading', name='检查记录', exact=True)).to_be_visible()

        def geometry(width, height):
            page.set_viewport_size({'width': width, 'height': height})
            settle()
            if width < 1000:
                close = page.get_by_role('button', name='Close sidebar', exact=True)
                if close.count() and close.is_visible():
                    close.click()
                    settle()
            dimensions = page.evaluate('({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth})')
            assert max(dimensions['body'], dimensions['document']) <= width + 1, dimensions
            header = page.locator('.st-key-record_header')
            expect(header).to_be_visible()
            save = page.get_by_role('button', name='保存', exact=True)
            grid = page.locator('.st-key-_records_main')
            header_box, save_box, grid_box = header.bounding_box(), save.bounding_box(), grid.bounding_box()
            assert header_box and save_box and grid_box
            assert grid_box['y'] + grid_box['height'] < header_box['y'] + 1, (grid_box, header_box)
            assert 220 <= grid_box['height'] <= 280, grid_box
            record = dict(width=width, height=height, dimensions=dimensions,
                          header=header_box, save=save_box, grid=grid_box,
                          record_to_save_height=save_box['y'] + save_box['height'] - header_box['y'])
            if width >= 1000:
                assert record['record_to_save_height'] < height, record
                main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
                assert main and grid_box['width'] >= .88 * (main['width'] - 64), (grid_box, main)
            save.scroll_into_view_if_needed()
            visible_save = save.bounding_box()
            assert visible_save and visible_save['x'] >= 0 and visible_save['x'] + visible_save['width'] <= width + 1
            screenshot(f'viewport-{width}x{height}.png')
            report['viewports'].append(record)

        secondary = None
        try:
            start()
            search('第 ２１７８０９ 行')
            expect(page.get_by_text('当前筛选范围内匹配 1 条，已置顶；其他记录保留。', exact=True)).to_be_visible()
            expect(page.get_by_text('1,000 条符合条件 · 点击一行打开', exact=True)).to_be_visible()
            select_first()
            expect(page.get_by_text('输入 CSV 第 113 行', exact=False)).to_be_visible()
            expect(page.get_by_role('button', name=re.compile('查看完整评分依据$'))).to_be_visible()
            expect(page.get_by_text('看到了什么', exact=False).first).to_be_visible()
            expect(page.get_by_text('为什么值得核查', exact=False).first).to_be_visible()
            expect(page.get_by_text('下一步查什么', exact=False).first).to_be_visible()
            assert page.locator('[data-testid="stVegaLiteChart"]').count() >= 1, 'RF/LR score ruler missing'
            ruler_labels = page.locator('[data-testid="stVegaLiteChart"]').first.locator('svg text').evaluate_all(
                "elements => elements.filter(e => ['RF','LR'].includes(e.textContent.trim())).map(e => {const r=e.getBoundingClientRect();return {model:e.textContent.trim(),x:r.x,y:r.y,width:r.width,height:r.height}})"
            )
            by_model = {label['model']: label for label in ruler_labels}
            assert set(by_model) == {'RF', 'LR'}, ruler_labels
            assert abs(by_model['RF']['y'] - by_model['LR']['y']) >= 25, ruler_labels
            report['ruler_label_geometry'] = ruler_labels
            expect(page.get_by_text('距参考下界约 4.34%', exact=False)).to_have_count(0)
            expect(page.get_by_text('RF 改回 P50', exact=False)).to_have_count(0)
            screenshot('01-compact-list-217809.png')
            page.locator('.st-key-record_header').evaluate("element => element.scrollIntoView({block:'start'})")
            screenshot('01-compact-record-217809.png')
            checks['unicode_search_and_default_compact_reasoning_with_ruler'] = True

            # A nonfirst page and a submitted quick-search term must both survive
            # leaving the review page; the active record itself stays selected.
            page.get_by_role('button', name='下一页', exact=True).click()
            settle()
            expect(page.get_by_text('第 2 / 20 页', exact=True)).to_be_visible()
            set_status()
            fill_note(note)
            detail()
            sidebar = page.locator('[data-testid="stSidebar"]')
            expect(sidebar.get_by_role('button', name=re.compile('检查记录$'))).to_have_attribute('kind', 'primary')
            expect(sidebar.get_by_role('button', name='评分依据', exact=True)).to_have_count(0)
            for label in ('评分图解', '证据明细', '原始数据与方法'):
                expect(page.get_by_role('tab', name=label, exact=True)).to_be_visible()
            expect(page.get_by_text('RF 分数是怎样组成的', exact=True)).to_be_visible()
            expect(page.get_by_text('基准 ', exact=False).first).to_be_visible()
            screenshot('02-score-detail.png')
            page.get_by_role('tab', name='证据明细', exact=True).click()
            settle()
            # The complete interval evidence belongs here, not on the main form.
            evidence_choice = page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(re.compile('查看.*证据')))
            if evidence_choice.count():
                evidence_choice.get_by_role('combobox').click()
                option = page.get_by_role('option', name=re.compile('平均返回包长度|Bwd Pkt Len Mean'))
                if option.count():
                    option.first.click()
                else:
                    page.keyboard.press('Escape')
                settle()
            if not page.get_by_text('4.34%', exact=False).count():
                # Some evidence selectors use a concise label. Search the known
                # source field directly without depending on its ordinal prefix.
                combos = page.get_by_role('combobox')
                if combos.count():
                    combos.first.click()
                    option = page.get_by_role('option', name=re.compile('平均返回包长度|Bwd Pkt Len Mean'))
                    if option.count(): option.first.click()
                    else: page.keyboard.press('Escape')
                    settle()
            expect(page.get_by_text('4.34%', exact=False).first).to_be_visible()
            expect(page.get_by_text(re.compile(r'^原值.*接近参考边界'))).to_be_visible()
            screenshot('03-compact-evidence-detail.png')
            page.get_by_role('tab', name='原始数据与方法', exact=True).click()
            settle()
            screenshot('04-original-fields-and-method.png')
            back()
            expect(page.get_by_role('textbox', name='备注', exact=True)).to_have_value(note)
            expect(page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('我的判断', exact=True))).to_contain_text('需要补充信息（待补证）')
            expect(page.get_by_role('textbox', name='检索记录、系统提示或我的判断', exact=True)).to_have_value('第 ２１７８０９ 行')
            expect(page.get_by_text('第 2 / 20 页', exact=True)).to_be_visible()
            expect(page.get_by_text('输入 CSV 第 113 行', exact=False)).to_be_visible()
            checks['detail_roundtrip_keeps_unsaved_status_note_identity_search_and_page'] = True

            nav('下载结果')
            expect(page.get_by_role('button', name='生成下载文件', exact=True)).to_be_disabled()
            page.get_by_role('button', name='返回填写位置', exact=True).click()
            settle()
            expect(page.get_by_role('textbox', name='备注', exact=True)).to_have_value(note)
            checks['detail_is_unsaved_and_export_guard_remains_active'] = True

            # A second independent browser session writes a labelled fixture to
            # the same record. The first session must retain its unsaved draft.
            secondary = new_context()
            other = secondary.new_page()
            other.set_default_timeout(40000)
            start(other)
            search('217809', other)
            select_first(other)
            set_status(target=other)
            fill_note(external_note, other)
            other.get_by_role('button', name='保存', exact=True).click()
            settle(other)
            expect(other.get_by_text('已保存。', exact=True)).to_be_visible()
            page.get_by_role('button', name='保存', exact=True).click()
            settle()
            expect(page.get_by_text('这条记录已被其他窗口修改。', exact=False).first).to_be_visible()
            expect(page.get_by_role('textbox', name='备注', exact=True)).to_have_value(note)
            expect(page.get_by_text('输入 CSV 第 113 行', exact=False)).to_be_visible()
            screenshot('05-concurrent-draft-preserved.png')
            page.get_by_role('button', name='放弃填写并读取已保存记录', exact=True).click()
            settle()
            expect(page.get_by_role('textbox', name='备注', exact=True)).to_have_value(external_note)
            fill_note(note)
            page.get_by_role('button', name='保存', exact=True).click()
            settle()
            expect(page.get_by_text('已保存。', exact=True)).to_be_visible()
            checks['concurrent_write_requires_reload_and_draft_is_preserved'] = True
            secondary.close()
            secondary = None

            search('does-not-match-[.*]')
            expect(page.get_by_text('未找到匹配，原表保留。', exact=True)).to_be_visible()
            expect(page.get_by_text('1,000 条符合条件 · 点击一行打开', exact=True)).to_be_visible()
            search('609344')
            select_first()
            expect(page.get_by_text('输入 CSV 第 473 行', exact=False)).to_be_visible()
            expect(page.locator('[data-testid="stMarkdownContainer"] > p').filter(has_text=re.compile('缺失|缺少')).first).to_be_visible()
            screenshot('06-compact-missing-609344.png')
            checks['missing_record_has_visible_quality_reasoning'] = True
            for width, height in ((1440, 900), (1280, 800), (390, 844)):
                geometry(width, height)
                detail()
                for label, slug in (('评分图解', 'scores'), ('证据明细', 'evidence'),
                                    ('原始数据与方法', 'raw-methods')):
                    page.get_by_role('tab', name=label, exact=True).click()
                    settle()
                    dimensions = page.evaluate('({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth})')
                    assert max(dimensions['body'], dimensions['document']) <= width + 1, (label, dimensions)
                    if width == 390 and label == '评分图解':
                        svg = page.locator('[data-testid="stVegaLiteChart"]').first.locator('svg').first
                        legend = svg.locator('text').filter(has_text=re.compile('^无加减$'))
                        expect(legend).to_be_visible()
                        svg_box, legend_box = svg.bounding_box(), legend.bounding_box()
                        assert svg_box and legend_box
                        assert (legend_box['width'] > 0 and legend_box['height'] > 0
                                and legend_box['x'] >= svg_box['x'] - 1
                                and legend_box['y'] >= svg_box['y'] - 1
                                and legend_box['x'] + legend_box['width'] <= svg_box['x'] + svg_box['width'] + 1
                                and legend_box['y'] + legend_box['height'] <= svg_box['y'] + svg_box['height'] + 1), (svg_box, legend_box)
                        report['narrow_legend_geometry'] = {'svg': svg_box, 'label': legend_box}
                        checks['narrow_score_legend_is_fully_inside_svg'] = True
                    if width == 390 and label == '证据明细':
                        # The selected missing-value card is actual visible prose;
                        # native dataframe accessibility text is intentionally excluded.
                        expect(page.locator('[data-testid="stMarkdownContainer"] > p').filter(
                            has_text=re.compile(r'^原值[：:]\s*原始缺失'))).to_be_visible()
                        checks['missing_value_card_uses_clear_missing_label'] = True
                    screenshot(f'detail-{slug}-{width}x{height}.png')
                back()
            checks['top_full_width_240px_grid_compact_save_and_no_page_overflow'] = True
            checks['all_detail_tabs_have_no_page_overflow_at_three_viewports'] = True
            page.set_viewport_size({'width': 1440, 'height': 900})
            settle()
            set_status()
            fill_note(missing_note)
            page.get_by_role('button', name='保存并下一条', exact=True).click()
            settle()
            expect(page.get_by_text('输入 CSV 第 113 行', exact=False)).to_be_visible()
            checks['save_next_follows_promoted_display_order'] = True
            history = page.locator('[data-testid="stExpander"]').filter(has=page.get_by_text('操作历史', exact=True))
            history.locator('summary').click()
            settle()
            page.get_by_role('button', name='查看当时证据', exact=True).click()
            settle()
            expect(history.locator('[data-testid="stMarkdownContainer"] > p').filter(has_text=re.compile('^已保存当时证据$'))).to_be_visible()
            checks['saved_history_snapshot_can_be_read_on_demand'] = True
            nav('下载结果')
            page.get_by_role('button', name='生成下载文件', exact=True).click()
            settle()
            for key, label, suffix in (
                ('predictions', '下载检查结果', '.csv'),
                ('history', '下载已保存的操作记录', '.json'),
                ('manifest', '下载本次运行信息', '.json'),
                ('confirmed', '下载复核结论单', '.csv'),
            ):
                with page.expect_download() as event:
                    page.get_by_role('button', name=label, exact=True).click()
                event.value.save_as(str(out / (key + suffix)))
            import pandas as pd
            predictions = pd.read_csv(out / 'predictions.csv')
            records = json.loads((out / 'history.json').read_text(encoding='utf-8'))
            assert len(predictions) == 1000
            assert predictions.loc[predictions.source_row.eq(217809), 'input_csv_row'].tolist() == [113]
            assert predictions.loc[predictions.source_row.eq(609344), 'input_csv_row'].tolist() == [473]
            for value in (note, external_note):
                saved = [r for r in records if r['note'] == value]
                assert saved and all(r['evidence_snapshot_status'] == 'saved' and r['evidence_snapshot']['identity']['source_row'] == 217809 and r['evidence_snapshot']['input_csv_row'] == 113 for r in saved)
            missing = [r for r in records if r['note'] == missing_note]
            assert missing and all(r['evidence_snapshot']['selection']['missing_fields'] == ['Flow Byts/s', 'Flow Pkts/s'] and r['evidence_snapshot']['identity']['source_row'] == 609344 for r in missing)
            assert pd.read_csv(out / 'confirmed.csv').empty
            checks['all_saved_snapshots_and_download_identity_are_compatible'] = True
            screenshot('07-downloads.png')
            assert not report['page_errors'], report['page_errors']
            assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in report['source_sha256'].items()), 'source changed during acceptance'
            report['passed'] = True
        except BaseException as error:
            report['error'] = repr(error)
            page.screenshot(path=str(out / 'failure.png'), full_page=True)
            (out / 'failure-ui.txt').write_text(page.locator('body').inner_text(), encoding='utf-8')
            raise
        finally:
            if secondary is not None:
                secondary.close()
            context.tracing.stop(path=str(out / 'trace.zip'))
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            browser.close()
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.url, args.output)
