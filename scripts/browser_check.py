"""Exercise a running isolated app: python -m scripts.browser_check --url URL."""
import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

from liushao.data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8502')
    parser.add_argument('--verify-restart', action='store_true')
    args = parser.parse_args()
    out = Path('test-results')
    out.mkdir(exist_ok=True)
    checked, console_errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        page.on('pageerror', lambda error: console_errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_load_state('networkidle')
        page.get_by_role('button', name='检查数据并预测', exact=True).click()
        expect(page.get_by_role('combobox', name=re.compile('复核结论'))).to_be_visible()
        page.get_by_role('button', name='导出预测与复核 CSV', exact=True).wait_for()
        if args.verify_restart:
            expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value('浏览器自动验收：证据已核对，待补充设备背景。')
            result = json.loads(Path('docs/results/browser_verification.json').read_text(encoding='utf-8'))
            result['checks'].append('review_survives_backend_process_restart')
            write_json('docs/results/browser_verification.json', result)
            browser.close()
            print('Backend restart persistence passed')
            return
        checked.append('real_model_demo_prediction')
        page.screenshot(path=str(out / 'app-overview.png'))
        page.get_by_text('待核实', exact=True).click()
        expect(page.get_by_text(re.compile('符合条件 16 条'))).to_be_visible()
        checked.append('disagreement_review_queue')
        page.get_by_text('高异常得分', exact=True).click()
        expect(page.get_by_text(re.compile('符合条件 1,000 条'))).to_be_visible()
        page.get_by_role('combobox', name=re.compile('复核结论')).click()
        page.get_by_role('option', name='待补证', exact=True).click()
        page.get_by_role('textbox', name='复核备注', exact=True).fill('浏览器自动验收：证据已核对，待补充设备背景。')
        page.get_by_role('button', name='保存复核', exact=True).click()
        expect(page.get_by_text('已保存复核，修改历史已追加。')).to_be_visible()
        page.get_by_role('button', name='保存复核', exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(out / 'app-review.png'))
        checked.append('detail_evidence_review_save')
        page.reload()
        page.get_by_role('button', name='检查数据并预测', exact=True).click()
        expect(page.get_by_role('textbox', name='复核备注', exact=True)).to_have_value('浏览器自动验收：证据已核对，待补充设备背景。')
        checked.append('review_survives_new_browser_session')
        for button, filename in [('导出预测与复核 CSV', 'browser_predictions.csv'),
                                 ('导出完整复核历史', 'browser_history.json'), ('导出运行清单', 'browser_manifest.json')]:
            with page.expect_download() as download:
                page.get_by_role('button', name=button, exact=True).click()
            download.value.save_as(out / filename)
        exported = pd.read_csv(out / 'browser_predictions.csv')
        assert len(exported) == 1000
        assert not any('label' in c.lower() or 'ground_truth' in c for c in exported.columns)
        reviewed = exported[exported.review_note.fillna('').str.contains('浏览器自动验收')]
        assert len(reviewed) >= 1 and set(reviewed.review_status) == {'待补证'}
        audit = json.loads((out / 'browser_history.json').read_text(encoding='utf-8'))
        assert set(reviewed.record_id) <= {item['record_id'] for item in audit}
        checked.append('exports_prediction_history_manifest_consistent')
        sample = pd.read_csv('artifacts/demo.csv').iloc[:3].astype(object)
        sample.iloc[2, :] = 'bad'
        sample['Label'] = ['Benign', 'Bot', 'Bot']
        csv_lines = sample.to_csv(index=False).splitlines()
        payload = ('\n'.join(csv_lines[:2] + [''] + csv_lines[2:]) + '\n').encode('utf-8')
        page.get_by_text('上传 CSV', exact=True).click()
        page.locator('input[type=file]').set_input_files({'name': 'browser_test.csv', 'mimeType': 'text/csv', 'buffer': payload})
        expect(page.locator('[data-testid="stMain"]').get_by_text('browser_test.csv', exact=True)).to_be_visible()
        page.get_by_role('button', name='检查数据并预测', exact=True).click()
        expect(page.get_by_role('button', name='导出无法判断记录', exact=True)).to_be_visible()
        expect(page.get_by_text(re.compile('无法判断比例 33.33%'))).to_be_visible()
        checked.append('upload_partial_invalid_rows_remain_unjudgeable')
        with page.expect_download() as downloaded:
            page.get_by_role('button', name='导出预测与复核 CSV', exact=True).click()
        downloaded.value.save_as(out / 'browser_upload_predictions.csv')
        uploaded_predictions = pd.read_csv(out / 'browser_upload_predictions.csv')
        assert uploaded_predictions.source_row.tolist() == [2, 4]
        checked.append('physical_line_numbers_survive_blank_lines')
        page.screenshot(path=str(out / 'app-upload.png'))
        duplicated = (csv_lines[0] + ',' + csv_lines[0].split(',')[0] + '\n' + csv_lines[1] + ',999\n').encode('utf-8')
        page.locator('input[type=file]').set_input_files({'name': 'duplicate_columns.csv', 'mimeType': 'text/csv', 'buffer': duplicated})
        expect(page.locator('[data-testid="stMain"]').get_by_text('duplicate_columns.csv', exact=True)).to_be_visible()
        page.get_by_role('button', name='检查数据并预测', exact=True).click()
        expect(page.get_by_text(re.compile('无法判断：CSV 表头包含重复字段'))).to_be_visible()
        checked.append('duplicate_raw_headers_rejected')
        broken = sample.drop(columns=sample.columns[0]).to_csv(index=False).encode('utf-8')
        page.locator('input[type=file]').set_input_files({'name': 'missing_column.csv', 'mimeType': 'text/csv', 'buffer': broken})
        expect(page.locator('[data-testid="stMain"]').get_by_text('missing_column.csv', exact=True)).to_be_visible()
        page.get_by_role('button', name='检查数据并预测', exact=True).click()
        expect(page.get_by_text(re.compile('无法判断：缺少必需特征'))).to_be_visible()
        expect(page.get_by_role('button', name='导出预测与复核 CSV', exact=True)).to_have_count(0)
        checked.append('missing_column_rejected_without_stale_predictions')
        # Mobile smoke check: widgets remain operable; tables intentionally scroll horizontally.
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(out / 'app-mobile.png'))
        assert not console_errors, console_errors
        browser.close()
    write_json('docs/results/browser_verification.json', {'checks': checked, 'page_errors': console_errors,
                                                        'browser': 'Microsoft Edge / Playwright', 'url': args.url})
    print(json.dumps(checked, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
