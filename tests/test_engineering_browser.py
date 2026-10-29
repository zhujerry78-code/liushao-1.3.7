"""Real browser checks for screenshot anchors during Streamlit-style rerenders."""

import pytest
from playwright.sync_api import sync_playwright

from scripts.engineering_browser import scroll_to_capture_anchor


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as playwright:
        with playwright.chromium.launch(channel='msedge', headless=True) as instance:
            yield instance


def test_capture_waits_for_obsolete_heading_to_disappear(browser):
    page = browser.new_page(viewport={'width': 800, 'height': 600})
    try:
        page.set_content('<h3 id="old">Review</h3><div style="height:2000px"></div>'
                         '<h3 id="current">Review</h3>')
        page.evaluate("setTimeout(() => document.getElementById('old').remove(), 250)")
        scroll_to_capture_anchor(page.get_by_role('heading', name='Review', exact=True),
                                 timeout=3000)
        assert page.locator('#old').count() == 0
        bounds = page.locator('#current').bounding_box()
        assert bounds is not None and 0 <= bounds['y'] < 600
    finally:
        page.close()


def test_capture_rejects_persistent_duplicate_headings(browser):
    page = browser.new_page()
    try:
        page.set_content('<h3>Review</h3><h3>Review</h3>')
        with pytest.raises(AssertionError, match='expected to have count'):
            scroll_to_capture_anchor(page.get_by_role('heading', name='Review', exact=True),
                                     timeout=200)
    finally:
        page.close()
