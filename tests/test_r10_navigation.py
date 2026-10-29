"""Route migration preserves the actual review workflow."""
import pytest
from streamlit.testing.v1 import AppTest

from app_navigation import PAGES, page_id
from app_ui_helpers import start, edit, navigate


@pytest.mark.parametrize('page', PAGES[:4] + PAGES[5:])
def test_legacy_page_name_migrates_without_replacing_state(r9_app, page):
    app = AppTest.from_file(str(r9_app[0]), default_timeout=45).run()
    state = app.session_state.ui
    state.page = page.label
    state.filters.line = '113'
    app.run()
    assert not app.exception
    assert app.session_state.ui.page == page.id
    assert app.session_state.ui.filters.line == '113'
    assert app.title[0].value == page.label


def test_navigation_preserves_unsaved_draft_and_handles_legacy_callback(r9_app):
    app = start(r9_app)
    navigate(app, '检查记录')
    edit(app, note='尚未保存的依据')
    navigate(app, '操作说明')
    assert app.session_state.ui.page == 'help'
    assert app.session_state.ui.editor.note == '尚未保存的依据'
    app.session_state.ui.page = '检查记录'
    app.run()
    assert app.session_state.ui.page == 'review'
    assert app.text_area(key='_note').value == '尚未保存的依据'
    assert page_id('不存在的页面') == 'overview'
