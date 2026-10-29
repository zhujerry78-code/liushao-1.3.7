"""Real AppTest integration for compact-detail navigation and draft safety."""
from pathlib import Path
import shutil

import pytest

from app_ui_helpers import edit, messages, navigate, start
from liushao.reviews import history


@pytest.fixture
def compact_app(r9_app):
    # Every presentation helper belongs to the byte-identical isolated app.
    source = Path(__file__).resolve().parents[1]
    for helper in source.glob('app*.py'):
        shutil.copyfile(helper, r9_app[0].parent / helper.name)
    return r9_app


def button_label(app, label):
    return next(button for button in app.button if button.label == label)


def test_detail_roundtrip_keeps_real_dirty_editor_and_search_state(compact_app):
    app = start(compact_app)
    navigate(app, '检查记录')
    original = app.session_state.ui.selected
    app.selectbox(key='_filter_queue').select('原始顺序').run()
    app.button(key='quick_search_toggle').click().run()
    app.text_input(key='_quick_search_input').input('第１０１行').run()
    edit(app, '待补证', '合成工程测试：评分详情往返草稿；非真人记录。')
    previous = dict(app.session_state.ui.filters)
    button_label(app, '查看完整评分依据').click().run()
    assert not app.exception
    assert app.session_state.ui.page == 'evidence_detail'
    assert app.session_state.ui.selected == original
    assert app.session_state.ui.editor.record_id == original
    assert [tab.label for tab in app.tabs] == ['评分图解', '证据明细', '原始数据与方法']
    assert history(compact_app[2]) == []
    button_label(app, '返回复核').click().run()
    assert not app.exception
    state = app.session_state.ui
    assert state.page == 'review' and state.selected == original
    assert state.search_keyword == '第１０１行' and state.search_open
    assert dict(state.filters) == previous
    assert app.selectbox(key='_status').value == '待补证'
    assert app.text_area(key='_note').value == '合成工程测试：评分详情往返草稿；非真人记录。'
    navigate(app, '下载结果')
    assert app.button(key='prepare_exports').disabled
    assert history(compact_app[2]) == []


def test_detail_navigation_keeps_sidebar_group_and_never_saves(compact_app):
    from app_navigation import page_id, page_label
    assert page_id('evidence_detail') == 'evidence_detail'
    assert page_label('evidence_detail') == '评分依据'
    app = start(compact_app)
    navigate(app, '检查记录')
    original = app.session_state.ui.selected
    button_label(app, '查看完整评分依据').click().run()
    assert not app.exception
    assert app.button(key='nav_检查记录').proto.type == 'primary'
    assert not any(button.key == 'nav_evidence_detail' for button in app.button)
    navigate(app, '检查记录')
    assert app.session_state.ui.selected == original
    assert history(compact_app[2]) == []


def test_invalidated_model_cannot_leave_previous_record_on_detail_page(compact_app):
    app = start(compact_app)
    navigate(app, '检查记录')
    edit(app, '待补证', '合成工程测试：模型失效保留草稿；非真人记录。')
    button_label(app, '查看完整评分依据').click().run()
    (compact_app[1] / 'model.json').write_text('{invalid fixture descriptor}', encoding='utf-8')
    app.run()
    assert not app.exception
    state = app.session_state.ui
    assert state.result is None and state.selected is None and state.editor is None
    assert state.suspended_draft is not None
    assert state.suspended_draft.note == '合成工程测试：模型失效保留草稿；非真人记录。'
    assert not any(button.label == '查看完整评分依据' for button in app.button)
    assert '暂存' in messages(app)
    assert history(compact_app[2]) == []


def test_failed_evidence_explanation_remains_explicit_and_returnable(compact_app, monkeypatch):
    import app_evidence

    def unavailable(*args, **kwargs):
        raise ValueError('合成工程测试：解释计算失败')

    monkeypatch.setattr(app_evidence, 'build_record_evidence', unavailable)
    app = start(compact_app)
    navigate(app, '检查记录')
    original = app.session_state.ui.selected
    edit(app, '待补证', '合成工程测试：解释失败时保留填写；非真人记录。')
    button_label(app, '查看完整评分依据').click().run()
    assert not app.exception
    assert '不可用' in messages(app)
    assert app.session_state.ui.selected == original
    button_label(app, '返回复核').click().run()
    assert not app.exception
    assert app.text_area(key='_note').value == '合成工程测试：解释失败时保留填写；非真人记录。'
    assert app.session_state.ui.selected == original
    assert history(compact_app[2]) == []


def test_saved_history_snapshot_stays_open_after_load_and_default_is_collapsed(compact_app):
    app = start(compact_app)
    navigate(app, '检查记录')
    original = app.session_state.ui.selected
    history_expander = next(item for item in app.expander if item.label == '操作历史')
    assert not history_expander.proto.expanded
    edit(app, '待补证', '合成工程测试：历史证据读取后持续展开；非真人记录。')
    app.button(key='save_review').click().run()
    assert not app.exception
    assert len(history(compact_app[2])) == 1
    history_expander = next(item for item in app.expander if item.label == '操作历史')
    assert not history_expander.proto.expanded
    app.button(key='read_history_' + original).click().run()
    assert not app.exception
    history_expander = next(item for item in app.expander if item.label == '操作历史')
    assert history_expander.proto.expanded
    assert any(item.value == '已保存当时证据' for item in app.info)
    assert app.session_state.ui.selected == original
    assert len(history(compact_app[2])) == 1
