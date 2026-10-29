"""Navigation event tests; final scroll geometry is checked in a real browser."""
import app_navigation_scroll as scroll

from app_ui_helpers import edit, navigate, start


def test_scroll_request_is_consumed_once_and_only_for_its_page(monkeypatch):
    rendered = []
    monkeypatch.setattr(scroll, 'component_html', lambda html, **kw: rendered.append((html, kw)))
    state = {}
    scroll.queue_navigation_scroll(state, 'evidence_detail')
    assert scroll.render_navigation_scroll(state, 'evidence_detail')
    assert not state and not scroll.render_navigation_scroll(state, 'evidence_detail')
    assert len(rendered) == 1 and '"target":"top"' in rendered[0][0]
    assert rendered[0][1] == dict(height=0, width=0, scrolling=False)
    scroll.queue_navigation_scroll(state, 'review')
    assert not scroll.render_navigation_scroll(state, 'overview')
    assert not state and len(rendered) == 1


def test_unknown_or_malformed_navigation_never_executes_html(monkeypatch):
    rendered = []
    monkeypatch.setattr(scroll, 'component_html', lambda *args, **kw: rendered.append(args))
    state = {}
    scroll.queue_navigation_scroll(state, 'about')
    assert not state
    for request in (None, {}, {'page':'review', 'target':'record', 'token':'</script>'},
                    {'page':'review', 'target':'elsewhere', 'token':'a'*32}):
        state[scroll._KEY] = request
        assert not scroll.render_navigation_scroll(state, 'review')
    assert not rendered


def test_only_detail_roundtrip_queues_scroll_and_keeps_draft(r9_app, monkeypatch):
    rendered = []
    monkeypatch.setattr(scroll, 'component_html', lambda html, **kw: rendered.append(html))
    app = start(r9_app)
    navigate(app, '检查记录')
    original = app.session_state.ui.selected
    edit(app, '待补证', '合成工程检查：往返定位仍保留备注。')
    assert not rendered
    app.button(key='open_evidence_detail').click().run()
    assert not app.exception and len(rendered) == 1
    assert '"target":"top"' in rendered[-1]
    app.run()
    assert len(rendered) == 1
    app.button(key='back_to_review').click().run()
    assert not app.exception and len(rendered) == 2
    assert '"target":"record"' in rendered[-1]
    assert app.session_state.ui.selected == original
    assert app.text_area(key='_note').value == '合成工程检查：往返定位仍保留备注。'
    edit(app, '待补证', '合成工程检查：继续填写。')
    app.run()
    assert len(rendered) == 2
    assert app.text_area(key='_note').value == '合成工程检查：继续填写。'
