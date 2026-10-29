"""Storage outages preserve drafts, original baselines and record ownership."""
import sqlite3
import pytest
from app_ui_helpers import start,navigate,edit,upload
from liushao import reviews as store


def outage(monkeypatch):
    original=store.history
    def fail(*a,**kw):raise sqlite3.OperationalError('injected storage outage')
    monkeypatch.setattr(store,'history',fail)
    return original


def test_read_outage_restores_draft_and_original_baseline_without_automatic_save(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','连续填写')
    rid=app.session_state.ui.selected;original=outage(monkeypatch);app.run()
    assert app.button(key='save_review').disabled
    monkeypatch.setattr(store,'history',original);app.button(key='retry_reviews').click().run()
    assert app.session_state.ui.editor.baseline==0 and app.session_state.ui.editor.record_id==rid
    assert app.text_area(key='_note').value=='连续填写' and store.history(r9_app[2])==[]


def test_outage_filter_change_keeps_draft_when_cancelled(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','保留填写')
    original=outage(monkeypatch);app.run();app.selectbox(key='_filter_status').select('确认异常').run()
    assert app.button(key='pending_save').disabled
    app.button(key='pending_cancel').click().run();assert app.text_area(key='_note').value=='保留填写'
    monkeypatch.setattr(store,'history',original)


def test_concurrent_write_during_outage_requires_conflict_and_explicit_reload(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','本窗口');rid=app.session_state.ui.selected
    original=outage(monkeypatch);app.run()
    store.save_review(r9_app[2],rid,r9_app[4]['model_version'],'确认正常','另一窗口')
    monkeypatch.setattr(store,'history',original);app.button(key='retry_reviews').click().run();app.button(key='save_review').click().run()
    assert app.session_state.ui.editor.conflict and len(store.history(r9_app[2]))==1
    app.button(key='reload_review').click().run();assert app.text_area(key='_note').value=='另一窗口'


def test_outage_draft_does_not_follow_another_input(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','旧输入草稿')
    original=outage(monkeypatch);app.run();app=upload(r9_app,monkeypatch,b'a,b\n3,4\n',app=app)
    assert app.session_state.ui.pending is not None
    app.button(key='pending_discard').click().run()
    monkeypatch.setattr(store,'history',original);navigate(app,'检查记录')
    assert app.text_area(key='_note').value=='' and store.history(r9_app[2])==[]


def test_write_failure_draft_survives_later_read_outage(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录');edit(app,'待补证','写失败填写')
    save=store.save_review
    def fail(*a,**kw):raise sqlite3.OperationalError('injected write failure')
    monkeypatch.setattr(store,'save_review',fail);app.button(key='save_review').click().run()
    original=outage(monkeypatch);app.run();assert app.session_state.ui.editor.note=='写失败填写'
    monkeypatch.setattr(store,'history',original);monkeypatch.setattr(store,'save_review',save)
    app.button(key='retry_reviews').click().run();assert store.history(r9_app[2])==[]
