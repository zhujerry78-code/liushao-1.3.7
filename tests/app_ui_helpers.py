"""AppTest helpers using the actual r9 controls; no session-state injection."""
import hashlib
import io
import json
import streamlit as st
from streamlit.testing.v1 import AppTest


def trust(artifacts, monkeypatch):
    monkeypatch.setenv('LIUSHAO_EXPECTED_MODEL_SHA256', hashlib.sha256((artifacts/'model.joblib').read_bytes()).hexdigest())
    monkeypatch.setenv('LIUSHAO_EXPECTED_DESCRIPTOR_SHA256', hashlib.sha256((artifacts/'model.json').read_bytes()).hexdigest())


def start(fixture):
    app = AppTest.from_file(str(fixture[0]),default_timeout=45).run()
    assert not app.exception
    app.button(key='run').click().run()
    assert not app.exception and app.session_state.ui.result is not None
    return app


def navigate(app, page):
    app.button(key='nav_'+page).click().run()
    assert not app.exception


def prepare(app, downloads):
    navigate(app, '下载结果')
    downloads.clear()
    app.button(key='prepare_exports').click().run()
    assert not app.exception
    return downloads


def edit(app, status='待补证', note='工程验证依据'):
    app.selectbox(key='_status').select(status).run()
    app.text_area(key='_note').input(note).run()


def upload(fixture, monkeypatch, content, name='uploaded.csv', app=None):
    value=io.BytesIO(content);value.name=name
    monkeypatch.setattr(st,'file_uploader',lambda *a,**kw:value)
    app=app or AppTest.from_file(str(fixture[0]),default_timeout=45).run()
    navigate(app,'查看结果')
    if app.session_state.ui.result is not None:
        app.button(key='change_file').click().run()
    app.radio(key='_source').set_value('上传 CSV').run()
    app.button(key='run').click().run()
    assert not app.exception
    return app


def messages(app):
    return '\n'.join(e.value for kind in (app.error,app.warning,app.info,app.markdown,app.caption) for e in kind)


def select_record_id(app, rid):
    """AppTest lacks native grid click events; Edge acceptance covers those events."""
    app.session_state.ui.selected=rid
    app.session_state.ui.editor=None
    app.run()
    assert not app.exception
    return app


def open_record(app, source_row):
    table=app.session_state.ui.result.predictions
    rid=table.loc[table.source_row.eq(source_row),'record_id'].iloc[0]
    return select_record_id(app,rid)
