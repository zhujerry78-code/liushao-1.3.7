"""Real frozen engine AppTest save/restart/history flow; automated, not a user study."""
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app_ui_helpers import navigate, messages
from liushao.reviews import history

ROOT=Path(__file__).resolve().parents[1]


def test_real_app_engine_snapshot_survives_new_session_and_exports(tmp_path,monkeypatch):
    if not (ROOT/'artifacts/engine.joblib').exists():
        pytest.skip('Source-only distribution excludes frozen binary assets')
    database=tmp_path/'automated-engine-review.sqlite'
    monkeypatch.setenv('LIUSHAO_REVIEW_DB',str(database))
    monkeypatch.setenv('LIUSHAO_ARTIFACTS_DIR',str(ROOT/'artifacts'))
    for name in ('LIUSHAO_EXPECTED_MODEL_SHA256','LIUSHAO_EXPECTED_DESCRIPTOR_SHA256'):
        monkeypatch.delenv(name,raising=False)
    st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    app.button(key='run').click().run()
    navigate(app,'检查记录')
    assert 'ECOD 采用' in messages(app) and '精确计算' in messages(app)
    rid=app.session_state.ui.selected
    app.selectbox(key='_status').select('待补证').run()
    app.text_area(key='_note').input('自动化工程验收：保存并重启，不是真人试用。').run()
    app.button(key='save_review').click().run()
    assert not app.exception and not app.error
    saved=history(database)[0]['evidence_snapshot']
    assert saved['engine']['status']=='verified'
    assert saved['engine']['identity']['engine_inference_version']=='r13-exact-va-v1'
    fresh=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    fresh.button(key='run').click().run()
    navigate(fresh,'检查记录')
    assert fresh.session_state.ui.selected==rid
    fresh.button(key='read_history_'+rid).click().run()
    assert not fresh.exception and '保存时引擎' in messages(fresh)
    assert fresh.session_state['_opened_history']['evidence_snapshot']==saved
    navigate(fresh,'下载结果')
    fresh.button(key='prepare_exports').click().run()
    assert not fresh.exception
    exported=json.loads(fresh.session_state.ui.exports.history)
    assert exported[0]['evidence_snapshot']==saved
    assert fresh.session_state.ui.exports.meta['engine_identity']['engine_inference_version']=='r13-exact-va-v1'
    st.cache_resource.clear()
