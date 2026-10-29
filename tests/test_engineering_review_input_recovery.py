"""Input failures retain identified old results and saved review associations."""
from app_ui_helpers import start,navigate,upload,edit
from liushao.reviews import history


def test_same_record_review_recovers_after_invalid_upload(r9_app,monkeypatch):
    content=(r9_app[1]/'demo.csv').read_bytes();app=upload(r9_app,monkeypatch,content)
    navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'确认正常','已保存来源依据');app.button(key='save_review').click().run()
    key=app.session_state.ui.active.key;app=upload(r9_app,monkeypatch,b'wrong\n1\n',app=app)
    assert app.session_state.ui.active.key==key and app.session_state.ui.run_error
    app=upload(r9_app,monkeypatch,content,app=app);navigate(app,'检查记录')
    assert app.session_state.ui.selected==rid and app.text_area(key='_note').value=='已保存来源依据'
    assert len(history(r9_app[2]))==1


def test_review_recovers_after_empty_filter_without_overwriting_continuous_draft(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'待补证','连续草稿');app.selectbox(key='_filter_status').select('确认异常').run()
    app.button(key='pending_cancel').click().run()
    assert app.session_state.ui.selected==rid and app.text_area(key='_note').value=='连续草稿'
