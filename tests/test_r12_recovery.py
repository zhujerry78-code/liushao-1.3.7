import json
from pathlib import Path

import pytest

import app_session as cache
from app_input import present_error
from app_ui_helpers import start, navigate, edit, upload
from app_ui_state import dirty, literal_markdown


def test_discarded_editor_stays_discarded_across_navigation_and_new_input(r9_app, monkeypatch):
    app = start(r9_app)
    navigate(app, '检查记录')
    edit(app, '待补证', '应当明确放弃的工程草稿')
    app.button(key='reload_review').click().run()
    assert not app.exception and not dirty(app.session_state.ui.editor)
    assert app.text_area(key='_note').value == ''
    navigate(app, '查看结果')
    app = upload(r9_app, monkeypatch, b'a,b\n1,2\n', name='__新文件__[1].csv', app=app)
    assert app.session_state.ui.pending is None
    assert app.session_state.ui.active.name == '__新文件__[1].csv'
    assert len(app.session_state.ui.result.predictions) == 1


@pytest.mark.parametrize('content', [None, [], True, 'text', {'format': 1},
    {'format': True, 'generation': 'a'*32, 'current': None}])
def test_malformed_cache_manifest_is_clearable_and_raises_a_useful_error(tmp_path, content):
    db = tmp_path/'reviews.sqlite'
    root = cache.session_root(db)
    root.mkdir()
    (root/'state.json').write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(ValueError, match='恢复清单'):
        cache.saved_upload(db)
    cache.clear_upload(db)
    assert cache.saved_upload(db) is None


@pytest.mark.parametrize('stamp', [None, '', 'bad', '2026-10-22T08:00:00'])
def test_cache_time_requires_an_explicit_timezone(tmp_path, stamp):
    db = tmp_path/'r.sqlite'
    token = cache.consent_generation(db)
    cache.save_upload(db, b'content', 'test.csv', 'm', token)
    path = cache.session_root(db)/'state.json'
    state = json.loads(path.read_bytes())
    state['current']['saved_at'] = stamp
    path.write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(ValueError, match='保存时间'):
        cache.saved_upload(db)


@pytest.mark.parametrize('name,model', [('', 'm'), ('a.csv', None), ('a.csv', ''), (None, 'm')])
def test_invalid_save_metadata_does_not_create_a_cache(tmp_path, name, model):
    db = tmp_path/'r.sqlite'
    with pytest.raises(ValueError):
        cache.save_upload(db, b'content', name, model, 'a'*32)
    assert not cache.session_root(db).exists()


def test_input_names_cannot_become_markdown_or_remote_embeds():
    assert literal_markdown('__记录__[1].csv') == r'\_\_记录\_\_\[1\]\.csv'
    assert literal_markdown('![x](https://example.invalid/a)') == r'\!\[x\]\(https://example\.invalid/a\)'


def test_local_file_error_retains_details_without_guessing_unknown_causes():
    error = present_error(PermissionError('private path'), '保留文件')
    assert error.reason == '本机目录暂时无法读写。' and 'PermissionError: private path' in error.detail
    unknown = present_error(RuntimeError('opaque'), '保留文件')
    assert '暂不能确定原因' in unknown.reason and 'RuntimeError: opaque' in unknown.detail
