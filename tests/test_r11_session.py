from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import pytest
import app_session as cache
from app_ui_helpers import upload, navigate
from streamlit.testing.v1 import AppTest

def test_default_read_never_creates_directory_and_database_paths_are_isolated(tmp_path):
    a, b = tmp_path/'a.sqlite', tmp_path/'b.sqlite'
    assert cache.saved_upload(a) is None and not cache.session_root(a).exists()
    token = cache.consent_generation(a)
    assert cache.save_upload(a, b'a,b\n1,2\n', 'a.csv', 'model', token)
    assert cache.saved_upload(b) is None
    saved = cache.saved_upload(a)
    assert cache.load_upload(a, (saved['generation'], saved['id']), 'model') == (b'a,b\n1,2\n', 'a.csv', '上传 CSV')

def test_clear_revokes_an_inflight_save(tmp_path):
    db = tmp_path/'reviews.sqlite'
    generation = cache.consent_generation(db)
    ready, finish = threading.Event(), threading.Event()
    def delayed():
        ready.set(); assert finish.wait(5)
        return cache.save_upload(db, b'data', 'same.csv', 'm', generation)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(delayed)
        assert ready.wait(5)
        cache.clear_upload(db); finish.set()
        assert future.result() is False
    assert cache.saved_upload(db) is None
    assert not list(cache.session_root(db).glob('*.input'))

def test_two_writers_leave_one_valid_last_commit(tmp_path):
    db = tmp_path/'reviews.sqlite'; generation = cache.consent_generation(db)
    with ThreadPoolExecutor() as pool:
        assert all(pool.map(lambda n: cache.save_upload(db, str(n).encode(), f'{n}.csv', 'm', generation), range(4)))
    saved = cache.saved_upload(db)
    content, name, _ = cache.load_upload(db, (saved['generation'], saved['id']), 'm')
    assert content.decode() == name.split('.')[0]
    assert len(list(cache.session_root(db).glob('*.input'))) == 1

def test_failed_manifest_commit_keeps_previous_input(tmp_path, monkeypatch):
    db = tmp_path/'r.sqlite'; generation = cache.consent_generation(db)
    cache.save_upload(db, b'old', 'old.csv', 'm', generation)
    before = cache.saved_upload(db)
    def fail(*a): raise OSError('injected manifest failure')
    monkeypatch.setattr(cache, '_write', fail)
    with pytest.raises(OSError): cache.save_upload(db, b'new', 'new.csv', 'm', generation)
    assert cache.saved_upload(db) == before
    assert len(list(cache.session_root(db).glob('*.input'))) == 1

@pytest.mark.parametrize('damage', ['source', 'path', 'oversize', 'length', 'model', 'bytes'])
def test_corrupt_or_wrong_identity_cache_is_not_restored(tmp_path, damage):
    db = tmp_path/'r.sqlite'; generation = cache.consent_generation(db)
    cache.save_upload(db, b'old', 'old.csv', 'm', generation)
    before = cache.saved_upload(db); reference = (generation, before['id'])
    root = cache.session_root(db); state_path = root/'state.json'
    state = json.loads(state_path.read_bytes())
    if damage == 'source': state['current']['source'] = '内置演示数据'
    if damage == 'path': state['current']['id'] = '../outside'
    if damage == 'length': state['current']['bytes'] = 99
    if damage == 'bytes': (root/(before['id']+'.input')).write_bytes(b'bad')
    if damage == 'oversize': state['padding'] = 'x'*65536
    state_path.write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(ValueError): cache.load_upload(db, reference, 'other' if damage == 'model' else 'm')
    cache.clear_upload(db)
    assert cache.saved_upload(db) is None

def test_cleanup_failure_reports_incomplete_and_revokes_restore(tmp_path, monkeypatch):
    db = tmp_path/'r.sqlite'; generation = cache.consent_generation(db)
    cache.save_upload(db, b'old', 'old.csv', 'm', generation)
    monkeypatch.setattr(cache, '_cleanup', lambda *a: (_ for _ in ()).throw(PermissionError('busy')))
    with pytest.raises(OSError, match='清除未完成'): cache.clear_upload(db)
    assert cache.saved_upload(db) is None
    assert cache.save_upload(db, b'new', 'new.csv', 'm', generation) is False

def test_upload_opt_in_restore_and_clear_are_explicit(r9_app, monkeypatch):
    content = b'a,b\n1,2\n3,4\n'
    app = upload(r9_app, monkeypatch, content)
    assert cache.saved_upload(r9_app[2]) is None
    app.button(key='change_file').click().run()
    app.checkbox(key='_remember').check().run()
    app.button(key='run').click().run()
    assert not app.exception and cache.saved_upload(r9_app[2])['name'] == 'uploaded.csv'
    restarted = AppTest.from_file(str(r9_app[0]), default_timeout=45).run()
    assert restarted.session_state.ui.result is None
    restarted.button(key='restore_last').click().run()
    assert not restarted.exception and len(restarted.session_state.ui.result.predictions) == 2
    restarted.button(key='change_file').click().run()
    restarted.button(key='clear_last').click().run()
    assert cache.saved_upload(r9_app[2]) is None
