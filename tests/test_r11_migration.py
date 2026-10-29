import hashlib
import sqlite3
import pytest
from delivery.migrate_reviews import migrate
from liushao.reviews import save_review, history
from liushao.reviews import connect

def setup(tmp_path, monkeypatch):
    monkeypatch.delenv('LIUSHAO_REVIEW_DB', raising=False)
    old, new = tmp_path/'旧目录', tmp_path/'新目录'
    old.mkdir(); new.mkdir()
    db = old/'demo-state/reviews.sqlite'
    save_review(db, 'a', 'm', '确认正常', '原始依据')
    save_review(db, 'b', 'other-model', '待补证', 'other')
    return old, new, db

def test_migration_preserves_all_rows_and_next_id(tmp_path, monkeypatch):
    old, new, source = setup(tmp_path, monkeypatch)
    before = source.read_bytes()
    result = migrate(old, new, checker=lambda root: False)
    target = new/'demo-state/reviews.sqlite'
    assert result['records'] == 2 and history(source) == history(target)
    assert source.read_bytes() == before
    save_review(target, 'c', 'm', '待补证', 'new')
    assert history(target)[-1]['id'] == 3 and len(history(source)) == 2

def test_backup_includes_committed_wal(tmp_path, monkeypatch):
    old, new, source = setup(tmp_path, monkeypatch)
    connection = sqlite3.connect(source)
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute("INSERT INTO reviews(record_id,model_version,status,note,created_at) VALUES('wal','m','待补证','WAL','2020-01-01T00:00:00+00:00')")
        connection.commit()
        assert migrate(old, new, checker=lambda root: False)['records'] == 3
    finally: connection.close()

@pytest.mark.parametrize('problem', ['target', 'running', 'corrupt', 'extra', 'custom_env', 'missing'])
def test_failed_migration_preserves_source_and_target(tmp_path, monkeypatch, problem):
    old, new, source = setup(tmp_path, monkeypatch)
    target = new/'demo-state/reviews.sqlite'
    if problem == 'target': target.parent.mkdir(); target.write_bytes(b'keep')
    if problem == 'corrupt': source.write_bytes(b'not sqlite')
    if problem == 'extra':
        with sqlite3.connect(source) as db: db.execute('CREATE TABLE extra(x)')
    if problem == 'custom_env': monkeypatch.setenv('LIUSHAO_REVIEW_DB', str(tmp_path/'else.sqlite'))
    if problem == 'missing': source.unlink()
    before = source.read_bytes() if source.exists() else None
    with pytest.raises(Exception): migrate(old, new, checker=lambda root: problem == 'running')
    assert (source.read_bytes() if source.exists() else None) == before
    assert target.read_bytes() == b'keep' if problem == 'target' else not target.exists()
    assert not list(new.rglob('migration-*'))

def test_empty_database_and_deleted_latest_id_preserve_sequence(tmp_path, monkeypatch):
    old, new, source = setup(tmp_path, monkeypatch)
    with sqlite3.connect(source) as db: db.execute('DELETE FROM reviews')
    assert migrate(old, new, checker=lambda root: False)['records'] == 0
    target = new/'demo-state/reviews.sqlite'
    save_review(target, 'next', 'm', '待补证', '')
    assert history(target)[0]['id'] == 3

@pytest.mark.parametrize('race', [False, True])
def test_interrupted_publication_keeps_old_and_any_racing_target(tmp_path, monkeypatch, race):
    from delivery import migrate_reviews as module
    old, new, source = setup(tmp_path, monkeypatch)
    before = source.read_bytes()
    def fail(temporary, target):
        if race: target.write_bytes(b'other owner')
        raise FileExistsError('injected publication failure')
    monkeypatch.setattr(module.os, 'rename', fail)
    with pytest.raises(FileExistsError): migrate(old, new, checker=lambda root: False)
    assert source.read_bytes() == before
    target = new/'demo-state/reviews.sqlite'
    assert target.read_bytes() == b'other owner' if race else not target.exists()
    assert not list(new.rglob('migration-*'))

def test_source_lock_blocks_copy(tmp_path, monkeypatch):
    from delivery.local_lock import file_lock
    old, new, source = setup(tmp_path, monkeypatch)
    with file_lock(old/'demo-state/install.lock'):
        with pytest.raises(TimeoutError): migrate(old, new, checker=lambda root: False)
    assert not (new/'demo-state/reviews.sqlite').exists()
