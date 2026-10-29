"""Opt-in last upload, isolated by database; no predictions or reviews are cached."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid
from datetime import datetime, timezone

MAX_BYTES = 50 * 1024 * 1024
MAX_MANIFEST = 64 * 1024

def session_root(database):
    db = Path(database).resolve()
    return db.with_name(db.name + '.last-session')

def _plain(path):
    if path.is_symlink() or path.is_junction(): raise ValueError('恢复缓存含文件链接，已停止操作。')
    return path

@contextmanager
def _locked(root):
    import msvcrt
    _plain(root).mkdir(parents=True, exist_ok=True)
    with _plain(root/'cache.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell(): handle.write(b'0'); handle.flush()
        deadline = time.monotonic() + 2
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline: raise TimeoutError('另一窗口正在更新恢复缓存，请稍后重试。')
                time.sleep(.02)
        try: yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

def _read(root):
    path = _plain(root/'state.json')
    if not path.exists(): return dict(format=1, generation=uuid.uuid4().hex, current=None)
    with path.open('rb') as handle: content = handle.read(MAX_MANIFEST+1)
    if len(content) > MAX_MANIFEST: raise ValueError('恢复清单超过 64 KiB。')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('恢复清单含重复字段。')
            result[key] = value
        return result
    state = json.loads(content, object_pairs_hook=unique)
    if not isinstance(state, dict) or not {'format', 'generation', 'current'}.issubset(state):
        raise ValueError('恢复清单结构无效。')
    if type(state['format']) is not int or state['format'] != 1 or not re.fullmatch('[0-9a-f]{32}', str(state.get('generation', ''))):
        raise ValueError('恢复清单版本或代次无效。')
    current = state.get('current')
    if current is not None:
        if (not isinstance(current, dict) or not re.fullmatch('[0-9a-f]{32}', str(current.get('id', '')))
                or current.get('source') != '上传 CSV' or type(current.get('bytes')) is not int
                or not 0 < current['bytes'] <= MAX_BYTES
                or not re.fullmatch('[0-9a-f]{64}', str(current.get('sha256', '')))
                or not isinstance(current.get('model'), str) or not current['model']
                or not isinstance(current.get('name'), str) or not current['name']
                or len(current['name']) > 1024):
            raise ValueError('恢复清单内容无效。')
        try:
            if datetime.fromisoformat(current['saved_at']).tzinfo is None:
                raise ValueError('missing timezone')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('恢复清单保存时间无效。') from error
    return state

def _write(root, state):
    content = json.dumps(state, ensure_ascii=False).encode('utf-8')
    if len(content) > MAX_MANIFEST: raise ValueError('恢复清单超过 64 KiB。')
    temp = root/(uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        _plain(root/'state.json')
        os.replace(temp, root/'state.json')
    finally:
        if temp.exists(): temp.unlink()

def _cleanup(root, keep=None):
    for path in root.iterdir():
        if re.fullmatch('[0-9a-f]{32}\\.(?:input|tmp)', path.name) and path.name != keep:
            _plain(path).unlink()

def consent_generation(database):
    root = session_root(database)
    with _locked(root):
        state = _read(root)
        if not (root/'state.json').exists(): _write(root, state)
        return state['generation']

def saved_upload(database):
    root = session_root(database)
    if not root.exists(): return None
    with _locked(root):
        state = _read(root)
        return dict(state['current'], generation=state['generation']) if state['current'] else None

def save_upload(database, content, name, model, generation):
    if (not isinstance(content, bytes) or not 0 < len(content) <= MAX_BYTES
            or not isinstance(name, str) or not name or len(name) > 1024
            or not isinstance(model, str) or not model):
        raise ValueError('保留文件大小或文件名无效。')
    root = session_root(database)
    with _locked(root):
        state = _read(root)
        if state['generation'] != generation: return False
        file_id = uuid.uuid4().hex
        path = root/(file_id+'.input')
        committed = False
        try:
            with path.open('xb') as handle:
                handle.write(content); handle.flush(); os.fsync(handle.fileno())
            state['current'] = dict(id=file_id, name=name, source='上传 CSV', bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(), model=model,
                saved_at=datetime.now(timezone.utc).isoformat())
            _write(root, state)
            committed = True
        finally:
            if not committed and path.exists(): path.unlink()
        try: _cleanup(root, path.name)
        except OSError as error: raise OSError('文件已保留，但旧缓存清理未完成。') from error
        return True

def clear_upload(database):
    root = session_root(database)
    if not root.exists(): return
    with _locked(root):
        # A corrupt manifest must still be clearable without following its contents.
        _write(root, dict(format=1, generation=uuid.uuid4().hex, current=None))
        try: _cleanup(root)
        except (OSError, ValueError) as error: raise OSError('恢复入口已撤销，但清除未完成，请检查本机缓存目录。') from error

def load_upload(database, reference, model):
    root = session_root(database)
    if not root.exists(): raise ValueError('已保留文件已清除。')
    with _locked(root):
        state = _read(root)
        current = state['current']
        if current is None or (state['generation'], current['id']) != tuple(reference):
            raise ValueError('已保留文件已改变或清除，请重新选择。')
        if current['model'] != model: raise ValueError('保留文件的模型身份与当前模型不一致，请重新上传原文件。')
        with _plain(root/(current['id']+'.input')).open('rb') as handle: content = handle.read(MAX_BYTES+1)
        if len(content) != current['bytes'] or hashlib.sha256(content).hexdigest() != current['sha256']:
            raise ValueError('保留文件长度或内容校验失败，请重新上传原文件。')
        return content, current['name'], '上传 CSV'
