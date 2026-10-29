import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from delivery import service_probe, local_lock, oneclick, preflight, desktop_start

def test_native_probe_distinguishes_ownership_stale_pid_and_unknown(tmp_path, monkeypatch):
    target = str(tmp_path / '.venv/Scripts/python.exe')
    monkeypatch.setattr(service_probe, 'windows_processes', lambda: iter([(123, target, 100.)]))
    assert local_lock.service_state(tmp_path).state == 'running'
    monkeypatch.setattr(service_probe, 'windows_processes', lambda: iter([(123, str(tmp_path.parent / 'other/python.exe'), 100.)]))
    assert local_lock.service_state(tmp_path).state == 'stopped'
    monkeypatch.setattr(service_probe, 'windows_processes', lambda: iter([(123, None, None)]))
    assert local_lock.service_state(tmp_path).state == 'unknown'
    with pytest.raises(RuntimeError, match='权限不足'):
        local_lock.active_service(tmp_path)

def test_session_pid_reuse_does_not_prove_service_running(tmp_path, monkeypatch):
    state = tmp_path / 'demo-state'
    state.mkdir()
    session = state / 'desktop-session.json'
    session.write_text(json.dumps({'root': str(tmp_path), 'server_pid': 100, 'server_created': 2.}))
    monkeypatch.setattr(service_probe, 'windows_processes', lambda: iter([(100, str(tmp_path / 'runtime/python/python.exe'), 3.)]))
    assert local_lock.service_state(tmp_path).state == 'stopped'
    session.write_text('broken')
    assert local_lock.service_state(tmp_path).state == 'unknown'

@pytest.mark.parametrize('length', [100, 101, 120, 121])
def test_path_warning_and_rejection_boundaries(tmp_path, capsys, length):
    # No files are created: the fixture directory itself may already exceed 100.
    base = Path(tmp_path.anchor)
    path = base / ('x' * (length-len(str(base))))
    assert len(str(path)) == length
    if length > 120:
        with pytest.raises(RuntimeError, match='120 字符上限'):
            local_lock.check_supported_path(path)
    else:
        local_lock.check_supported_path(path)
    output = capsys.readouterr().out
    assert ('警告' in output) == (100 < length <= 120)
    if 100 < length <= 120:
        assert '仍允许继续' in output and str(length) in output


def test_sync_directory_is_still_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('OneDriveCommercial', str(tmp_path))
    with pytest.raises(RuntimeError, match='OneDrive'):
        local_lock.check_supported_path(tmp_path / 'sync')

def test_top_level_install_failure_remains_useful_when_logs_unwritable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(oneclick, 'ROOT', tmp_path)
    monkeypatch.setattr(oneclick, 'setup_console', lambda: None)
    monkeypatch.setattr(oneclick, '_main', lambda: (_ for _ in ()).throw(PermissionError('TEMP不可写')))
    (tmp_path / 'install-logs').write_text('blocks directory')
    assert oneclick.main() == 1
    output = capsys.readouterr().out
    assert 'TEMP不可写' in output and '3_环境自检.cmd' in output and 'Traceback' not in output

def test_preflight_detects_half_install_and_moved_directory_without_imports(tmp_path):
    with pytest.raises(RuntimeError, match='安装未完成'):
        preflight.check(tmp_path)
    environment = tmp_path / '.venv'
    (environment / 'Scripts').mkdir(parents=True)
    (environment / 'Scripts/python.exe').write_bytes(b'not used')
    (environment / 'pyvenv.cfg').write_text('home = Z:\\missing-runtime\nexecutable = Z:\\missing-runtime\\python.exe\n')
    with pytest.raises(RuntimeError, match='移动或改名'):
        preflight.check(tmp_path)


def test_legacy_copied_install_is_detected_even_if_old_runtime_exists(tmp_path):
    root = tmp_path / 'new'
    environment = root / '.venv'
    (environment / 'Scripts').mkdir(parents=True)
    (environment / 'Scripts/python.exe').write_bytes(b'never executed')
    old = tmp_path / 'old/runtime/python'
    old.mkdir(parents=True)
    (old / 'python.exe').write_bytes(b'never executed')
    (environment / 'pyvenv.cfg').write_text(f'home = {old}\nexecutable = {old / "python.exe"}\n')
    with pytest.raises(RuntimeError, match='移动或复制'):
        preflight.check(root)

def test_timeout_keeps_only_verified_healthy_service_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_start, 'open_interface', lambda *a, **kw: (_ for _ in ()).throw(TimeoutError('slow browser')))
    monkeypatch.setattr(desktop_start, 'owned_session', lambda root: ({'server_pid': 10, 'port': 8501}, None))
    monkeypatch.setattr(desktop_start, 'listener_owned', lambda *a: True)
    monkeypatch.setattr(desktop_start, 'service_ready', lambda *a: True)
    result = desktop_start.open_or_pending(tmp_path, 'http://127.0.0.1:8501', 10, 1.)
    assert result['state'] == 'frontend_pending' and result['frontend_ready'] is False
    monkeypatch.setattr(desktop_start, 'listener_owned', lambda *a: False)
    with pytest.raises(TimeoutError):
        desktop_start.open_or_pending(tmp_path, 'http://127.0.0.1:8501', 10, 1.)
