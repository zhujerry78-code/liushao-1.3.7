"""Delivery boundary regressions without installing into the real environment."""
import hashlib
import json
from pathlib import Path
import zipfile
import os
import subprocess
import sys
import venv

import pytest

from delivery import setup_demo, selfcheck, oneclick
from delivery.local_lock import file_lock, active_service, check_supported_path
from delivery.safe_files import safe_name, validate_names, read_json
from scripts import build_windows_demo


@pytest.mark.parametrize('name', ['../a', '/a', 'C:/a', 'a\\b', 'a:x', 'a./b', 'NUL.txt', 'x//y', 'x/./y'])
def test_reject_unsafe_portable_names(name):
    with pytest.raises(ValueError): safe_name(name)


def test_reject_duplicate_aliases_and_json_keys():
    with pytest.raises(ValueError): validate_names(['A.py','a.py'])
    with pytest.raises(ValueError): validate_names(['Pkg/a.py','pkg/b.py'])
    with pytest.raises(ValueError): read_json('{"files":{},"files":{}}')


@pytest.mark.parametrize('failure', ['extra', 'missing', 'duplicate', 'traversal', 'hash'])
def test_wheel_set_is_exact(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(setup_demo, 'ROOT', tmp_path)
    wheels = tmp_path/'wheels'; wheels.mkdir()
    (wheels/'one.whl').write_bytes(b'wheel fixture')
    digest = hashlib.sha256(b'wheel fixture').hexdigest()
    content = digest+'  one.whl\n'
    if failure == 'extra': (wheels/'extra.whl').write_bytes(b'extra')
    if failure == 'missing': (wheels/'one.whl').unlink()
    if failure == 'duplicate': content += content
    if failure == 'traversal': content = digest+'  ../one.whl\n'
    if failure == 'hash': (wheels/'one.whl').write_bytes(b'changed')
    (wheels/'SHA256SUMS.txt').write_text(content)
    with pytest.raises(RuntimeError): setup_demo.verify_wheels()


def test_install_lock_excludes_another_owner(tmp_path):
    with file_lock(tmp_path/'install.lock'):
        with pytest.raises(TimeoutError):
            with file_lock(tmp_path/'install.lock', timeout=.2):
                pytest.fail('second owner acquired the lock')
    with file_lock(tmp_path/'install.lock', timeout=.2):
        pass


def test_archive_failure_stops_reinstall(tmp_path, monkeypatch):
    monkeypatch.setattr(oneclick, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='已停止安装'):
        oneclick.park_broken_venv()


def test_diagnostics_allowlist_size_and_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(selfcheck, 'ROOT', tmp_path)
    logs = tmp_path/'install-logs'; logs.mkdir()
    (logs/'install-test.log').write_bytes(b'x'*(3*1024*1024))
    (logs/'secret.txt').write_text('must never be collected')
    (tmp_path/'reviews.sqlite').write_bytes(b'actual records must never be collected')
    monkeypatch.setattr(selfcheck, 'manifest_state', lambda: (_ for _ in ()).throw(ValueError('broken fixture')))
    report = selfcheck.build_report()
    assert '检查未完成：ValueError' in report
    archive = selfcheck.build_archive(report, tmp_path/'诊断报告-fixture.txt')
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) <= 20
        assert 'install-logs/secret.txt' not in z.namelist()
        assert 'reviews.sqlite' not in z.namelist()
        assert len(z.read('install-logs/install-test.log')) == 2*1024*1024
        assert '截断' in z.read('收集说明.txt').decode()


def test_snapshot_requires_independent_digest_and_rejects_changed_source(tmp_path):
    manifest = b'{"source_commit":"' + b'0'*40 + b'","files":{"app.py":"' + b'0'*64 + b'"}}'
    (tmp_path/'SOURCE_MANIFEST.json').write_bytes(manifest)
    (tmp_path/'app.py').write_text('changed')
    with pytest.raises(ValueError, match='independently'):
        build_windows_demo.read_source(snapshot=tmp_path)
    with pytest.raises(ValueError, match='Changed source'):
        build_windows_demo.read_source(snapshot=tmp_path, expected_digest=hashlib.sha256(manifest).hexdigest())


def test_diagnostic_tail_rejects_directory_junction(tmp_path, monkeypatch):
    outside = tmp_path/'outside'; outside.mkdir()
    (outside/'install-private.log').write_text('private fixture must not be collected')
    root = tmp_path/'package'; root.mkdir()
    subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(root/'install-logs'),str(outside)],
                   check=True,capture_output=True)
    monkeypatch.setattr(selfcheck,'ROOT',root)
    assert 'private fixture' not in '\n'.join(selfcheck.latest_log_tail())
    archive = selfcheck.build_archive('report',root/'诊断报告-test.txt')
    with zipfile.ZipFile(archive) as z:
        assert not any(name.endswith('.log') for name in z.namelist())


def test_running_same_environment_blocks_archive_even_without_session_json(tmp_path):
    root = tmp_path/'运行 空格(测试)'; root.mkdir()
    venv.EnvBuilder(with_pip=False).create(root/'.venv')
    module = root/'streamlit'; module.mkdir();(module/'__init__.py').write_text('')
    (module/'__main__.py').write_text('import time; time.sleep(45)')
    python = root/'.venv/Scripts/python.exe'
    process = subprocess.Popen([str(python),'-m','streamlit','run','app.py'],cwd=root,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        assert active_service(root)
    finally:
        import psutil
        from delivery.desktop_start import stop_process_tree
        try: stop_process_tree(psutil.Process(process.pid))
        except psutil.NoSuchProcess: pass
    assert active_service(root) is False


def test_unsupported_install_paths_fail_explicitly():
    for path in (Path('x'*181),Path(r'\\server\share')):
        with pytest.raises(RuntimeError): check_supported_path(path)


def test_foreign_health_response_does_not_prove_service_ownership():
    from delivery.desktop_start import wait_for_service
    from http.server import HTTPServer,BaseHTTPRequestHandler
    import threading
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200);self.end_headers();self.wfile.write(b'ok')
        def log_message(self,*args): pass
    listener = HTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=listener.serve_forever,daemon=True);thread.start()
    process = subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
    try:
        with pytest.raises(TimeoutError):
            wait_for_service(process,f'http://127.0.0.1:{listener.server_port}',timeout=.3)
    finally:
        process.terminate();process.wait(timeout=5)
        listener.shutdown();listener.server_close();thread.join(timeout=2)


@pytest.mark.parametrize('flag',[None,'--offline','--auto'])
def test_legacy_install_modes_always_use_verified_offline_setup(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(oneclick,'ROOT',tmp_path)
    for name in ('setup_console','check_python','check_disk','check_path'):
        monkeypatch.setattr(oneclick,name,lambda:None)
    verified=[];modes=[]
    monkeypatch.setattr(oneclick,'verify',lambda root:verified.append(root))
    monkeypatch.setattr(oneclick,'already_installed',lambda:False)
    monkeypatch.setattr(oneclick,'run_setup',lambda mode:modes.append(mode) or 0)
    monkeypatch.setattr(sys,'argv',['oneclick.py','--no-launch']+([flag] if flag else []))
    assert oneclick.main()==0 and verified==[tmp_path] and modes==['--offline']


def test_interpreter_probe_preserves_literal_percent_and_exclamation_path(tmp_path):
    root=tmp_path/'探测 %R9_ALIAS% &! (目录)';root.mkdir()
    venv.EnvBuilder(with_pip=False).create(root/'.venv')
    delivery=root/'delivery';delivery.mkdir()
    import shutil
    shutil.copyfile(Path(oneclick.__file__).with_name('find-python.cmd'),delivery/'find-python.cmd')
    (root/'probe.cmd').write_bytes(b'@echo off\r\nsetlocal DisableDelayedExpansion\r\ncall "delivery\\find-python.cmd"\r\nif not defined PYEXE exit /b 1\r\n"%PYEXE%" -X utf8 -c "import sys;print(sys.executable)"\r\n')
    expected=root/'.venv/Scripts/python.exe'
    result=subprocess.run(['cmd.exe','/d','/v:on','/c','probe.cmd'],cwd=root,
        env={**os.environ,'LIUSHAO_PYTHON':str(expected),'R9_ALIAS':'SHOULD_NOT_EXPAND'},
        capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr.decode(errors='replace')
    assert Path(result.stdout.decode('utf-8').strip()).resolve()==expected.resolve()
