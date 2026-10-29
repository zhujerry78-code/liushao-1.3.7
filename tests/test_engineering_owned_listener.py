"""Health responses must come from the server owned by this acceptance run."""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import threading
from types import SimpleNamespace

import pandas as pd
import psutil
import pytest

from scripts import engineering_browser, engineering_ci, engineering_p0_browser
from scripts import engineering_run
from scripts.engineering_run import AcceptanceRun, OwnedProcessTree, start_owned_process


@pytest.fixture
def foreign_health_listener(monkeypatch):
    """Occupy the released port while the legitimate child is still starting."""
    servers = []

    class ForeignHealth(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')

        def log_message(self, *_):
            pass

    def launch(argv, **kwargs):
        port = int(next(value.split('=', 1)[1] for value in argv
                        if value.startswith('--server.port=')))
        server = HTTPServer(('127.0.0.1', port), ForeignHealth)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        # Preserve the real Streamlit invocation, delaying only module startup.
        code = ('import time,runpy,sys; time.sleep(.8); sys.argv='
                + repr(['streamlit', *argv[3:]])
                + '; runpy.run_module("streamlit",run_name="__main__")')
        return start_owned_process([sys.executable, '-c', code], **kwargs)

    for module in (engineering_ci, engineering_browser, engineering_p0_browser):
        monkeypatch.setattr(module, 'start_owned_process', launch)
    try:
        yield servers
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_startup_does_not_accept_an_unrelated_healthy_listener(tmp_path, foreign_health_listener):
    result = engineering_ci.startup_check(tmp_path, timeout=4)
    assert not result['passed'], result
    assert result['owned_process_stopped'] and result['cleanup']['passed']
    assert foreign_health_listener[0].fileno() >= 0


def test_standard_browser_rejects_foreign_health_before_opening_the_page(
    tmp_path, monkeypatch, foreign_health_listener,
):
    artifacts = tmp_path / 'models'
    artifacts.mkdir()
    (artifacts / 'model.joblib').write_bytes(b'fixture')
    (artifacts / 'model.json').write_bytes(b'{}')
    (artifacts / 'demo.csv').write_bytes(b'fixture\n1\n')
    monkeypatch.setattr(engineering_browser, 'prepare_model', lambda *a: (b'fixture', '{}', 'x'))
    monkeypatch.setattr(engineering_browser, 'load_model_bundle', lambda *a: {'model_version': 'x'})
    monkeypatch.setattr(engineering_browser, 'read_csv_checked', lambda *a: None)
    monkeypatch.setattr(engineering_browser, 'infer', lambda *a: (pd.DataFrame(index=range(1000)),
                                                               None, pd.DataFrame()))

    def browser_must_not_start():
        raise AssertionError('Browser launched after accepting a foreign listener')

    monkeypatch.setattr(engineering_browser, 'sync_playwright', browser_must_not_start)
    output = tmp_path / 'evidence'
    acceptance = AcceptanceRun(output, engineering_browser.ROOT, report_name='report.json')
    args = SimpleNamespace(workspace=engineering_browser.ROOT, output=output, artifacts=artifacts,
                           expected_model_sha256=hashlib.sha256(b'fixture').hexdigest())
    with pytest.raises(RuntimeError, match='Isolated Streamlit process exited'):
        engineering_browser.run_browser(args, acceptance)
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    assert report['passed'] is False and report['server_cleanup'][0]['passed']
    assert foreign_health_listener[0].fileno() >= 0


def p0_start_fixture(tmp_path):
    runner = engineering_p0_browser.P0Browser.__new__(engineering_p0_browser.P0Browser)
    runner.out, runner.starts = tmp_path, []
    runner.process = runner.process_tree = runner.log = None
    runner.models = {'frozen': {'directory': tmp_path}}
    runner.page = SimpleNamespace(goto=lambda *a: None)
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        runner.port = listener.getsockname()[1]
    runner.url = f'http://127.0.0.1:{runner.port}'
    return runner


def test_p0_browser_rejects_foreign_health_before_navigation(tmp_path, foreign_health_listener):
    runner = p0_start_fixture(tmp_path)
    navigations = []
    runner.page.goto = navigations.append
    try:
        with pytest.raises(RuntimeError, match='Owned Streamlit service exited'):
            runner.start('frozen')
        assert navigations == []
    finally:
        runner.stop()
    assert runner.starts[0]['cleanup']['passed']
    assert foreign_health_listener[0].fileno() >= 0


def test_legitimate_startup_health_still_passes_and_cleans_up(tmp_path):
    result = engineering_ci.startup_check(tmp_path, timeout=20)
    assert result['passed'], result
    assert result['owned_process_stopped'] and result['cleanup']['passed']


def test_p0_restart_accepts_both_legitimate_owned_listeners(tmp_path):
    runner = p0_start_fixture(tmp_path)
    try:
        runner.start('frozen')
        runner.start('frozen')
        assert len(runner.starts) == 2
        assert runner.starts[0]['cleanup']['passed']
        assert runner.process.poll() is None
    finally:
        runner.stop()
    assert runner.starts[1]['cleanup']['passed']


@pytest.fixture
def listener_owner(monkeypatch):
    state = {'created': 1, 'member': True}
    process = SimpleNamespace(create_time=lambda: state['created'], status=lambda: 'running')
    monkeypatch.setattr(engineering_run.psutil, 'Process', lambda pid: process)
    job = SimpleNamespace(layout={'fixture': True},
                          pids=lambda: [123] if state['member'] else [],
                          contains=lambda pid: state['member'])
    owner = OwnedProcessTree(SimpleNamespace(pid=123, poll=lambda: None), job=job)
    return owner, state


def listener(pid=123, host='127.0.0.1', port=8501):
    return SimpleNamespace(pid=pid, status=psutil.CONN_LISTEN,
                           laddr=SimpleNamespace(ip=host, port=port))


@pytest.mark.parametrize('host', ['127.0.0.1', '0.0.0.0'])
def test_listener_accepts_verified_owned_exact_or_wildcard_socket(
    listener_owner, monkeypatch, host,
):
    owner, _ = listener_owner
    monkeypatch.setattr(engineering_run.psutil, 'net_connections', lambda **kw: [listener(host=host)])
    assert owner.owns_tcp_listener('127.0.0.1', 8501)


@pytest.mark.parametrize('pid', [999, None])
def test_owned_wildcard_cannot_hide_a_foreign_or_unknown_specific_listener(
    listener_owner, monkeypatch, pid,
):
    owner, _ = listener_owner
    monkeypatch.setattr(engineering_run.psutil, 'net_connections', lambda **kw: [
        listener(host='0.0.0.0'), listener(pid=pid)])
    assert not owner.owns_tcp_listener('127.0.0.1', 8501)


@pytest.mark.parametrize('error', [psutil.AccessDenied(), RuntimeError('GetExtendedTcpTable failed')])
def test_unreadable_listener_table_cannot_count_as_owned(listener_owner, monkeypatch, error):
    owner, _ = listener_owner

    def unreadable(**kwargs):
        raise error

    monkeypatch.setattr(engineering_run.psutil, 'net_connections', unreadable)
    assert not owner.owns_tcp_listener('127.0.0.1', 8501)
    assert owner.errors and 'Cannot inspect TCP listener ownership' in owner.errors[-1]


@pytest.mark.parametrize('still_in_job', [False, True])
def test_listener_pid_reuse_during_socket_query_cannot_count_as_owned(
    listener_owner, monkeypatch, still_in_job,
):
    owner, state = listener_owner

    def reused(**kwargs):
        state.update(created=2, member=still_in_job)
        return [listener()]

    monkeypatch.setattr(engineering_run.psutil, 'net_connections', reused)
    assert not owner.owns_tcp_listener('127.0.0.1', 8501)
    assert owner.identities[123] == 1
