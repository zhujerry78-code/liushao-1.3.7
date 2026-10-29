"""Local protocol/unit checks; no browser UI or user process is controlled."""
from contextlib import nullcontext
import http.client
import io
import json
import os
import re
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from delivery import desktop_start as desktop


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows URL association contract')


@pytest.fixture(autouse=True)
def isolated_native_activation_environment(monkeypatch):
    monkeypatch.delenv('LIUSHAO_BROWSER_ACTIVATION_ID', raising=False)


@pytest.fixture
def no_browser_process(monkeypatch):
    """A unit test must never fall through to a real browser process."""
    def forbidden(*args, **kwargs):
        raise AssertionError('Direct browser Popen is not the Windows URL activation path')

    monkeypatch.setattr(desktop.subprocess, 'Popen', forbidden)
    return forbidden


def test_windows_standard_url_activation_precedes_direct_app_process(monkeypatch, tmp_path, no_browser_process):
    opened = []
    monkeypatch.setattr(desktop.os, 'startfile', lambda url: opened.append(url))
    monkeypatch.setattr(desktop.webbrowser, 'open', no_browser_process)
    existing_profile = tmp_path / 'existing-profile'
    existing_profile.mkdir()
    sentinel = existing_profile / 'keep-user-profile.txt'
    sentinel.write_text('existing profile must remain untouched', encoding='utf-8')
    result = desktop.open_interface('http://127.0.0.1:8503', profile=existing_profile)
    assert opened == ['http://127.0.0.1:8503']
    assert isinstance(result, str) and result
    assert sentinel.read_text(encoding='utf-8') == 'existing profile must remain untouched'
    assert sorted(p.name for p in existing_profile.iterdir()) == [sentinel.name]


@pytest.mark.parametrize('url', [
    'https://127.0.0.1:8503',
    'http://localhost:8503',
    'http://[::1]:8503',
    'http://example.invalid:8503',
    'http://127.0.0.1.example.invalid:8503',
    'http://127.0.0.1@evil.invalid:8503',
])
def test_non_literal_loopback_url_is_rejected_before_native_activation(url, monkeypatch, tmp_path, no_browser_process):
    opened = []
    monkeypatch.setattr(desktop.os, 'startfile', lambda value: opened.append(value))
    monkeypatch.setattr(desktop.webbrowser, 'open', no_browser_process)
    with pytest.raises((ValueError, RuntimeError)):
        desktop.open_interface(url, root=tmp_path)
    assert opened == []


def test_native_activation_without_frontend_ack_cannot_be_reported_as_success(monkeypatch, tmp_path, no_browser_process):
    opened = []
    monkeypatch.setattr(desktop.os, 'startfile', lambda value: opened.append(value))
    monkeypatch.setattr(desktop.webbrowser, 'open', no_browser_process)

    def no_ack(self, timeout=None):
        raise TimeoutError('合成工程测试：浏览器未返回页面就绪确认')

    monkeypatch.setattr(desktop.FrontendReady, 'wait', no_ack)
    with pytest.raises((TimeoutError, RuntimeError), match='确认|就绪|ready|ack|ACK'):
        desktop.open_interface('http://127.0.0.1:8503', root=tmp_path)
    assert len(opened) == 1
    query = parse_qs(urlsplit(opened[0]).query)
    assert len(query.get('_liushao_launch', [])) == 1
    assert re.fullmatch('[0-9a-f]{64}', query['_liushao_launch'][0])
    assert not list((tmp_path / 'demo-state').glob('frontend-*.json'))


def test_stopping_service_does_not_kill_users_default_browser(monkeypatch, tmp_path):
    server = SimpleNamespace(pid=12001)
    user_browser = SimpleNamespace(pid=12002, info={
        'name': 'msedge.exe',
        'cmdline': ['msedge.exe', '--profile-directory=Default'],
    })
    stopped = []
    monkeypatch.setattr(desktop, 'startup_lock', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(desktop, 'owned_session', lambda root: ({'token': 'fixture-owned'}, server))
    monkeypatch.setattr(desktop, 'stop_process_tree', lambda process: stopped.append(process.pid))
    monkeypatch.setattr(desktop, 'remove_session', lambda *args: None)
    monkeypatch.setattr(desktop.psutil, 'process_iter', lambda *args: [user_browser])
    assert desktop.stop_desktop(tmp_path) == 0
    assert stopped == [server.pid]


def callback_get(url, headers=None):
    """Real local HTTP, explicitly bypassing all system proxy settings."""
    parsed = urlsplit(url)
    assert parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
    try:
        connection.request('GET', parsed.path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        return response.status, dict(response.getheaders()), body
    finally:
        connection.close()


def test_real_callback_ack_persists_receipt_and_context_cleans_request_and_server(tmp_path):
    url = 'http://127.0.0.1:8503'
    with desktop.FrontendReady(tmp_path, url, timeout=2) as ready:
        assert ready.server.server_address[0] == '127.0.0.1'
        assert re.fullmatch('[0-9a-f]{64}', ready.token)
        request = json.loads(ready.request_path.read_text(encoding='utf-8'))
        assert request['token'] == ready.token and request['callback'] == ready.callback
        assert parse_qs(urlsplit(ready.launch_url).query) == {'_liushao_launch': [ready.token]}
        response, headers, body = callback_get(ready.callback, {'Origin': url, 'Referer': ready.launch_url})
        assert response == 204 and body == b''
        assert headers['Access-Control-Allow-Origin'] == url
        ready.wait(timeout=.5)
        receipt = json.loads((tmp_path / 'demo-state/browser-last-ready.json').read_text(encoding='utf-8'))
        assert receipt['root'] == str(tmp_path.resolve())
        assert receipt['url'] == url and receipt['token'] == ready.token
        assert receipt['source'] == 'browser-rendered-callback'
        callback, request_path, thread = ready.callback, ready.request_path, ready.thread
    assert not request_path.exists()
    assert not thread.is_alive()
    with pytest.raises(OSError):
        callback_get(callback)
    assert (tmp_path / 'demo-state/browser-last-ready.json').is_file()


def test_callback_wrong_nonce_cannot_set_ready_event_or_persist_receipt(tmp_path):
    with desktop.FrontendReady(tmp_path, 'http://127.0.0.1:8503', timeout=2) as ready:
        other = ('a' if ready.token[0] != 'a' else 'b') + ready.token[1:]
        response, _, _ = callback_get(ready.callback.replace(ready.token, other))
        assert response == 404
        with pytest.raises(TimeoutError):
            ready.wait(timeout=.001)
        assert not (tmp_path / 'demo-state/browser-last-ready.json').exists()


def test_expired_callback_is_rejected_even_when_nonce_is_correct(tmp_path):
    with desktop.FrontendReady(tmp_path, 'http://127.0.0.1:8503', timeout=.01) as ready:
        time.sleep(.03)
        response, _, _ = callback_get(ready.callback)
        assert response == 410
        with pytest.raises(TimeoutError):
            ready.wait(timeout=.001)
        assert not (tmp_path / 'demo-state/browser-last-ready.json').exists()


@pytest.mark.parametrize('headers', [
    {'Origin': 'http://evil.invalid'},
    {'Origin': 'null'},
    {'Origin': 'http://127.0.0.1:8504'},
    {'Referer': 'http://evil.invalid/page'},
    {'Referer': 'http://127.0.0.1.evil.invalid:8503/page'},
    {'Referer': 'https://127.0.0.1:8503/page'},
])
def test_callback_rejects_foreign_origin_or_referer_without_ack(headers, tmp_path):
    with desktop.FrontendReady(tmp_path, 'http://127.0.0.1:8503', timeout=2) as ready:
        response, _, _ = callback_get(ready.callback, headers)
        assert response == 403
        with pytest.raises(TimeoutError):
            ready.wait(timeout=.001)
        assert not (tmp_path / 'demo-state/browser-last-ready.json').exists()


@pytest.mark.parametrize('invalid_part', ['host', 'client_address'])
def test_callback_rejects_nonliteral_host_or_nonloopback_client(invalid_part, tmp_path):
    with desktop.FrontendReady(tmp_path, 'http://127.0.0.1:8503', timeout=2) as ready:
        if invalid_part == 'host':
            response, _, _ = callback_get(ready.callback, {'Host': 'evil.invalid'})
        else:
            # A loopback-only listener cannot receive a real remote peer. Exercise
            # the actual handler with a synthetic socket and remote peer tuple.
            # No remote network interface is bound or contacted.
            class RequestSocket:
                def __init__(self, request):
                    self.request = request
                    self.output = bytearray()

                def makefile(self, *args, **kwargs):
                    return io.BytesIO(self.request)

                def sendall(self, value):
                    self.output.extend(value)

            host = urlsplit(ready.callback).netloc
            raw = f'GET /{ready.token} HTTP/1.0\r\nHost: {host}\r\n\r\n'.encode('ascii')
            connection = RequestSocket(raw)
            ready.server.RequestHandlerClass(connection, ('203.0.113.5', 43210), ready.server)
            response = int(bytes(connection.output).split(b' ', 2)[1])
        assert response == 403
        with pytest.raises(TimeoutError):
            ready.wait(timeout=.001)
        assert not (tmp_path / 'demo-state/browser-last-ready.json').exists()


def test_mock_native_activation_requires_real_ack_and_leaves_no_request_or_listener(monkeypatch, tmp_path, no_browser_process):
    callbacks = []
    launched = []

    def standard_browser_activation(url):
        launched.append(url)
        request = json.loads((tmp_path / 'demo-state/frontend-launch.json').read_text(encoding='utf-8'))
        assert parse_qs(urlsplit(url).query) == {'_liushao_launch': [request['token']]}
        response, _, _ = callback_get(request['callback'], {'Origin': request['url'], 'Referer': url})
        assert response == 204
        callbacks.append(request['callback'])

    monkeypatch.setattr(desktop.os, 'startfile', standard_browser_activation)
    monkeypatch.setattr(desktop.webbrowser, 'open', no_browser_process)
    result = desktop.open_interface('http://127.0.0.1:8503', root=tmp_path, timeout=2)
    assert result == 'Windows URL association (frontend confirmed)'
    assert len(launched) == len(callbacks) == 1
    assert not (tmp_path / 'demo-state/frontend-launch.json').exists()
    assert (tmp_path / 'demo-state/browser-last-ready.json').is_file()
    with pytest.raises(OSError):
        callback_get(callbacks[0])


def test_original_cmd_can_activate_published_url_and_python_still_requires_real_ack(monkeypatch, tmp_path, no_browser_process):
    """Simulate the native CMD reader, never a real browser or human action."""
    monkeypatch.setenv('LIUSHAO_BROWSER_ACTIVATION_ID', '123-456-789-10')
    monkeypatch.setattr(desktop.os, 'startfile', no_browser_process)
    original_wait = desktop.FrontendReady.wait
    published = tmp_path/'demo-state/browser-open-123-456-789-10.url.txt'
    profile_request = tmp_path/'demo-state/browser-open-123-456-789-10.profile-id.txt'

    def cmd_reader_ack(ready, timeout=None):
        value = published.read_text(encoding='ascii')
        assert value == ready.launch_url+'\n'
        assert profile_request.read_text(encoding='ascii') == desktop.native_profile_id(tmp_path)+'\n'
        assert re.fullmatch('[0-9a-f]{24}', desktop.native_profile_id(tmp_path))
        assert re.fullmatch(r'http://127[.]0[.]0[.]1:8503/\?_liushao_launch=[0-9a-f]{64}\n', value)
        assert callback_get(ready.callback)[0] == 204
        original_wait(ready, timeout)

    monkeypatch.setattr(desktop.FrontendReady, 'wait', cmd_reader_ack)
    assert desktop.open_interface('http://127.0.0.1:8503', root=tmp_path, timeout=2).endswith('(frontend confirmed)')
    assert not published.exists()
    assert not profile_request.exists()
    assert not (tmp_path/'demo-state/frontend-launch.json').exists()


def test_delegated_url_is_cleaned_on_ack_timeout(monkeypatch, tmp_path, no_browser_process):
    monkeypatch.setenv('LIUSHAO_BROWSER_ACTIVATION_ID', '1-2-3-4')
    monkeypatch.setattr(desktop.os, 'startfile', no_browser_process)
    published = tmp_path/'demo-state/browser-open-1-2-3-4.url.txt'

    def no_ack(ready, timeout=None):
        assert published.is_file()
        raise TimeoutError('no frontend ACK')

    monkeypatch.setattr(desktop.FrontendReady, 'wait', no_ack)
    with pytest.raises(TimeoutError, match='no frontend ACK'):
        desktop.open_interface('http://127.0.0.1:8503', root=tmp_path, timeout=.01)
    assert not published.exists()
    assert not list((tmp_path/'demo-state').glob('browser-open-*'))
    assert not list((tmp_path/'demo-state').glob('*.tmp'))


@pytest.mark.parametrize('activation_id', ['../outside', '1-2', '1-2-3-4&start', '1-2-3-4\n'])
def test_invalid_delegation_id_cannot_become_a_file_path(activation_id, monkeypatch, tmp_path, no_browser_process):
    monkeypatch.setenv('LIUSHAO_BROWSER_ACTIVATION_ID', activation_id)
    opened=[]

    def activation(url):
        opened.append(url)
        ready=json.loads((tmp_path/'demo-state/frontend-launch.json').read_bytes())
        assert callback_get(ready['callback'])[0] == 204

    monkeypatch.setattr(desktop.os, 'startfile', activation)
    desktop.open_interface('http://127.0.0.1:8503', root=tmp_path, timeout=2)
    assert len(opened)==1
    assert not list((tmp_path/'demo-state').glob('browser-open-*'))


def test_noninteractive_failure_never_waits_for_a_modal_dialog(monkeypatch, no_browser_process):
    import ctypes
    monkeypatch.setenv('LIUSHAO_NONINTERACTIVE','1')
    monkeypatch.setattr(ctypes.windll.user32,'MessageBoxW',no_browser_process)
    desktop.show_error('synthetic failure')


def test_stop_closes_only_this_installations_dedicated_browser(monkeypatch,tmp_path):
    server=SimpleNamespace(pid=17001)
    own=SimpleNamespace(pid=17002,info={'name':'msedge.exe','cmdline':[
        'msedge.exe','--user-data-dir='+str(desktop.native_browser_profile(tmp_path))]})
    other=SimpleNamespace(pid=17003,info={'name':'msedge.exe','cmdline':[
        'msedge.exe','--user-data-dir='+str(desktop.native_browser_profile(tmp_path/'other'))]})
    normal=SimpleNamespace(pid=17004,info={'name':'msedge.exe','cmdline':['msedge.exe']})
    stopped=[]
    monkeypatch.setattr(desktop,'startup_lock',lambda *a,**k:nullcontext())
    monkeypatch.setattr(desktop,'owned_session',lambda root:({'token':'test'},server))
    monkeypatch.setattr(desktop,'stop_process_tree',lambda p:stopped.append(p.pid))
    monkeypatch.setattr(desktop,'remove_session',lambda *a:None)
    monkeypatch.setattr(desktop.psutil,'process_iter',lambda *a:[own,other,normal])
    assert desktop.stop_desktop(tmp_path)==0
    assert stopped==[17001,17002]


def test_native_profile_reused_for_live_service_and_rotated_after_cold_restart(tmp_path):
    state=tmp_path/'demo-state'
    state.mkdir()
    session=state/'desktop-session.json'
    session.write_text(json.dumps({'root':str(tmp_path.resolve()),'token':'a'*32}),encoding='utf-8')
    first=desktop.native_profile_id(tmp_path)
    assert desktop.native_profile_id(tmp_path)==first
    (state/'browser-profile-state.json').write_text(json.dumps({'root':str(tmp_path.resolve()),'profile_id':first}),encoding='utf-8')
    session.unlink()
    assert desktop.native_profile_id(tmp_path)==first
    session.write_text(json.dumps({'root':str(tmp_path.resolve()),'token':'b'*32}),encoding='utf-8')
    assert desktop.native_profile_id(tmp_path)!=first
