"""Validate local launch ownership and browser-only readiness acknowledgement."""
import copy
import json
from pathlib import Path

import pytest

import app_desktop_ready as ready


TOKEN = 'a9' * 32


def request(base_root, **changes):
    value = dict(root=str(Path(base_root).resolve()), url='http://127.0.0.1:8790', token=TOKEN,
                 callback='http://127.0.0.1:53177/' + TOKEN, expires=1090.)
    value.update(changes)
    return value


def test_valid_request_returns_only_browser_fields_without_mutating_it(tmp_path):
    raw = request(tmp_path, server_pid=1234); before = copy.deepcopy(raw)
    parsed = ready.validate_ready_request(raw, tmp_path, now=1000.)
    assert parsed == dict(url=raw['url'], token=TOKEN, callback=raw['callback'], expires=1090.)
    assert raw == before


@pytest.mark.parametrize('change', [dict(expires=1000.), dict(expires=999.), dict(expires=True),
    dict(expires=float('inf')), dict(expires=float('nan')), dict(expires='1090'),
    dict(root='another root'), dict(token='abc'), dict(token='a'*63), dict(token='g'*64),
    dict(token='a'*64+'\n'), dict(token='</script><script>alert(1)</script>')])
def test_expired_wrong_owner_or_injected_token_never_validates(tmp_path, change):
    assert ready.validate_ready_request(request(tmp_path, **change), tmp_path, now=1000.) is None


@pytest.mark.parametrize('url', ['https://127.0.0.1:8790', 'http://localhost:8790',
    'http://127.1:8790', 'http://127.0.0.2:8790', 'http://192.168.1.2:8790',
    'http://example.com:8790', 'http://127.0.0.1', 'http://127.0.0.1:0',
    'http://127.0.0.1:65536', 'http://127.0.0.1:08790', 'http://127.0.0.1:8790/@evil',
    'http://user@127.0.0.1:8790', 'http://127.0.0.1:8790?x=1', 'http://127.0.0.1:8790#x',
    'http://127.0.0.1:8790/</script>'])
def test_launch_url_must_be_plain_explicit_loopback_origin(tmp_path, url):
    assert ready.validate_ready_request(request(tmp_path, url=url), tmp_path, now=1000.) is None


@pytest.mark.parametrize('callback', ['http://example.com:53177/' + TOKEN,
    'http://127.0.0.1:53177/wrong', 'http://127.0.0.1:53177/%61' + TOKEN[1:],
    'http://127.0.0.1:53177/' + TOKEN + '?secret=1', 'http://127.0.0.1:53177/' + TOKEN + '#x',
    'file:///tmp/' + TOKEN, 'http://127.0.0.1:53177/' + TOKEN + '/more',
    'http://user:pass@127.0.0.1:53177/' + TOKEN])
def test_callback_cannot_change_host_protocol_path_or_add_data(tmp_path, callback):
    assert ready.validate_ready_request(request(tmp_path, callback=callback), tmp_path, now=1000.) is None


def test_script_requires_parent_launch_token_and_settled_app_before_fetch(tmp_path):
    parsed = ready.validate_ready_request(request(tmp_path), tmp_path, now=1000.)
    script = ready._ready_html(parsed)
    assert 'requestAnimationFrame' not in script
    assert script.index('tokens[0] === launch.token') < script.index('async function check()')
    assert script.index('querySelector(\'[data-testid="stApp"]\')') < script.index('fetch(launch.callback')
    for phrase in ('parentURL.origin === expected.origin', 'getAll("_liushao_launch")',
                   'tokens.length === 1', 'Date.now() / 1000 < launch.expires',
                   'if (!app || !app.isConnected) return false', 'mode: "cors"',
                   'cache: "no-store"', 'credentials: "omit"'):
        assert phrase in script
    assert 'localStorage' not in script and 'sessionStorage' not in script
    assert 'document.cookie' not in script


def test_transient_rendering_and_network_failures_retry_only_until_launch_expires(tmp_path):
    script = ready._ready_html(ready.validate_ready_request(request(tmp_path), tmp_path, now=1000.))
    for phrase in ('if (matchesParent()) window.setTimeout(check, 250)',
                   'if (!matchesParent()) return', 'if (!current || !stable)',
                   'app.getAttribute("data-test-script-state") !== "notRunning"',
                   'if (!response.ok) throw', 'controller.abort(), 2000',
                   'window.clearTimeout(timeout)'):
        assert phrase in script
    assert 'stable = false;\n      retry();' in script


def test_absent_malformed_or_oversized_requests_do_not_render_any_iframe(tmp_path, monkeypatch):
    calls=[]; monkeypatch.setattr(ready, 'component_html', lambda *a, **k: calls.append((a,k)))
    assert ready.render_desktop_ready(tmp_path) is False
    path=tmp_path/'demo-state'/'frontend-launch.json';path.parent.mkdir()
    for text in ('{', '[]', json.dumps(request(tmp_path, root='wrong')), ' ' * 4097):
        path.write_text(text, encoding='utf-8')
        assert ready.render_desktop_ready(tmp_path) is False
        assert path.read_text(encoding='utf-8') == text
    assert calls == []


def test_valid_request_renders_zero_size_html_and_never_changes_launch_file(tmp_path, monkeypatch):
    calls=[]; monkeypatch.setattr(ready, 'component_html', lambda *a, **k: calls.append((a,k)))
    monkeypatch.setattr(ready.time, 'time', lambda: 1000.)
    path=tmp_path/'demo-state'/'frontend-launch.json';path.parent.mkdir()
    path.write_text(json.dumps(request(tmp_path)),encoding='utf-8');before=path.read_bytes()
    assert ready.render_desktop_ready(tmp_path) is True
    assert path.read_bytes() == before
    assert len(calls)==1 and calls[0][1] == dict(height=0, width=0, scrolling=False)
    assert 'fetch(launch.callback' in calls[0][0][0]


def test_root_comparison_is_exact_not_a_prefix(tmp_path):
    assert ready.validate_ready_request(request(tmp_path, root=str(tmp_path.resolve())+'-other'), tmp_path, now=1000.) is None
    assert ready.validate_ready_request(request(tmp_path, root=str(tmp_path.resolve())+'/..'), tmp_path, now=1000.) is None



def test_readiness_requires_a_connected_sized_main_title_and_no_app_exception(tmp_path):
    parsed = ready.validate_ready_request(request(tmp_path), tmp_path, now=1000.)
    script = ready._ready_html(parsed)
    callback = script.index('fetch(launch.callback')
    for gate in ('querySelector(\'[data-testid="stMain"] h1\')',
                 'querySelector(\'[data-testid="stException"]\')',
                 'if (!title || !title.isConnected || exception) return false',
                 'title.getBoundingClientRect()',
                 'return bounds.width > 0 && bounds.height > 0'):
        assert script.index(gate) < callback
    assert 'visibilityState' not in script
