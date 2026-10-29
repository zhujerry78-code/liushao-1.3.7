"""Browser-executed acknowledgement for one short-lived local desktop launch.

This helper only reads a launch request.  A Python health check never counts as
browser readiness: the callback is sent exclusively by the rendered iframe JS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from streamlit.components.v1 import html as component_html


_TOKEN = re.compile(r'[0-9a-fA-F]{64}\Z')
_MAX_REQUEST_BYTES = 4096


def _loopback_http(value, *, path=None):
    if not isinstance(value, str) or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
                or type(port) is not int or not 1 <= port <= 65535
                or parsed.netloc != f'127.0.0.1:{port}'
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            return None
        if path is None:
            if parsed.path not in ('', '/'):
                return None
        elif parsed.path != path:
            return None
        return f'http://127.0.0.1:{port}' + (path or '')
    except (TypeError, ValueError):
        return None


def validate_ready_request(request, root, *, now=None):
    """Return only constrained browser fields, or None for any invalid request."""
    if not isinstance(request, dict):
        return None
    try:
        if request.get('root') != str(Path(root).resolve()):
            return None
        token = request.get('token')
        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            return None
        expires = request.get('expires')
        current = time.time() if now is None else now
        if (type(expires) not in (int, float) or not math.isfinite(expires)
                or type(current) not in (int, float) or not math.isfinite(current)
                or expires <= current):
            return None
        url = _loopback_http(request.get('url'))
        callback = _loopback_http(request.get('callback'), path='/' + token)
        if url is None or callback is None:
            return None
        return dict(url=url, callback=callback, token=token, expires=expires)
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _ready_html(request):
    # Only validate_ready_request output reaches this function.  JSON encoding
    # also protects the script boundary if future display fields are added.
    data = json.dumps(request, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    data = data.replace('<', r'\u003c').replace('>', r'\u003e').replace('&', r'\u0026')
    return """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;width:0;height:0;overflow:hidden}</style>
</head><body><script>
(() => {
  "use strict";
  const launch = """ + data + """;
  const expected = new URL(launch.url);
  function matchesParent() {
    try {
      const parentURL = new URL(window.parent.location.href);
      const tokens = parentURL.searchParams.getAll("_liushao_launch");
      return window.parent !== window && parentURL.origin === expected.origin &&
        tokens.length === 1 && tokens[0] === launch.token && Date.now() / 1000 < launch.expires;
    } catch (_) { return false; }
  }
  // Off-screen, zero-size component frames must not depend on animation frames.
  // Require a settled app on two checks; retry transient rendering/network races
  // only while this exact launch request is still valid.
  let stable = false;
  function rendered() {
      let app, title, exception;
      try {
        const doc = window.parent.document;
        app = doc.querySelector('[data-testid="stApp"]');
        title = doc.querySelector('[data-testid="stMain"] h1');
        exception = doc.querySelector('[data-testid="stException"]');
      } catch (_) { return false; }
      if (!app || !app.isConnected) return false;
      if (app.getAttribute("data-test-script-state") !== "notRunning") return false;
      if (!title || !title.isConnected || exception) return false;
      const bounds = title.getBoundingClientRect();
      return bounds.width > 0 && bounds.height > 0;
  }
  function retry() {
    if (matchesParent()) window.setTimeout(check, 250);
  }
  async function check() {
    if (!matchesParent()) return;
    const current = rendered();
    if (!current || !stable) {
      stable = current;
      retry();
      return;
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 2000);
    try {
      const response = await fetch(launch.callback, {method: "GET", mode: "cors",
        cache: "no-store", credentials: "omit", signal: controller.signal});
      if (!response.ok) throw new Error("Readiness was not accepted");
    } catch (_) {
      stable = false;
      retry();
    } finally {
      window.clearTimeout(timeout);
    }
  }
  check();
})();
</script></body></html>"""


def render_desktop_ready(root):
    """Render a zero-size acknowledgement iframe only for a valid launch file."""
    try:
        root = Path(root).resolve()
        path = root / 'demo-state' / 'frontend-launch.json'
        if not path.is_file() or path.stat().st_size > _MAX_REQUEST_BYTES:
            return False
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        request = validate_ready_request(raw, root)
    except (OSError, TypeError, ValueError, OverflowError):
        return False
    if request is None:
        return False
    component_html(_ready_html(request), height=0, width=0, scrolling=False)
    return True
