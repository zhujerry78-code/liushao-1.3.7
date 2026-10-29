"""One-shot positioning for the review / evidence-detail round trip."""
from __future__ import annotations

import json
from uuid import uuid4

from streamlit.components.v1 import html as component_html


_KEY = '_detail_navigation_scroll'
_DESTINATIONS = {'evidence_detail': 'top', 'review': 'record'}


def queue_navigation_scroll(state, destination):
    if destination in _DESTINATIONS:
        state[_KEY] = dict(page=destination, target=_DESTINATIONS[destination], token=uuid4().hex)


def _scroll_html(request):
    # All values come from the fixed destination map and a generated token.
    data = json.dumps(request, ensure_ascii=True, separators=(',', ':'))
    return '''<!doctype html><html><body style="margin:0"><script>
(() => {
  "use strict";
  const request = ''' + data + ''';
  let doc;
  try { doc = window.parent.document; } catch (_) { return; }
  if (window.parent === window) return;
  window.parent.__liushaoDetailScroll = request.token;
  let frames = 0, cancelled = false;
  const events = ['wheel', 'touchstart', 'pointerdown', 'keydown'];
  function cleanup() { events.forEach(name => doc.removeEventListener(name, cancel, true)); }
  function cancel() { cancelled = true; cleanup(); }
  events.forEach(name => doc.addEventListener(name, cancel, {capture:true, once:true}));
  function position() {
    if (cancelled || window.parent.__liushaoDetailScroll !== request.token || ++frames > 180) {
      cleanup(); return;
    }
    const app = doc.querySelector('[data-testid="stApp"]');
    const main = doc.querySelector('[data-testid="stMain"]');
    const title = main && main.querySelector('h1');
    const expected = request.page === 'evidence_detail' ? '评分依据' : '检查记录';
    if (!app || !main || !title || title.textContent.trim() !== expected ||
        app.getAttribute('data-test-script-state') !== 'notRunning') {
      window.requestAnimationFrame(position); return;
    }
    const record = request.target === 'record' && main.querySelector('.st-key-record_header');
    const top = record ? Math.max(0, main.scrollTop + record.getBoundingClientRect().top - main.getBoundingClientRect().top - 72) : 0;
    cleanup();
    main.scrollTo({top, behavior:'instant'});
  }
  window.requestAnimationFrame(() => window.requestAnimationFrame(position));
})();
</script></body></html>'''


def render_navigation_scroll(state, page):
    """Consume once after rendering; ordinary reruns do not reposition content."""
    request = state.pop(_KEY, None)
    if not isinstance(request, dict) or request.get('page') != page or page not in _DESTINATIONS:
        return False
    if request.get('target') != _DESTINATIONS[page]:
        return False
    token = request.get('token')
    if not isinstance(token, str) or len(token) != 32 or any(ch not in '0123456789abcdef' for ch in token):
        return False
    component_html(_scroll_html(dict(page=page, target=_DESTINATIONS[page], token=token)),
                   height=0, width=0, scrolling=False)
    return True
