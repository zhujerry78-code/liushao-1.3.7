"""Local desktop startup: one verified service per folder, then a browser window."""
from __future__ import annotations

from contextlib import contextmanager
import json
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import secrets
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import uuid
import webbrowser

import psutil
try:
    from .verify_demo import MODEL, DESCRIPTOR
    from .local_lock import file_lock
except ImportError:
    from verify_demo import MODEL, DESCRIPTOR
    from local_lock import file_lock

START_TIMEOUT = 150.0
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def service_ready(url: str) -> bool:
    """Bypass system proxies; a listening socket alone is not a ready web app."""
    try:
        with HTTP.open(url + '/_stcore/health', timeout=1) as response:
            return response.status == 200 and response.read(32).strip() == b'ok'
    except (OSError, ValueError):
        return False


def listener_owned(pid, port):
    try:
        root = psutil.Process(pid)
        for process in [root, *root.children(recursive=True)]:
            if any(c.status == psutil.CONN_LISTEN and c.laddr.ip == '127.0.0.1'
                   and c.laddr.port == port for c in process.net_connections(kind='inet')):
                return True
    except psutil.Error:
        pass
    return False


def recover_session(root):
    candidates = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            if (len(command) < 5 or command[1:5] != ['-m', 'streamlit', 'run', 'app.py']
                    or Path(command[0]).resolve() != (root / '.venv/Scripts/python.exe').resolve()
                    or Path(process.cwd()).resolve() != root.resolve()
                    or '--server.address=127.0.0.1' not in command):
                continue
            ports = [int(a.split('=',1)[1]) for a in command if a.startswith('--server.port=')]
            if len(ports) != 1 or not 1024 <= ports[0] <= 65535:
                raise RuntimeError('发现同目录服务但无法确认端口，请先关闭后重试。')
            candidates.append((process, ports[0]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates: return None
    if len(candidates) != 1:
        raise RuntimeError('发现多个同目录服务，请先关闭后重试。')
    process, port = candidates[0]
    if not listener_owned(process.pid, port):
        raise RuntimeError('发现同目录服务但无法确认监听状态，请稍后重试。')
    state = dict(root=str(root.resolve()), token=uuid.uuid4().hex, port=port,
                 url=f'http://127.0.0.1:{port}', server_pid=process.pid,
                 server_created=process.create_time())
    atomic_json(root / 'demo-state/desktop-session.json', state)
    return state, process


def wait_for_service(server: subprocess.Popen, url: str, timeout: float = START_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError('程序在准备过程中退出，请查看启动日志。')
        if listener_owned(server.pid, int(url.rsplit(':', 1)[1])) and service_ready(url):
            return
        time.sleep(.2)
    raise TimeoutError('程序准备时间过长，请查看启动日志后重试。')


def browser_candidates() -> list[Path]:
    """Use installed browsers directly instead of relying only on URL associations."""
    candidates = []
    for name, tail in [('msedge.exe', 'Microsoft/Edge/Application/msedge.exe'),
                       ('chrome.exe', 'Google/Chrome/Application/chrome.exe')]:
        for variable in ('PROGRAMFILES(X86)', 'PROGRAMFILES', 'LOCALAPPDATA'):
            if base := os.environ.get(variable):
                candidates.append(Path(base) / tail)
        if os.name == 'nt':
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                    try:
                        with winreg.OpenKey(hive, 'Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\' + name,
                                            0, winreg.KEY_READ | view) as key:
                            candidates.append(Path(winreg.QueryValue(key, None)))
                    except OSError:
                        pass
    found = []
    for path in candidates:
        if path.is_file() and path not in found:
            found.append(path)
    return found


def local_app_url(url: str) -> str:
    """A launcher must never send a callback token to a non-local website."""
    try:
        value = urlsplit(url)
        port = value.port
        if (value.scheme != 'http' or value.hostname != '127.0.0.1'
                or port is None or not 1024 <= port <= 65535
                or value.netloc != f'127.0.0.1:{port}' or value.path not in ('', '/')
                or value.query or value.fragment
                or any(ord(c) <= 32 or ord(c) == 127 for c in url)):
            raise ValueError('Only the literal local application URL is allowed')
        return f'http://127.0.0.1:{port}'
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('无效的本机界面地址') from error


def native_profile_id(root: Path) -> str:
    root = Path(root).resolve()
    identity = str(root).casefold()
    # Reuse a profile during one service lifetime. A cold restart gets a fresh
    # browser configuration instead of restoring a terminated browser session.
    try:
        state = json.loads((root / 'demo-state/desktop-session.json').read_text(encoding='utf-8'))
        if isinstance(state, dict) and state.get('root') == str(root) and re.fullmatch('[0-9a-f]{32}', state.get('token', '')):
            identity += '\0' + state['token']
            return hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
    except (OSError, TypeError, ValueError):
        pass
    try:
        previous = json.loads((root / 'demo-state/browser-profile-state.json').read_text(encoding='utf-8'))
        if isinstance(previous, dict) and previous.get('root') == str(root) and re.fullmatch('[0-9a-f]{24}', previous.get('profile_id', '')):
            return previous['profile_id']
    except (OSError, TypeError, ValueError):
        pass
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]


def native_browser_profile(root: Path) -> Path | None:
    base = os.environ.get('LOCALAPPDATA')
    return Path(base) / 'Liushao/BrowserProfiles' / native_profile_id(root) if base else None


class FrontendReady:
    """One launch, one loopback receiver, acknowledged only by browser JS."""

    def __init__(self, root: Path, url: str, timeout: float = 60.0):
        self.root = Path(root).resolve()
        self.url = local_app_url(url)
        self.timeout = min(START_TIMEOUT, max(.01, float(timeout)))
        self.token = secrets.token_hex(32)
        self.launch_url = self.url + '/?_liushao_launch=' + self.token
        self.request_path = self.root / 'demo-state/frontend-launch.json'
        self.event = threading.Event()
        self.expires = 0.0
        self.server = None
        self.thread = None

    def __enter__(self):
        owner = self

        class Receiver(BaseHTTPRequestHandler):
            def do_GET(self):
                expected_host = f'127.0.0.1:{self.server.server_port}'
                if self.client_address[0] != '127.0.0.1' or self.headers.get('Host') != expected_host:
                    self.send_error(403)
                    return
                if not hmac.compare_digest(self.path, '/' + owner.token):
                    self.send_error(404)
                    return
                if time.time() >= owner.expires:
                    self.send_error(410)
                    return
                if self.headers.get('Origin') not in (None, owner.url):
                    self.send_error(403)
                    return
                referer = self.headers.get('Referer')
                if referer:
                    parsed = urlsplit(referer)
                    if f'{parsed.scheme}://{parsed.netloc}' != owner.url:
                        self.send_error(403)
                        return
                receipt = dict(root=str(owner.root), url=owner.url, token=owner.token,
                               ready_at=time.time(), source='browser-rendered-callback')
                atomic_json(owner.root / 'demo-state/browser-last-ready.json', receipt)
                self.send_response(204)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', owner.url)
                self.end_headers()
                owner.event.set()

            def log_message(self, *args):
                pass

        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
        self.server.daemon_threads = True
        self.callback = f'http://127.0.0.1:{self.server.server_port}/{self.token}'
        self.expires = time.time() + self.timeout
        atomic_json(self.request_path, dict(root=str(self.root), url=self.url, token=self.token,
                                            callback=self.callback, expires=self.expires))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .1}, daemon=True)
        self.thread.start()
        return self

    def wait(self, timeout=None):
        if not self.event.wait(self.timeout if timeout is None else timeout):
            raise TimeoutError('浏览器尚未确认页面就绪，不能记为启动成功。请关闭空白页后重试；启动日志已保留。')

    def __exit__(self, *args):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)
        try:
            if json.loads(self.request_path.read_text(encoding='utf-8')).get('token') == self.token:
                self.request_path.unlink()
        except (OSError, ValueError):
            pass


def open_interface(url: str, *, profile: Path | None = None, root: Path | None = None,
                   timeout: float = 60.0) -> str:
    """Use the standard OS browser activation, keeping its normal sandbox."""
    url = local_app_url(url)

    def activate(target):
        if os.name == 'nt':
            os.startfile(target)
        elif not webbrowser.open(target, new=1):
            raise RuntimeError('未能打开系统浏览器，请确认默认浏览器已安装。')

    # profile remains accepted for old callers. Neither it nor the user's normal
    # browser profile is modified or deleted by a system URL activation.
    if root is None:
        activate(url)
        return 'Windows URL association (request sent)'
    activation_id = os.environ.get('LIUSHAO_BROWSER_ACTIVATION_ID', '')
    delegated = re.fullmatch(r'[0-9]{1,5}(?:-[0-9]{1,5}){3}', activation_id) is not None
    request_path = None
    temporary = None
    published = None
    profile_request = None
    profile_id = None
    try:
        with FrontendReady(root, url, timeout=timeout) as ready:
            if delegated:
                # The original desktop CMD opens the URL. A Low-labelled Python
                # must not be the parent of the browser's sandbox broker.
                request_path = Path(root) / 'demo-state' / f'browser-open-{activation_id}.url.txt'
                profile_request = request_path.with_name(f'browser-open-{activation_id}.profile-id.txt')
                profile_id = native_profile_id(root) + '\n'
                atomic_json(Path(root) / 'demo-state/browser-profile-state.json',
                    dict(root=str(Path(root).resolve()), profile_id=profile_id.strip()))
                temporary = profile_request.with_name(profile_request.name + '.' + uuid.uuid4().hex + '.tmp')
                temporary.write_text(profile_id, encoding='ascii')
                temporary.replace(profile_request)
                temporary = request_path.with_name(request_path.name + '.' + uuid.uuid4().hex + '.tmp')
                published = ready.launch_url + '\n'
                temporary.write_text(published, encoding='ascii')
                temporary.replace(request_path)
            else:
                activate(ready.launch_url)
            ready.wait()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if request_path is not None:
            try:
                if request_path.read_text(encoding='ascii') == published:
                    request_path.unlink()
            except (OSError, UnicodeError):
                pass
        if profile_request is not None:
            try:
                if profile_request.read_text(encoding='ascii') == profile_id:
                    profile_request.unlink()
            except (OSError, UnicodeError):
                pass
    return 'Windows URL association (frontend confirmed)'


def free_port(requested: int) -> int:
    for port in dict.fromkeys([requested, *range(8501, 8521)]):
        if not 1024 <= port <= 65535:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('可用的本机端口已被占用，请关闭其他流哨窗口后重试。')


@contextmanager
def startup_lock(runtime: Path, timeout: float = START_TIMEOUT):
    import msvcrt
    with (runtime / 'startup.lock').open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        deadline = time.monotonic() + timeout
        while True:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('已有启动操作正在进行，请稍后重试。')
                time.sleep(.2)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def owned_session(root: Path) -> tuple[dict, psutil.Process] | None:
    """Never reuse/stop another folder's service, or a PID reused by Windows."""
    try:
        state = json.loads((root / 'demo-state/desktop-session.json').read_text(encoding='utf-8'))
        if Path(state['root']).resolve() != root.resolve():
            return None
        port = int(state['port'])
        if not 1024 <= port <= 65535 or state['url'] != f'http://127.0.0.1:{port}':
            return None
        process = psutil.Process(state['server_pid'])
        command = process.cmdline()
        if (abs(process.create_time() - state['server_created']) > .01
                or Path(process.cwd()).resolve() != root.resolve()
                or not command or Path(command[0]).resolve() != (root / '.venv/Scripts/python.exe').resolve()
                or command[1:5] != ['-m', 'streamlit', 'run', 'app.py']
                or '--server.address=127.0.0.1' not in command
                or f'--server.port={port}' not in command):
            return None
        return state, process
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return None


def stop_process_tree(process: psutil.Process) -> None:
    """The Windows virtualenv redirector can have a separate Python child."""
    try:
        targets = list(reversed(process.children(recursive=True))) + [process]
    except psutil.NoSuchProcess:
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=8)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)


def remove_session(root: Path, token: str) -> None:
    path = root / 'demo-state/desktop-session.json'
    try:
        if json.loads(path.read_text(encoding='utf-8')).get('token') == token:
            path.unlink()
    except (OSError, ValueError):
        pass


def close_app_browser(root: Path) -> None:
    profiles = [root / 'demo-state/browser-profile']
    if profile := native_browser_profile(root):
        profiles.append(profile)
    profile_arguments = {'--user-data-dir=' + str(path) for path in profiles}
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            if (process.info['name'].lower() in ('msedge.exe', 'chrome.exe')
                    and profile_arguments.intersection(process.info['cmdline'] or [])):
                stop_process_tree(process)
        except (AttributeError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def open_or_pending(root, url, server_pid, timeout, *, open_browser=True):
    if not open_browser:
        return dict(browser='headless', frontend_ready=False, state='frontend_pending')
    try:
        browser = open_interface(url, root=root, timeout=timeout)
        return dict(browser=browser, frontend_ready=True, state='ready')
    except (TimeoutError, OSError, RuntimeError) as error:
        existing = owned_session(root)
        if (not existing or existing[0]['server_pid'] != server_pid
                or not listener_owned(server_pid, existing[0]['port']) or not service_ready(url)):
            raise
        return dict(browser=str(error), frontend_ready=False, state='frontend_pending')


def desktop_worker(root: Path, requested: int, result: Path, verify, deadline=None, *, open_browser=True) -> int:
    server = None
    token = uuid.uuid4().hex
    runtime = root / 'demo-state'
    deadline = deadline or time.monotonic() + START_TIMEOUT
    try:
        with startup_lock(runtime, max(.01, deadline-time.monotonic())), file_lock(runtime / 'install.lock', max(.01, deadline-time.monotonic())):
            verify(root)
            existing = owned_session(root) or recover_session(root)
            if existing and listener_owned(existing[1].pid, existing[0]['port']) and service_ready(existing[0]['url']):
                state = existing[0]
                frontend = open_or_pending(root, state['url'], state['server_pid'],
                    min(60.0, max(.01, deadline-time.monotonic())), open_browser=open_browser)
                atomic_json(result, dict(ok=True, reused=True, url=state['url'],
                                         server_pid=state['server_pid'], **frontend))
                return 0
            if existing:
                # A known process that no longer answers may still be recovering.
                # Do not kill it or spawn a second service behind its active page.
                raise RuntimeError('当前程序暂时没有响应。请保存已有填写，双击“4_关闭流哨.cmd”后再启动。')
            for attempt in range(3):
                try:
                    port = free_port(requested)
                    url = f'http://127.0.0.1:{port}'
                    env = {**os.environ, 'PYTHONUTF8': '1',
                           'LIUSHAO_ARTIFACTS_DIR': str(root / 'artifacts'),
                           'LIUSHAO_EVALUATION_SUMMARY': str(root / 'resources/evaluation/summary.json'),
                           'LIUSHAO_REVIEW_DB': os.environ.get('LIUSHAO_REVIEW_DB', str(runtime / 'reviews.sqlite')),
                           'LIUSHAO_EXPECTED_MODEL_SHA256': MODEL,
                           'LIUSHAO_EXPECTED_DESCRIPTOR_SHA256': DESCRIPTOR}
                    server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py',
                        '--server.address=127.0.0.1', f'--server.port={port}',
                        '--server.headless=true', '--browser.gatherUsageStats=false'],
                        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                    wait_for_service(server, url, max(.01, deadline-time.monotonic()))
                    break
                except (RuntimeError, TimeoutError):
                    if server is not None and server.poll() is None:
                        stop_process_tree(psutil.Process(server.pid))
                    if attempt == 2 or time.monotonic() >= deadline:
                        raise
                    requested = requested + 1
            process = psutil.Process(server.pid)
            atomic_json(runtime / 'desktop-session.json', dict(root=str(root.resolve()), token=token,
                port=port, url=url, server_pid=server.pid, server_created=process.create_time()))
            frontend = open_or_pending(root, url, server.pid,
                min(60.0, max(.01, deadline-time.monotonic())), open_browser=open_browser)
            atomic_json(result, dict(ok=True, reused=False, url=url, server_pid=server.pid, **frontend))
            print(f'Service ready: {url}; frontend={frontend["state"]}', flush=True)
        return server.wait()
    except Exception as error:
        print(f'Startup failed: {type(error).__name__}: {error}', flush=True)
        atomic_json(result, dict(ok=False, state='failed', message=str(error)))
        return 1
    finally:
        if server is not None:
            if server.poll() is None:
                try:
                    stop_process_tree(psutil.Process(server.pid))
                except psutil.NoSuchProcess:
                    pass
            remove_session(root, token)


def show_error(message: str) -> None:
    if os.name == 'nt' and os.environ.get('LIUSHAO_NONINTERACTIVE') != '1':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, '流哨未能启动', 0x10)


def launch_desktop(root: Path, requested: int) -> int:
    """The CMD exits once the verified worker has opened the ready interface."""
    runtime = root / 'demo-state'
    try:
        logs = runtime / 'launch-logs'
        logs.mkdir(parents=True, exist_ok=True)
        key = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        log_path = logs / (key + '.log')
        result = logs / (key + '.json')
        deadline = time.monotonic() + START_TIMEOUT
        worker = None
        with log_path.open('w', encoding='utf-8') as log:
            worker = subprocess.Popen([sys.executable, '-X', 'utf8', str(root / 'delivery/run_demo.py'),
                '--desktop-worker', '--port', str(requested), '--result-file', str(result),
                '--startup-deadline', str(deadline)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        while time.monotonic() < deadline:
            if result.exists():
                status = json.loads(result.read_text(encoding='utf-8'))
                if status.get('ok'):
                    if not status.get('frontend_ready'):
                        message = '服务已就绪，窗口未确认。可重新打开窗口，或再次运行「2_启动流哨.cmd」复用当前服务。\n' + status['url']
                        print(message, flush=True)
                        if os.name == 'nt' and os.environ.get('LIUSHAO_NONINTERACTIVE') != '1':
                            import ctypes
                            if ctypes.windll.user32.MessageBoxW(None, message + '\n\n重新打开窗口？', '流哨服务运行中', 4) == 6:
                                os.startfile(local_app_url(status['url']))
                        return 0
                    print('流哨界面已加载完成，此窗口将自动关闭。', flush=True)
                    return 0
                raise RuntimeError(status.get('message', '启动失败，请查看日志。'))
            if worker.poll() is not None:
                raise RuntimeError('启动过程意外结束，请查看日志。')
            time.sleep(.15)
        stop_process_tree(psutil.Process(worker.pid))
        raise TimeoutError('启动等待时间过长，请稍后重试。')
    except Exception as error:
        if 'worker' in locals() and worker is not None and worker.poll() is None:
            try:
                stop_process_tree(psutil.Process(worker.pid))
            except psutil.NoSuchProcess:
                pass
        message = str(error)
        if 'log_path' in locals():
            message += f'\n\n启动日志：{log_path}'
        print(message, flush=True)
        show_error(message)
        return 1


def stop_desktop(root: Path) -> int:
    runtime = root / 'demo-state'
    runtime.mkdir(exist_ok=True)
    with startup_lock(runtime):
        existing = owned_session(root) or recover_session(root)
        if not existing:
            close_app_browser(root)
            print('当前文件夹没有正在运行的流哨。', flush=True)
            return 0
        state, process = existing
        stop_process_tree(process)
        close_app_browser(root)
        remove_session(root, state['token'])
    print('流哨已关闭，已保存的记录仍保留在本机。', flush=True)
    return 0
