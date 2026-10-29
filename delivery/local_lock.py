"""Standard-library Windows lock, usable before dependencies are installed."""
from contextlib import contextmanager
import os
from pathlib import Path
import time
import subprocess


@contextmanager
def file_lock(path: Path, timeout=600):
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('已有安装或启动正在进行，请稍后重试。')
                time.sleep(.2)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def active_service(root):
    """Compatibility boolean; an unknown state is never treated as stopped."""
    return bool(service_state(root))


def service_state(root):
    try:
        from .service_probe import native_service_state, ServiceState
    except ImportError:
        from service_probe import native_service_state, ServiceState
    return native_service_state(root)


def check_supported_path(root):
    root = Path(root).resolve()
    text = str(root)
    if text.startswith('\\\\'):
        raise RuntimeError('本地交付暂不支持 UNC 网络目录，请使用本机可写目录。')
    if len(text) > 120:
        raise RuntimeError(f'安装目录有 {len(text)} 个字符，超过 120 字符上限；请解压到较短的本机可写目录（建议不超过 100 字符）。')
    if os.name == 'nt':
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(root.anchor)) == 4:
            raise RuntimeError('不支持映射网络盘，请使用本机可写目录。')
    for key, value in os.environ.items():
        if key.upper().startswith('ONEDRIVE') and value and root.is_relative_to(Path(value).resolve()):
            raise RuntimeError('请将安装目录迁出 OneDrive 同步目录后再安装。')
    if len(text) > 100:
        print(f'警告：安装目录有 {len(text)} 个字符，超过建议的 100 字符，仍允许继续（最长 120 字符）。'
              '较深的依赖路径可能受 Windows 限制，建议安装前解压到更短的本机可写目录。', flush=True)
