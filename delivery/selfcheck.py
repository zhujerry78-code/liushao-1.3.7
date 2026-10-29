"""环境自检：把安装/运行情况写成一份报告，并打包成可发送的诊断包。

只使用标准库，Python 3.12 即可运行；缺少任何组件都不会中断，缺失项会写进报告。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile
from uuid import uuid4
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = '-' * 64


def say(message: str = '') -> None:
    print(message, flush=True)


def run(command: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=timeout, cwd=str(ROOT))
    except (OSError, subprocess.SubprocessError) as error:
        return 1, f'无法执行：{error}'
    return completed.returncode, completed.stdout.decode('utf-8', errors='replace').strip()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def registry_value(root, key: str, name: str):
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(root, key) as handle:
            return winreg.QueryValueEx(handle, name)[0]
    except OSError:
        return None


def python_candidates() -> list[tuple[str, str]]:
    """按一键脚本的顺序列出可能的解释器。"""
    candidates: list[tuple[str, list[str]]] = []
    if os.environ.get('LIUSHAO_PYTHON'):
        candidates.append(('环境变量 LIUSHAO_PYTHON', [os.environ['LIUSHAO_PYTHON']]))
    candidates.append(('py -3.12', ['py', '-3.12']))
    candidates.append(('本目录 .venv', [str(ROOT / '.venv' / 'Scripts' / 'python.exe')]))
    found = []
    for label, command in candidates:
        executable = command[0]
        if os.sep in executable and not Path(executable).is_file():
            found.append((label, '（不存在）'))
            continue
        if os.sep in executable:
            found.append((label, executable))
            continue
        code, output = run([executable, '-c', 'import sys;print(sys.executable)'], timeout=30)
        found.append((label, output.splitlines()[-1] if code == 0 and output else '（不可用）'))
    for label, path, version in (
            ('HKCU 注册表', None, '3.12'), ('HKLM 注册表', None, '3.12')):
        try:
            import winreg
        except ImportError:
            break
        hives = [winreg.HKEY_CURRENT_USER] if label.startswith('HKCU') else [winreg.HKEY_LOCAL_MACHINE]
        for hive in hives:
            value = registry_value(hive, rf'Software\Python\PythonCore\{version}\InstallPath',
                                   'ExecutablePath')
            if value:
                found.append((label, str(value)))
    for path in (Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Python/Python312/python.exe',
                 Path('C:/Python312/python.exe'), Path('C:/Program Files/Python312/python.exe')):
        if path.is_file():
            found.append(('常见安装目录', str(path)))
    which = shutil.which('python')
    found.append(('PATH 中的 python', which or '（不存在）'))
    return found


def python_details(path: str) -> str:
    if not Path(path).is_file():
        return '（不可用）'
    code, output = run([path, '-c',
                        'import sys,platform;print("版本:",sys.version.replace(chr(10)," "));'
                        'print("位数:", 64 if sys.maxsize>2**32 else 32);'
                        'print("位置:", sys.executable)'], timeout=60)
    return output if code == 0 else f'（启动失败）{output}'


def directory_state(path: Path) -> str:
    if not path.exists():
        return '不存在'
    if not path.is_dir():
        return '存在但不是文件夹'
    files = [p for p in path.rglob('*') if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return f'存在，{len(files)} 个文件，{total / 1024 ** 2:.1f} MB'


def venv_state() -> list[str]:
    environment = ROOT / '.venv'
    notes = [f'目录状态：{directory_state(environment)}']
    config = environment / 'pyvenv.cfg'
    if config.is_file():
        notes.append('pyvenv.cfg: ' + config.read_text(encoding='utf-8', errors='replace').replace('\n', ' | '))
    python = environment / 'Scripts' / 'python.exe'
    if python.is_file():
        code, output = run([str(python), '-m', 'pip', '--version'], timeout=90)
        notes.append(f'pip: {output if code == 0 else "不可用 " + output}')
        code, output = run([str(python), '-m', 'pip', 'check'], timeout=180)
        notes.append(f'pip check (退出码 {code}): {output}')
        code, output = run([str(python), '-X', 'utf8', str(ROOT / 'delivery' / 'setup_demo.py'),
                            '--check-environment'], timeout=180)
        notes.append(f'锁定依赖核对 (退出码 {code}): {output}')
    else:
        notes.append('缺少 .venv\\Scripts\\python.exe（环境不完整）')
    return notes


def wheels_state() -> list[str]:
    directory = ROOT / 'wheels'
    if not directory.is_dir():
        return ['没有 wheels 目录（本包为在线安装模式）']
    wheels = sorted(directory.glob('*.whl'))
    notes = [f'{len(wheels)} 个 wheel，合计 {sum(p.stat().st_size for p in wheels) / 1024 ** 2:.1f} MB']
    sums = directory / 'SHA256SUMS.txt'
    if not sums.is_file():
        notes.append('缺少 wheels\\SHA256SUMS.txt')
        return notes
    expected = {}
    for line in sums.read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, _, name = line.partition('  ')
            expected[name.strip()] = digest.strip().lower()
    missing, changed = [], []
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file():
            missing.append(name)
        elif sha256_of(path) != digest:
            changed.append(name)
    notes.append(f'校验：应到 {len(expected)} 个，缺失 {len(missing)}，内容不符 {len(changed)}')
    if missing:
        notes.append('缺失：' + ', '.join(missing[:8]))
    if changed:
        notes.append('不符：' + ', '.join(changed[:8]))
    return notes


def manifest_state() -> list[str]:
    path = ROOT / 'PACKAGE_MANIFEST.json'
    if not path.is_file():
        return ['缺少 PACKAGE_MANIFEST.json']
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        return [f'无法读取清单：{error}']
    files = manifest.get('files') or {}
    missing, changed = [], []
    for name, entry in files.items():
        target = ROOT / name
        if not target.is_file():
            missing.append(name)
        elif sha256_of(target) != entry.get('sha256'):
            changed.append(name)
    notes = [f'清单记录 {len(files)} 个文件；缺失 {len(missing)}，内容不符 {len(changed)}']
    if missing:
        notes.append('缺失：' + ', '.join(missing[:8]))
    if changed:
        notes.append('不符：' + ', '.join(changed[:8]))
    notes.append(f'清单自身 SHA-256：{sha256_of(path)}')
    return notes


def port_state() -> list[str]:
    notes = []
    for port in range(8501, 8511):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                notes.append(f'端口 {port} 已被占用')
    return notes or ['8501-8510 端口均空闲']


def temp_state() -> str:
    directory = tempfile.gettempdir()
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=True):
            return f'可写：{directory}'
    except OSError as error:
        return f'不可写：{directory}（{error}）'


def long_path_state() -> str:
    try:
        import winreg
    except ImportError:
        return '非 Windows，不适用'
    value = registry_value(winreg.HKEY_LOCAL_MACHINE,
                           r'SYSTEM\CurrentControlSet\Control\FileSystem', 'LongPathsEnabled')
    return {1: '已启用', 0: '未启用', None: '未设置'}.get(value, str(value))


def log_state() -> list[str]:
    notes = []
    for name in ('install-logs', 'demo-state'):
        directory = ROOT / name
        if not directory.is_dir():
            notes.append(f'{name}：尚未生成')
            continue
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
        notes.append(f'{name}：{len(entries)} 项')
        for entry in entries[-12:]:
            if entry.is_file():
                notes.append(f'    {entry.name}  {entry.stat().st_size} 字节')
            else:
                notes.append(f'    {entry.name}\\')
    return notes


def desktop_dir() -> Path:
    try:
        import winreg
    except ImportError:
        return Path.home() / 'Desktop'
    for hive in (winreg.HKEY_CURRENT_USER,):
        value = registry_value(hive, r'Software\Microsoft\Windows\CurrentVersion\Explorer'
                                     r'\User Shell Folders', 'Desktop')
        if value:
            return Path(os.path.expandvars(value))
    return Path.home() / 'Desktop'


def runtime_state() -> list[str]:
    """The bundled interpreter is what the installer uses; report its health."""
    directory = ROOT / 'runtime' / 'python'
    python = directory / 'python.exe'
    if not python.is_file():
        return ['缺失：runtime\\python\\python.exe（重新完整解压原包）']
    files = [p for p in directory.rglob('*') if p.is_file()]
    notes = [f'存在：{len(files)} 个文件，{sum(p.stat().st_size for p in files) / 1024 ** 2:.1f} MB']
    code, output = run([str(python), '-c',
                        'import sys;print(sys.version.split()[0], 64 if sys.maxsize>2**32 else 32)'],
                       timeout=60)
    notes.append(f'版本与位数：{output if code == 0 else "无法运行 " + output}')
    for module in ('ssl', 'sqlite3', 'decimal', 'ctypes', 'lzma', 'zoneinfo'):
        code, _ = run([str(python), '-c', f'import {module}'], timeout=60)
        if code != 0:
            notes.append(f'标准库缺失：{module}')
    notes.append(f'python.exe SHA-256：{sha256_of(python)}')
    return notes


def path_state() -> list[str]:
    text = str(ROOT)
    notes = [f'路径长度：{len(text)} 个字符']
    notes.append('含非 ASCII 字符：' + ('是（一般可用，出问题时可换到纯英文路径）'
                                      if any(ord(ch) > 127 for ch in text) else '否'))
    notes.append('含空格：' + ('是' if ' ' in text else '否'))
    for mark in ('&', '%', '!', '^'):
        if mark in text:
            notes.append(f'注意：路径含特殊字符 {mark}（如异常请换目录）')
    return notes


def local_plain_file(path):
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()): return False
    for parent in [path, *path.parents]:
        if parent.is_symlink() or parent.is_junction(): return False
        if parent == ROOT: break
    return True


def latest_log_tail(lines: int = 12) -> list[str]:
    directory = ROOT / 'install-logs'
    if not directory.is_dir():
        return ['没有 install-logs']
    logs = sorted((p for p in directory.glob('install-*.log') if local_plain_file(p)),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        return ['install-logs 为空']
    newest = logs[-1]
    notes = [f'{newest.name}（{newest.stat().st_size} 字节）最后 {lines} 行：']
    try:
        with newest.open('rb') as stream:
            stream.seek(max(0, newest.stat().st_size - 65536))
            content = stream.read(65536).decode('utf-8', errors='replace').splitlines()
    except OSError as error:
        return notes + [f'读取失败：{error}']
    notes += ['    ' + line for line in content[-lines:]]
    return notes


def build_report() -> str:
    lines = ['流哨环境自检报告', f'生成时间：{datetime.now().astimezone().isoformat()}', LINE,
             f'系统：{platform.platform()}；Python：{sys.version}', f'安装目录：{ROOT}']
    checks = [('临时目录', temp_state), ('Python', lambda: python_details(sys.executable)),
              ('安装环境', venv_state), ('离线依赖', wheels_state), ('软件文件', manifest_state),
              ('端口', port_state), ('日志', log_state), ('内置运行时', runtime_state),
              ('路径', path_state), ('最近安装日志', latest_log_tail),
              ('软件版本', lambda: (ROOT / 'VERSION').read_text(encoding='utf-8').strip())]
    for label, check in checks:
        lines.append('[' + label + ']')
        try:
            result = check()
            lines.extend(result if isinstance(result, list) else [str(result)])
        except Exception as error:
            lines.append(f'检查未完成：{type(error).__name__}: {error}')
    lines.append('仅收集本机排查信息；不包含数据库、上传数据或浏览器资料。')
    return '\n'.join(lines)


def build_archive(report: str, report_path: Path) -> Path:
    destination = report_path.with_name(report_path.stem.replace('诊断报告', '诊断包') + '.zip')
    limit, total, count = 2*1024*1024, 0, 1
    names = ['VERSION', 'PACKAGE_MANIFEST.json', 'SHA256SUMS.txt', 'requirements-lock.txt', '.venv/pyvenv.cfg', 'wheels/SHA256SUMS.txt']
    logs = []
    for directory, pattern in [('install-logs', 'install-*.log'), ('install-logs', 'bootstrap-*.log'),
                                ('demo-state/launch-logs', '*.log')]:
        try:
            logs += sorted((ROOT / directory).glob(pattern), key=lambda p:p.stat().st_mtime, reverse=True)[:4]
        except OSError:
            pass
    names += [p.relative_to(ROOT).as_posix() for p in logs]
    notes = []
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        payload = report.encode('utf-8')[:limit]
        archive.writestr(report_path.name, payload)
        total += len(payload)
        for name in names:
            path = ROOT / name
            try:
                if not local_plain_file(path):
                    continue
                if not path.resolve().is_relative_to(ROOT.resolve()): continue
                relative = path.relative_to(ROOT)
                if any((ROOT / Path(*relative.parts[:i])).is_symlink() or
                       (ROOT / Path(*relative.parts[:i])).is_junction()
                       for i in range(1, len(relative.parts)+1)): continue
                if count >= 19 or total >= 18*1024*1024:
                    notes.append('达到文件数或总大小限制，后续文件省略。')
                    break
                with path.open('rb') as stream: data = stream.read(limit)
                archive.writestr(name, data)
                total += len(data); count += 1
                if path.stat().st_size > limit: notes.append(name + ' 已截断至 2 MiB。')
            except OSError as error:
                notes.append(name + ' 读取失败：' + str(error))
        archive.writestr('收集说明.txt', '\n'.join(notes) or '白名单文件收集完成。')
    return destination


def write_outputs(report: str, stamp: str) -> tuple[Path, Path] | None:
    """桌面写不了就退回安装目录，保证无论如何都留下证据。"""
    failures = []
    for directory in (desktop_dir(), ROOT, Path.cwd()):
        if not directory.is_dir():
            continue
        report_path = directory / f'诊断报告_流哨_{stamp}.txt'
        try:
            with report_path.open('x', encoding='utf-8', newline='\r\n') as output:
                output.write(report)
            archive = build_archive(report, report_path)
        except OSError as error:
            failures.append(f'{directory}：{error}')
            continue
        return report_path, archive
    say('无法写入诊断文件，试过这些位置：')
    for failure in failures:
        say(f'  {failure}')
    return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    say('流哨环境自检：正在收集信息（约 10 秒，wheels 校验需要读取大文件）…')
    report = build_report()
    say()
    say(report)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8]
    written = write_outputs(report, stamp)
    if written is None:
        say('请把上面整段文字复制保存成 txt 发给组织者。')
        return 1
    report_path, archive = written
    say()
    say(LINE)
    say('自检完成。请把下面这个文件发给组织者（不要只发截图）：')
    say(f'  {archive}')
    say(f'（同一目录还有纯文本报告：{report_path.name}）')
    say('里面只有安装信息、日志和文件校验结果，不含复核内容和聊天记录。')
    say(LINE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
