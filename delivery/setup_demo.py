"""Install the frozen Windows demo in its own Python 3.12 environment."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import importlib.util
import locale
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import traceback
from typing import TextIO
from uuid import uuid4
import venv
from contextlib import nullcontext
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .local_lock import file_lock
except ImportError:
    from local_lock import file_lock

ROOT = Path(__file__).resolve().parents[1]
STAGE_START = datetime.now()


def stage(title: str) -> None:
    """Announce a stage with wall-clock time so a stalled step is obvious."""
    elapsed = (datetime.now() - STAGE_START).total_seconds()
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Stage: {title} (+{elapsed:.0f}s)', flush=True)


class Tee:
    """Write the exact UTF-8 log even when the console cannot display a character."""

    def __init__(self, console: TextIO, log: TextIO):
        self.console = console
        self.log = log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        try:
            self.console.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.console, 'encoding', None) or 'ascii'
            self.console.write(text.encode(encoding, errors='backslashreplace').decode(encoding))
        return len(text)

    def flush(self) -> None:
        self.log.flush()
        self.console.flush()


def decode_output(output: str | bytes) -> str:
    """Prefer Python's UTF-8 output, then the local Windows encoding."""
    if isinstance(output, str):
        return output
    for encoding in ('utf-8', locale.getencoding()):
        try:
            return output.decode(encoding)
        except UnicodeDecodeError:
            pass
    return output.decode('utf-8', errors='backslashreplace')


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Stream child output through the console/log tee and keep its exit status."""
    print(f'Command: {subprocess.list2cmdline(command)}', flush=True)
    environment = dict(os.environ if env is None else env)
    environment['PYTHONUTF8'] = '1'
    environment['PYTHONIOENCODING'] = 'utf-8'
    output = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env=environment) as process:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = decode_output(raw_line)
            output.append(line)
            print(line, end='', flush=True)
        returncode = process.wait()
    print(f'Exit code: {returncode}', flush=True)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, output=''.join(output))


def print_diagnostics() -> None:
    """Log an explicit allowlist; never dump the environment or pip credentials."""
    details = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'Package': str(ROOT),
        'Interpreter': sys.executable,
        'Base interpreter': getattr(sys, '_base_executable', None),
        'Python version': sys.version,
        'Base prefix': sys.base_prefix,
        'Platform': platform.platform(),
        'Architecture': platform.machine(),
        'Interpreter bits': 64 if sys.maxsize > 2**32 else 32,
        'UTF-8 mode': sys.flags.utf8_mode,
        'TEMP': os.environ.get('TEMP'),
        'TMP': os.environ.get('TMP'),
    }
    for name, value in details.items():
        print(f'{name}: {value}')
    try:
        print(f'Resolved temporary directory: {tempfile.gettempdir()}')
    except OSError as error:
        print(f'Temporary directory lookup failed: {error}')
    spec = importlib.util.find_spec('ensurepip')
    print(f'ensurepip module: {spec.origin if spec else "not found"}')
    if spec and spec.origin:
        bundled = Path(spec.origin).parent / '_bundled'
        print(f'Bundled pip wheels: {sorted(path.name for path in bundled.glob("*.whl"))}')


def run_installer() -> int:
    """Persist failures, including output hidden inside venv's ensurepip exception."""
    log_directory = ROOT / 'install-logs'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    log_path = log_directory / f'install-{stamp}-{uuid4().hex[:8]}.log'
    try:
        log_directory.mkdir(exist_ok=True)
        log = log_path.open('x', encoding='utf-8', newline='\n')
    except OSError as error:
        print(f'Cannot create installation log: {error}. Extract into a writable fresh folder.',
              file=sys.stderr)
        return 1
    with log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        print(f'Installation log: {log_path}')
        try:
            print_diagnostics()
            parent_owns = os.environ.get('LIUSHAO_INSTALL_PARENT') == str(os.getppid())
            with (nullcontext() if parent_owns else file_lock(ROOT / 'demo-state/install.lock')):
                return main()
        except (Exception, KeyboardInterrupt) as error:
            print(f'Installation failed: {error}', file=sys.stderr)
            if isinstance(error, subprocess.CalledProcessError):
                print(f'Failed command: {error.cmd!r}', file=sys.stderr)
                print(f'Exit code: {error.returncode}', file=sys.stderr)
                for name, output in [('stdout / combined output', error.output),
                                     ('stderr', error.stderr)]:
                    if output:
                        print(f'Captured {name}:\n{decode_output(output)}', file=sys.stderr)
            traceback.print_exc(file=log)
            print('Keep this folder and install-logs for diagnosis. '
                  'If .venv is incomplete, extract the original ZIP into a fresh folder; '
                  'do not copy .venv or overwrite existing review data.', file=sys.stderr)
            print('保留本文件夹和 install-logs；不要删除 install-logs，也不要把 .venv 复制到别的电脑。',
                  file=sys.stderr)
            print('下一步：双击同目录的「3_环境自检.cmd」生成诊断包，连同本窗口截图发给组织者。',
                  file=sys.stderr)
            print(f'Installation log: {log_path}', file=sys.stderr)
            return 130 if isinstance(error, KeyboardInterrupt) else 1


def installation_environment() -> dict[str, str]:
    """Keep pip settings from redirecting installs outside this package's venv."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith('PIP_')}
    environment['PIP_CONFIG_FILE'] = os.devnull
    return environment


def check_environment() -> None:
    """Require every locked distribution in the interpreter used to run the demo."""
    environment = (ROOT / '.venv').resolve()
    if Path(sys.prefix).resolve() != environment or sys.prefix == sys.base_prefix:
        raise RuntimeError('Installation check must use this package\'s .venv Python.')
    count = 0
    for requirement in (ROOT / 'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        requirement = requirement.strip()
        if not requirement or requirement.startswith('#'):
            continue
        name, separator, expected = requirement.partition('==')
        if not separator or not name or not expected:
            raise RuntimeError(f'Expected an exact locked dependency: {requirement}')
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f'Locked dependency missing from .venv: {requirement}') from error
        if distribution.version != expected:
            raise RuntimeError(f'Locked dependency version mismatch: {requirement}')
        if not Path(distribution.locate_file('')).resolve().is_relative_to(environment):
            raise RuntimeError(f'Locked dependency is outside this package\'s .venv: {name}')
        count += 1
    if not count:
        raise RuntimeError('The dependency lock is empty.')
    print(f'Verified {count} locked dependencies inside this package\'s .venv.')


def verify_wheels() -> int:
    """Reject modified or missing bundles before pip is allowed to run."""
    directory = ROOT / 'wheels'
    sums = directory / 'SHA256SUMS.txt'
    if not sums.is_file():
        raise RuntimeError('缺少 wheels/SHA256SUMS.txt，无法核对离线依赖；请重新解压原包。')
    expected = {}
    for line in sums.read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, _, name = line.partition('  ')
            name = name.strip()
            if (not re.fullmatch(r'[0-9a-fA-F]{64}', digest) or not name.endswith('.whl')
                    or Path(name).name != name or any(c in name for c in ':\\/')
                    or name.casefold() in {n.casefold() for n in expected}):
                raise RuntimeError('离线依赖清单包含重复或不安全条目')
            expected[name] = digest.lower()
    if not expected:
        raise RuntimeError('wheels/SHA256SUMS.txt 是空的；请重新解压原包。')
    actual = {p.name for p in directory.glob('*.whl')}
    if actual != set(expected):
        raise RuntimeError('离线依赖集合与清单不一致（含额外或缺失文件）')
    problems = []
    for name, digest in expected.items():
        path = directory / name
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(directory.resolve()):
            raise RuntimeError('离线依赖不能是链接')
        if not path.is_file():
            problems.append(f'缺少 {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f'内容不符 {name}')
    if problems:
        raise RuntimeError('离线依赖校验未通过：' + '；'.join(problems[:6])
                           + '。请重新解压原包或联系组织者，不要手工替换 wheel。')
    print(f'Verified {len(expected)} offline wheels against wheels/SHA256SUMS.txt.')
    return len(expected)


def resolve_mode(offline: bool, online: bool) -> str:
    """Prefer the bundled wheels; fall back to PyPI only when they are absent."""
    has_wheels = any((ROOT / 'wheels').glob('*.whl'))
    if online:
        return 'online'
    if offline and not has_wheels:
        raise RuntimeError('指定了 --offline，但 wheels 目录里没有 .whl 文件。')
    return 'offline' if has_wheels else 'online'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--offline', action='store_true', help='只用包内 wheels 离线安装')
    choice.add_argument('--online', action='store_true', help='强制从 PyPI 联网安装')
    parser.add_argument('--auto', action='store_true',
                        help='有 wheels 时离线安装，没有时联网安装（默认行为）')
    parser.add_argument('--check-environment', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise RuntimeError('Use 64-bit Python 3.12. Set LIUSHAO_PYTHON to its python.exe.')
    if args.check_environment:
        check_environment()
        return 0
    mode = resolve_mode(args.offline, args.online)
    print(f'Installation mode: {mode}')
    stage('verify package integrity（校验软件文件）')
    run_command([sys.executable, '-I', '-X', 'utf8', str(ROOT / 'delivery/verify_demo.py')])
    if mode == 'offline':
        stage('verify offline wheels（校验离线依赖）')
        verify_wheels()
    environment = ROOT / '.venv'
    python = environment / 'Scripts/python.exe'
    if environment.exists() and not python.is_file():
        raise RuntimeError('Incomplete .venv: keep it for diagnosis and extract into a fresh folder.')
    if not environment.exists():
        stage('create .venv and bootstrap pip with ensurepip（创建独立环境）')
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    stage('verify the package environment and pip（核对环境）')
    run_command([str(python), '-I', '-X', 'utf8', '-c',
                 'import sys; from pathlib import Path; '
                 'assert sys.version_info[:2] == (3, 12); assert sys.maxsize > 2**32; '
                 'assert sys.prefix != sys.base_prefix; '
                 'assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()',
                 str(environment)])
    env = installation_environment()
    run_command([str(python), '-I', '-X', 'utf8', '-m', 'pip', '--version'], env=env)
    stage('install locked dependencies（安装 50 个锁定依赖）')
    command = [str(python), '-I', '-X', 'utf8', '-m', 'pip', '--disable-pip-version-check', 'install',
               '--no-cache-dir', '--only-binary=:all:', '--retries', '5', '--timeout', '60']
    if mode == 'offline':
        command += ['--no-index', '--find-links', str(ROOT / 'wheels')]
    else:
        command += ['--index-url', 'https://pypi.org/simple']
    run_command(command + ['-r', str(ROOT / 'requirements-lock.txt')], env=env)
    stage('verify dependency consistency and locked installation locations（核对依赖位置）')
    run_command([str(python), '-I', '-X', 'utf8', '-m', 'pip', 'check'], env=env)
    run_command([str(python), '-I', '-X', 'utf8', str(ROOT / 'delivery/setup_demo.py'),
                 '--check-environment'], env=env)
    state = ROOT / 'demo-state'
    state.mkdir(exist_ok=True)
    (state / 'installation.json').write_text(__import__('json').dumps(
        {'root': str(ROOT.resolve())}, ensure_ascii=False), encoding='utf-8')
    print('Installation complete. Run run-demo.cmd. No GitHub account is required.')
    print(f'安装完成（{mode}，用时 {(datetime.now() - STAGE_START).total_seconds() / 60:.1f} 分钟）。')
    return 0


if __name__ == '__main__':
    # Internal verification is already captured by its parent installation process.
    raise SystemExit(main() if '--check-environment' in sys.argv[1:] else run_installer())
