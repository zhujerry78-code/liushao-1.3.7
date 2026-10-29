"""一键安装并启动流哨：环境预检 → 安装依赖 → 启动本地演示。

本脚本只负责编排；真正的安装日志、文件校验与依赖核对都在
delivery/setup_demo.py 里完成，安装结果仍以 Installation complete. 为准。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import tempfile
import traceback
from uuid import uuid4
try:
    from .local_lock import file_lock, active_service, check_supported_path
    from .verify_demo import verify
except ImportError:
    from local_lock import file_lock, active_service, check_supported_path
    from verify_demo import verify
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
SETUP = ROOT / 'delivery' / 'setup_demo.py'
RUN = ROOT / 'delivery' / 'run_demo.py'
LINE = '=' * 64

BANNER = f"""
{LINE}
  流哨 · 一键安装并启动
  软件 {(ROOT / 'VERSION').read_text(encoding='utf-8').strip()} · 本机离线交付
  本包自带全部依赖，安装过程不需要联网。
{LINE}
"""

FAILED_STEPS = """\
安装没有完成。请按下面三步处理，不需要自己改代码或重装系统：

  1. 不要删除或移动这个文件夹，也不要重新解压覆盖。
  2. 双击同目录下的「3_环境自检.cmd」，会生成一个诊断包 zip。
  3. 把诊断包和这一步的截图发给组织者，并说明卡在哪一行。

日志目录：install-logs\\（每次安装一个文件，全部保留）
常见原因：文件夹不可写、磁盘空间不足、杀毒软件拦截、手工改过文件。
"""


def setup_console() -> None:
    """让中文尽量在任何控制台代码页下都可读。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    os.system('')


def say(message: str = '') -> None:
    print(message, flush=True)


def check_python() -> None:
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise SystemExit(
            '需要 64 位 Python 3.12。\n'
            '请双击「1_一键安装并启动.cmd」，它会使用包内自带的 Python 3.12。'
        )


def check_disk() -> None:
    try:
        free = shutil.disk_usage(ROOT).free
    except OSError:
        return
    if free < 1_200 * 1024 * 1024:
        say(f'提示：磁盘剩余 {free / 1024 ** 3:.1f} GB，安装与运行约需 1.2 GB。')


def check_path() -> None:
    check_supported_path(ROOT)


def wheels_present() -> bool:
    try:
        return any((ROOT / 'wheels').glob('*.whl'))
    except OSError:
        return False


def park_broken_venv() -> None:
    """保留失败环境作为证据，同时让重试可以继续。"""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    target = ROOT / f'.venv.broken-{stamp}-{uuid4().hex[:8]}'
    try:
        ROOT.joinpath('.venv').rename(target)
    except OSError as error:
        raise RuntimeError(f'无法归档旧环境，已停止安装：{error}') from error
    say(f'检测到上一次没装完的环境，已改名为 {target.name}（保留证据，装好后可删除）。')


def already_installed() -> bool:
    """只有 50 个锁定依赖都在本目录 .venv 里才算装好。"""
    if not VENV_PYTHON.is_file():
        return False
    try:
        try:
            from .preflight import check
        except ImportError:
            from preflight import check
        check(ROOT)
        result = subprocess.run(
            [str(VENV_PYTHON), '-X', 'utf8', str(SETUP), '--check-environment'],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def run_setup(mode: str) -> int:
    # A Windows venv redirector inserts a PID between this owner and setup.
    # Launch its base interpreter directly so the lock handoff has the actual parent.
    command = [getattr(sys, '_base_executable', None) or sys.executable, '-X', 'utf8', str(SETUP), mode]
    say(f'实际命令：{subprocess.list2cmdline(command)}')
    say()
    try:
        return subprocess.call(command, cwd=ROOT, env={**os.environ, 'LIUSHAO_INSTALL_PARENT': str(os.getpid())})
    except KeyboardInterrupt:
        say()
        say('安装被中断。保留下载和日志，重新双击本文件即可继续。')
        return 130


def launch(port: int | None) -> int:
    command = [str(VENV_PYTHON), str(RUN)]
    if port is not None:
        command += ['--port', str(port)]
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        say()
        say('已停止服务。已保存的复核记录仍在 demo-state 目录。')
        return 0


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-launch', action='store_true', help='只安装，不启动')
    parser.add_argument('--port', type=int, default=None, help='指定端口，默认自动选择')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline', action='store_true', help='仅使用包内离线依赖（默认）')
    mode.add_argument('--auto', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    setup_console()
    say(BANNER)
    if args.auto:
        say('提示：本交付的 --auto 仍仅使用包内已校验的离线依赖。')
    check_python()
    check_disk()
    check_path()
    for directory, label in ((ROOT, '安装目录'), (Path(os.environ.get('TEMP') or tempfile.gettempdir()), 'TEMP临时目录')):
        try:
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError as error:
            raise PermissionError(f'{label}不可写：{directory}；{error}') from error
    say(f'安装位置：{ROOT}')
    say('安装方式：离线（仅使用包内已校验依赖）')
    say()

    started = time.monotonic()
    with file_lock(ROOT / 'demo-state/install.lock'):
        manifest = verify(ROOT)
        if isinstance(manifest, dict):
            say('包修订：' + manifest.get('package_revision', '本地交付'))
        skipped = already_installed()
        if not skipped and (ROOT / '.venv').exists():
            say('检测到本目录已有装不完的旧环境，先归档再重装。')
            if active_service(ROOT):
                raise RuntimeError('当前程序正在运行或无法确认状态，请先关闭本目录流哨。')
            park_broken_venv()
        if skipped:
            say('检测到本目录已安装完成，跳过安装步骤。')
            code = 0
        else:
            code = run_setup('--offline')
    elapsed = time.monotonic() - started

    if code != 0:
        say()
        say(LINE)
        say(FAILED_STEPS)
        say(LINE)
        return code

    say()
    say(LINE)
    if skipped:
        say(f'  安装状态正常，无需重装（检查耗时 {elapsed:.1f} 秒）。')
    else:
        say(f'  安装完成，用时 {elapsed / 60:.1f} 分钟。')
    say(LINE)
    leftovers = sorted(ROOT.glob('.venv.broken-*'))
    if leftovers:
        say(f'提示：目录里有 {len(leftovers)} 个旧的失败环境（.venv.broken-*），'
            '确认安装正常后可以删除，不影响使用。')
    if args.no_launch:
        say('以后双击「2_启动流哨.cmd」即可开始使用；想放桌面就右键它 → 发送到 → 桌面快捷方式。')
        return 0

    say()
    say('正在打开流哨界面；打开后此安装窗口会自动关闭。')
    return launch(args.port)


def main() -> int:
    try:
        return _main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        setup_console()
        say(f'安装未完成：{type(error).__name__}：{error}')
        try:
            logs = ROOT / 'install-logs'
            logs.mkdir(exist_ok=True)
            (logs / ('failure-' + uuid4().hex + '.log')).write_text(traceback.format_exc(), encoding='utf-8')
        except OSError:
            say('日志目录也不可写，请保存本窗口的具体错误。')
        say(FAILED_STEPS)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
