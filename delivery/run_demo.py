"""启动流哨：检查文件、等待网页就绪，自动打开界面。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys

from verify_demo import verify
from local_lock import check_supported_path
from desktop_start import desktop_worker, free_port, launch_desktop, stop_desktop

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--headless', action='store_true', help='技术检查：在当前窗口运行，不打开界面')
    parser.add_argument('--stop', action='store_true', help='关闭当前文件夹启动的流哨')
    parser.add_argument('--desktop-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--startup-deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--result-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
        raise RuntimeError('64-bit Python 3.12 is required')
    if not 1024 <= args.port <= 65535:
        raise ValueError('Port must be between 1024 and 65535')
    check_supported_path(ROOT)
    if args.stop:
        return stop_desktop(ROOT)
    if args.desktop_worker:
        if args.result_file is None:
            parser.error('--result-file is required for the worker')
        return desktop_worker(ROOT, args.port, args.result_file, verify, args.startup_deadline)
    if not args.headless:
        print('正在准备流哨，界面会自动打开，请稍候……', flush=True)
        return launch_desktop(ROOT, args.port)

    # Headless technical runs use the same ownership, lock and stop protocol.
    import uuid
    runtime = ROOT / 'demo-state'
    runtime.mkdir(exist_ok=True)
    result = runtime / ('headless-' + uuid.uuid4().hex + '.json')
    return desktop_worker(ROOT, args.port, result, verify, open_browser=False)



if __name__ == '__main__':
    raise SystemExit(main())
