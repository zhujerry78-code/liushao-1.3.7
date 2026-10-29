"""Pure standard-library checks, run by the bundled base Python before the venv."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(root):
    root = Path(root).resolve()
    environment = root / '.venv'
    config = environment / 'pyvenv.cfg'
    if not config.is_file() or not (environment / 'Scripts/python.exe').is_file():
        raise RuntimeError('安装未完成，请重跑「1_一键安装并启动.cmd」。')
    values = dict(line.split(' = ', 1) for line in config.read_text(encoding='utf-8').splitlines() if ' = ' in line)
    for key in ('home', 'executable'):
        if not values.get(key) or not Path(values[key]).exists():
            raise RuntimeError('目录被移动或改名；请保留 demo-state 中的复核库，在最终位置重新解压并安装。')
    marker = root / 'demo-state/installation.json'
    home = Path(values['home']).resolve()
    if (not marker.exists() and home.name.casefold() == 'python'
            and home.parent.name.casefold() == 'runtime' and home.parent.parent != root):
        raise RuntimeError('旧安装目录被移动或复制；请保留复核库，在最终位置重新解压并安装。')
    if marker.exists() and Path(json.loads(marker.read_bytes())['root']).resolve() != root:
        raise RuntimeError('安装目录已移动或复制；请保留复核库，在最终位置重新解压并安装。')
    code = '''import importlib.metadata as m,sys,json
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve()==root/'.venv' and sys.prefix!=sys.base_prefix
for line in (root/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
 if line.strip() and not line.lstrip().startswith('#'):
  name,version=line.strip().split('==',1)
  assert m.version(name)==version, name
import psutil,streamlit,numpy,pandas,sklearn,joblib
'''
    result = subprocess.run([str(environment / 'Scripts/python.exe'), '-I', '-X', 'utf8', '-c', code, str(root)],
                            capture_output=True, timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('安装未完成或依赖损坏，请重跑「1_一键安装并启动.cmd」。关闭操作未执行，运行记录已保留。')


def main():
    try:
        check(ROOT)
        return 0
    except subprocess.TimeoutExpired:
        print('环境检查超时。请保留复核库，运行「3_环境自检.cmd」或重新运行「1_一键安装并启动.cmd」。', flush=True)
        return 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
