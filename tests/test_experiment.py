from pathlib import Path
import subprocess
import sys


def test_training_command_exposes_reproducible_cli():
    assert Path('scripts/train.py').exists(), '训练命令尚未实现'
    result = subprocess.run([sys.executable, '-m', 'scripts.train', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--data' in result.stdout and '--output' in result.stdout
