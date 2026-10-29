"""Report unavailable historical audit prerequisites without claiming a passed run."""
import json
from pathlib import Path
import subprocess


def unavailable(root, paths, *, git_ref=None, output=None):
    root = Path(root)
    missing = [name for name in paths if not (root / name).is_file()]
    if git_ref:
        try:
            subprocess.run(['git', 'cat-file', '-e', git_ref], cwd=root,
                           check=True, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            missing.append('Git history: ' + git_ref)
    if not missing:
        return False
    result = {'status': 'not_run', 'passed': False, 'missing': missing,
              'reason': '缺少本次执行所需输入；历史报告不等于当前运行通过。'}
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return True
