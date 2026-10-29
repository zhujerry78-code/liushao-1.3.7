"""Use a verified, source-equivalent CRLF runtime for both handoff packages.

This records the current LF-checkout compatibility gap without changing the guard,
descriptors, models or repository source. Only trusted, byte-verified sources execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.compare_release_models import NEW_ZIP_SHA, ROOT, require


def same_source(left, right):
    return left.replace(b'\r\n', b'\n') == right.replace(b'\r\n', b'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--original-zip', type=Path, required=True)
    parser.add_argument('--rebuilt-zip', type=Path, required=True)
    parser.add_argument('--external-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=False)
    require(hashlib.sha256(args.rebuilt_zip.read_bytes()).hexdigest() == NEW_ZIP_SHA, 'A包SHA不一致')
    evidence = {}
    with zipfile.ZipFile(args.rebuilt_zip) as archive:
        for name in ('liushao/__init__.py', 'liushao/model.py', 'liushao/data.py'):
            content = archive.read(name)
            local = (ROOT/name).read_bytes()
            require(same_source(content, local), '存在非换行源码变更，禁止构造共同运行环境: '+name)
            target = args.runtime/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            evidence[name] = {'source_sha256': hashlib.sha256(content).hexdigest(),
                'lf_sha256': hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest(),
                'local_sha256': hashlib.sha256(local).hexdigest(), 'line_endings_only': True}
    for name in ('app_model.py', 'requirements-lock.txt', 'liushao/reviews.py'):
        (args.runtime/name).write_bytes((ROOT/name).read_bytes())
    (args.runtime/'RUNTIME_EQUIVALENCE.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    code = ('import sys;sys.path.insert(0,'+repr(str(args.runtime.resolve()))+');'
            'from scripts.compare_release_models import main;raise SystemExit(main())')
    command = [sys.executable, '-c', code, '--original-zip', str(args.original_zip.resolve()),
               '--rebuilt-zip', str(args.rebuilt_zip.resolve()), '--output', str(args.output.resolve())]
    if args.external_source:
        command += ['--external-source', str(args.external_source.resolve())]
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
