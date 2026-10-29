"""Build a verified new runtime directory from an explicit source and runtime template.

In-place synchronization is disabled. Both input directories remain unchanged;
checksums establish file consistency and are not digital signatures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from delivery import verify_demo
from delivery.safe_files import read_json, regular_file, validate_names

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {'PACKAGE_MANIFEST.json', 'SOURCE_MANIFEST.json', 'SHA256SUMS.txt'}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _plain_path(path: Path) -> Path:
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError(f'不接受符号链接或目录联接: {item}')
    return path.resolve()


def _source_snapshot(source_root: Path) -> tuple[dict, bytes, dict[str, bytes]]:
    raw = regular_file(source_root, 'SOURCE_MANIFEST.json').read_bytes()
    manifest = read_json(raw)
    if (manifest.get('format') != 1
            or not re.fullmatch(r'[0-9a-f]{40}', manifest.get('source_commit', ''))
            or not isinstance(manifest.get('files'), dict)):
        raise ValueError('源码清单格式或提交身份无效')
    validate_names(manifest['files'])
    if GENERATED.intersection(manifest['files']):
        raise ValueError('源码文件列表包含应由构建生成的清单')
    contents = {}
    for name, expected in manifest['files'].items():
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError(f'源码哈希格式无效: {name}')
        content = regular_file(source_root, name).read_bytes()
        if sha256(content) != expected:
            raise ValueError(f'源文件哈希与 SOURCE_MANIFEST.json 不一致: {name}')
        contents[name] = content
    if not {'VERSION', 'RELEASE_DATE'}.issubset(contents):
        raise ValueError('源码清单缺少 VERSION 或 RELEASE_DATE')
    return manifest, raw, contents


def sync_runtime_directory(target_dir: Path, source_root: Path = ROOT,
                           output_dir: Path | None = None, *,
                           package_revision: str | None = None,
                           run_pytest: bool = False) -> dict:
    """Publish a new directory only after validation; never overwrite a runtime."""
    if output_dir is None:
        raise ValueError('原地同步已禁用；请用 --output 指定不存在的新目录')
    template = _plain_path(target_dir)
    source = _plain_path(source_root)
    output = _plain_path(output_dir)
    if not template.is_dir() or not source.is_dir():
        raise ValueError('源码目录和运行包模板必须已经存在')
    if output.exists():
        raise FileExistsError(f'输出目录已存在，不会覆盖: {output}')
    if not output.parent.is_dir():
        raise ValueError('输出目录的父目录必须已经存在')
    if output.is_relative_to(template) or output.is_relative_to(source):
        raise ValueError('输出目录必须位于源码和运行包模板目录之外')

    src_manifest, src_bytes, source_files = _source_snapshot(source)
    # Validate the exact template before copying any of its files. Only listed
    # immutable files are copied: .venv, database, caches and logs stay behind.
    original = verify_demo.verify(template)
    old_source = read_json(regular_file(template, 'SOURCE_MANIFEST.json').read_bytes())
    old_source_names = set(old_source['files'])
    version = source_files['VERSION'].decode('utf-8-sig').strip()
    release_date = source_files['RELEASE_DATE'].decode('utf-8-sig').strip()
    revision = package_revision or original['package_revision']
    if package_revision is None and version != original['software_version']:
        raise ValueError('源码版本已改变；必须明确指定 --package-revision')
    manifest = {k: v for k, v in original.items() if k != 'files'}
    manifest.update(software_sha=src_manifest['source_commit'], software_version=version,
                    release_date=release_date, package_revision=revision, files={})
    staged = Path(tempfile.mkdtemp(prefix='.liushao-build-', dir=output.parent))
    published = False
    try:
        def put(name, content, origin):
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            manifest['files'][name] = {'bytes': len(content), 'sha256': sha256(content), 'origin': origin}

        for name, expected in original['files'].items():
            if name in old_source_names or name in source_files or name in GENERATED or name == 'README.md':
                continue
            content = regular_file(template, name).read_bytes()
            if len(content) != expected['bytes'] or sha256(content) != expected['sha256']:
                raise ValueError(f'运行包模板在构建期间发生变化: {name}')
            put(name, content, expected.get('origin', 'verified-assets-or-generated'))
        for name, content in source_files.items():
            put(name, content, 'source-snapshot')
        put('SOURCE_MANIFEST.json', src_bytes, 'verified-assets-or-generated')
        if 'README.md' not in source_files:
            readme = (f'# 流哨 {version} / {revision}\n\n'
                      '解压到独立可写目录，双击 1_一键安装并启动.cmd；首次安装使用包内依赖。\n'
                      '以后使用 2_启动流哨.cmd；关闭使用 4_关闭流哨.cmd；排查使用 3_环境自检.cmd。\n'
                      '这是新建的干净运行目录，不包含模板的虚拟环境、复核数据库或历史日志。\n'
                      '旧版复核记录迁移请参阅 docs/usage/MIGRATION.md；不要覆盖旧目录。\n'
                      '模型提示与评分解释辅助人工核查，不是已确认攻击结论。\n'
                      '运行包与源码身份见 PACKAGE_MANIFEST.json 和 SOURCE_MANIFEST.json。\n'
                      'SHA256 校验用于核对文件一致性，不是数字签名。\n')
            put('README.md', readme.encode('utf-8'), 'verified-assets-or-generated')
        manifest['files'] = dict(sorted(manifest['files'].items()))
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        (staged / 'PACKAGE_MANIFEST.json').write_bytes(manifest_bytes)
        sums = [f"{info['sha256']}  {name}\n" for name, info in manifest['files'].items()]
        sums.append(f'{sha256(manifest_bytes)}  PACKAGE_MANIFEST.json\n')
        (staged / 'SHA256SUMS.txt').write_text(''.join(sums), encoding='utf-8', newline='\n')
        verified = verify_demo.verify(staged)
        if run_pytest:
            proc = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider',
                                   'tests/test_compact_evidence_ui.py',
                                   'tests/test_r12_interval_attribution.py',
                                   'tests/test_review_guidance.py', '-q'], cwd=staged,
                                  env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            if proc.returncode:
                raise RuntimeError(f'新目录测试失败，未发布输出（退出码 {proc.returncode}）')
            verified = verify_demo.verify(staged)
        # On Windows a rename refuses an existing destination. Recheck explicitly
        # as well, so a newly created user directory is never intentionally replaced.
        if output.exists():
            raise FileExistsError(f'输出目录在构建期间已出现，不会覆盖: {output}')
        staged.rename(output)
        published = True
        return {'template_dir': str(template), 'target_dir': str(output),
                'source_dir': str(source), 'in_place_update': False,
                'updated_source_files': len(source_files),
                'total_verified_files': len(verified['files']),
                'software_sha': verified['software_sha'],
                'package_revision': revision,
                'source_manifest_sha256': sha256(src_bytes),
                'package_manifest_sha256': sha256(manifest_bytes)}
    finally:
        if not published and staged.exists():
            # Only remove the unique staging directory created by this invocation.
            if staged.parent.resolve() != output.parent.resolve() or not staged.name.startswith('.liushao-build-'):
                raise RuntimeError('临时目录边界校验失败，已保留现场')
            shutil.rmtree(staged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-package', required=True, type=Path,
                        help='已有完整运行包目录，仅作为只读资源模板')
    parser.add_argument('--source-snapshot', default=ROOT, type=Path,
                        help='具有完整且已核验 SOURCE_MANIFEST.json 的源码目录')
    parser.add_argument('--output', required=True, type=Path,
                        help='不存在的新目录；禁止覆盖原运行目录')
    parser.add_argument('--package-revision', help='新包修订号；必须受交付验证器支持')
    parser.add_argument('--run-pytest', action='store_true',
                        help='用当前 Python 在新目录运行白盒回归测试，通过后才发布')
    args = parser.parse_args()
    report = sync_runtime_directory(args.target_package, args.source_snapshot, args.output,
                                    package_revision=args.package_revision, run_pytest=args.run_pytest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
