"""Portable standard-library entry point for a complete local material bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile


MANIFEST = 'MATERIAL_MANIFEST.json'
RESERVED = {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}


def sha_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def contained(root, name):
    root = Path(root).resolve()
    parts = PurePosixPath(name)
    if (not name or parts.is_absolute() or '\\' in name or ':' in name
            or any(p in ('', '.', '..') or p.endswith((' ', '.'))
                   or p.split('.')[0].upper() in RESERVED or any(ord(c)<32 or c in '<>"|?*' for c in p)
                   for p in name.split('/'))):
        raise ValueError('材料路径无效：'+str(name))
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('材料路径越界：'+name)
    if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError('材料路径含文件链接：'+name)
    return path


def load_manifest(root):
    with (Path(root)/MANIFEST).open('rb') as handle:
        content = handle.read(4*1024*1024+1)
    if len(content) > 4*1024*1024:
        raise ValueError('材料清单过大。')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('材料清单有重复字段。')
            value[key] = item
        return value
    result = json.loads(content, object_pairs_hook=unique)
    if not isinstance(result, dict) or type(result.get('format')) is not int or result['format'] != 1:
        raise ValueError('材料清单版本无效。')
    if not isinstance(result.get('files'), dict) or not result['files']:
        raise ValueError('材料清单缺少文件。')
    return result


def verify_materials(root):
    root = Path(root).resolve()
    manifest = load_manifest(root)
    folded = set()
    for name, expected in manifest['files'].items():
        path = contained(root, name)
        if name.casefold() in folded or not path.is_file():
            raise ValueError('材料重名或缺失：'+name)
        folded.add(name.casefold())
        if (not isinstance(expected, dict) or type(expected.get('bytes')) is not int
                or expected['bytes'] < 0 or path.stat().st_size != expected['bytes']
                or sha_file(path) != expected.get('sha256')):
            raise ValueError('材料已改变：'+name)
    for key in ('runtime_archive', 'source_archive', 'assets_archive', 'evaluation_summary'):
        if manifest.get(key) not in manifest['files']:
            raise ValueError('材料清单缺少入口：'+key)
    return manifest


def extract_new(archive, destination):
    """Only extract into a new contained directory; never replace existing user files."""
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError('展开目录已存在，请使用新目录：'+str(destination))
    with zipfile.ZipFile(archive) as source:
        names, roots, total = set(), set(), 0
        for entry in source.infolist():
            name = entry.filename.rstrip('/')
            contained(destination, name)
            folded = name.casefold()
            if folded in names or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('压缩包包含重名或链接：'+name)
            names.add(folded)
            roots.add(PurePosixPath(name).parts[0])
            total += entry.file_size
            if total > 2*1024*1024*1024:
                raise ValueError('压缩包展开大小超过上限。')
        if len(roots) != 1:
            raise ValueError('压缩包应只有一个根目录。')
        destination.mkdir(parents=True)
        for entry in source.infolist():
            path = contained(destination, entry.filename.rstrip('/'))
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as reader, path.open('xb') as writer:
                    shutil.copyfileobj(reader, writer, 1024*1024)
    return destination/next(iter(roots))


def run(root, action):
    root = Path(root).resolve()
    manifest = verify_materials(root)
    package = contained(root, '运行文件/'+manifest['runtime_root'])
    python = package/'runtime/python/python.exe'
    if not python.is_file():
        raise FileNotFoundError('请先使用完整材料中的“1_准备运行文件.cmd”。')
    env = {**os.environ, 'PYTHONUTF8': '1'}
    subprocess.run([str(python), '-I', '-X', 'utf8', str(package/'delivery/verify_demo.py')],
                   cwd=package, env=env, check=True)
    if action in ('prepare', 'verify'):
        print('材料与运行文件校验通过；未创建复核数据库。')
        print('运行目录：', package)
        return
    if action == 'migrate':
        subprocess.run([str(python), str(package/'delivery/migrate_reviews.py'), '--interactive'],
                       cwd=package, env=env, check=True)
        return
    arguments = ['--offline'] + (['--no-launch'] if action == 'rebuild' else [])
    subprocess.run([str(python), str(package/'delivery/oneclick.py'), *arguments],
                   cwd=package, env=env, check=True)
    if action == 'start':
        return
    source = extract_new(contained(root, manifest['source_archive']), root/('源码展开-'+uuid.uuid4().hex[:8]))
    output = root/'重建输出'
    if output.exists():
        raise FileExistsError('“重建输出”已存在，请另存已有结果后使用全新材料目录重建。')
    local_python = package/'.venv/Scripts/python.exe'
    subprocess.run([str(local_python), '-m', 'scripts.build_windows_demo', '--source-snapshot', '.',
        '--source-manifest-sha256', manifest['source_manifest_sha256'],
        '--assets-zip', str(contained(root, manifest['assets_archive'])),
        '--package-revision', manifest['package_revision'],
        '--evaluation-summary', str(contained(root, manifest['evaluation_summary'])),
        '--evaluation-summary-sha256', manifest['files'][manifest['evaluation_summary']]['sha256'],
        '--output', str(output)], cwd=source, env=env, check=True)
    for key in ('runtime_archive', 'source_archive'):
        original = manifest[key]
        rebuilt = output/PurePosixPath(original).name
        if sha_file(rebuilt) != manifest['files'][original]['sha256']:
            raise ValueError('重建文件与交付不一致：'+rebuilt.name)
    print('源码重建通过：运行包与源码包均逐字节一致。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--action', choices=['prepare','migrate','start','verify','rebuild'], required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.action)
        return 0
    except Exception as error:
        print('操作未完成：', type(error).__name__, str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
