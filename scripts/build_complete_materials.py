"""Assemble a self-contained, explicitly inventoried local project material bundle."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from delivery.material_entry import MANIFEST, contained, sha_file, verify_materials
from scripts.build_windows_demo import ASSETS_SHA256, ARTIFACTS

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROLES = {'user-guide', 'project-summary', 'acceptance-index', 'engineering-evidence', 'screenshots'}
FORBIDDEN_PARTS = {'demo-state', 'browser-profile', '.venv', '__pycache__', '.git'}


def inspect_runtime(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in archive.infolist()):
            raise ValueError('运行包含文件链接。')
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError('运行包含重名文件。')
        roots = {PurePosixPath(name).parts[0] for name in names}
        if len(roots) != 1:
            raise ValueError('运行包根目录不唯一。')
        root = next(iter(roots))
        manifest = json.loads(archive.read(root+'/PACKAGE_MANIFEST.json'))
        expected = set(manifest['files']) | {'PACKAGE_MANIFEST.json', 'SHA256SUMS.txt'}
        if {name[len(root)+1:] for name in names} != expected:
            raise ValueError('运行包含未列明文件或缺少文件。')
        for name, record in manifest['files'].items():
            contained(Path.cwd(), name)
            with archive.open(root+'/'+name) as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != record['sha256'] or archive.getinfo(root+'/'+name).file_size != record['bytes']:
                raise ValueError('运行包文件校验失败：'+name)
        for name, digest in ARTIFACTS.items():
            if manifest['files'].get('artifacts/'+name, {}).get('sha256') != digest:
                raise ValueError('运行包冻结资源身份不符：'+name)
        provenance = archive.read(root+'/SOURCE_MANIFEST.json')
        if json.loads(provenance)['source_commit'] != manifest['software_sha']:
            raise ValueError('运行包来源不一致。')
        return root, manifest, hashlib.sha256(provenance).hexdigest()


def validate_attachment(item):
    if not isinstance(item, dict) or not all(item.get(k) for k in ('path','destination','role','source_version','sha256')):
        raise ValueError('附件必须明确路径、目标、用途、来源版本与SHA256。')
    contained(Path.cwd(), item['destination'])
    parts = PurePosixPath(item['destination']).parts
    if (any(p in FORBIDDEN_PARTS or p.endswith('.last-session') for p in parts)
            or Path(item['destination']).suffix.lower() in {'.sqlite','.sqlite3','.db','.input','.tmp','.wal','.shm'}):
        raise ValueError('附件包含运行中数据或缓存。')
    path = Path(item['path'])
    if not path.is_file() or path.is_symlink() or sha_file(path) != item['sha256']:
        raise ValueError('附件来源缺失或已改变：'+str(path))


def build(delivery, assets_zip, evaluation_summary, attachments_path, output, bundle_date, *, full=True):
    delivery, output = Path(delivery), Path(output)
    if output.exists():
        raise FileExistsError('材料输出目录已存在，请使用新目录。')
    when = date.fromisoformat(bundle_date)
    archives = sorted(delivery.glob('*.zip'))
    if len(archives) != 2:
        raise ValueError('交付目录应明确包含一份运行ZIP和一份源码ZIP。')
    runtime = next((p for p in archives if '_源码_' not in p.name), None)
    source = next((p for p in archives if '_源码_' in p.name), None)
    if runtime is None or source is None:
        raise ValueError('缺少运行包或源码包。')
    sums = {line.split('  ',1)[1]:line.split('  ',1)[0]
            for line in (delivery/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()}
    if any(sha_file(path) != sums.get(path.name) for path in archives):
        raise ValueError('运行包或源码包与独立清单不一致。')
    runtime_root, package, source_digest = inspect_runtime(runtime)
    if source_digest != (delivery/'SOURCE_MANIFEST.sha256').read_text().split()[0]:
        raise ValueError('源码清单独立摘要不一致。')
    with zipfile.ZipFile(source) as archive:
        source_names = archive.namelist()
        if len(source_names) != len({n.casefold() for n in source_names}):
            raise ValueError('源码包含重名文件。')
        marker = next(n for n in source_names if n.endswith('/SOURCE_MANIFEST.json'))
        content = archive.read(marker)
        source_manifest = json.loads(content)
        if hashlib.sha256(content).hexdigest() != source_digest or source_manifest['source_commit'] != package['software_sha']:
            raise ValueError('两份ZIP不是同一源码版本。')
        prefix = marker.removesuffix('SOURCE_MANIFEST.json')
        for name in source_names:
            if not name.startswith(prefix):
                raise ValueError('源码包有额外根目录。')
            relative = name[len(prefix):]
            contained(Path.cwd(), relative)
            if relative in source_manifest['files'] or relative in {'SOURCE_MANIFEST.json','SOURCE_README.txt'}:
                continue
            expected = package['files'].get(relative)
            if not expected or hashlib.sha256(archive.read(name)).hexdigest() != expected['sha256']:
                raise ValueError('源码包有未列明文件：'+relative)
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in archive.infolist()):
            raise ValueError('源码包含文件链接。')
        for name, digest in source_manifest['files'].items():
            if hashlib.sha256(archive.read(prefix+name)).hexdigest() != digest:
                raise ValueError('源码文件校验失败：'+name)
        for name in ('scripts/build_complete_materials.py','delivery/material_entry.py','delivery/materials_bootstrap.ps1'):
            if (name not in source_manifest['files'] or archive.read(prefix+name).replace(b'\r\n',b'\n')
                    != (ROOT/name).read_bytes().replace(b'\r\n',b'\n')):
                raise ValueError('请使用所选源码中的材料构建工具：'+name)
    if sha_file(assets_zip) != ASSETS_SHA256 or ASSETS_SHA256 != package['assets_zip_sha256']:
        raise ValueError('原始资源ZIP身份不符。')
    if sha_file(evaluation_summary) != package['evaluation_summary_sha256']:
        raise ValueError('评测摘要身份不符。')
    attachments = json.loads(Path(attachments_path).read_text(encoding='utf-8-sig'))
    if not isinstance(attachments, list):
        raise ValueError('附件清单必须是列表。')
    for item in attachments:
        validate_attachment(item)
    roles = {item['role'] for item in attachments}
    required = REQUIRED_ROLES | ({'presentation', 'historical-trial-materials'} if full else set())
    if not required <= roles:
        raise ValueError('材料缺少必需用途：'+', '.join(sorted(required-roles)))
    accepted = [json.loads(Path(item['path']).read_text(encoding='utf-8-sig'))
                for item in attachments if item['role'] == 'acceptance-index']
    if not accepted or any(not x.get('passed') or x.get('software_sha') != package['software_sha'] for x in accepted):
        raise ValueError('验收索引未通过或不属于本次源码。')
    title = f'流哨_完整材料_{bundle_date}_{package["software_sha"][:7]}'
    stage = output/title
    stage.mkdir(parents=True)
    metadata = {}
    def add(path, name, role, version):
        target = contained(stage, name)
        if name.casefold() in {n.casefold() for n in metadata}:
            raise ValueError('材料目标重名：'+name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        metadata[name] = dict(bytes=target.stat().st_size,sha256=sha_file(target),role=role,source_version=version)
    def text(name, content, role='entry'):
        target = contained(stage, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8-sig' if name.endswith(('.txt','.cmd')) else 'utf-8')
        metadata[name] = dict(bytes=target.stat().st_size,sha256=sha_file(target),role=role,source_version=package['software_sha'])
    runtime_name, source_name = '01_软件/'+runtime.name, '02_源码/'+source.name
    add(runtime,runtime_name,'runtime',package['software_sha'])
    add(source,source_name,'source',package['software_sha'])
    add(delivery/'SOURCE_MANIFEST.sha256','02_源码/SOURCE_MANIFEST.sha256','source-hash',package['software_sha'])
    assets_name, evaluation_name = '03_重建资源/verified-assets.zip', '03_重建资源/evaluation-summary.json'
    add(assets_zip,assets_name,'assets',ASSETS_SHA256)
    add(evaluation_summary,evaluation_name,'evaluation',package['model_version'])
    add(ROOT/'delivery/material_entry.py','tools/material_entry.py','entry',package['software_sha'])
    add(ROOT/'delivery/materials_bootstrap.ps1','tools/materials_bootstrap.ps1','entry',package['software_sha'])
    for number, label, action in [(1,'准备运行文件','prepare'),(2,'迁移旧版复核','migrate'),
                                   (3,'安装并启动','start'),(4,'校验全部材料','verify'),(5,'从源码重建','rebuild')]:
        # ASCII batch source works under Windows cmd before it selects UTF-8.
        command = ('@echo off\r\nsetlocal DisableDelayedExpansion\r\nchcp 65001 >nul\r\n'
            'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\\materials_bootstrap.ps1" -Action '+action+
            '\r\nset "MATERIAL_EXIT=%errorlevel%"\r\nif not "%LIUSHAO_NONINTERACTIVE%"=="1" pause\r\nexit /b %MATERIAL_EXIT%\r\n')
        name=f'{number}_{label}.cmd';target=contained(stage,name);target.write_bytes(command.encode('ascii'))
        metadata[name]=dict(bytes=target.stat().st_size,sha256=sha_file(target),role='entry',source_version=package['software_sha'])
    for item in attachments:
        add(item['path'],item['destination'],item['role'],item['source_version'])
    text('00_先读我.txt',f'流哨完整材料\n软件：{package["software_version"]} / {package["package_revision"]}\n源码：{package["software_sha"]}\n'
        '顺序：1_准备运行文件 → 如需旧复核，2_迁移旧版复核 → 3_安装并启动。\n'
        '首次迁移必须在新包创建数据库前完成；旧库保留。\n'
        '4_校验全部材料核对所有已列明文件；5_从源码重建不需要另找原资源ZIP，也不需要系统Python。\n'
        '请先完整解压到较短的本机可写目录，再操作。重建时使用包内环境和本地资源，不联网下载。\n'
        '01是本次软件，02是配套源码，03提供独立重建依赖；其他目录按用途和原版本标识。\n'
        '历史展示及试用资料保留原版本，不能作为本次代码的新试用结果；工程合成记录不是真人研究。\n'
        '运行时新增的数据库、缓存、实际导出和浏览器资料不属于发行材料。\n'
        '清单不等于数字签名；请同时核对压缩包旁独立保存的SHA256。\n')
    table = io.StringIO(newline='');writer=csv.writer(table);writer.writerow(['文件','用途','来源版本','字节数','SHA256'])
    for name,item in sorted(metadata.items()):writer.writerow([name,item['role'],item['source_version'],item['bytes'],item['sha256']])
    text('材料目录.csv',table.getvalue(),'inventory')
    text('FILES_SHA256.txt',''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(metadata.items())),'inventory')
    manifest=dict(format=1,software_sha=package['software_sha'],software_version=package['software_version'],
        package_revision=package['package_revision'],release_date=package['release_date'],bundle_date=bundle_date,
        scope='software-and-existing-showcase' if full else 'software-and-technical-evidence',runtime_archive=runtime_name,
        runtime_directory='运行文件',runtime_root=runtime_root,source_archive=source_name,
        assets_archive=assets_name,evaluation_summary=evaluation_name,source_manifest_sha256=source_digest,
        required_roles=sorted(required),files=metadata,manifest_excludes_itself=True)
    (stage/MANIFEST).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verify_materials(stage)
    destination=output/(title+'.zip')
    with zipfile.ZipFile(destination,'w',allowZip64=True) as archive:
        for path in sorted(p for p in stage.rglob('*') if p.is_file()):
            name=title+'/'+path.relative_to(stage).as_posix()
            info=zipfile.ZipInfo(name,(*when.timetuple()[:3],0,0,0));info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {'.zip','.pptx','.xlsx','.docx','.pdf','.mp4','.png'} else zipfile.ZIP_DEFLATED
            with path.open('rb') as reader,archive.open(info,'w',force_zip64=True) as writer:
                shutil.copyfileobj(reader,writer,1024*1024)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('完整材料ZIP的CRC检查未通过。')
        expected={title+'/'+n for n in metadata}|{title+'/'+MANIFEST}
        if set(archive.namelist()) != expected:
            raise ValueError('完整材料ZIP目录与清单不一致。')
    digest=sha_file(destination)
    destination.with_suffix('.zip.sha256').write_text(digest+'  '+destination.name+'\n',encoding='utf-8')
    (output/'MATERIAL_MANIFEST.sha256').write_text(sha_file(stage/MANIFEST)+'  '+MANIFEST+'\n',encoding='utf-8')
    print(json.dumps(dict(zip=str(destination),sha256=digest,files=len(metadata),software_sha=package['software_sha']),ensure_ascii=False))
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('delivery','assets-zip','evaluation-summary','attachments','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--bundle-date',required=True)
    parser.add_argument('--software-only',action='store_true')
    args=parser.parse_args()
    build(args.delivery,args.assets_zip,args.evaluation_summary,args.attachments,args.output,args.bundle_date,full=not args.software_only)


if __name__=='__main__':
    main()
