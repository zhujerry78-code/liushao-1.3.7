"""Deterministic runtime/source ZIPs from a verified, already-sealed directory.

No network, dependency installation, training, fake git commit, or copied runtime
checksums in the independent source archive. Existing output ZIPs are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAMP = (2026, 10, 29, 0, 0, 0)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def source_member(name):
    return not (name.startswith(('runtime/','wheels/')) or name in
                ('artifacts/model.joblib','artifacts/engine.joblib','PACKAGE_MANIFEST.json','SHA256SUMS.txt'))


def source_contents(root, runtime):
    members = {name:(root/name).read_bytes() for name in runtime['files'] if source_member(name)}
    files = {name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                   'origin':runtime['files'][name]['origin']} for name,data in sorted(members.items())}
    manifest = dict(runtime, package_kind='source', files=files,
                    excluded_runtime_assets=['runtime/','wheels/','artifacts/model.joblib','artifacts/engine.joblib'])
    members['PACKAGE_MANIFEST.json'] = encoded(manifest)
    sums = {name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
    members['SHA256SUMS.txt'] = ''.join(f'{sums[n]}  {n}\n' for n in sorted(sums)).encode('utf-8')
    return members


def write_zip(path, prefix, members):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(members.items()):
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member: ' + name)
            item = zipfile.ZipInfo(prefix+'/'+name,STAMP)
            item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=0o100644 << 16
            archive.writestr(item,data)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(root):
    from delivery.safe_files import validate_names, regular_file
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_bytes())
    if manifest.get('package_kind') != 'source': raise ValueError('Not an independent source package')
    validate_names(manifest['files'])
    for name,meta in manifest['files'].items():
        data=regular_file(root,name).read_bytes()
        if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
            raise ValueError('Source package mismatch: '+name)
    checksums={}
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if name in checksums: raise ValueError('Duplicate source checksum: '+name)
        checksums[name]=digest
        if hashlib.sha256(regular_file(root,name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source checksum mismatch: '+name)
    if set(checksums) != set(manifest['files'])|{'PACKAGE_MANIFEST.json'}:
        raise ValueError('Incomplete independent source checksums')
    source=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
    if source.get('source_commit')!=manifest['software_sha']:
        raise ValueError('Source snapshot identity mismatch')
    for name,digest in source['files'].items():
        if manifest['files'].get(name,{}).get('sha256')!=digest:
            raise ValueError('Source provenance mismatch: '+name)
    return {'package_files':len(manifest['files']),'source_files':len(source['files']),
            'software_sha':manifest['software_sha']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-source',type=Path)
    args=parser.parse_args()
    if args.verify_source:
        print(json.dumps(verify_source(args.verify_source),ensure_ascii=False)); return
    if args.output is None: parser.error('--output is required for packaging')
    from delivery.verify_demo import verify
    manifest=verify(ROOT)
    args.output.mkdir(parents=True,exist_ok=True)
    runtime={name:(ROOT/name).read_bytes() for name in manifest['files']}
    runtime.update({name:(ROOT/name).read_bytes() for name in ('PACKAGE_MANIFEST.json','SHA256SUMS.txt')})
    names={'runtime':'Liushao-1.3.7-Windows-r13-contest','source':'Liushao-1.3.7-source-r13-contest'}
    outputs={kind:args.output/(name+'.zip') for kind,name in names.items()}
    if any(path.exists() for path in outputs.values()) or (args.output/'release_hashes.json').exists():
        raise FileExistsError('Release outputs already exist; choose a fresh directory')
    report={}
    for kind,members in (('runtime',runtime),('source',source_contents(ROOT,manifest))):
        path=outputs[kind]
        digest=write_zip(path,names[kind],members)
        report[kind]={'file':path.name,'bytes':path.stat().st_size,'sha256':digest,'files':len(members)}
    report['source_snapshot']=manifest['software_sha']
    (args.output/'release_hashes.json').write_bytes(encoded(report))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
