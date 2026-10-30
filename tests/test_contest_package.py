"""Independent source manifests cover included files, not omitted binary assets."""
import hashlib
import json

import pytest
from scripts.package_contest_release import source_contents, verify_source, write_zip


def test_source_package_regenerates_checksums_without_runtime_omissions(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    contents={'app.py':b'print("fixture")\n','runtime/python.exe':b'python',
              'wheels/a.whl':b'wheel','artifacts/model.joblib':b'model','artifacts/engine.joblib':b'engine'}
    contents['SOURCE_MANIFEST.json']=json.dumps({'format':1,'source_commit':'a'*40,
        'files':{'app.py':hashlib.sha256(contents['app.py']).hexdigest()}}).encode()
    for name,data in contents.items():
        path=runtime/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest={'software_sha':'a'*40,'files':{n:{'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),
                                             'origin':'fixture'} for n,d in contents.items()}}
    members=source_contents(runtime,manifest)
    assert set(members)=={'app.py','SOURCE_MANIFEST.json','PACKAGE_MANIFEST.json','SHA256SUMS.txt'}
    root=tmp_path/'source';root.mkdir()
    for name,data in members.items(): (root/name).write_bytes(data)
    assert verify_source(root)['package_files']==2
    assert 'model.joblib' not in (root/'SHA256SUMS.txt').read_text()
    (root/'app.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'):verify_source(root)


def test_archives_are_deterministic_and_existing_outputs_preserved(tmp_path):
    members={'b.txt':b'b','a.txt':b'a'}
    a=tmp_path/'a.zip';b=tmp_path/'b.zip'
    assert write_zip(a,'source',members)==write_zip(b,'source',dict(reversed(list(members.items()))))
    before=a.read_bytes()
    with pytest.raises(FileExistsError):write_zip(a,'source',members)
    assert a.read_bytes()==before
