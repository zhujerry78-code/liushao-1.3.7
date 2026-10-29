import hashlib
import json
import stat
import zipfile

import pytest

from delivery.material_entry import MANIFEST, contained, extract_new, verify_materials
from scripts.build_complete_materials import validate_attachment


@pytest.mark.parametrize('name', ['../out','/outside','a/../b','C:/x','a\\b','CON.txt','aux','x.','x ','a//b','a/evil\x00.txt'])
def test_material_paths_reject_windows_aliases_and_escape(tmp_path, name):
    with pytest.raises((ValueError, OSError)):
        contained(tmp_path, name)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('problem', ['escape','case','link'])
def test_source_extraction_fails_before_creating_destination(tmp_path, problem):
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        if problem == 'escape': z.writestr('../outside.txt','no')
        elif problem == 'case': z.writestr('src/a.py','a');z.writestr('src/A.py','b')
        else:
            info=zipfile.ZipInfo('src/link');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,'../../outside')
    destination=tmp_path/'new'
    with pytest.raises(ValueError):extract_new(archive,destination)
    assert not destination.exists() and not (tmp_path.parent/'outside.txt').exists()


def test_source_extraction_never_overwrites_existing_files(tmp_path):
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('src/a.py','new')
    destination=tmp_path/'existing';destination.mkdir();(destination/'keep').write_text('old')
    with pytest.raises(FileExistsError):extract_new(archive,destination)
    assert (destination/'keep').read_text()=='old'
    assert (extract_new(archive,tmp_path/'new')/'a.py').read_text()=='new'


def test_material_verification_binds_all_required_inputs_and_detects_corruption(tmp_path):
    names=['runtime.zip','source.zip','assets.zip','summary.json']
    files={}
    for name in names:
        (tmp_path/name).write_bytes(name.encode());files[name]=dict(bytes=len(name),sha256=hashlib.sha256(name.encode()).hexdigest())
    manifest=dict(format=1,files=files,**dict(zip(('runtime_archive','source_archive','assets_archive','evaluation_summary'),names)))
    (tmp_path/MANIFEST).write_text(json.dumps(manifest),encoding='utf-8')
    assert verify_materials(tmp_path)['files']==files
    (tmp_path/'source.zip').write_bytes(b'altered')
    with pytest.raises(ValueError,match='材料已改变'):verify_materials(tmp_path)


def test_attachments_require_explicit_identity_and_exclude_live_data(tmp_path):
    path=tmp_path/'report.txt';path.write_bytes(b'evidence')
    item=dict(path=str(path),destination='proof/report.txt',role='engineering-evidence',source_version='v',sha256=hashlib.sha256(b'evidence').hexdigest())
    validate_attachment(item)
    for change in [dict(sha256='0'*64),dict(destination='demo-state/private.txt'),dict(destination='proof/private.sqlite'),dict(destination='../out')]:
        with pytest.raises(ValueError):validate_attachment({**item,**change})
