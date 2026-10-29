"""A release guard needs an independently retained digest."""
import hashlib
import json
import pytest
from scripts.release_source_guard import verify_source

@pytest.fixture
def release(tmp_path):
    files = {'app.py': b'# reviewed application\n', 'liushao/model.py': b'# reviewed adapter\n'}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = tmp_path / 'SOURCE_MANIFEST.json'
    manifest.write_text(json.dumps({'source_commit': 'a'*40, 'files': {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}), encoding='utf-8')
    return tmp_path, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()

def test_explicit_digest_checks_registered_source(release):
    report = verify_source(*release)
    assert report['passed'] and report['checked_count'] == 2 and report['absent_local_artifacts']

def test_missing_digest_cannot_auto_approve(release):
    root, manifest, digest = release
    report = verify_source(root, manifest, None)
    assert not report['passed'] and report['status'] == 'not_run'

@pytest.mark.parametrize('name', ['app.py', 'liushao/model.py'])
def test_changed_registered_source_fails(release, name):
    root, manifest, digest = release
    (root / name).write_bytes(b'unapproved replacement')
    report = verify_source(root, manifest, digest)
    assert not report['passed'] and report['failures'][0]['path'] == name

def test_replacement_manifest_cannot_replace_trusted_digest(release):
    root, manifest, digest = release
    data = json.loads(manifest.read_bytes())
    (root / 'app.py').write_bytes(b'changed')
    data['files']['app.py'] = hashlib.sha256(b'changed').hexdigest()
    manifest.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='独立保存'):
        verify_source(root, manifest, digest)

def test_missing_registered_file_fails(release):
    root, manifest, digest = release
    (root / 'app.py').unlink()
    assert not verify_source(root, manifest, digest)['passed']

def test_newline_only_checkout_is_accepted(release):
    root, manifest, digest = release
    (root / 'app.py').write_bytes(b'# reviewed application\r\n')
    assert verify_source(root, manifest, digest)['passed']
