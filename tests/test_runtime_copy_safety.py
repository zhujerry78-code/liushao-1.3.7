"""A failed or invalid update must leave every existing runtime byte unchanged."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import sync_runtime_package as sync


def digest(content):
    return hashlib.sha256(content).hexdigest()


def tree_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def write_source(root, files):
    root.mkdir(exist_ok=True)
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    manifest = {'format': 1, 'source_commit': 'a' * 40,
                'files': {name: digest(content) for name, content in files.items()}}
    (root / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')


@pytest.fixture
def packages(tmp_path, monkeypatch):
    template, source, output = (tmp_path / name for name in ['template', 'source', 'output'])
    common = {'VERSION': b'1.3.0', 'RELEASE_DATE': b'2026-10-24', 'app.py': b'old content'}
    write_source(template, common)
    contents = tree_bytes(template)
    contents['asset.bin'] = b'trusted resource'
    (template / 'asset.bin').write_bytes(contents['asset.bin'])
    manifest = {'software_version': '1.3.0', 'software_sha': 'a' * 40,
                'package_revision': 'r11-final-readable', 'release_date': '2026-10-24',
                'files': {n: {'bytes': len(v), 'sha256': digest(v), 'origin': 'test'} for n, v in contents.items()}}
    (template / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    # Unlisted user state and caches must not become part of a new release.
    (template / 'demo-state').mkdir()
    (template / 'demo-state/reviews.sqlite').write_bytes(b'keep user data')
    (template / '.venv').mkdir()
    (template / '.venv/pyvenv.cfg').write_text('home = C:\\old-machine', encoding='utf-8')
    write_source(source, {**common, 'app.py': b'new content'})

    def verify(directory):
        m = json.loads((directory / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
        if m['package_revision'] not in ('r11-final-readable', 'r11-whitebox-polished'):
            raise ValueError('Unsupported package revision')
        for name, expected in m['files'].items():
            content = (directory / name).read_bytes()
            if digest(content) != expected['sha256'] or len(content) != expected['bytes']:
                raise ValueError('hash mismatch')
        return m

    monkeypatch.setattr(sync.verify_demo, 'verify', verify)
    return template, source, output


def test_success_publishes_clean_new_directory_and_preserves_inputs(packages):
    template, source, output = packages
    before_template, before_source = tree_bytes(template), tree_bytes(source)
    result = sync.sync_runtime_directory(template, source, output)
    assert tree_bytes(template) == before_template
    assert tree_bytes(source) == before_source
    assert (output / 'app.py').read_bytes() == b'new content'
    assert (output / 'asset.bin').read_bytes() == b'trusted resource'
    assert not (output / '.venv').exists()
    assert not (output / 'demo-state').exists()
    assert result['in_place_update'] is False
    assert not list(output.parent.glob('.liushao-build-*'))


def test_missing_output_argument_never_enables_in_place_update(packages):
    template, source, _ = packages
    before = tree_bytes(template)
    with pytest.raises(ValueError, match='原地同步已禁用'):
        sync.sync_runtime_directory(template, source)
    assert tree_bytes(template) == before


@pytest.mark.parametrize('mode', ['changed_hash', 'missing_late_file', 'unsafe_path'])
def test_source_preflight_failure_has_no_partial_output(packages, mode):
    template, source, output = packages
    before = tree_bytes(template)
    if mode == 'changed_hash':
        (source / 'app.py').write_bytes(b'unverified edit')
    else:
        path = source / 'SOURCE_MANIFEST.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['files']['../outside.txt' if mode == 'unsafe_path' else 'missing.py'] = '0' * 64
        path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError)):
        sync.sync_runtime_directory(template, source, output)
    assert tree_bytes(template) == before
    assert not output.exists()
    assert not list(output.parent.glob('.liushao-build-*'))


def test_existing_output_is_preserved(packages):
    template, source, output = packages
    output.mkdir()
    (output / 'keep.txt').write_bytes(b'unchanged')
    with pytest.raises(FileExistsError):
        sync.sync_runtime_directory(template, source, output)
    assert (output / 'keep.txt').read_bytes() == b'unchanged'


def test_nested_output_is_rejected(packages):
    template, source, _ = packages
    with pytest.raises(ValueError, match='之外'):
        sync.sync_runtime_directory(template, source, template / 'nested')


def test_final_verification_failure_discards_only_staging(packages):
    template, source, output = packages
    before = tree_bytes(template)
    with pytest.raises(ValueError, match='Unsupported'):
        sync.sync_runtime_directory(template, source, output, package_revision='unsupported')
    assert tree_bytes(template) == before
    assert not output.exists()
    assert not list(output.parent.glob('.liushao-build-*'))


def test_changed_version_requires_explicit_revision(packages):
    template, source, output = packages
    files = {'VERSION': b'1.3.1', 'RELEASE_DATE': b'2026-10-24', 'app.py': b'new'}
    write_source(source, files)
    with pytest.raises(ValueError, match='package-revision'):
        sync.sync_runtime_directory(template, source, output)
    assert not output.exists()
    result = sync.sync_runtime_directory(template, source, output, package_revision='r11-whitebox-polished')
    manifest = json.loads((output / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['software_version'] == '1.3.1'
    assert '1.3.1 / r11-whitebox-polished' in (output / 'README.md').read_text(encoding='utf-8')
    assert result['package_revision'] == 'r11-whitebox-polished'
