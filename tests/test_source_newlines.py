"""LF/CRLF interoperability must not admit non-newline changes or model bytes."""
import hashlib
from pathlib import Path

import pytest

import app_model
from scripts.engineering_external import verify_source_module


SOURCE = b'value = 1\n# inference fixture\n'


def digest(content):
    return hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize('working_eol', [b'\n', b'\r\n'])
@pytest.mark.parametrize('registered_eol', [b'\n', b'\r\n'])
def test_source_registration_accepts_only_matching_lf_crlf_content(
    tmp_path, monkeypatch, working_eol, registered_eol,
):
    package = tmp_path / 'liushao'
    package.mkdir()
    names = ['__init__.py', 'model.py', 'data.py']
    for name in names:
        (package / name).write_bytes(SOURCE.replace(b'\n', working_eol))
    monkeypatch.setattr(app_model.liushao, '__file__', str(package / '__init__.py'))
    descriptor = {'source_code_sha256': {
        'liushao/' + name: digest(SOURCE.replace(b'\n', registered_eol)) for name in names
    }}
    context = app_model._runtime_context(descriptor)
    assert set(context['source_sha256'].values()) == {digest(SOURCE)}
    evidence = verify_source_module(package / 'model.py', descriptor['source_code_sha256']['liushao/model.py'])
    assert evidence['expected_sha256'] == descriptor['source_code_sha256']['liushao/model.py']
    assert evidence['lf_sha256'] == digest(SOURCE)


@pytest.mark.parametrize('changed', [
    SOURCE.replace(b'1', b'2'), SOURCE + b'# extra comment\n',
    SOURCE.replace(b'value =', b'value  ='), SOURCE.replace(b'\n', b'\r'),
])
@pytest.mark.parametrize('registered_eol', [b'\n', b'\r\n'])
def test_real_source_changes_are_rejected_by_app_and_cli(
    tmp_path, monkeypatch, changed, registered_eol,
):
    package = tmp_path / 'liushao'
    package.mkdir()
    for name in ['__init__.py', 'model.py', 'data.py']:
        (package / name).write_bytes(changed)
    monkeypatch.setattr(app_model.liushao, '__file__', str(package / '__init__.py'))
    expected = digest(SOURCE.replace(b'\n', registered_eol))
    descriptor = {'source_code_sha256': {'liushao/' + name: expected
                  for name in ['__init__.py', 'model.py', 'data.py']}}
    with pytest.raises(app_model.CompatibilityError, match='源码不匹配'):
        app_model._runtime_context(descriptor)
    with pytest.raises(ValueError, match='源码内容不匹配'):
        verify_source_module(package / 'model.py', expected)


def test_model_binary_hash_is_always_byte_exact():
    content = b'model\r\nbytes'
    descriptor = {'model_sha256': digest(content)}
    assert app_model._check_model_bytes(content, descriptor) == digest(content)
    with pytest.raises(app_model.CompatibilityError, match='模型哈希不匹配'):
        app_model._check_model_bytes(content.replace(b'\r\n', b'\n'), descriptor)
