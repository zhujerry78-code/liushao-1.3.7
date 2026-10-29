"""Entry points must load and cite the immutable bytes they validated."""
import hashlib
import io
import pytest
from scripts import engineering_run

from scripts import engineering_external as external
from scripts import engineering_model_compatibility as compatibility
from test_engineering_external import inputs  # noqa: F401
from test_engineering_model_compatibility import synthetic_bundle, write_package  # noqa: F401


def test_external_loads_verified_bytes_and_cites_descriptor_snapshot(inputs, monkeypatch):
    args, _ = inputs
    model_bytes = args.model.read_bytes()
    descriptor_bytes = args.descriptor.read_bytes()
    original_load = external.joblib.load

    def replace_paths_then_load(stream):
        assert isinstance(stream, io.BytesIO)
        assert stream.getvalue() == model_bytes
        args.model.write_bytes(b'replaced after validation')
        args.descriptor.write_bytes(b'replaced after validation')
        return original_load(stream)

    monkeypatch.setattr(external.joblib, 'load', replace_paths_then_load)
    bundle, _, provenance = external.load_frozen_inputs(args)
    assert bundle['model_version']
    assert provenance['model']['sha256'] == hashlib.sha256(model_bytes).hexdigest()
    assert provenance['model']['descriptor_sha256'] == hashlib.sha256(descriptor_bytes).hexdigest()


def test_compatibility_loads_verified_bytes_and_cites_descriptor_snapshot(
    tmp_path, synthetic_bundle, monkeypatch,
):
    model, descriptor, _ = write_package(tmp_path, synthetic_bundle)
    original_bytes = model.read_bytes()
    descriptor_bytes = descriptor.read_bytes()
    original_load = compatibility.joblib.load

    def replace_paths_then_load(stream):
        assert isinstance(stream, io.BytesIO)
        assert stream.getvalue() == original_bytes
        model.write_bytes(b'replaced after validation')
        descriptor.write_bytes(b'replaced after validation')
        return original_load(stream)

    monkeypatch.setattr(compatibility.joblib, 'load', replace_paths_then_load)
    report = compatibility.inspect_model(model, descriptor)
    assert report['compatible'], report['checks']
    assert report['model_sha256'] == hashlib.sha256(original_bytes).hexdigest()
    assert report['descriptor_sha256'] == hashlib.sha256(descriptor_bytes).hexdigest()


def test_trusted_descriptor_identity_is_checked_before_loading(tmp_path, synthetic_bundle, monkeypatch):
    model, descriptor, _ = write_package(tmp_path, synthetic_bundle)
    expected = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    descriptor.write_bytes(descriptor.read_bytes() + b'\n')

    def never_load(*_args, **_kwargs):
        pytest.fail('untrusted descriptor reached deserialization')

    monkeypatch.setattr(compatibility.joblib, 'load', never_load)
    report = compatibility.inspect_model(model, descriptor, expected_descriptor_sha256=expected)
    assert not report['compatible']
    assert '可信交接' in report['checks'][-1]['error']


@pytest.mark.parametrize('fault', ['demo.csv', 'demo.json', 'demo_origins.parquet'])
def test_frozen_handoff_requires_every_original_demo_source(tmp_path, fault):
    names = ['model.joblib', 'model.json', 'demo.csv', 'demo.json', 'demo_origins.parquet']
    files = []
    for name in names:
        content = ('synthetic handoff ' + name).encode()
        (tmp_path / name).write_bytes(content)
        files.append({'path': 'artifacts/' + name, 'bytes': len(content),
                      'sha256': hashlib.sha256(content).hexdigest()})
    manifest = {'files': files}
    assert len(compatibility.verify_frozen_handoff(
        tmp_path / 'model.joblib', tmp_path / 'model.json', tmp_path / 'demo.csv', manifest)) == 5
    (tmp_path / fault).write_bytes(b'changed source')
    with pytest.raises(compatibility.CompatibilityError, match='可信交接文件不匹配'):
        compatibility.verify_frozen_handoff(
            tmp_path / 'model.joblib', tmp_path / 'model.json', tmp_path / 'demo.csv', manifest)


def test_exclusive_report_publication_is_complete_and_never_overwrites(tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    original_link = engineering_run.os.link
    observed = []

    def checked_link(source, target):
        import json
        assert not output.exists()
        assert json.loads(type(output)(source).read_text(encoding='utf-8')) == {'status': 'unfinished'}
        observed.append('complete before publication')
        original_link(source, target)

    monkeypatch.setattr(engineering_run.os, 'link', checked_link)
    engineering_run.atomic_json(output, {'status': 'unfinished'}, exclusive=True)
    assert observed == ['complete before publication']
    monkeypatch.setattr(engineering_run.os, 'link', original_link)
    with pytest.raises(FileExistsError):
        engineering_run.atomic_json(output, {'status': 'completed'}, exclusive=True)
    assert output.read_text(encoding='utf-8').strip() == '{\n  "status": "unfinished"\n}'
    assert [item.name for item in tmp_path.iterdir()] == ['report.json']


def test_exclusive_report_publication_failure_leaves_no_partial_json(tmp_path, monkeypatch):
    output = tmp_path / 'report.json'

    def denied(*_args):
        raise OSError('synthetic publication failure')

    monkeypatch.setattr(engineering_run.os, 'link', denied)
    with pytest.raises(OSError, match='publication failure'):
        engineering_run.atomic_json(output, {'status': 'unfinished'}, exclusive=True)
    assert list(tmp_path.iterdir()) == []
