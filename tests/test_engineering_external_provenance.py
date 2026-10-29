"""Missing source provenance must stop evaluation before model deserialization."""
import pytest

from liushao.data import write_json
from scripts import engineering_external as external
from test_engineering_external import inputs, load  # noqa: F401


REQUIRED_SOURCES = ('liushao/__init__.py', 'liushao/model.py', 'liushao/data.py')


@pytest.mark.parametrize('fault', [
    'missing', 'empty', 'not_mapping', 'null',
    *REQUIRED_SOURCES,
    'outside_module', 'not_python', 'invalid_digest', 'non_string_digest', 'changed_source',
])
def test_invalid_source_registration_stops_before_deserialization(inputs, monkeypatch, fault):
    args, _ = inputs
    descriptor = load(args.descriptor)
    sources = descriptor['source_code_sha256']
    if fault == 'missing':
        del descriptor['source_code_sha256']
    elif fault == 'empty':
        descriptor['source_code_sha256'] = {}
    elif fault == 'not_mapping':
        descriptor['source_code_sha256'] = []
    elif fault == 'null':
        descriptor['source_code_sha256'] = None
    elif fault in REQUIRED_SOURCES:
        del sources[fault]
    elif fault == 'outside_module':
        sources['../outside.py'] = '0' * 64
    elif fault == 'not_python':
        sources['liushao/metadata.json'] = '0' * 64
    elif fault == 'invalid_digest':
        sources['liushao/model.py'] = 'G' * 64
    elif fault == 'non_string_digest':
        sources['liushao/model.py'] = None
    elif fault == 'changed_source':
        sources['liushao/model.py'] = '0' * 64
    write_json(args.descriptor, descriptor)

    def never_deserialize(*_args, **_kwargs):
        pytest.fail('Invalid source provenance reached model deserialization')

    monkeypatch.setattr(external.joblib, 'load', never_deserialize)
    with pytest.raises(ValueError, match='源码|source_code_sha256'):
        external.run(args)
    out = external.OUTPUT_ROOT / args.run_id
    assert load(out / 'manifest.json')['status'] == 'failed'
    assert not (out / 'metrics.json').exists()
    assert not (out / 'predictions.csv').exists()


@pytest.mark.parametrize('windows_paths', [False, True])
def test_complete_source_registration_preserves_frozen_inputs(inputs, windows_paths):
    args, expected_bundle = inputs
    descriptor = load(args.descriptor)
    if windows_paths:
        descriptor['source_code_sha256'] = {
            name.replace('/', '\\'): digest
            for name, digest in descriptor['source_code_sha256'].items()
        }
        write_json(args.descriptor, descriptor)

    bundle, references, provenance = external.load_frozen_inputs(args)

    assert bundle['config'] == expected_bundle['config']
    assert bundle['model_version'] == expected_bundle['model_version']
    assert len(references['ids']) == 8
    verification = provenance['model']['source_verification']
    assert set(verification) == set(descriptor['source_code_sha256'])
    for name, digest in descriptor['source_code_sha256'].items():
        assert verification[name]['expected_sha256'] == digest
        assert verification[name]['match'] == 'raw'


@pytest.mark.parametrize('windows_paths', [False, True])
def test_every_registered_source_is_checked_including_reviews(inputs, windows_paths):
    """A registered digest is an actual check even for non-inference modules."""
    args, expected_bundle = inputs
    descriptor = load(args.descriptor)
    review_path = 'liushao/reviews.py'
    if windows_paths:
        review_path = review_path.replace('/', '\\')
    descriptor['source_code_sha256'][review_path] = '0' * 64
    write_json(args.descriptor, descriptor)

    with pytest.raises(ValueError, match='源码内容不匹配'):
        external.load_frozen_inputs(args)


@pytest.mark.parametrize('fault', ['invalid_digest', 'outside_module', 'not_python'])
def test_non_inference_registration_still_validates_metadata(inputs, monkeypatch, fault):
    args, _ = inputs
    descriptor = load(args.descriptor)
    name = {'invalid_digest': 'liushao/reviews.py', 'outside_module': '../reviews.py',
            'not_python': 'liushao/reviews.json'}[fault]
    descriptor['source_code_sha256'][name] = 'invalid' if fault == 'invalid_digest' else '0' * 64
    write_json(args.descriptor, descriptor)

    def never_deserialize(*_args, **_kwargs):
        pytest.fail('Malformed non-inference registration reached deserialization')

    monkeypatch.setattr(external.joblib, 'load', never_deserialize)
    with pytest.raises(ValueError, match='源码'):
        external.load_frozen_inputs(args)
