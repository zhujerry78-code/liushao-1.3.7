"""Synthetic release metadata exercises refined integrity without loading a model."""
import hashlib
import json
from pathlib import Path

import pytest

from delivery import verify_demo
from scripts import build_windows_demo as builder


SOURCES = (
    'docs/release/FROZEN_TEST_20261016.json',
    'docs/release/MODEL_COMPARISON_20261016.json',
    'docs/results/metrics.json',
    'docs/innovation/study.json',
    'docs/phase2/audit.json',
    'docs/results/data_summary.json',
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def digest(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def refined_package(tmp_path, monkeypatch):
    """A complete synthetic manifest tests verification, not estimator semantics."""
    names = set(builder.FILES) | {
        'SOURCE_MANIFEST.json', 'delivery/desktop_start.py', 'delivery/evaluation_resources.py',
        'delivery/preflight.py', 'delivery/service_probe.py', 'delivery/migrate_reviews.py',
        'delivery/browser-activate.cmd', 'run-demo.cmd', 'install.cmd',
        '4_关闭流哨.cmd', '5_迁移旧版复核.cmd', 'liushao/model.py', 'liushao/data.py',
        'liushao/reviews.py', 'liushao/reference.py', 'liushao/feature_metadata.py',
        'artifacts/model.joblib', 'artifacts/model.json', 'artifacts/model.original.json',
        'artifacts/demo.csv', 'artifacts/demo.json', 'artifacts/demo_origins.parquet',
        'resources/evaluation/summary.json', 'resources/evaluation/model.original.json',
        'resources/evaluation/model.release.json', 'resources/evaluation/reevaluation.json',
        'resources/examples/格式练习_缺少必需列.csv', 'resources/examples/格式练习_缺值与无效记录.csv',
        'scripts/reevaluate_frozen.py', 'EVALUATION_RESOURCES.json', *SOURCES,
    }
    members = {name: ('synthetic file: ' + name).encode('utf-8') for name in names}
    members['VERSION'] = b'1.3.6\n'
    members['RELEASE_DATE'] = b'2026-10-24\n'
    model_sha = digest(members['artifacts/model.joblib'])
    original = {
        'model_sha256': model_sha, 'model_version': 'synthetic-refined-metadata',
        'config': {'features': ['Flow Duration'], 'rf_threshold': .12, 'lr_threshold': .84,
                   'near_margin': .1, 'data_version': 'synthetic-data-version'},
        'reference': {'Flow Duration': {'q01': 0., 'median': 1., 'q99': 2.}},
        'training_rows': 3, 'normal_reference_rows': 2,
    }
    original_bytes = encoded(original)
    original_sha = digest(original_bytes)
    descriptor = dict(original, training_descriptor_sha256=original_sha)
    descriptor_bytes = encoded(descriptor)
    for name in ('artifacts/model.original.json', 'resources/evaluation/model.original.json'):
        members[name] = original_bytes
    for name in ('artifacts/model.json', 'resources/evaluation/model.release.json'):
        members[name] = descriptor_bytes
    identity = {'model_version': original['model_version'], 'model_sha256': model_sha,
                'descriptor_sha256': original_sha, 'data_version': original['config']['data_version']}
    info = {key: original[key] for key in ('training_rows', 'normal_reference_rows')}
    info.update({key: original['config'][key] for key in ('features', 'rf_threshold', 'lr_threshold', 'near_margin')})
    summary = {'format': 1, 'identity': identity, 'model_info': info,
               'sources': [{'path': name, 'sha256': digest(members[name])} for name in SOURCES]}
    members['resources/evaluation/summary.json'] = encoded(summary)
    evaluation_names = {'resources/evaluation/summary.json', 'resources/evaluation/model.original.json',
                        'resources/evaluation/model.release.json', 'resources/evaluation/reevaluation.json', *SOURCES}
    members['EVALUATION_RESOURCES.json'] = encoded({'format': 1, 'identity': identity,
        'files': {name: {'bytes': len(members[name]), 'sha256': digest(members[name])}
                  for name in evaluation_names}})
    members['SOURCE_MANIFEST.json'] = encoded({'format': 1, 'source_commit': 'a'*40,
        'files': {'app.py': digest(members['app.py']), 'liushao/model.py': digest(members['liushao/model.py'])}})
    manifest = {'format': 1, 'software_sha': 'a'*40, 'software_version': '1.3.6',
                'release_date': '2026-10-24', 'package_revision': 'r11-whitebox-refined',
                'evaluation_summary_sha256': digest(members['resources/evaluation/summary.json']),
                'files': {name: {'bytes': len(value), 'sha256': digest(value)} for name, value in members.items()}}
    for name, value in members.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (tmp_path/'PACKAGE_MANIFEST.json').write_bytes(encoded(manifest))
    monkeypatch.setattr(verify_demo, 'MODEL', model_sha)
    monkeypatch.setattr(verify_demo, 'DESCRIPTOR', digest(descriptor_bytes))
    return tmp_path, manifest


def save_manifest(root, manifest):
    (root/'PACKAGE_MANIFEST.json').write_bytes(encoded(manifest))


def test_refined_metadata_and_all_six_sources_verify(refined_package):
    root, manifest = refined_package
    result = verify_demo.verify(root)
    assert result['software_version'] == '1.3.6'
    assert set(SOURCES) <= set(result['files'])


@pytest.mark.parametrize('name', SOURCES)
def test_each_missing_historical_source_gets_chinese_recovery(refined_package, name):
    root, manifest = refined_package
    (root/name).unlink()
    with pytest.raises(FileNotFoundError) as raised:
        verify_demo.verify(root)
    text = str(raised.value)
    assert name in text and '交付文件缺失' in text
    assert '保留旧目录和复核库' in text and '重新完整解压' in text


@pytest.mark.parametrize('name', SOURCES)
def test_each_changed_historical_source_remains_blocking(refined_package, name):
    root, manifest = refined_package
    (root/name).write_bytes(b'changed historical evidence')
    with pytest.raises(ValueError, match='交付文件内容与清单不符'):
        verify_demo.verify(root)


@pytest.mark.parametrize('name', ['liushao/reference.py', 'liushao/feature_metadata.py', 'artifacts/model.original.json'])
def test_refined_manifest_cannot_omit_new_required_files(refined_package, name):
    root, manifest = refined_package
    del manifest['files'][name]
    save_manifest(root, manifest)
    with pytest.raises(ValueError, match='Incomplete file manifest'):
        verify_demo.verify(root)


def test_refined_cannot_rehash_an_old_version_into_its_manifest(refined_package):
    root, manifest = refined_package
    content = b'1.3.5\n'
    (root/'VERSION').write_bytes(content)
    manifest['software_version'] = '1.3.5'
    manifest['files']['VERSION'] = {'bytes': len(content), 'sha256': digest(content)}
    save_manifest(root, manifest)
    with pytest.raises(ValueError, match='Unexpected software version'):
        verify_demo.verify(root)


def test_aligned_keeps_its_own_descriptor_pin(refined_package, monkeypatch):
    root, manifest = refined_package
    content = b'1.3.5\n'
    (root/'VERSION').write_bytes(content)
    manifest.update(software_version='1.3.5', package_revision='r11-whitebox-aligned')
    manifest['files']['VERSION'] = {'bytes':len(content), 'sha256':digest(content)}
    save_manifest(root, manifest)
    monkeypatch.setattr(verify_demo, 'ALIGNED_DESCRIPTOR', '0'*64)
    with pytest.raises(ValueError, match='Unexpected trusted descriptor identity'):
        verify_demo.verify(root)
    monkeypatch.setattr(verify_demo, 'ALIGNED_DESCRIPTOR', manifest['files']['artifacts/model.json']['sha256'])
    assert verify_demo.verify(root)['software_version'] == '1.3.5'


def test_refined_builder_demands_evaluation_before_assets(tmp_path, monkeypatch):
    source = {name: (builder.ROOT/name).read_bytes() for name in
              ('scripts/build_windows_demo.py', 'delivery/safe_files.py')}
    monkeypatch.setattr(builder, 'read_source', lambda *args: ('a'*40, source))
    with pytest.raises(ValueError, match='require an evaluation summary'):
        builder.build(tmp_path/'output', package_revision='r11-whitebox-refined', assets_zip=tmp_path/'absent.zip')
    assert not (tmp_path/'output').exists()
