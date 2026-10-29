"""Competition packaging contracts without network or actual installation."""
from pathlib import Path
import sys

import pytest

from delivery.evaluation_resources import check_identity, model_info
from scripts import build_windows_demo as builder
from test_r10_evaluation import evaluation_data


def test_summary_identity_is_checked_before_installed_dependencies(r9_app, evaluation_data):
    import hashlib
    import json
    descriptor_bytes = (r9_app[1]/'model.json').read_bytes()
    descriptor = json.loads(descriptor_bytes)
    evaluation_data['identity']['data_version'] = descriptor['config']['data_version']
    assert check_identity(evaluation_data, descriptor, hashlib.sha256(descriptor_bytes).hexdigest())
    evaluation_data['model_info']['training_rows'] += 1
    with pytest.raises(ValueError):
        check_identity(evaluation_data, descriptor, hashlib.sha256(descriptor_bytes).hexdigest())


@pytest.mark.parametrize('revision', ['r9', 'r10-competition'])
def test_cli_passes_summary_and_preserves_r9_default(monkeypatch, tmp_path, revision):
    calls = []
    args = ['builder', '--source-ref', 'HEAD', '--assets-zip', 'offline.zip', '--output', str(tmp_path/'out')]
    if revision != 'r9':
        args += ['--package-revision', revision, '--evaluation-summary', 'summary.json', '--evaluation-summary-sha256', 'a'*64]
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(builder, 'build', lambda *args, **kwargs: calls.append(kwargs))
    builder.main()
    assert calls[0]['package_revision'] == revision
    assert calls[0]['evaluation_summary'] == (Path('summary.json') if revision != 'r9' else None)
    assert calls[0]['evaluation_summary_sha256'] == ('a'*64 if revision != 'r9' else None)


def test_competition_requires_summary_before_resource_read(tmp_path, monkeypatch):
    source = {name: (builder.ROOT/name).read_bytes() for name in ('scripts/build_windows_demo.py','delivery/safe_files.py')}
    monkeypatch.setattr(builder, 'read_source', lambda *args: ('a'*40, source))
    with pytest.raises(ValueError, match='require an evaluation summary'):
        builder.build(tmp_path/'out', package_revision='r10-competition', assets_zip=tmp_path/'absent.zip')


def test_setup_lock_handoff_uses_actual_parent_even_from_venv(tmp_path, monkeypatch):
    from delivery import oneclick
    child = tmp_path/'check_parent.py'
    child.write_text("import os\nassert os.environ['LIUSHAO_INSTALL_PARENT'] == str(os.getppid())\n", encoding='utf-8')
    monkeypatch.setattr(oneclick, 'ROOT', tmp_path)
    monkeypatch.setattr(oneclick, 'SETUP', child)
    assert oneclick.run_setup('--offline') == 0
