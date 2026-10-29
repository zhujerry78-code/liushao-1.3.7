"""The full engineering gate must not inherit a developer's pytest selection."""
import os
import sys

import pytest

from scripts import engineering_ci as ci


@pytest.mark.parametrize('selection', ['-k test_selected', '-m gate_selected'])
def test_gate_runs_required_failure_despite_inherited_selection(
    tmp_path, monkeypatch, selection,
):
    (tmp_path / 'pytest.ini').write_text(
        '[pytest]\nmarkers =\n    gate_selected: selected by a developer shell\n',
        encoding='utf-8',
    )
    (tmp_path / 'test_gate.py').write_text(
        'import pytest\n'
        '@pytest.mark.gate_selected\n'
        'def test_selected():\n    assert True\n'
        'def test_required_other():\n    assert False, "required regression ran"\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(ci, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTEST_ADDOPTS', selection)
    junit = tmp_path / 'pytest.xml'
    result = ci.run_command(
        [sys.executable, '-m', 'pytest', 'test_gate.py', '-q', '-p', 'no:cacheprovider',
         f'--junitxml={junit}'],
        tmp_path / 'pytest.log', timeout=30,
    )

    assert result['passed'] is False
    assert result['returncode'] == 1 and result['cleanup']['passed']
    counts = ci.test_counts(junit)
    assert counts['valid'] and not counts['successful']
    assert counts['total'] == 2 and counts['failures'] == 1
    assert os.environ['PYTEST_ADDOPTS'] == selection


def test_other_commands_retain_inherited_pytest_options(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k developer_filter')
    result = ci.run_command(
        [sys.executable, '-c', 'import os; print(os.environ["PYTEST_ADDOPTS"])'],
        tmp_path / 'other.log', timeout=30,
    )

    assert result['passed'] and result['returncode'] == 0
    assert (tmp_path / 'other.log').read_text(encoding='utf-8').strip() == '-k developer_filter'
    assert os.environ['PYTEST_ADDOPTS'] == '-k developer_filter'
