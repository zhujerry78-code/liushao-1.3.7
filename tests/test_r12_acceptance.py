import json
import sys

import pytest

from scripts import engineering_r11_acceptance as acceptance


def test_failed_browser_keeps_prior_stage_evidence_and_never_claims_completion(tmp_path, monkeypatch):
    package = tmp_path/'package';package.mkdir()
    (package/'PACKAGE_MANIFEST.json').write_text('{}')
    output = tmp_path/'proof'
    monkeypatch.setattr(acceptance, 'verified_imports', lambda root: ({'software_sha':'a'*40},None,None,None))
    monkeypatch.setattr(acceptance, 'prepare_migration', lambda *a: dict(passed=True,records=2))
    def failed(*a):raise RuntimeError('injected browser failure')
    monkeypatch.setattr(acceptance, 'browser_flow', failed)
    monkeypatch.setattr(sys, 'argv', ['acceptance','--package',str(package),'--output',str(output),'--prepare-migration','--browser'])
    with pytest.raises(RuntimeError, match='injected browser failure'):
        acceptance.main()
    result = json.loads((output/'acceptance.json').read_bytes())
    assert result['software_sha']=='a'*40 and result['passed'] is False
    assert result['checks']['migration']==dict(passed=True,records=2)
    assert 'injected browser failure' in result['error']


def test_acceptance_requires_a_real_stage_instead_of_empty_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['acceptance','--package',str(tmp_path),'--output',str(tmp_path/'proof')])
    with pytest.raises(SystemExit) as error:acceptance.main()
    assert error.value.code==2 and not (tmp_path/'proof').exists()


def test_stage_returning_failure_cannot_turn_into_success(tmp_path, monkeypatch):
    package=tmp_path/'package';package.mkdir();(package/'PACKAGE_MANIFEST.json').write_text('{}')
    output=tmp_path/'proof'
    monkeypatch.setattr(acceptance,'verified_imports',lambda root:({'software_sha':'b'*40},None,None,None))
    monkeypatch.setattr(acceptance,'prepare_migration',lambda *a:dict(passed=False))
    monkeypatch.setattr(sys,'argv',['acceptance','--package',str(package),'--output',str(output),'--prepare-migration'])
    with pytest.raises(AssertionError,match='required acceptance stage'):acceptance.main()
    assert json.loads((output/'acceptance.json').read_bytes())['passed'] is False
