"""Replay guard regressions use frozen Git bytes and an isolated temporary workspace."""
import pytest

from scripts import engineering_reproduce as reproduction


@pytest.fixture(scope='module')
def frozen_sources():
    import subprocess
    try: archive = reproduction.source_archive()
    except subprocess.CalledProcessError: pytest.skip('Original historical Git object is absent from the supplied source snapshot; not verified')
    return {name: archive[name] for name in ['configs/data.lock.json', 'app.py']}


def isolated_workspace(tmp_path, monkeypatch, sources):
    work = tmp_path / 'replay'
    monkeypatch.setattr(reproduction, 'WORK', work)
    monkeypatch.setattr(reproduction, 'source_archive', lambda: sources)
    reproduction.prepare_workspace(reuse=False)
    return work


def test_replay_accepts_unchanged_actual_baseline_archive(tmp_path, monkeypatch, frozen_sources):
    work = isolated_workspace(tmp_path, monkeypatch, frozen_sources)
    assert (work / 'configs/data.lock.json').read_bytes() == frozen_sources['configs/data.lock.json']
    assert reproduction.prepare_workspace(reuse=True)['source_hashes_verified']


@pytest.mark.parametrize('reference_ending', [b'\n', b'\r\n'], ids=['reference-LF', 'reference-CRLF'])
@pytest.mark.parametrize('working_ending', [b'\n', b'\r\n'], ids=['working-LF', 'working-CRLF'])
def test_replay_lock_allows_line_endings_on_both_sides(
    tmp_path, monkeypatch, frozen_sources, reference_ending, working_ending
):
    # Derive both representations from the real frozen lock; no parameters are changed.
    normalized = frozen_sources['configs/data.lock.json'].replace(b'\r\n', b'\n')
    sources = {**frozen_sources,
               'configs/data.lock.json': normalized.replace(b'\n', reference_ending)}
    work = isolated_workspace(tmp_path, monkeypatch, sources)
    (work / 'configs/data.lock.json').write_bytes(normalized.replace(b'\n', working_ending))
    assert reproduction.prepare_workspace(reuse=True)['source_hashes_verified']


def test_replay_lock_parameter_change_is_rejected(tmp_path, monkeypatch, frozen_sources):
    work = isolated_workspace(tmp_path, monkeypatch, frozen_sources)
    original = frozen_sources['configs/data.lock.json']
    changed = original.replace(b'"seed": 42', b'"seed": 43')
    assert changed != original, 'Frozen baseline seed must be present in this regression fixture'
    (work / 'configs/data.lock.json').write_bytes(changed)
    with pytest.raises(ValueError, match='configs/data.lock.json'):
        reproduction.prepare_workspace(reuse=True)


def test_replay_app_source_change_is_rejected(tmp_path, monkeypatch, frozen_sources):
    work = isolated_workspace(tmp_path, monkeypatch, frozen_sources)
    (work / 'app.py').write_bytes(frozen_sources['app.py'] + b'\n# changed application source\n')
    with pytest.raises(ValueError, match='app.py'):
        reproduction.prepare_workspace(reuse=True)
