"""Publishing a side branch cannot exempt new report versions from inspection."""
import json

import pytest

from test_engineering_push_guard import (  # noqa: F401
    commit, git, guard, published_merge, repository, ZERO,
)


def test_published_side_reports_still_block_against_explicit_main_base(published_merge):
    repo, base, _, _, incoming = published_merge
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert incoming <= {item['commit'] for item in report['findings']}


@pytest.mark.parametrize('previous_ref', ['new', 'existing'])
def test_hook_main_boundary_cannot_exempt_published_then_deleted_report(
    repository, tmp_path, previous_ref,
):
    repo, base = repository
    remote = tmp_path / 'remote-main.git'
    git(repo, 'clone', '--bare', str(repo), str(remote))
    git(repo, 'remote', 'add', 'origin', str(remote))
    git(repo, 'checkout', '-b', 'side', base)
    forbidden = commit(repo, 'reports/local.md', '# test report\nlocal fixture\n')
    git(repo, 'rm', 'reports/local.md')
    git(repo, 'commit', '-m', 'test: remove from final tree')
    git(repo, 'push', 'origin', 'side')
    git(repo, 'checkout', 'work')
    commit(repo, 'app.py', 'pass\n')
    git(repo, 'merge', '--no-ff', 'side', '-m', 'test: merge published side')
    head = git(repo, 'rev-parse', 'HEAD')
    if previous_ref == 'existing':
        git(repo, 'push', 'origin', 'work')
        previous = head
        head = commit(repo, 'README.md', '# Usage\n')
    else:
        previous = ZERO
    update = f'refs/heads/work {head} refs/heads/work {previous}\n'
    result = guard(repo, '--hook', 'origin', str(remote), stdin=update)
    assert result.returncode == 1, result.stderr + result.stdout
    assert forbidden in {item['commit'] for item in json.loads(result.stdout)['findings']}


@pytest.mark.parametrize('failure', ['missing-main', 'main-object-not-fetched'])
def test_hook_blocks_when_remote_main_cannot_be_locally_verified(repository, tmp_path, failure):
    repo, base = repository
    remote = tmp_path / 'remote.git'
    git(repo, 'clone', '--bare', str(repo), str(remote))
    if failure == 'missing-main':
        git(remote, 'update-ref', '-d', 'refs/heads/main')
    else:
        publisher = tmp_path / 'main-publisher'
        git(repo, 'clone', str(remote), str(publisher))
        git(publisher, 'checkout', 'main')
        git(publisher, 'config', 'user.name', 'Main fixture')
        git(publisher, 'config', 'user.email', 'fixture@example.invalid')
        commit(publisher, 'README.md', '# Main update\n')
        git(publisher, 'push', 'origin', 'main')
    update = f'refs/heads/work {base} refs/heads/work {ZERO}\n'
    result = guard(repo, '--hook', 'origin', str(remote), stdin=update)
    assert result.returncode == 2
    assert 'Push blocked:' in result.stderr
