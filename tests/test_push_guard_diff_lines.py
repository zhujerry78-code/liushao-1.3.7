"""Changed-line inspection must distinguish hunk content from diff file headers."""
import json

import pytest

from test_engineering_push_guard import commit, git, guard, repository  # noqa: F401


CHAT_MARKERS = [
    ('<' + 'assistant' + '>fixture</' + 'assistant' + '>', 'conversation_markup'),
    ('{"type": "' + 'response_item' + '", "payload": {}}', 'conversation_rollout'),
]


@pytest.mark.parametrize('path', ['guide.md', 'docs/PROGRESS.md'])
@pytest.mark.parametrize('marker, expected_rule', CHAT_MARKERS)
def test_plus_prefixed_conversation_content_is_not_a_diff_header(
    repository, path, marker, expected_rule,
):
    repo, base = repository
    content = '++' + marker + '\n'
    commit(repo, path, content)

    result = guard(repo, '--base', base)

    assert result.returncode == 1, result.stdout + result.stderr
    findings = json.loads(result.stdout)['findings']
    assert any(item['path'] == path and item['rule'] == expected_rule for item in findings)
    assert marker not in result.stdout + result.stderr


def test_safe_doc_edit_with_existing_marker_checks_only_real_added_lines(repository):
    repo, _ = repository
    original = '# Usage\n' + CHAT_MARKERS[0][0] + '\n'
    base = commit(repo, 'guide.md', original)
    commit(repo, 'guide.md', original + '+++ usage detail\n')

    result = guard(repo, '--base', base)

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['findings'] == []


def test_conversation_detection_ignores_configured_diff_colors(repository):
    repo, base = repository
    git(repo, 'config', 'color.ui', 'always')
    commit(repo, 'guide.md', CHAT_MARKERS[0][0] + '\n')

    result = guard(repo, '--base', base)

    assert result.returncode == 1, result.stdout + result.stderr
    assert any(item['path'] == 'guide.md' and item['rule'] == 'conversation_markup'
               for item in json.loads(result.stdout)['findings'])


@pytest.mark.parametrize('path', ['guide.md', 'docs/PROGRESS.md'])
def test_plus_prefixed_content_keeps_complete_blob_secret_checks(repository, path):
    repo, _ = repository
    token = 'ghp_' + 'a' * 36
    original = '# Project progress\n++' + token + '\n'
    base = commit(repo, path, original)
    commit(repo, path, original + 'Application checkpoint.\n')

    result = guard(repo, '--base', base)

    assert result.returncode == 1, result.stdout + result.stderr
    assert any(item['path'] == path and item['rule'] == 'github_token'
               for item in json.loads(result.stdout)['findings'])
    assert token not in result.stdout + result.stderr


def test_formal_progress_safe_update_remains_allowed(repository):
    repo, base = repository
    commit(repo, 'docs/PROGRESS.md', '# Project progress\nApplication checkpoint.\n')

    result = guard(repo, '--base', base)

    assert result.returncode == 0, result.stdout + result.stderr
