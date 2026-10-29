"""Exercise outgoing history in disposable Git repositories, never the real remote."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/engineering_push_guard.py'
ZERO = '0' * 40


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo).decode().strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / 'working copy'
    repo.mkdir()
    git(repo, 'init', '-b', 'work')
    git(repo, 'config', 'user.name', 'Guard fixture')
    git(repo, 'config', 'user.email', 'guard@example.invalid')
    commit(repo, 'docs/PROGRESS.md', '# historical progress\n')
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'branch', 'main', base)
    return repo, base


def commit(repo, path, content, message='test: fixture change'):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding='utf-8')
    git(repo, 'add', '--', path)
    git(repo, 'commit', '-m', message)
    return git(repo, 'rev-parse', 'HEAD')


def guard(repo, *args, stdin=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=repo,
                          input=stdin, capture_output=True, text=True, encoding='utf-8')


@pytest.fixture
def published_merge(repository, tmp_path):
    """A local bare remote already owns legacy reports on a separate branch."""
    repo, base = repository
    remote = tmp_path / 'published remote.git'
    git(repo, 'clone', '--bare', str(repo), str(remote))
    git(repo, 'remote', 'add', 'origin', str(remote))
    git(repo, 'checkout', '-b', 'mvp', base)
    progress = commit(repo, 'docs/engineering/PROGRESS.md',
                      '# historical progress\npublished checkpoint\n')
    verification = commit(repo, 'docs/handoff/VERIFICATION.json', '{"legacy": true}\n')
    side = commit(repo, 'docs/handoff/ACCEPTANCE.md',
                  '# Verification evidence\nPublished legacy acceptance.\n')
    git(repo, 'push', 'origin', 'mvp')
    git(repo, 'checkout', 'work')
    commit(repo, 'app.py', 'pass\n')
    git(repo, 'merge', '--no-ff', 'mvp', '-m', 'test: inherit published baseline')
    return repo, base, remote, side, {progress, verification, side}


def test_published_side_reports_are_checked_against_explicit_base(published_merge):
    repo, base, _, _, incoming = published_merge
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert incoming <= {item['commit'] for item in json.loads(result.stdout)['findings']}


def test_normal_code_and_usage_docs_pass_with_historical_report_untouched(repository):
    repo, base = repository
    commit(repo, 'app.py', 'print("hello")\n')
    commit(repo, 'README.md', '# Usage\nKeep local reports outside the repository.\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['commits_checked'] == 2 and report['findings'] == []


@pytest.mark.parametrize('existing', [False, True], ids=['create', 'update'])
def test_formal_progress_path_allows_ordinary_project_checkpoints(repository, existing):
    repo, base = repository
    if not existing:
        git(repo, 'rm', 'docs/PROGRESS.md')
        git(repo, 'commit', '-m', 'test: remove fixture checkpoint')
        base = git(repo, 'rev-parse', 'HEAD')
    commit(repo, 'docs/PROGRESS.md', '# Project progress\nApplication checkpoint.\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['findings'] == []


@pytest.mark.parametrize('path', [
    'progress.md', 'PROGRESS.md', 'docs/progress.md', 'DOCS/PROGRESS.md',
    'nested/docs/PROGRESS.md', 'docs/engineering/PROGRESS.md',
    'reports/docs/PROGRESS.md', 'docs\\PROGRESS.md',
])
def test_formal_progress_path_does_not_allow_other_locations_or_aliases(path):
    from scripts.engineering_push_guard import local_path

    assert local_path(path)


@pytest.mark.parametrize('content, expected_rule', [
    ('ghp_' + 'a' * 36, 'github_token'),
    ('sk-' + 'a' * 40, 'openai_style_key'),
    ('AKIA' + 'A' * 16, 'aws_access_id'),
    ('-----BEGIN ' + 'PRIVATE KEY-----', 'private_key'),
    ('password = "' + 'a' * 24 + '"', 'credential_assignment'),
    ('# ' + 'Progress report\nlocal-only fixture', 'local_report_heading'),
    ('User' + ': local-only fixture', 'conversation_role'),
], ids=['github', 'api-key', 'aws', 'private-key', 'password', 'report', 'conversation'])
def test_formal_progress_file_still_rejects_unsafe_content(repository, content, expected_rule):
    repo, base = repository
    commit(repo, 'docs/PROGRESS.md', content + '\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    findings = json.loads(result.stdout)['findings']
    assert any(item['path'] == 'docs/PROGRESS.md' and item['rule'] == expected_rule
               for item in findings)
    assert not any(item['rule'] == 'local_only_path' for item in findings)
    assert content not in result.stdout + result.stderr


def test_formal_progress_scans_existing_secrets_even_when_only_safe_lines_are_added(repository):
    repo, _ = repository
    content = '# Project progress\n' + 'ghp_' + 'a' * 36 + '\n'
    base = commit(repo, 'docs/PROGRESS.md', content)
    commit(repo, 'docs/PROGRESS.md', content + 'Application checkpoint.\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert any(item['rule'] == 'github_token'
               for item in json.loads(result.stdout)['findings'])
    assert content not in result.stdout + result.stderr


def test_formal_progress_file_still_obeys_payload_size_limit(repository):
    repo, base = repository
    commit(repo, 'docs/PROGRESS.md', 'x' * (5 * 1024 * 1024 + 1))
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert any(item['rule'] == 'large_blob'
               for item in json.loads(result.stdout)['findings'])


@pytest.mark.parametrize('path', [
    'progress.md', 'docs/other/PROGRESS.md', 'progress-reports/checkpoint.md',
    'test-results/checkpoint.md', 'artifacts/model.joblib', 'data/reviews.sqlite',
    '.env', 'config/secrets.toml', 'config/private.key',
])
def test_progress_exception_preserves_report_and_payload_path_blocks(repository, path):
    repo, base = repository
    commit(repo, path, 'local-only fixture\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert any(item['path'] == path for item in json.loads(result.stdout)['findings'])


def test_unchanged_conversation_example_in_usage_doc_does_not_block_prose_edit(repository):
    repo, _ = repository
    base = commit(repo, 'README.md', '# Usage\nUser: existing public example\n')
    commit(repo, 'README.md', '# Usage\nUser: existing public example\nNew usage detail.\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('path', [
    'reports/result.md', 'nested/chat-history/session.jsonl',
    'docs/test-report.txt.bak', 'docs/测试报告.pdf', 'docs/对话摘要.md',
    'memory-snapshots/request.md', 'docs/engineering/VERIFICATION.json',
    'docs/engineering/PROGRESS.md', 'docs/要求快照.zip',
    'chat.json', 'docs/test_report.md.backup',
])
def test_forbidden_paths_are_blocked_even_when_gitignore_allows_them(repository, path):
    repo, base = repository
    bad = commit(repo, path, 'local-only fixture\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    findings = json.loads(result.stdout)['findings']
    assert any(item['commit'] == bad and item['path'] == path for item in findings)


def test_checks_intermediate_commits_after_file_was_deleted(repository):
    repo, base = repository
    bad = commit(repo, 'chat-logs/session.txt', 'local-only fixture\n')
    git(repo, 'rm', 'chat-logs/session.txt')
    git(repo, 'commit', '-m', 'test: remove fixture')
    assert git(repo, 'diff', '--name-only', base, 'HEAD') == ''
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert json.loads(result.stdout)['findings'][0]['commit'] == bad


def test_editing_historical_report_is_blocked_but_it_is_never_deleted(repository):
    repo, base = repository
    base = commit(repo, 'docs/PROGRESS.md', '# Progress report\nlocal-only fixture\n')
    commit(repo, 'docs/PROGRESS.md', '# Project progress\nnew local record\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert any(item['rule'] == 'local_report_heading'
               for item in json.loads(result.stdout)['findings'])
    assert (repo / 'docs/PROGRESS.md').is_file()


def test_custom_hook_directory_is_preserved(repository, tmp_path):
    repo, _ = repository
    shared = tmp_path / 'shared-hooks'
    shared.mkdir()
    git(repo, 'config', 'core.hooksPath', str(shared))
    assert guard(repo, '--install').returncode == 2
    assert list(shared.iterdir()) == []


@pytest.mark.parametrize('content', [
    '# ' + '测试报告\nlocal-only fixture\n',
    'User' + ': local-only fixture\nAssistant' + ': reply\n',
    '{"type": "' + 'response_item", "payload": {}}\n',
])
def test_report_or_conversation_pasted_in_ordinary_file_is_blocked(repository, content):
    repo, base = repository
    commit(repo, 'README.md', '# Usage\n' + content)
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert 'local-only fixture' not in result.stdout + result.stderr


def test_conversation_in_commit_message_is_blocked_without_echoing_message(repository):
    repo, base = repository
    commit(repo, 'app.py', 'pass\n', 'fix: example\n\nUser' + ': private fixture')
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert 'private fixture' not in result.stdout + result.stderr


SECRET_CASES = [
    ('ghp_' + 'a' * 36, 'github_token'),
    ('sk-' + 'a' * 40, 'openai_style_key'),
    ('AKIA' + 'A' * 16, 'aws_access_id'),
    ('-----BEGIN ' + 'PRIVATE KEY-----', 'private_key'),
    ('password = "' + 'a' * 24 + '"', 'credential_assignment'),
]


@pytest.mark.parametrize('content, expected_rule', SECRET_CASES)
def test_secret_in_commit_message_is_blocked_without_echoing_message(
    repository, content, expected_rule,
):
    repo, base = repository
    commit(repo, 'app.py', 'pass\n', 'fix: fixture\n\n' + content)
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert any(item['path'] == '(commit message)' and item['rule'] == expected_rule
               for item in json.loads(result.stdout)['findings'])
    assert content not in result.stdout + result.stderr


@pytest.mark.parametrize('encoding, bom', [
    ('utf-16-le', b'\xff\xfe'), ('utf-16-be', b'\xfe\xff'),
])
@pytest.mark.parametrize('content, expected_rule', SECRET_CASES)
def test_utf16_formal_progress_scans_complete_file_for_secrets(
    repository, encoding, bom, content, expected_rule,
):
    repo, _ = repository
    original = '# Project progress\n' + content + '\n'
    base = commit(repo, 'docs/PROGRESS.md', bom + original.encode(encoding))
    updated = original + 'Application checkpoint.\n'
    commit(repo, 'docs/PROGRESS.md', bom + updated.encode(encoding))
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert any(item['path'] == 'docs/PROGRESS.md' and item['rule'] == expected_rule
               for item in json.loads(result.stdout)['findings'])
    assert content not in result.stdout + result.stderr


@pytest.mark.parametrize('encoding, bom', [
    ('utf-16-le', b'\xff\xfe'), ('utf-16-be', b'\xfe\xff'),
])
def test_utf16_formal_progress_allows_safe_project_updates(repository, encoding, bom):
    repo, base = repository
    content = '# Project progress\nApplication checkpoint.\n'
    commit(repo, 'docs/PROGRESS.md', bom + content.encode(encoding))
    result = guard(repo, '--base', base)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('path, content', [
    ('README.md', '# Usage\n用户: private fixture\n助手: reply\n'.encode('gb18030')),
    ('docs/PROGRESS.md', b'# Project progress\n\xffinvalid text\n'),
    ('app.py', b'# source\n\xffinvalid text\n'),
    ('docs/PROGRESS.md', b'\xff\xfe#\x00\x01'),
    ('docs/PROGRESS.md', b'\xfe\xff\x00#\x01'),
    ('docs/checkpoint', b'\xff\xfe#\x00\x01'),
    ('docs/checkpoint', b'\xfe\xff\x00#\x01'),
])
def test_undecodable_project_text_blocks_without_echoing_content(repository, path, content):
    repo, base = repository
    commit(repo, path, content)
    result = guard(repo, '--base', base)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'Push blocked:' in result.stderr
    assert 'private fixture' not in result.stdout + result.stderr
    assert 'invalid text' not in result.stdout + result.stderr


def test_reencoding_legacy_usage_text_does_not_block_a_valid_update(repository):
    repo, _ = repository
    original = '# 使用说明\n运行应用。\n'
    base = commit(repo, 'README.md', original.encode('gb18030'))
    commit(repo, 'README.md', original + '补充新的启动说明。\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reencoding_legacy_report_does_not_exempt_its_heading(repository):
    repo, _ = repository
    original = '# ' + '进度报告\n本地记录。\n'
    base = commit(repo, 'docs/PROGRESS.md', original.encode('gb18030'))
    commit(repo, 'docs/PROGRESS.md', '# Project progress\nApplication checkpoint.\n')
    result = guard(repo, '--base', base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert any(item['rule'] == 'local_report_heading'
               for item in json.loads(result.stdout)['findings'])


@pytest.mark.parametrize('encoding, bom', [
    ('utf-16-le', b''), ('utf-16-be', b''),
    ('utf-32-le', b'\xff\xfe\x00\x00'), ('utf-32-be', b'\x00\x00\xfe\xff'),
])
@pytest.mark.parametrize('content', [
    'User' + ': private fixture\n', 'ghp_' + 'a' * 36 + '\n',
])
def test_nul_encoded_text_cannot_hide_conversations_or_secrets(
    repository, encoding, bom, content,
):
    repo, base = repository
    commit(repo, 'README.md', bom + content.encode(encoding))
    result = guard(repo, '--base', base)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'Push blocked:' in result.stderr
    assert 'private fixture' not in result.stdout + result.stderr
    assert content.strip() not in result.stdout + result.stderr


@pytest.mark.parametrize('encoding', ['utf-16-le', 'utf-16-be'])
def test_nul_encoded_commit_message_is_rejected(repository, encoding):
    repo, base = repository
    tree = git(repo, 'rev-parse', f'{base}^{{tree}}')
    headers = (
        f'tree {tree}\nparent {base}\n'
        'author Guard fixture <fixture@example.invalid> 1 +0000\n'
        'committer Guard fixture <fixture@example.invalid> 1 +0000\n\n'
    ).encode('ascii')
    message = ('User' + ': private fixture\n').encode(encoding)
    # Git plumbing can store byte messages that ordinary `git commit -m` rejects.
    result = subprocess.run(
        ['git', 'hash-object', '--literally', '-w', '-t', 'commit', '--stdin'],
        cwd=repo, input=headers + message, capture_output=True, check=True,
    )
    git(repo, 'update-ref', 'HEAD', result.stdout.decode('ascii').strip())
    result = guard(repo, '--base', base)
    assert result.returncode == 2, result.stdout + result.stderr
    assert '(commit message)' in result.stderr
    assert 'private fixture' not in result.stdout + result.stderr


def test_binary_png_remains_allowed_when_it_cannot_decode_as_text(repository):
    import base64

    repo, base = repository
    image = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8'
        '/x8AAusB9Wl2nDAAAAAASUVORK5CYII='
    )
    commit(repo, 'docs/images/pixel.png', image)
    result = guard(repo, '--base', base)
    assert result.returncode == 0, result.stdout + result.stderr


def test_merged_side_branch_history_is_checked(repository):
    repo, base = repository
    git(repo, 'checkout', '-b', 'side')
    bad = commit(repo, 'progress-reports/local.txt', 'local-only fixture')
    git(repo, 'rm', 'progress-reports/local.txt')
    git(repo, 'commit', '-m', 'test: remove fixture')
    git(repo, 'checkout', 'work')
    commit(repo, 'app.py', 'pass\n')
    git(repo, 'merge', '--no-ff', 'side', '-m', 'test: merge fixture')
    result = guard(repo, '--base', base)
    assert result.returncode == 1
    assert any(item['commit'] == bad for item in json.loads(result.stdout)['findings'])


def test_hook_checks_all_updates_against_actual_remote_main(repository, tmp_path):
    repo, base = repository
    bad = commit(repo, 'reports/result.txt', 'local-only fixture')
    git(repo, 'update-ref', 'refs/remotes/origin/work', bad)
    remote = tmp_path / 'remote-main.git'
    git(repo, 'clone', '--bare', str(repo), str(remote))
    updates = f'refs/heads/work {bad} refs/heads/work {base}\n'
    result = guard(repo, '--hook', 'origin', str(remote), stdin=updates)
    assert result.returncode == 1
    updates = (f'refs/heads/work {base} refs/heads/work {base}\n'
               f'refs/heads/side {bad} refs/heads/side {base}\n')
    assert guard(repo, '--hook', 'origin', str(remote), stdin=updates).returncode == 1


def test_empty_range_and_deletion_pass_but_missing_base_or_malformed_input_fail(repository):
    repo, base = repository
    assert guard(repo, '--base', base).returncode == 0
    assert guard(repo, '--hook', 'origin', 'unused',
                 stdin=f'(delete) {ZERO} refs/heads/work {base}\n').returncode == 0
    assert guard(repo, '--base', 'missing-ref').returncode == 2
    assert guard(repo, '--hook', 'origin', 'unused', stdin='malformed\n').returncode == 2


def test_new_remote_branch_uses_remote_main_as_history_boundary(repository, tmp_path):
    repo, _ = repository
    remote = tmp_path / 'remote.git'
    git(repo, 'clone', '--bare', str(repo), str(remote))
    commit(repo, 'app.py', 'pass\n')
    head = git(repo, 'rev-parse', 'HEAD')
    update = f'refs/heads/work {head} refs/heads/new {ZERO}\n'
    result = guard(repo, '--hook', 'origin', str(remote), stdin=update)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['commits_checked'] == 1


def test_installed_hook_runs_on_real_local_push_and_preserves_existing_hooks(repository, tmp_path):
    repo, base = repository
    scripts = repo / 'scripts'
    scripts.mkdir()
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    shutil.copyfile(SCRIPT.with_name('git_payload_phase2.py'), scripts / 'git_payload_phase2.py')
    # A test-only launcher uses the same interpreter as the project test process.
    python = repo / '.venv/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    if sys.platform == 'win32':
        import venv
        venv.EnvBuilder(with_pip=False).create(repo / '.venv')
    else:
        python = repo / '.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
    installed = guard(repo, '--install')
    assert installed.returncode == 0, installed.stderr
    assert guard(repo, '--install').returncode == 0
    remote = tmp_path / 'remote.git'
    git(repo, 'clone', '--bare', str(repo), str(remote))
    git(repo, 'remote', 'add', 'origin', str(remote))
    good = commit(repo, 'docs/PROGRESS.md', '# Project progress\nApplication checkpoint.\n')
    assert subprocess.run(['git', 'push', 'origin', 'work'], cwd=repo,
                          capture_output=True).returncode == 0
    assert git(remote, 'rev-parse', 'refs/heads/work') == good
    commit(repo, 'reports/result.md', 'local-only fixture')
    rejected = subprocess.run(['git', 'push', 'origin', 'work'], cwd=repo,
                              capture_output=True)
    assert rejected.returncode != 0
    assert git(remote, 'rev-parse', 'refs/heads/work') == good
    hook = repo / '.git/hooks/pre-push'
    hook.write_text('#!/bin/sh\n# custom hook\n', encoding='utf-8')
    assert guard(repo, '--install').returncode == 2
    assert hook.read_text(encoding='utf-8') == '#!/bin/sh\n# custom hook\n'
