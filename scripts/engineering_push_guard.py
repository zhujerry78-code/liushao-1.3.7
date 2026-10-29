"""Check every outgoing commit before an ordinary push; use only the standard library.

Default: compare HEAD with the locally cached upstream. Git's pre-push hook instead
checks every updated head against the current remote main, including prior side-branch
history. Published side branches and their later deletions cannot exempt report commits.
Findings contain locations and rule names, never file contents or commit messages.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

if __package__:
    from .git_payload_phase2 import (
        PayloadDecodeError, decode_payload_text, inspect_blob, inspect_secrets,
    )
else:  # Git's hook executes this file directly.
    from git_payload_phase2 import (
        PayloadDecodeError, decode_payload_text, inspect_blob, inspect_secrets,
    )


LOCAL_DIRECTORIES = {
    'reports', 'test-reports', 'progress-reports', 'execution-records',
    'validation-evidence', 'verification-evidence', 'chat-logs', 'chat-history',
    'conversations', 'memory-snapshots', 'codex-progress', 'test-results',
}
LOCAL_NAMES = {'progress.md', 'verification.json', 'delivery_evidence.md'}
LOCAL_NAME_PATTERN = re.compile(
    r'测试报告|进度报告|执行记录|验证证据|聊天记录|对话记录|对话摘要|要求快照|需求快照|'
    r'(?:^|[-_. ])(?:test[-_ ]?reports?|progress[-_ ]?reports?|'
    r'execution[-_ ]?(?:records?|logs?)|(?:validation|verification)[-_ ]?evidence|'
    r'chat(?:[-_ ]?(?:logs?|history|records?))?|'
    r'conversation(?:[-_ ]?(?:logs?|summar(?:y|ies)|history))?|'
    r'requirements?[-_ ]?snapshots?)(?:$|[-_. ])', re.IGNORECASE,
)
CONTENT_RULES = {
    'local_report_heading': re.compile(
        r'^\s*#{1,6}\s*(?:测试报告|进度报告|执行记录|验证证据|聊天记录|对话摘要|要求快照|'
        r'(?:test|progress)\s+report|execution\s+record|verification\s+evidence)'
        r'(?:\s|[：:·/\-]|$)', re.MULTILINE | re.IGNORECASE),
    'conversation_role': re.compile(
        r'^\s*(?:#{1,6}\s*)?(?:用户|助手|User|Assistant)\s*[:：]',
        re.MULTILINE | re.IGNORECASE),
    'conversation_markup': re.compile(
        r'<(?:user|assistant|system|developer)(?:>|\s)|<\|(?:im_start|im_end)\|>',
        re.IGNORECASE),
    'conversation_rollout': re.compile(
        r'"type"\s*:\s*"(?:session_meta|response_item|event_msg)"'),
}
HOOK = '''#!/bin/sh
# liushao-push-guard-v1: installed locally; never bypass a failed check.
root=$(git rev-parse --show-toplevel) || exit 1
if [ -x "$root/.venv/Scripts/python.exe" ]; then
    python="$root/.venv/Scripts/python.exe"
elif [ -x "$root/.venv/bin/python" ]; then
    python="$root/.venv/bin/python"
else
    echo "Push blocked: restore the project .venv before checking outgoing commits." >&2
    exit 1
fi
cd "$root" || exit 1
exec "$python" "$root/scripts/engineering_push_guard.py" --hook "$@"
'''


class GuardError(RuntimeError):
    """The outgoing range could not be checked, so the push must stop."""


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(['git', *args], cwd=repo, capture_output=True)
    if result.returncode:
        raise GuardError(f'Git {args[0]} failed; check refs and fetch missing remote objects.')
    return result.stdout


def commit_oid(repo: Path, ref: str) -> str:
    return git(repo, 'rev-parse', '--verify', '--end-of-options',
               f'{ref}^{{commit}}').decode().strip()


def content_rules(content: bytes, path: str | None = None) -> set[str]:
    text = decode_payload_text(content, path)
    if text is None:
        return set()
    return {name for name, pattern in CONTENT_RULES.items() if pattern.search(text)}


def historical_report_rules(content: bytes, path: str) -> set[str]:
    """Recognize old report headings without imposing new encoding rules on main."""
    try:
        return content_rules(content, path) & {'local_report_heading'}
    except PayloadDecodeError:
        # Only the predecessor may use this legacy encoding. Every outgoing version
        # is separately required to pass the current strict text/payload checks.
        try:
            text = content.decode('gb18030')
        except UnicodeError:
            return set()
        if CONTENT_RULES['local_report_heading'].search(text):
            return {'local_report_heading'}
        return set()


def local_path(path: str) -> bool:
    parts = path.replace('\\', '/').lower().split('/')
    return (any(part.replace('_', '-') in LOCAL_DIRECTORIES for part in parts[:-1])
            or (parts[-1] in LOCAL_NAMES and path != 'docs/PROGRESS.md')
            or any(LOCAL_NAME_PATTERN.search(part) for part in parts))


def added_patch_lines(patch: bytes) -> bytes:
    """Read additions inside diff hunks, including content whose line starts with ++."""
    added = []
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith(b'diff --git '):
            in_hunk = False
        elif line.startswith(b'@@ '):
            in_hunk = True
        elif in_hunk and line.startswith(b'+'):
            added.append(line[1:])
    return b'\n'.join(added)


def inspect_commit(repo: Path, oid: str) -> list[dict]:
    findings = []
    # Formatted Git log output can truncate an embedded NUL in a stored message.
    _, separator, message = git(repo, 'cat-file', 'commit', oid).partition(b'\n\n')
    if not separator:
        raise GuardError('Commit message could not be separated from its headers.')
    message_rules = content_rules(message) | inspect_secrets(message, decode_payload_text(message))
    for rule in sorted(message_rules):
        findings.append({'commit': oid, 'path': '(commit message)', 'rule': rule})
    parents = git(repo, 'rev-list', '--parents', '-n', '1', oid).decode().split()[1:]
    comparison = [parents[0], oid] if parents else ['--root', oid]
    entries = git(repo, 'diff-tree', '--no-commit-id', '--raw', '-r', '-z',
                  '--no-renames', '--diff-filter=AMT', *comparison).split(b'\0')
    for index in range(0, len(entries) - 1, 2):
        metadata, raw_path = entries[index:index + 2]
        previous_mode, mode, previous, current, _ = metadata.split()
        path = os.fsdecode(raw_path)
        if local_path(path):
            findings.append({'commit': oid, 'path': path, 'rule': 'local_only_path'})
        if mode == b'160000':
            raise GuardError('Changed submodule content cannot be inspected by this guard.')
        content = git(repo, 'cat-file', 'blob', current.decode())
        # The formal progress path is still subject to every payload/content check.
        findings.extend({'commit': oid, 'path': path, 'rule': item['type']}
                        for item in inspect_blob(path, content))
        rules = content_rules(content, path)
        if previous_mode in (b':100644', b':100755', b':120000'):
            previous_content = git(repo, 'cat-file', 'blob', previous.decode())
            rules |= historical_report_rules(previous_content, path)
        if rules:
            report_rules = rules & {'local_report_heading'}
            # Unchanged historical markers do not make an ordinary code/doc edit a report.
            # A report heading identifies the whole file; any edit must remain checked.
            # Conversation markers in ordinary docs are checked in added lines only.
            if parents and not content.startswith((b'\xff\xfe', b'\xfe\xff')):
                patch = git(repo, 'diff', '--no-ext-diff', '--no-textconv', '--no-color', '--text',
                            '--no-renames', '--unified=0', parents[0], oid, '--', path)
                content = added_patch_lines(patch)
                if b'\x00' not in content:
                    rules = content_rules(content, path)
                # Binary UTF-16 changes are conservatively checked as complete files.
            rules |= report_rules
            for rule in sorted(rules):
                findings.append({'commit': oid, 'path': path, 'rule': rule})
    return findings


def remote_main_base(repo: Path, remote: str) -> str:
    """Use only advertised main as the protected history boundary; require its object."""
    advertised = git(repo, 'ls-remote', '--refs', '--', remote, 'refs/heads/main')
    entries = [line.split() for line in advertised.splitlines()]
    if len(entries) != 1 or len(entries[0]) != 2 or entries[0][1] != b'refs/heads/main':
        raise GuardError('Remote main is unavailable; fetch and verify main before pushing.')
    oid = entries[0][0].decode('ascii')
    if re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', oid) is None:
        raise GuardError('Remote main has an invalid object ID; no push is allowed.')
    return commit_oid(repo, oid)


def hook_ranges(repo: Path, remote: str, lines: str) -> list[tuple[str, list[str]]]:
    ranges = []
    main_base = None
    for line in lines.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise GuardError('Malformed pre-push input; no push is allowed without a checked range.')
        _, local, _, previous = fields
        if any(re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', oid) is None
               for oid in (local, previous)):
            raise GuardError('Invalid object ID in pre-push input.')
        if set(local) == {'0'}:  # A deletion introduces no commits.
            continue
        if main_base is None:
            main_base = remote_main_base(repo, remote)
        # Even an existing destination can already contain reports from a side branch.
        # Its advertised previous tip must never narrow the main..head inspection.
        ranges.append((commit_oid(repo, local), [main_base]))
    return ranges


def inspect_ranges(repo: Path, ranges: list[tuple[str, list[str]]]) -> dict:
    """Inspect every new commit version; remote publication grants no exemption."""
    commits = set()
    for head, bases in ranges:
        commits.update(git(repo, 'rev-list', head, '--not', *bases).decode().splitlines())
    findings = [finding for oid in sorted(commits) for finding in inspect_commit(repo, oid)]
    return {'passed': not findings, 'commits_checked': len(commits), 'findings': findings,
            'scope': 'Every outgoing commit message and changed file version; '
                     'rule-based inspection does not certify arbitrary prose.'}


def install_hook(repo: Path) -> dict:
    custom = subprocess.run(['git', 'config', '--get', 'core.hooksPath'], cwd=repo,
                            capture_output=True)
    if custom.returncode != 1:
        raise GuardError('Custom core.hooksPath was preserved; integrate the guard explicitly.')
    path = Path(os.fsdecode(git(repo, 'rev-parse', '--git-path', 'hooks/pre-push')).strip())
    if not path.is_absolute():
        path = repo / path
    if path.exists():
        if path.read_bytes() != HOOK.encode('utf-8'):
            raise GuardError('An existing pre-push hook was preserved; integrate it explicitly.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(HOOK.encode('utf-8'))
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {'installed': True, 'hook': str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--install', action='store_true', help='Install only a local pre-push hook.')
    action.add_argument('--hook', nargs=2, metavar=('REMOTE', 'URL'))
    action.add_argument('--base', help='Explicit local comparison base; otherwise use cached upstream.')
    parser.add_argument('--head', default='HEAD')
    args = parser.parse_args()
    repo = Path.cwd()
    try:
        repo = Path(os.fsdecode(git(repo, 'rev-parse', '--show-toplevel')).strip())
        if args.install:
            report = install_hook(repo)
        else:
            ranges = hook_ranges(repo, args.hook[1], sys.stdin.read()) if args.hook else [
                (commit_oid(repo, args.head), [commit_oid(repo, args.base or '@{upstream}')])
            ]
            report = inspect_ranges(repo, ranges)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if report.get('installed') or report['passed'] else 1
    except (GuardError, OSError, PayloadDecodeError) as error:
        print(f'Push blocked: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
