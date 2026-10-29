"""Read-only audit of all files in the history intended for ordinary push."""
import argparse
import json
import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = {
    'github_token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'openai_style_key': rb'sk-(?:proj-)?[A-Za-z0-9_-]{32,}',
    'aws_access_id': rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'credential_assignment': rb'''(?i)(?:api_key|access_token|client_secret|password)\s*[=:]\s*["'][A-Za-z0-9_+/=-]{20,}["']''',
}
BLOCKED_EXTENSIONS = {'.joblib', '.pickle', '.pkl', '.parquet', '.feather', '.pcap', '.pcapng', '.csv', '.sqlite', '.sqlite3', '.db', '.pem', '.key'}
TEXT_EXTENSIONS = {
    '.py', '.pyi', '.md', '.rst', '.txt', '.json', '.jsonl', '.ndjson', '.yaml', '.yml',
    '.toml', '.ini', '.cfg', '.conf', '.csv', '.tsv', '.html', '.htm', '.xml', '.svg',
    '.css', '.js', '.ts', '.sh', '.ps1', '.bat', '.cmd', '.sql', '.lock', '.log',
}
TEXT_NAMES = {
    '.gitignore', '.gitattributes', '.editorconfig', '.dockerignore',
    'dockerfile', 'makefile', 'license', 'notice',
}


class PayloadDecodeError(ValueError):
    """Required text could not be inspected without losing its original characters."""


def decode_payload_text(content: bytes, path: str | None = None) -> str | None:
    """Decode supported text strictly; leave undecodable binary assets as raw bytes."""
    encoding = 'utf-16' if content.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    name = path.replace('\\', '/').lower().rsplit('/', 1)[-1] if path else ''
    required_text = (encoding == 'utf-16' or path is None
                     or Path(name).suffix in TEXT_EXTENSIONS or name in TEXT_NAMES
                     or name == '.env' or name.startswith('.env.'))
    try:
        text = content.decode(encoding)
        if '\x00' in text:
            # BOM-less UTF-16 and misread UTF-32 must not masquerade as valid UTF-8.
            if not required_text:
                return None
            raise UnicodeError('NUL characters are not supported project text.')
        return text
    except UnicodeError as error:
        if required_text:
            location = path if path is not None else '(commit message)'
            raise PayloadDecodeError(
                f'Cannot decode text at {location}; expected UTF-8 or BOM-marked UTF-16.'
            ) from error
        return None


def inspect_secrets(content: bytes, decoded_text: str | None = None) -> set[str]:
    """Return rule names only, scanning raw bytes and any supported decoded text."""
    candidates = [content]
    if decoded_text is not None:
        candidates.append(decoded_text.encode('utf-8'))
    return {name for name, pattern in SECRET_PATTERNS.items()
            if any(re.search(pattern, candidate) for candidate in candidates)}


def inspect_blob(path, content, max_bytes=5 * 1024 * 1024):
    findings = []
    normalized = path.replace('\\', '/')
    parts = normalized.lower().split('/')
    base = parts[-1]
    if parts[0] in {'.venv', 'venv', 'data', 'artifacts', 'test-results', 'build'}:
        findings.append({'path': path, 'type': 'excluded_directory'})
    if Path(base).suffix in BLOCKED_EXTENSIONS or base in {'.env', 'secrets.toml'} or (base.startswith('.env.') and base != '.env.example') or base.endswith(('.sqlite-wal', '.sqlite-shm', '.db-wal', '.db-shm')):
        findings.append({'path': path, 'type': 'excluded_file_type'})
    if len(content) > max_bytes:
        findings.append({'path': path, 'type': 'large_blob', 'bytes': len(content)})
    decoded_text = decode_payload_text(content, path)
    for name in sorted(inspect_secrets(content, decoded_text)):
        findings.append({'path': path, 'type': name})
    return findings


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    from liushao.data import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/phase2/git_safety.json')
    args = parser.parse_args()
    target = 'zhujerry78-code/liushao'
    repo = json.loads(subprocess.check_output(['gh', 'repo', 'view', target, '--json', 'nameWithOwner,isPrivate,url']))
    remote = git('remote', 'get-url', 'origin').decode().strip()
    if remote.removesuffix('.git') != f'https://github.com/{target}' or repo['nameWithOwner'] != target or repo['isPrivate'] is not True:
        raise RuntimeError('目标URL或Private状态不匹配，禁止推送')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD']).returncode != 0:
        raise RuntimeError('远程main不是当前历史祖先，停止同步')
    commits = git('rev-list', 'HEAD').decode().splitlines()
    entries = set()
    for commit in commits:
        for entry in git('ls-tree', '-r', '-l', '-z', commit).split(b'\0'):
            if not entry:
                continue
            descriptor, path = entry.split(b'\t', 1)
            _, kind, oid, size = descriptor.split()
            if kind == b'blob':
                entries.add((oid.decode(), path.decode('utf-8'), int(size)))
    content_cache, findings = {}, []
    for oid, path, size in sorted(entries):
        if oid not in content_cache:
            content_cache[oid] = git('cat-file', 'blob', oid)
        findings.extend({**item, 'object': oid} for item in inspect_blob(path, content_cache[oid]))
    branch = git('branch', '--show-current').decode().strip()
    result = {'repository': repo, 'origin': remote, 'branch': branch,
              'head': git('rev-parse', 'HEAD').decode().strip(),
              'remote_main': git('rev-parse', 'origin/main').decode().strip(),
              'remote_main_is_ancestor': True, 'history_commits_scanned': len(commits),
              'unique_file_versions_scanned': len(entries), 'unique_blobs_scanned': len(content_cache),
              'largest_blob_bytes': max(size for _, _, size in entries), 'findings': findings,
              'pending_commits': git('log', '--format=%H %s', 'origin/main..HEAD').decode('utf-8').splitlines(),
              'limits': 'local pattern and path/size scan, not a guarantee against all secrets or semantic sensitive data; raw matching values are never printed',
              'push_policy': 'ordinary push of current branch only; no force, no history rewrite, no main update'}
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ['branch', 'head', 'history_commits_scanned', 'unique_file_versions_scanned', 'largest_blob_bytes', 'findings']}, indent=2))
    if findings:
        raise RuntimeError('发现潜在敏感载荷，停止推送；报告不包含秘密原值')


if __name__ == '__main__':
    main()
