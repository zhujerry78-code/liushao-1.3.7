"""Reseal a prepared runtime directory; call before launch checks and final ZIP.

Regenerates SOURCE_MANIFEST.json, PACKAGE_MANIFEST.json and SHA256SUMS.txt from the
files on disk.  The registered file set is the previous manifest plus the
explicitly added paths; unregistered local state (.venv, demo-state, logs) stays out.
Usage:
    python -m scripts.seal_release --version 1.3.7 --revision r13-contest --date 2026-10-29
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCE_GLOBS = ['liushao_engine/*.py', 'scripts/fit_engine.py', 'scripts/evaluate_engine.py',
                    'scripts/seal_release.py', 'tests/test_engine.py',
                    'app_engine_*.py','tests/test_contest_*.py','scripts/package_contest_release.py',
                    'docs/release/CONTEST_CHANGELOG.md','docs/usage/CONTEST_SOURCE.md']
NEW_GENERATED = ['artifacts/engine.joblib', 'artifacts/engine.json', 'docs/results/engine_evaluation.json',
                 'docs/usage/ENGINE.md','docs/results/contest_*']
SELF = {'PACKAGE_MANIFEST.json', 'SHA256SUMS.txt'}


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--date', required=True)
    args = p.parse_args()
    old_package = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    old_source = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    (ROOT / 'VERSION').write_text(args.version + '\n', encoding='utf-8')
    (ROOT / 'RELEASE_DATE').write_text(args.date + '\n', encoding='utf-8')

    source_names = set(old_source['files'])
    for pattern in NEW_SOURCE_GLOBS:
        source_names.update(path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern))
    missing = sorted(name for name in source_names if not (ROOT / name).is_file())
    if missing:
        raise ValueError('Registered source files missing: ' + repr(missing))
    source_names = sorted(source_names)
    source_files = {name: digest(name) for name in source_names}
    # Source snapshot identity: SHA-1 over the canonical file map (40 hex like a commit id).
    snapshot = hashlib.sha1(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    canonical_sha256 = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    baseline = old_source.get('base_source_snapshot',old_source['source_commit'])
    source_manifest = {'format': 1, 'source_commit': snapshot,
                       'source_identity_kind':'content-snapshot-sha1-not-git-commit',
                       'source_snapshot_sha256':canonical_sha256,
                       'base_source_snapshot':baseline, 'files': source_files}
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')

    generated = {path.relative_to(ROOT).as_posix() for pattern in NEW_GENERATED for path in ROOT.glob(pattern)}
    names = (set(old_package['files']) | set(source_names) | generated) - SELF
    missing = sorted(name for name in names if not (ROOT / name).is_file())
    if missing:
        raise ValueError('Registered package files missing: ' + repr(missing))
    names = sorted(names)
    files = {}
    for name in names:
        content = (ROOT / name).read_bytes()
        files[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                       'origin': 'source-snapshot' if name in source_files else 'verified-assets-or-generated'}
    previous = old_package.get('previous_release') if old_package.get('package_revision') == args.revision else None
    package = dict(old_package, software_version=args.version, software_sha=snapshot, release_date=args.date,
                   package_kind='runtime', software_identity_kind='content-snapshot-sha1-not-git-commit',
                   source_snapshot_sha256=canonical_sha256, base_source_snapshot=baseline,
                   package_revision=args.revision, previous_release=previous or {
                       'software_version': old_package['software_version'],
                       'software_sha': old_package['software_sha'],
                       'package_revision': old_package['package_revision']}, files=files)
    (ROOT / 'PACKAGE_MANIFEST.json').write_text(json.dumps(package, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')
    sums = {name: meta['sha256'] for name, meta in files.items()}
    sums['PACKAGE_MANIFEST.json'] = digest('PACKAGE_MANIFEST.json')
    (ROOT / 'SHA256SUMS.txt').write_text(''.join(f'{sums[n]}  {n}\n' for n in sorted(sums)), encoding='utf-8')
    print(json.dumps({'version': args.version, 'revision': args.revision, 'source_snapshot': snapshot,
                      'source_files': len(source_files), 'package_files': len(files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
