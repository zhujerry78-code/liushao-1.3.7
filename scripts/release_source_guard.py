"""Verify a release manifest against a separately supplied, explicit trusted digest."""
import hashlib
import json
from pathlib import Path
import re
from delivery.safe_files import validate_names, regular_file

IMMUTABLE_ARTIFACTS = {
    'artifacts/model.joblib': '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7',
    'resources/evaluation/model.original.json': 'e280995a00084234c538720541db6475bb51382c4e2ffc9c0e4db8cbde933880',
}


def verify_source(root, manifest_path, trusted_digest):
    if not trusted_digest or not re.fullmatch('[0-9a-f]{64}', trusted_digest):
        return {'passed': False, 'status': 'not_run', 'missing': ['independently retained source manifest SHA256']}
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != trusted_digest:
        raise ValueError('发布源码清单与独立保存的摘要不一致')
    manifest = json.loads(raw)
    expected = manifest['files']
    validate_names(expected)
    failures, records, absent = [], [], []
    for name, digest in {**expected, **IMMUTABLE_ARTIFACTS}.items():
        if not (Path(root) / name).exists() and name in IMMUTABLE_ARTIFACTS:
            absent.append(name)
            continue
        try:
            content = regular_file(Path(root), name).read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            normalized = hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest()
            valid = actual == digest or (name in expected and normalized == digest)
            records.append({'path': name, 'sha256': actual, 'expected_sha256': digest, 'passed': valid})
            if not valid:
                failures.append({'path': name, 'reason': 'protected_content_changed'})
        except FileNotFoundError:
            failures.append({'path': name, 'reason': 'missing_protected_file'})
    return {'passed': not failures, 'status': 'failed' if failures else 'passed',
            'source_commit': manifest['source_commit'], 'source_manifest_sha256': trusted_digest,
            'checked_count': len(records), 'historical_manifest_count': len(expected) + len(IMMUTABLE_ARTIFACTS),
            'absent_local_artifacts': absent, 'files': records, 'failures': failures}
