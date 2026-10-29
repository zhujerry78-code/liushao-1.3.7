import importlib.util


def test_payload_scan_flags_secrets_and_binaries_without_echoing_secret():
    assert importlib.util.find_spec('scripts.git_payload_phase2') is not None, 'Git载荷审查尚未实现'
    from scripts.git_payload_phase2 import inspect_blob
    token = 'ghp_' + 'a' * 36
    findings = inspect_blob('config.txt', token.encode(), 1024)
    assert findings and token not in repr(findings)
    assert inspect_blob('data/traffic.csv', b'a,b\n1,2', 1024)
    assert inspect_blob('artifacts/model.joblib', b'not code', 1024)
    assert inspect_blob('src/safe.py', b'print("safe")', 1024) == []
    assert inspect_blob('large.txt', b'x' * 11, 10)
