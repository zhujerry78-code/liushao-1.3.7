"""Transport tests use mocked HTTP byte ranges; they do not fetch benchmark data."""
import hashlib
import json

import pytest

from scripts import engineering_fetch as fetch


class Response:
    def __init__(self, body, content_range, status=206):
        self.body = body
        self.status = status
        self.headers = {'Content-Range': content_range}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, length):
        return self.body[:length]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, 'OUTPUT_ROOT', tmp_path / 'engineering')
    data = b'abcdefghijk'
    source = {'url': fetch.BASE_URL + fetch.EXTERNAL_SOURCE, 'bytes': len(data),
              'sha256': hashlib.sha256(data).hexdigest(), 'etag': None}
    destination = fetch.OUTPUT_ROOT / fetch.EXTERNAL_SOURCE
    calls = []

    def opener(request, timeout):
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        calls.append((start, end))
        return Response(data[start:end + 1], f'bytes {start}-{end}/{len(data)}')

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', opener)
    return source, destination, calls, opener


def test_complete_fetch_and_existing_target_are_hash_verified(setup):
    source, destination, calls, _ = setup
    result = fetch.fetch_verified(source, destination, chunk_bytes=4, workers=2, retries=1)
    assert result['status'] == 'verified_complete'
    assert result['actual_sha256'] == source['sha256']
    assert destination.read_bytes() == b'abcdefghijk'
    assert sorted(calls) == [(0, 3), (4, 7), (8, 10)]
    calls.clear()
    assert fetch.fetch_verified(source, destination)['status'] == 'already_verified'
    assert calls == []
    destination.write_bytes(b'corruptdata')
    with pytest.raises(ValueError, match='SHA-256'):
        fetch.fetch_verified(source, destination)
    assert destination.read_bytes() == b'corruptdata'


def test_bad_content_range_keeps_no_complete_target(setup, monkeypatch):
    source, destination, _, _ = setup
    monkeypatch.setattr(fetch.urllib.request, 'urlopen',
                        lambda *_args, **_kwargs: Response(b'abcd', 'bytes 0-3/999'))
    with pytest.raises(ValueError, match='分段失败'):
        fetch.fetch_verified(source, destination, chunk_bytes=4, workers=1, retries=1)
    assert not destination.exists()
    report = json.loads(destination.with_suffix('.parts-4').joinpath('latest_attempt.json')
                        .read_text(encoding='utf-8'))
    assert len(report['failures']) == 3
    assert all('Content-Range' in item['error'] for item in report['failures'])


def test_complete_hash_mismatch_cannot_expose_csv(setup):
    source, destination, _, _ = setup
    source['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='完整文件 SHA-256'):
        fetch.fetch_verified(source, destination, chunk_bytes=4, workers=1, retries=1)
    assert not destination.exists()
    assert destination.with_suffix('.assembling').read_bytes() == b'abcdefghijk'


def test_retry_keeps_verified_parts_and_refetches_corruption(setup, monkeypatch):
    source, destination, calls, good_opener = setup
    def interrupted(request, timeout):
        if request.get_header('Range') == 'bytes=4-7':
            raise TimeoutError('synthetic dropped range')
        return good_opener(request, timeout)
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', interrupted)
    with pytest.raises(ValueError, match='分段失败'):
        fetch.fetch_verified(source, destination, chunk_bytes=4, workers=1, retries=1)
    parts = destination.with_suffix('.parts-4')
    assert (parts / '000000000000.part').exists()
    assert (parts / '000000000008.part').exists()
    assert not destination.exists()
    (parts / '000000000000.part').write_bytes(b'bad!')
    calls.clear()
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', good_opener)
    result = fetch.fetch_verified(source, destination, chunk_bytes=4, workers=1, retries=1)
    assert calls == [(0, 3), (4, 7)]
    assert result['parts_reused'] == 1
    assert destination.read_bytes() == b'abcdefghijk'


def test_allowlist_original_lock_and_output_confinement(setup):
    source, destination, _, _ = setup
    with pytest.raises(ValueError, match='未登记'):
        fetch.resolve_source('unknown.csv', '0' * 64)
    with pytest.raises(ValueError, match='原数据锁'):
        fetch.resolve_source(fetch.ORIGINAL_SOURCES[0], '0' * 64)
    with pytest.raises(ValueError, match='test-results/engineering'):
        fetch.fetch_verified(source, destination.parent.parent / destination.name)
    source['url'] = 'https://example.com/' + fetch.EXTERNAL_SOURCE
    with pytest.raises(ValueError, match='官方 HTTPS'):
        fetch.fetch_verified(source, destination)
