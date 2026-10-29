"""Verified resumable transport for three explicitly selected official CSV objects."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

from liushao.data import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'test-results/engineering'
LOCK_PATH = ROOT / 'configs/data.lock.json'
BASE_URL = ('https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/'
            'Processed%20Traffic%20Data%20for%20ML%20Algorithms/')
ORIGINAL_SOURCES = (
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
)
EXTERNAL_SOURCE = 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'
EXTERNAL_BYTES = 375945899
EXTERNAL_ETAG = 'eb1dc2ec76efe09c8995f28bfa4c8656-23'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_source(name, expected_sha256):
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is not None,
            '必须显式提供完整小写 SHA-256')
    require(name in (*ORIGINAL_SOURCES, EXTERNAL_SOURCE), '拒绝未登记的数据来源')
    if name in ORIGINAL_SOURCES:
        lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
        require(lock['base_url'] == BASE_URL, '原数据锁中的官方 URL 已改变')
        entries = [item for item in lock['files'] if item['name'] == name]
        require(len(entries) == 1, '原数据锁中的来源不唯一或缺失')
        item = entries[0]
        require(expected_sha256 == item['sha256'], '提供的 SHA-256 与原数据锁不一致')
        return {'url': BASE_URL + name, 'bytes': item['bytes'],
                'sha256': expected_sha256, 'etag': None}
    return {'url': BASE_URL + name, 'bytes': EXTERNAL_BYTES,
            'sha256': expected_sha256, 'etag': EXTERNAL_ETAG}


def confined_path(path):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(OUTPUT_ROOT.resolve()) and resolved != OUTPUT_ROOT.resolve(),
            '下载产物必须位于 test-results/engineering 内')
    return resolved


def fetch_verified(source, destination, chunk_bytes=1024 * 1024, workers=12, retries=3):
    """Keep verified parts on failure; only expose the final CSV after full SHA verification."""
    require(source['url'].startswith(BASE_URL), '只允许已登记的官方 HTTPS 对象')
    require(source['url'][len(BASE_URL):] in (*ORIGINAL_SOURCES, EXTERNAL_SOURCE),
            '拒绝未登记的官方对象')
    require(re.fullmatch(r'[0-9a-f]{64}', source['sha256']) is not None,
            '必须提供完整 SHA-256')
    require(min(source['bytes'], chunk_bytes, workers, retries) > 0, '下载参数必须为正数')
    require(workers <= 32 and chunk_bytes <= 4 * 1024 * 1024, '并发最多 32，单块最多 4 MiB')
    target = confined_path(destination)
    require(target.name == source['url'][len(BASE_URL):], '目标文件名必须与登记对象一致')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(target.is_file() and target.stat().st_size == source['bytes'],
                '已有目标长度不匹配；保留原文件')
        actual = sha256(target)
        require(actual == source['sha256'], '已有目标 SHA-256 不匹配；保留原文件')
        return {**source, 'status': 'already_verified', 'actual_sha256': actual}
    started = time.perf_counter()
    parts = confined_path(target.with_suffix(f'.parts-{chunk_bytes}'))
    parts.mkdir(exist_ok=True)
    specification = {**source, 'chunk_bytes': chunk_bytes}
    spec_path = parts / 'source.json'
    if spec_path.exists():
        require(json.loads(spec_path.read_text(encoding='utf-8')) == specification,
                '断点目录属于不同来源或参数；保留原断点')
    else:
        write_json(spec_path, specification)

    def fetch_part(start):
        end = min(start + chunk_bytes, source['bytes']) - 1
        size = end - start + 1
        path = parts / f'{start:012d}.part'
        checksum = path.with_suffix('.sha256')
        if path.exists() and checksum.exists() and path.stat().st_size == size:
            if sha256(path) == checksum.read_text(encoding='ascii').strip():
                return {'start': start, 'bytes': size, 'reused': True}
        error = ''
        for attempt in range(retries):
            try:
                request = urllib.request.Request(source['url'], headers={
                    'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=20) as response:
                    require(response.status == 206, f'HTTP {response.status}，要求 206')
                    expected_range = f'bytes {start}-{end}/{source["bytes"]}'
                    require(response.headers.get('Content-Range') == expected_range,
                            'Content-Range 不匹配')
                    if source.get('etag'):
                        require(response.headers.get('ETag', '').strip('"') == source['etag'],
                                '官方对象 ETag 改变')
                    block = response.read(size + 1)
                require(len(block) == size, '分段响应长度不匹配')
                pending = path.with_suffix('.pending')
                pending.write_bytes(block)
                pending.replace(path)
                checksum.write_text(hashlib.sha256(block).hexdigest(), encoding='ascii')
                return {'start': start, 'bytes': size, 'reused': False}
            except Exception as caught:
                error = f'{type(caught).__name__}: {caught}'
                if attempt + 1 < retries:
                    time.sleep(min(attempt + 1, 3))
        return {'start': start, 'bytes': 0, 'error': error}

    starts = list(range(0, source['bytes'], chunk_bytes))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_part, start) for start in starts]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 16 == 0 or len(results) == len(starts):
                print(json.dumps({'completed_parts': len(results), 'parts': len(starts),
                                  'verified_bytes': sum(item['bytes'] for item in results),
                                  'errors': sum('error' in item for item in results)}), flush=True)
    failures = [item for item in results if 'error' in item]
    write_json(parts / 'latest_attempt.json', {'results': results, 'failures': failures})
    require(not failures, f'{len(failures)} 个分段失败；保留断点，可按同一命令重试')
    assembly = confined_path(target.with_suffix('.assembling'))
    with assembly.open('wb') as output:
        for start in starts:
            path = parts / f'{start:012d}.part'
            require(sha256(path) == path.with_suffix('.sha256').read_text(encoding='ascii').strip(),
                    f'分段在拼装前改变: {start}')
            with path.open('rb') as stream:
                output.write(stream.read())
    require(assembly.stat().st_size == source['bytes'], '完整文件长度不匹配')
    actual = sha256(assembly)
    require(actual == source['sha256'], '完整文件 SHA-256 不匹配；没有产生可用 CSV')
    assembly.replace(target)
    result = {**source, 'status': 'verified_complete', 'actual_sha256': actual,
              'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.perf_counter() - started,
              'parts': len(starts), 'parts_reused': sum(item['reused'] for item in results),
              'chunk_bytes': chunk_bytes, 'workers': workers}
    write_json(target.with_suffix('.source.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=(*ORIGINAL_SOURCES, EXTERNAL_SOURCE))
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--chunk-bytes', type=int, default=1024 * 1024)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(argv)
    source = resolve_source(args.source, args.sha256)
    print(json.dumps(fetch_verified(source, args.dest, args.chunk_bytes, args.workers),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
