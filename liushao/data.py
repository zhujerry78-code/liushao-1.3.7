"""CICFlowMeter 输入、稳定来源标识和分组划分。"""
from __future__ import annotations

import hashlib
import csv
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDED = {'Label', 'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port'}


def _csv_header(reader):
    header = next(reader, [])
    if reader.line_num != 1:
        raise ValueError('不支持多行 CSV 表头')
    header = [field.strip() for field in header]
    if not header or any(not field for field in header):
        raise ValueError('CSV 表头为空或包含空字段')
    if len(set(header)) != len(header):
        raise ValueError('CSV 表头包含重复字段')
    return header


def _csv_records(reader, width, allow_blank):
    previous = reader.line_num
    for row in reader:
        line = previous + 1
        if reader.line_num != line:
            raise ValueError(f'第 {line} 行包含多行字段，当前版本不支持')
        previous = reader.line_num
        if not row:
            if allow_blank:
                continue
            raise ValueError(f'第 {line} 行为空行；批量处理要求逐行记录')
        if len(row) != width:
            raise ValueError(f'第 {line} 行字段数不一致')
        yield line, row


def read_csv_checked(stream, max_rows=50000):
    """Validate raw headers before pandas can mangle them; retain physical lines."""
    content = stream.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8-sig')
    reader = csv.reader(io.StringIO(content, newline=''), strict=True)
    header = _csv_header(reader)
    lines, records = [], []
    for line, row in _csv_records(reader, len(header), allow_blank=True):
        if len(records) >= max_rows:
            raise ValueError(f'最多支持 {max_rows:,} 条记录，请分批导入')
        lines.append(line)
        records.append(row)
    result = pd.DataFrame(records, columns=header)
    result.attrs['source_rows'] = lines
    return result


def validate_csv_layout(path):
    """Streaming preflight for official files whose chunks use physical offsets."""
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = _csv_header(reader)
        for _ in _csv_records(reader, len(header), allow_blank=False):
            pass
    return header


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def version_id(config, sources, manifest_hash):
    normalized = {k: v for k, v in config.items() if k != 'files'}
    return hashlib.sha256(json.dumps({'sources': sources, 'config': normalized,
                                      'manifest': manifest_hash}, sort_keys=True).encode()).hexdigest()


def download(url, target, expected_sha256=None, expected_bytes=None):
    """Bounded HTTP ranges avoid long proxy stalls; each range is verified."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.download')
        if expected_bytes is None or not url.startswith('http'):
            with urllib.request.urlopen(url, timeout=45) as response, temporary.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
        else:
            step = 4 * 1024 * 1024
            def part(start):
                end = min(start + step, expected_bytes) - 1
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                        with urllib.request.urlopen(req, timeout=60) as response:
                            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{expected_bytes}':
                                raise ValueError('下载范围响应不匹配')
                            data = response.read()
                        if len(data) != end - start + 1:
                            raise ValueError('下载范围长度不匹配')
                        return data
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            with ThreadPoolExecutor(max_workers=4) as pool, temporary.open('wb') as out:
                for index, block in enumerate(pool.map(part, range(0, expected_bytes, step))):
                    out.write(block)
                    if index % 20 == 0:
                        print(f'{target.name}: {out.tell()}/{expected_bytes} bytes', flush=True)
        if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
            raise ValueError('下载文件长度不匹配')
        if expected_sha256 and sha256(temporary) != expected_sha256:
            raise ValueError('下载文件哈希不匹配')
        temporary.replace(target)
    digest = sha256(target)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('已下载文件哈希不匹配')
    if expected_bytes is not None and target.stat().st_size != expected_bytes:
        raise ValueError('已下载文件长度不匹配')
    return {'url': url, 'name': target.name, 'bytes': target.stat().st_size, 'sha256': digest}


def clean(raw, features, source_hash, source_name, row_offset=0):
    raw = raw.copy()
    raw.columns = raw.columns.str.strip()
    if raw.columns.duplicated().any():
        raise ValueError('存在重复字段名')
    missing = sorted(set(features) - set(raw.columns))
    if missing:
        raise ValueError('缺少必需特征: ' + ', '.join(missing))
    if set(features) & EXCLUDED:
        raise ValueError('特征列表包含标签或元数据')
    x = raw[features].apply(pd.to_numeric, errors='coerce').astype('float64')
    x = x.replace([np.inf, -np.inf], np.nan)
    # RF internally converts to float32. Values outside its range cannot be inferred.
    x = x.mask(x.abs() > np.finfo(np.float32).max)
    rows = np.asarray(raw.attrs.get('source_rows', np.arange(row_offset + 2, row_offset + len(raw) + 2)))
    meta = pd.DataFrame({'record_id': [f'{source_hash}:{r}' for r in rows],
                         'source_file': source_name, 'source_sha256': source_hash,
                         'source_row': rows}, index=raw.index)
    if 'Timestamp' in raw:
        meta['timestamp'] = pd.to_datetime(raw.Timestamp, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    else:
        meta['timestamp'] = pd.NaT
    if 'Label' in raw:
        meta['label'] = raw.Label.astype('string').str.strip()
    for field in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Flow ID']:
        if field in raw:
            meta[field] = raw[field]
    bad = x.isna().all(axis=1)
    rejected = meta[bad].copy()
    rejected['reason'] = '全部必需特征无效，无法判断'
    meta['missing_features'] = x.isna().sum(axis=1)
    return x[~bad].reset_index(drop=True), meta[~bad].reset_index(drop=True), rejected.reset_index(drop=True)


def fingerprints(x, near=False):
    # Six significant digits, same sentinel for missing cells. Not a distance metric.
    arr = x.to_numpy(dtype='float64', copy=True)
    if near:
        exponent = np.floor(np.log10(np.maximum(np.abs(arr), 1e-300)))
        factor = np.power(10., np.clip(5 - exponent, -300, 300))
        arr = np.round(arr * factor) / factor
    arr[arr == 0] = 0
    frame = pd.DataFrame(arr, columns=x.columns).fillna(0)
    values = pd.util.hash_pandas_object(frame, index=False).astype(str)
    masks = pd.util.hash_pandas_object(x.isna(), index=False).astype(str)
    return values + ':' + masks


def split_records(x, meta, seed=42, block_minutes=15):
    result = meta.copy()
    if result.timestamp.isna().any():
        raise ValueError('划分需要全部有效的采集时间')
    result['block_id'] = result.source_file + '|' + result.timestamp.dt.floor(f'{block_minutes}min').astype(str)
    blocks = sorted(result.block_id.unique())
    if len(blocks) < 3:
        raise ValueError('至少需要3个独立时间块')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(blocks)
    a = max(1, int(len(blocks) * .6))
    b = min(len(blocks) - 1, max(a + 1, int(len(blocks) * .8)))
    assignment = {key: ('train' if i < a else 'validation' if i < b else 'test') for i, key in enumerate(shuffled)}
    result['split'] = result.block_id.map(assignment)
    result['exact_fingerprint'] = fingerprints(x).values
    result['near_fingerprint'] = fingerprints(x, near=True).values
    result['purge_reason'] = ''
    # Prefer train, then validation, then test; never choose by target labels.
    rank = result.split.map({'train': 0, 'validation': 1, 'test': 2})
    for column in ['exact_fingerprint', 'near_fingerprint']:
        best = rank.groupby(result[column]).transform('min')
        bad = rank > best
        result.loc[bad, 'split'] = 'purged'
        result.loc[bad, 'purge_reason'] = column + '_cross_split'
    return result
