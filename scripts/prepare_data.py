"""Run from repository root: python -m scripts.prepare_data."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from liushao.data import EXCLUDED, clean, download, sha256, split_records, write_json, version_id, validate_csv_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data.json')
    parser.add_argument('--output', default='data/processed')
    args = parser.parse_args()
    start = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    xs, metas, audits, downloads, rejects = [], [], [], [], []
    features = None
    for item in config['files']:
        path = Path('data/raw') / item['name']
        info = download(config['base_url'] + item['name'], path, item.get('sha256'), item['bytes'])
        downloads.append(info)
        print('Inspecting', item['name'], info['sha256'], flush=True)
        validate_csv_layout(path)
        offset = 0
        raw_labels = {}
        sampled = 0
        nonfinite_cells = 0
        invalid_timestamps = 0
        columns = None
        for chunk in pd.read_csv(path, chunksize=config['chunk_rows'], dtype=str):
            chunk.columns = chunk.columns.str.strip()
            columns = chunk.columns.tolist()
            if features is None:
                features = [c for c in columns if c not in EXCLUDED]
            for label, count in chunk['Label'].value_counts(dropna=False).items():
                raw_labels[str(label)] = raw_labels.get(str(label), 0) + int(count)
            # Hash sampling independent of labels, values and chunk boundaries.
            keep = [int(hashlib.sha256(f'{config["seed"]}:{info["sha256"]}:{row}'.encode()).hexdigest()[:16], 16) % config['sample_modulus'] == 0
                    for row in range(offset + 2, offset + len(chunk) + 2)]
            x, meta, rejected = clean(chunk, features, info['sha256'], item['name'], offset)
            selected_rows = set(np.arange(offset + 2, offset + len(chunk) + 2)[keep])
            take = meta.source_row.isin(selected_rows)
            x, meta = x[take].reset_index(drop=True), meta[take].reset_index(drop=True)
            rejected = rejected[rejected.source_row.isin(selected_rows)]
            sampled += len(selected_rows)
            nonfinite_cells += int(x.isna().sum().sum())
            good = meta.timestamp.notna() & meta.label.notna() & ~meta.label.isin(['Label', ''])
            invalid_timestamps += int(meta.timestamp.isna().sum())
            if (~good).any():
                extra = meta[~good].copy()
                extra['reason'] = '无效时间或评测标签'
                rejects.append(extra)
            rejects.append(rejected)
            xs.append(x[good].reset_index(drop=True))
            metas.append(meta[good].reset_index(drop=True))
            offset += len(chunk)
        audits.append({'file': item['name'], 'raw_rows': offset, 'columns': columns,
                       'raw_labels': raw_labels, 'sampled_rows': sampled,
                       'sampled_missing_cells': nonfinite_cells, 'sampled_invalid_timestamps': invalid_timestamps})
    x = pd.concat(xs, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    manifest = split_records(x, meta, config['seed'], config['block_minutes'])
    x.to_parquet(out / 'features.parquet', index=False)
    # Labels exist only here and in evaluation outputs; inference does not load this manifest.
    manifest.to_parquet(out / 'manifest.parquet', index=False)
    pd.concat(rejects, ignore_index=True).to_csv(out / 'rejected.csv', index=False)
    summary = {'config': config, 'sources': downloads, 'audits': audits, 'features': features,
               'split_counts': {str(k): int(v) for k, v in manifest.split.value_counts().items()},
               'split_labels': pd.crosstab(manifest.split, manifest.label).to_dict(),
               'split_blocks': manifest.groupby('split').block_id.nunique().to_dict(),
               'cross_split_overlap_after_purge': {},
               'elapsed_seconds': time.perf_counter() - start, 'python': platform.python_version(),
               'scope': 'two capture days, seeded 15-minute grouped holdout; not chronological or cross-environment validation',
               'near_duplicate_definition': 'all features equal after rounding to 6 significant digits; no broader similarity guarantee'}
    valid = manifest[manifest.split != 'purged']
    for column in ['exact_fingerprint', 'near_fingerprint']:
        summary['cross_split_overlap_after_purge'][column] = int((valid.groupby(column).split.nunique() > 1).sum())
        assert summary['cross_split_overlap_after_purge'][column] == 0
    for name in ['train', 'validation', 'test']:
        subset = manifest[manifest.split == name]
        if (subset.label != 'Benign').nunique() != 2:
            raise ValueError(f'{name} 缺少一个二分类类别；停止实验，不自动重选划分')
    for filename in ['features.parquet', 'manifest.parquet', 'rejected.csv']:
        summary[filename + '_sha256'] = sha256(out / filename)
    summary['data_version'] = version_id(config, downloads, summary['manifest.parquet_sha256'])
    write_json(out / 'summary.json', summary)
    write_json('docs/results/data_summary.json', summary)
    write_json('configs/data.lock.json', {**config, 'files': [{**f, 'sha256': d['sha256']} for f, d in zip(config['files'], downloads)]})
    print(json.dumps({key: summary[key] for key in ['split_counts', 'split_labels', 'cross_split_overlap_after_purge', 'elapsed_seconds']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
