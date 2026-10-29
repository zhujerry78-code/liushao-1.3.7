"""Audit an explicit external CSV against frozen data without loading or running models."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import pandas as pd

from liushao.data import clean, fingerprints, sha256, write_json
from scripts import engineering_external as external


OUTPUT_ROOT = external.ROOT / 'test-results/engineering/external-preflight'


def numeric_quality(raw, features):
    """Describe raw numeric problems; clean remains the sole adaptation implementation."""
    values = raw[features]
    missing = values.isna() | values.apply(lambda column: column.str.strip().eq(''))
    numeric = values.apply(pd.to_numeric, errors='coerce').astype('float64')
    invalid = numeric.isna() & ~missing
    infinite = np.isinf(numeric)
    overflow = np.isfinite(numeric) & (numeric.abs() > np.finfo(np.float32).max)
    masks = {'empty_cells': missing, 'nonnumeric_cells': invalid,
             'infinite_cells': infinite, 'float32_overflow_cells': overflow,
             'cleaned_missing_cells': numeric.isna() | infinite | overflow}
    return {field: Counter({name: int(mask[field].sum()) for name, mask in masks.items()})
            for field in features}


def audit_source(args, features, references, header, labels, out):
    """Stream quality and intersections, then publish audit.json only after all checks pass."""
    quality = Counter(raw_rows=0, valid_rows=0, unjudgeable_rows=0, missing_cells=0,
                      rows_with_missing_features=0, record_id_overlap=0, invalid_timestamp_rows=0)
    per_field = {field: Counter() for field in features}
    overlaps = {kind: Counter({split: 0 for split in external.SPLITS})
                for kind in ('exact', 'near')}
    id_overlaps = Counter({split: 0 for split in external.SPLITS})
    class_counts = {name: Counter({label: 0 for label in labels}) for name in external.STRATA}
    seen = {'exact': set(), 'near': set()}
    duplicates = Counter(exact=0, near=0)
    rejected_labels, rejected_reasons, observed_labels = Counter(), Counter(), Counter()
    wrote_rejected = False
    for chunk in external.csv_chunks(args.source, header, args.chunk_rows):
        source_rows = chunk.attrs['source_rows']
        label_by_row = dict(zip(source_rows, chunk.Label.str.strip()))
        observed_labels.update(label_by_row.values())
        raw = chunk.drop(columns='Label')
        raw.attrs['source_rows'] = source_rows
        x, meta, rejected = clean(raw, features, args.source_sha256, Path(args.source).name)
        for field, counts in numeric_quality(raw, features).items():
            per_field[field].update(counts)
        quality.update(raw_rows=len(chunk), valid_rows=len(meta), unjudgeable_rows=len(rejected),
                       missing_cells=int(x.isna().sum().sum()),
                       rows_with_missing_features=int(x.isna().any(axis=1).sum()),
                       invalid_timestamp_rows=int(meta.timestamp.isna().sum()
                                                  + rejected.timestamp.isna().sum()))
        rejected_labels.update(label_by_row[int(row)] for row in rejected.source_row)
        rejected_reasons.update(rejected.reason)
        if len(rejected):
            rejected['ground_truth_evaluation_only'] = [label_by_row[int(row)]
                                                       for row in rejected.source_row]
            rejected.to_csv(out / 'rejected.csv', mode='a', header=not wrote_rejected, index=False)
            wrote_rejected = True
        truth_labels = np.array([label_by_row[int(row)] for row in meta.source_row])
        masks = {}
        for kind, near in (('exact', False), ('near', True)):
            fingerprint = fingerprints(x, near=near).astype(str)
            union = np.zeros(len(meta), dtype=bool)
            for split in external.SPLITS:
                mask = fingerprint.isin(references[kind][split]).to_numpy()
                overlaps[kind][split] += int(mask.sum())
                union |= mask
            masks[kind + '_overlap'] = union
            before = len(seen[kind])
            seen[kind].update(fingerprint)
            duplicates[kind] += len(fingerprint) - (len(seen[kind]) - before)
        masks['any_overlap'] = masks['exact_overlap'] | masks['near_overlap']
        masks['no_overlap'] = ~masks['any_overlap']
        masks['all'] = np.ones(len(meta), dtype=bool)
        quality['record_id_overlap'] += int(meta.record_id.isin(references['ids']).sum())
        for split in external.SPLITS:
            id_overlaps[split] += int(meta.record_id.isin(references['ids_by_split'][split]).sum())
        for name, mask in masks.items():
            class_counts[name].update(truth_labels[mask])
        print(f'已审计 {quality["raw_rows"]:,} 行；本审计阶段未运行预测', flush=True)
    external.require(observed_labels == labels, '审计标签数量与整文件预检不一致')
    external.require(quality['raw_rows'] == sum(labels.values()), '审计行数与预检不一致')
    external.require(quality['valid_rows'] + quality['unjudgeable_rows'] == quality['raw_rows'],
                     '审计行数不守恒')
    external.verify_file(args.source, args.source_sha256, args.source_bytes)
    result = {
        'status': 'completed', 'inference_performed': False, 'model_loading_performed_by_audit': False,
        'quality': dict(quality), 'numeric_quality_by_feature': {
            field: dict(counts) for field, counts in per_field.items()},
        'source_label_support': dict(labels),
        'unjudgeable_label_support': dict(rejected_labels),
        'unjudgeable_reasons': dict(rejected_reasons),
        'reference_overlap_rows': {kind: dict(counts) for kind, counts in overlaps.items()},
        'reference_record_id_overlap_rows': dict(id_overlaps),
        'within_source_duplicate_extra_rows': dict(duplicates),
        'strata_label_support': {name: dict(counts) for name, counts in class_counts.items()},
        'quality_semantics': {
            'missing_cells': 'After existing clean, judgeable rows only; includes invalid numbers.',
            'numeric_quality_by_feature': 'All raw rows, including rejected rows; no imputation.',
            'overlaps': 'Judgeable rows; exact and six-significant-digit fingerprints from clean.',
            'units': 'Not independently verified; preserve original source feature semantics.',
            'duplicates': 'Retained; counts are additional rows after first occurrence.'},
        'scope': 'Data audit only. No model performance, session isolation or deployment claim.',
    }
    if wrote_rejected:
        result['rejected_sha256'] = sha256(out / 'rejected.csv')
    write_json(out / 'audit.json', result)
    return result


def verify_evaluation_matches_audit(result, audit):
    """Reject success metrics when any evaluation population differs from the prior audit."""
    for name in ('quality', 'source_label_support', 'unjudgeable_label_support',
                 'unjudgeable_reasons', 'reference_overlap_rows',
                 'within_source_duplicate_extra_rows'):
        external.require(result[name] == audit[name], f'推理结果与预测前数据审计不一致: {name}')
    for name in external.STRATA:
        external.require(result['strata'][name]['label_support'] == audit['strata_label_support'][name],
                         f'推理结果与预测前数据审计分层不一致: {name}')


def run(args):
    external.require(re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) is not None,
                     'run-id 只能含字母数字下划线横线')
    external.require(args.chunk_rows > 0 and args.source_bytes > 0, '块大小和源文件大小必须为正数')
    out = OUTPUT_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'inference_performed': False, 'model_loaded': False,
        'script_sha256': sha256(__file__),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: version(name) for name in ('numpy', 'pandas', 'pyarrow')}},
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    write_json(out / 'manifest.json', manifest)
    try:
        try:
            manifest['git_head'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=external.ROOT, stderr=subprocess.PIPE, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sm_file = external.ROOT / 'SOURCE_MANIFEST.json'
            manifest['git_head'] = json.loads(sm_file.read_text(encoding='utf-8'))['source_commit'] if sm_file.is_file() else 'source-snapshot'
        manifest['code_sha256'] = {name: sha256(external.ROOT / name) for name in
                                  ('scripts/engineering_external.py', 'liushao/data.py')}
        summary, references, reference_provenance = external.load_reference_data(args.reference_data)
        manifest['reference'] = reference_provenance
        manifest['source'] = external.verify_source_inputs(args, summary)
        header, labels, dates = external.preflight_csv(args, summary['features'])
        manifest['source_timestamp_dates'] = dict(dates)
        manifest['feature_order_matches_source'] = [name for name in header
                                                    if name in summary['features']] == summary['features']
        audit_source(args, summary['features'], references, header, labels, out)
        manifest.update(status='completed', total_seconds=time.perf_counter() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        audit_sha256=sha256(out / 'audit.json'))
        write_json(out / 'manifest.json', manifest)
        return out
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        total_seconds=time.perf_counter() - started)
        write_json(out / 'manifest.json', manifest)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-data', 'source'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('source-url', 'source-sha256', 'run-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-bytes', required=True, type=int)
    parser.add_argument('--attack-label', required=True, action='append')
    parser.add_argument('--chunk-rows', type=int, default=20000)
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(run(parse_args()))
