"""Fit the 1.3.7 trusted triage engine on top of the frozen base model; test split is never read.

Usage:
    python -m scripts.fit_engine --data <processed data dir> --artifacts artifacts
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from liushao.data import sha256, write_json
from liushao_engine import engine_version, fit_engine


def load_split(data, bundle, names):
    summary = json.loads((data / 'summary.json').read_text(encoding='utf-8'))
    for name in ['features.parquet', 'manifest.parquet']:
        if sha256(data / name) != summary[name + '_sha256']:
            raise ValueError('数据哈希不匹配: ' + name)
    if summary['data_version'] != bundle['config']['data_version']:
        raise ValueError('数据版本与底座模型不一致')
    x, manifest = pd.read_parquet(data / 'features.parquet'), pd.read_parquet(data / 'manifest.parquet')
    if list(x.columns) != bundle['config']['features']:
        raise ValueError('特征顺序与底座模型不一致')
    y = (manifest.label != 'Benign').astype(int).to_numpy()
    out = {}
    for name in names:
        mask = manifest.split.eq(name).to_numpy()
        out[name] = (x[mask].reset_index(drop=True), y[mask], manifest[mask].reset_index(drop=True))
    return out, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed')
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    data, artifacts = Path(args.data), Path(args.artifacts)
    bundle = joblib.load(artifacts / 'model.joblib')
    descriptor = json.loads((artifacts / 'model.json').read_text(encoding='utf-8'))
    if sha256(artifacts / 'model.joblib') != descriptor['model_sha256']:
        raise ValueError('底座模型哈希不匹配')
    splits, summary = load_split(data, bundle, ['train', 'validation'])
    (xt, yt, _), (xv, yv, _) = splits['train'], splits['validation']
    tick = time.perf_counter()
    engine = fit_engine(bundle, xt, yt, xv, yv)
    seconds = time.perf_counter() - tick
    joblib.dump(engine, artifacts / 'engine.joblib', compress=3)
    write_json(artifacts / 'engine.json', {
        'format': engine['format'], 'engine_version': engine_version(engine),
        'engine_sha256': sha256(artifacts / 'engine.joblib'),
        'base_model_version': bundle['model_version'], 'base_model_sha256': descriptor['model_sha256'],
        'data_version': summary['data_version'],
        'data_sha256': {name: summary[name + '_sha256'] for name in ['features.parquet', 'manifest.parquet']},
        'chain': ['ECOD (train normal only)', 'Dempster-Shafer fusion of RF/LR/ECOD',
                  'Mondrian split conformal', 'Inductive Venn-Abers', 'Saabas tree-path attribution',
                  'MMD two-sample drift test'],
        'splits_used': {'ecod': 'train, label Benign only', 'fusion_calibration': 'validation half A (stratified, seed 42)',
                        'conformal_and_venn_abers': 'validation half B', 'test': 'not read'},
        'rows': engine['split'],
        'reliability': engine['fusion']['reliability'],
        'epsilon': engine['conformal']['epsilon'],
        'conflict_quantile': engine['conflict_quantile'], 'conflict_cut': engine['conflict_cut'],
        'fit_seconds': seconds,
        'source_code_sha256': {str(path).replace('\\', '/'): sha256(path)
                               for path in sorted(Path('liushao_engine').glob('*.py'))}})
    print(json.dumps({'engine_version': engine_version(engine), 'reliability': engine['fusion']['reliability'],
                      'conflict_cut': engine['conflict_cut'], 'rows': engine['split'], 'seconds': seconds},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
