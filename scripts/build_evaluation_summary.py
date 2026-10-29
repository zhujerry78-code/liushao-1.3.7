"""Aggregate existing local frozen evidence; never run training or evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from app_evaluation import validate_summary
from delivery.safe_files import read_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('docs/release/FROZEN_TEST_20261016.json',
           'docs/release/MODEL_COMPARISON_20261016.json', 'docs/results/metrics.json',
           'docs/innovation/study.json', 'docs/phase2/audit.json', 'docs/results/data_summary.json')
MODEL = '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7'
DESCRIPTOR = 'e280995a00084234c538720541db6475bb51382c4e2ffc9c0e4db8cbde933880'


def sha(content):
    return hashlib.sha256(content).hexdigest()


def assemble(repo, model_package):
    raw = {name: (repo / name).read_bytes() for name in SOURCES}
    docs = {name: read_json(content) for name, content in raw.items()}
    frozen, comparison, metrics, study, audit, data = (docs[name] for name in SOURCES)
    with zipfile.ZipFile(model_package) as archive:
        descriptors = [name for name in archive.namelist() if name.endswith('/artifacts/model.json')]
        if len(descriptors) != 1:
            raise ValueError('Expected one frozen descriptor in the local package')
        descriptor_bytes = archive.read(descriptors[0])
        model_bytes = archive.read(descriptors[0][:-len('model.json')] + 'model.joblib')
    if sha(model_bytes) != MODEL or sha(descriptor_bytes) != DESCRIPTOR:
        raise ValueError('Untrusted frozen model or descriptor')
    descriptor = read_json(descriptor_bytes)
    identity = {'model_version': descriptor['model_version'], 'model_sha256': MODEL,
                'descriptor_sha256': DESCRIPTOR, 'data_version': descriptor['config']['data_version']}
    for item in (frozen, comparison['models']['original']):
        if item['model_sha256'] != MODEL or item['descriptor_sha256'] != DESCRIPTOR:
            raise ValueError('Historical results refer to another model')
    for item in (metrics, study, audit, data):
        if item['data_version'] != identity['data_version']:
            raise ValueError('Historical results refer to another dataset')
    # These older audit results used a separately rebuilt model at the top level.
    # Only its split section is copied; it does not authenticate this model's identity.
    for name in (SOURCES[2], SOURCES[4], SOURCES[5]):
        expected = study['protected_hashes'][name.replace('/', '\\')]
        if sha(raw[name]) != expected:
            raise ValueError('Changed historical source: ' + name)
    if any(frozen['metrics'][key] != metrics[key] for key in ('rf', 'lr')):
        raise ValueError('Frozen classification metrics disagree')
    for key in ('model.joblib', 'model.json'):
        if study['protected_hashes']['artifacts\\' + key] != (MODEL if key.endswith('joblib') else DESCRIPTOR):
            raise ValueError('Innovation study has another model identity')
    info = {key: descriptor[key] for key in ('training_rows', 'normal_reference_rows')}
    info.update({key: descriptor['config'][key] for key in ('features', 'rf_threshold', 'lr_threshold', 'near_margin')})
    ext = {'evaluated_at': comparison['started_at_utc'], 'source_sha256': comparison['external']['source_sha256'],
           'quality': comparison['external']['audit_quality']}
    for key in ('all', 'any_overlap', 'no_overlap'):
        group = comparison['external']['strata'][key]
        ext[key] = {'records': sum(group['label_support'].values()), 'class_support': group['label_support'],
                    'metrics': group['models']['original']['metrics']}
    slow = comparison['external']['strata']['per_attack_vs_benign']['DoS attacks-Slowloris']['models']['original']['metrics']['rf']
    ext['slowloris'] = {'detected': slow['tp'], 'support': slow['attack_support']}
    split = {'counts': data['split_counts'], 'block_minutes': data['config']['block_minutes'],
             'seed': data['config']['seed'], 'cross_split_overlap': audit['split']['cross_split_overlap']}
    for key in ('chronological_holdout', 'source_disjoint_holdout', 'session_disjointness_verifiable'):
        split[key] = audit['split'][key]
    summary = {'format': 1, 'identity': identity, 'model_info': info,
               'held_out': {'evaluated_at': frozen['started_at_utc'], 'records': frozen['test_records'],
                            'class_support': frozen['class_support'], 'metrics': frozen['metrics']},
               'attack_ranking': next(row for row in metrics['ranking_score'] if row['k_requested'] == 100),
               'error_review': {'total_errors': study['total_errors'], 'budgets': study['budgets'],
                                'evaluated_at': study['date'], 'scope': study['analysis_status']},
               'external': ext, 'split': split,
               'sources': [{'path': name, 'sha256': sha(raw[name]),
                            'evaluated_at': docs[name].get('started_at_utc', docs[name].get('date', '原始数据处理记录')),
                            'git_head': docs[name].get('git_head_at_execution', docs[name].get('git_head', docs[name].get('baseline_git_head', '未记录')))}
                           for name in SOURCES]}
    derivatives = {name: docs[name]['publication'] for name in SOURCES if docs[name].get('publication')}
    if derivatives:
        from copy import deepcopy
        from scripts.publish_evidence import annotate_summary, encoded
        original_summary = deepcopy(summary)
        for source in original_summary['sources']:
            if source['path'] in derivatives:
                source['sha256'] = derivatives[source['path']]['original_sha256']
        annotate_summary(summary, derivatives, sha(encoded(original_summary)))
    return validate_summary(summary, MODEL, DESCRIPTOR, identity['data_version'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--model-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.repo.resolve()):
        parser.error('New evaluation summaries must stay outside the Git repository')
    summary = assemble(args.repo, args.model_package)
    content = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as file:
        file.write(content)
    print(json.dumps({'path': str(args.output), 'sha256': sha(content)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
