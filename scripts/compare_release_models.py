"""Paired, read-only evaluation of two specific trusted handoff artifacts; never fit."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd

from app_model import load_model_bundle, prepare_model
from liushao.data import clean, fingerprints, read_csv_checked, sha256, write_json
from liushao.model import classification_metrics, infer, ranked, review_ranked
from scripts import engineering_external as ext
from scripts.engineering_external_preflight import audit_source
from scripts.verify_frozen_release import checked_zip_member, PACKAGE_SHA, INNER_SHA, MODEL_SHA, DESCRIPTOR_SHA

ROOT = Path(__file__).resolve().parents[1]
NEW_ZIP_SHA = '7e88c7449b01f42a56c1341e48d780d316bb79990685534df1f5fe7978d87041'
NEW_MODEL_SHA = '40ee89927c853d4d0ada25f32b7ed464838562c3c0b2b52b737570d6d9d4bea6'
NEW_DESCRIPTOR_SHA = '8eb8c09b5eaa7e77bd044051c0df4e4a5abc9bc5d35907be97161f5bc1caf30a'
SOURCE_SHA = 'fa2947a8256d81ee9103ae16139d62d0e17aa23e696ee80d9e76fb51c01c9c4b'
SOURCE_NAME = 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_demo_input():
    with (ROOT/'artifacts/demo.csv').open('rb') as stream:
        return read_csv_checked(stream)


def aligned_truth(pred, labels_by_row):
    require(pred.source_row.is_unique, '真值关联的来源行重复')
    require(set(pred.source_row) <= labels_by_row.keys(), '真值关联缺少来源行')
    return np.array([labels_by_row[int(r)] != 'Benign' for r in pred.source_row], dtype=int)


def restore_test_identity(pred, meta):
    require(len(pred) == len(meta), '原test身份恢复分母不同')
    result = pred.reset_index(drop=True).copy()
    for field in ('record_id', 'source_file', 'source_sha256', 'source_row', 'timestamp'):
        result[field] = meta[field].to_numpy()
    return result


def label_support(truth, vocabulary):
    counts = Counter(truth)
    return {label: int(counts[label]) for label in vocabulary}


def review_summary(pred, y, budgets=(20, 50, 100)):
    pred = pred.reset_index(drop=True)
    y = np.asarray(y, dtype=bool)
    require(len(pred) == len(y), '评价分母不一致')
    fp = pred.rf_anomaly.to_numpy() & ~y
    fn = ~pred.rf_anomaly.to_numpy() & y
    error = fp | fn
    masks = {'disagreement': pred.disagreement.to_numpy(),
             'near_threshold': pred.near_threshold.to_numpy(),
             'union': (pred.disagreement | pred.near_threshold).to_numpy()}
    sets = {}
    for name, mask in masks.items():
        count, found = int(mask.sum()), int(error[mask].sum())
        sets[name] = {'records': count, 'errors': found, 'fp': int(fp[mask].sum()),
            'fn': int(fn[mask].sum()), 'error_fraction': found/count if count else None,
            'workload_fraction': count/len(pred) if len(pred) else None,
            'fp_coverage': int(fp[mask].sum())/int(fp.sum()) if fp.any() else None,
            'fn_coverage': int(fn[mask].sum())/int(fn.sum()) if fn.any() else None}
    base = ranked(pred).index.tolist()
    orders = {'union': review_ranked(pred).index.tolist()}
    distance = (pred.rf_score - pred.rf_threshold).abs()
    for name in ('disagreement', 'near_threshold'):
        subset = pred.loc[masks[name]].assign(distance=distance[masks[name]])
        orders[name] = subset.sort_values(['distance', 'record_id'], kind='stable').index.tolist()
    orders['rf_half_centrality'] = pred.assign(distance=(pred.rf_score-.5).abs()).sort_values(
        ['distance', 'record_id'], kind='stable').index.tolist()
    orders['rf_threshold_distance'] = pred.assign(distance=distance).sort_values(
        ['distance', 'record_id'], kind='stable').index.tolist()
    results = {}
    for name, order in orders.items():
        initial = len(order)
        selected = set(order)
        complete = order + [i for i in base if i not in selected]
        results[name] = []
        for k in budgets:
            take = complete[:k]
            results[name].append({'k': k, 'reviewed': len(take),
                'hint_records': min(k, initial) if name in masks else None,
                'errors_found': int(error[take].sum()), 'fp_found': int(fp[take].sum()),
                'fn_found': int(fn[take].sum())})
    return {'population': len(pred), 'rf_errors': int(error.sum()), 'rf_fp': int(fp.sum()),
        'rf_fn': int(fn.sum()), 'sets': sets, 'budgets': results,
        'random_expected_errors': {str(k): min(k, len(pred))*int(error.sum())/len(pred)
                                   if len(pred) else None for k in budgets}}


def unpack_and_load(old_zip, new_zip, out):
    old_content = Path(old_zip).read_bytes()
    inner = checked_zip_member(old_content, PACKAGE_SHA, '模型文件/liushao-frozen-c7a028b4e4cafedd.zip')
    old_model = checked_zip_member(inner, INNER_SHA, 'artifacts/model.joblib')
    old_desc = checked_zip_member(inner, INNER_SHA, 'artifacts/model.json')
    new_content = Path(new_zip).read_bytes()
    require(hashlib.sha256(new_content).hexdigest() == NEW_ZIP_SHA, 'A总包SHA不一致')
    with zipfile.ZipFile(io.BytesIO(new_content)) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'A包重复成员')
        manifest = json.loads(archive.read('manifest.json'))
        for item in manifest['files']:
            value = archive.read(item['path'])
            require(len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'],
                    'A包清单不匹配: ' + item['path'])
        new_model, new_desc = (archive.read('artifacts/'+n) for n in ('model.joblib', 'model.json'))
        lock = archive.read('requirements-lock.txt').decode('utf-8-sig')
        require([s.strip() for s in lock.splitlines() if s.strip()] ==
                [s.strip() for s in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8').splitlines() if s.strip()],
                'A包依赖锁不一致')
        order = [s.strip() for s in archive.read('feature_order.txt').decode('utf-8-sig').splitlines() if s.strip()]
        require(order == json.loads(new_desc)['config']['features'], 'A包特征顺序文件不一致')
        new_members = [{'path': i.filename, 'bytes': i.file_size,
                        'sha256': hashlib.sha256(archive.read(i)).hexdigest()} for i in archive.infolist()]
    bundles, identities = {}, {}
    for name, content, descriptor, model_sha, desc_sha, expected_version in (
        ('original', old_model, old_desc, MODEL_SHA, DESCRIPTOR_SHA, 'c7a028b4e4cafedd'),
        ('rebuilt', new_model, new_desc, NEW_MODEL_SHA, NEW_DESCRIPTOR_SHA, '15547c4b02b362e0')):
        require(hashlib.sha256(content).hexdigest() == model_sha, '模型SHA不一致')
        require(hashlib.sha256(descriptor).hexdigest() == desc_sha, '描述SHA不一致')
        folder = out / name
        folder.mkdir()
        (folder/'model.joblib').write_bytes(content)
        (folder/'model.json').write_bytes(descriptor)
        data, text, context = prepare_model(folder/'model.joblib', folder/'model.json')
        bundles[name] = load_model_bundle(data, text)
        d = json.loads(text)
        require(d['model_version'] == expected_version, '版本不一致')
        # prepare_model checks the actually imported runtime sources. Both packages
        # use the same verified source-equivalent runtime; do not recheck a different checkout.
        identities[name] = {'model_version': expected_version, 'model_sha256': model_sha,
            'descriptor_sha256': desc_sha, 'config': d['config'], 'training_rows': d['training_rows'],
            'normal_reference_rows': d['normal_reference_rows'], 'app_guard_context': context,
            'classes': {m: bundles[name][m].predictor.classes_.tolist() for m in ('rf', 'lr')},
            'pipeline_steps': {m: list(bundles[name][m].predictor.named_steps) for m in ('rf', 'lr')},
            'classifier_dimensions': {m: int(bundles[name][m].predictor.named_steps['classifier'].n_features_in_)
                                      for m in ('rf', 'lr')}, 'legacy_application_load_passed': True}
    return bundles, identities, new_members


def population_result(predictions, labels, masks, bundles):
    truth = np.asarray(labels)
    y = (truth != 'Benign').astype(int)
    result = {}
    for stratum, mask in masks.items():
        result[stratum] = {'label_support': label_support(truth[mask], sorted(set(truth))), 'models': {}}
        for name, pred in predictions.items():
            part = pred.loc[mask].reset_index(drop=True)
            result[stratum]['models'][name] = {
                'metrics': {m: classification_metrics(y[mask], part[m+'_score'], bundles[name]['config'][m+'_threshold'])
                            for m in ('rf', 'lr')}, 'review': review_summary(part, y[mask])}
    return result


def compare(args, out):
    installed = {}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.split('==')
        installed[name] = version(name)
        require(installed[name] == expected, '环境版本不一致: '+name)
    require(platform.python_version_tuple()[:2] == ('3', '12'), '需要Python3.12')
    bundles, identities, members = unpack_and_load(args.original_zip, args.rebuilt_zip, out)
    original = bundles['original']['config']
    rebuilt = bundles['rebuilt']['config']
    require(original['data_version'] == rebuilt['data_version'], '两模型声明的数据版本不同')
    require(original['features'] == rebuilt['features'], '两模型特征顺序不同')
    summary, references, provenance = ext.load_reference_data(ROOT/'data/processed')
    x = pd.read_parquet(ROOT/'data/processed/features.parquet')
    manifest = pd.read_parquet(ROOT/'data/processed/manifest.parquet')
    take = manifest.split.eq('test').to_numpy()
    xt, mt = x[take].reset_index(drop=True), manifest[take].reset_index(drop=True)
    demo = read_demo_input()
    demo_checks, predictions, times = {}, {}, {n: {m: [] for m in ('rf', 'lr')} for n in bundles}
    for name, bundle in bundles.items():
        dp, _, rejected = infer(demo.drop(columns=['Label'], errors='ignore'), bundle, 'demo', 'demo.csv')
        require(len(dp) == 1000 and not len(rejected), '应用演示输入未通过')
        demo_checks[name] = {'rows': len(dp), 'rejected': len(rejected), 'label_absent': 'label' not in dp}
        predictions[name], _, rejected = infer(xt, bundle, 'paired-test', 'original-test')
        require(not len(rejected), '原test拒绝数量改变')
        predictions[name] = restore_test_identity(predictions[name], mt)
        predictions[name].to_parquet(out/(name+'-test.parquet'), index=False)
    for repeat in range(5):
        for name in (list(bundles) if repeat % 2 == 0 else list(reversed(bundles))):
            for model in ('rf', 'lr'):
                start = time.perf_counter()
                bundles[name][model].predict_proba(xt)
                times[name][model].append(time.perf_counter()-start)
    results = {'models': identities, 'new_archive_members': members, 'packages': installed,
        'reference_data': provenance, 'app_demo_checks': demo_checks,
        'original_test': population_result(predictions, mt.label.to_numpy(), {'all': np.ones(len(mt), bool)}, bundles),
        'pipeline_seconds_15519_rows_5_warm_alternating_runs': times,
        'paired_original_test_differences': {m: {
            'max_absolute_score_difference': float(np.max(np.abs(predictions['original'][m+'_score']-predictions['rebuilt'][m+'_score']))),
            'decision_difference_rows': int((predictions['original'][m+'_anomaly'] != predictions['rebuilt'][m+'_anomaly']).sum())}
            for m in ('rf', 'lr')},
        'training_provenance_scope': 'Original frozen hashes verified; rebuilt data version/source and split provenance from A report, descriptor and same reference hashes, not independent observation of training.',
        'scope': 'Descriptive paired comparison. No retuning, independent training-process proof, deployment claim or model selection based on these scores.',
        'training_executed': False}
    if args.external_source:
        external_args = SimpleNamespace(source=args.external_source, source_sha256=SOURCE_SHA,
            source_url='https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/'+SOURCE_NAME,
            source_bytes=375945899, chunk_rows=20000,
            attack_label=['DoS attacks-GoldenEye', 'DoS attacks-Slowloris'])
        ext.verify_source_inputs(external_args, summary)
        header, labels, dates = ext.preflight_csv(external_args, original['features'])
        audit_folder = out/'external-audit'
        audit_folder.mkdir()
        audit = audit_source(external_args, original['features'], references, header, labels, audit_folder)
        all_pred = {n: [] for n in bundles}
        all_truth, all_overlap = [], []
        wall = {n: 0. for n in bundles}
        for block, chunk in enumerate(ext.csv_chunks(args.external_source, header, 20000)):
            labels_by_row = dict(zip(chunk.attrs['source_rows'], chunk.Label.str.strip()))
            raw = chunk.drop(columns='Label')
            raw.attrs['source_rows'] = chunk.attrs['source_rows']
            xp, meta, rejected = clean(raw, original['features'], SOURCE_SHA, SOURCE_NAME)
            overlap = np.zeros(len(xp), bool)
            for kind, near in (('exact', False), ('near', True)):
                fp = fingerprints(xp, near=near)
                overlap |= fp.isin(set().union(*references[kind].values())).to_numpy()
            for name in (list(bundles) if block % 2 == 0 else list(reversed(bundles))):
                tick = time.perf_counter()
                pred, _, bad = infer(raw, bundles[name], SOURCE_SHA, SOURCE_NAME)
                wall[name] += time.perf_counter()-tick
                require(pred.record_id.tolist() == meta.record_id.tolist() and len(bad) == len(rejected), '成对有效行不一致')
                pred.to_parquet(out/f'{name}-external-{block:03d}.parquet', index=False)
                all_pred[name].append(pred)
            all_truth.extend([labels_by_row[int(r)] for r in meta.source_row])
            all_overlap.extend(overlap.tolist())
            print(f'Paired inference {block+1}: {len(all_truth)} rows', flush=True)
        predictions = {n: pd.concat(v, ignore_index=True) for n, v in all_pred.items()}
        overlap = np.array(all_overlap)
        result = population_result(predictions, np.array(all_truth),
            {'all': np.ones(len(overlap), bool), 'any_overlap': overlap, 'no_overlap': ~overlap}, bundles)
        require(len(overlap) == audit['quality']['valid_rows'], '全量推理分母与审计不一致')
        for stratum in result:
            require(result[stratum]['label_support'] == audit['strata_label_support'][stratum], '分层推理与审计不一致')
        # Per-attack binary metrics use all benign rows plus that attack; never merge their denominators.
        truth = np.array(all_truth)
        result['per_attack_vs_benign'] = {label: population_result(predictions, truth,
            {'all': (truth == 'Benign') | (truth == label)}, bundles)['all'] for label in external_args.attack_label}
        results['external'] = {'source_sha256': SOURCE_SHA, 'source_bytes': 375945899,
            'source_label_support': dict(labels), 'timestamp_dates': dict(dates), 'audit_quality': audit['quality'],
            'data_audit_sha256': sha256(audit_folder/'audit.json'),
            'data_audit_completed_before_external_inference': True, 'strata': result,
            'infer_wall_seconds_including_cleaning': wall}
        ext.verify_file(args.external_source, SOURCE_SHA, 375945899)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-zip', required=True, type=Path)
    parser.add_argument('--rebuilt-zip', required=True, type=Path)
    parser.add_argument('--external-source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': sha256(__file__), 'python': platform.python_version(), 'platform': platform.platform(),
        'arguments': {k: str(v) if v is not None else None for k,v in vars(args).items()}}
    write_json(args.output/'manifest.json', result)
    try:
        result.update(compare(args, args.output), status='completed')
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    result.update(seconds=time.perf_counter()-started, finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output/'comparison.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'seconds', 'error')}, ensure_ascii=False))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
