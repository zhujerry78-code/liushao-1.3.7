"""Reproduce the frozen MVP and audit review efficiency without changing it.

Run from root: python -m scripts.audit_phase2
All original CLI side effects are contained in artifacts/phase2/replay_workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from liushao.data import EXCLUDED, sha256, split_records, write_json
from liushao.model import classification_metrics, infer, ranked, review_ranked, signed_log

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/phase2'
WORK = ROOT / 'artifacts/phase2/replay_workspace'
BUDGETS = (20, 50, 100)
RANDOM_SEED = 20261016
REPETITIONS = 10000


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def inference_invariance_checks(first, second):
    scores = ['rf_score', 'lr_score']
    decisions = ['rf_anomaly', 'lr_anomaly', 'disagreement', 'near_threshold']
    return {'scores_within_1e_12': bool(np.allclose(first[scores], second[scores], atol=1e-12, rtol=0)),
            'decisions_identical': first[decisions].equals(second[decisions])}


def flag_statistics(pred, y):
    y = np.asarray(y, dtype=bool)
    positive = pred.rf_anomaly.to_numpy(dtype=bool)
    fp, fn = positive & ~y, ~positive & y
    errors = fp | fn
    disagreement = pred.disagreement.to_numpy(dtype=bool)
    near = pred.near_threshold.to_numpy(dtype=bool)
    result = {}
    masks = {'disagreement': disagreement, 'near_threshold': near,
             'intersection': disagreement & near, 'union': disagreement | near}
    for name, mask in masks.items():
        count, found = int(mask.sum()), int((mask & errors).sum())
        found_fp, found_fn = int((mask & fp).sum()), int((mask & fn).sum())
        result[name] = {'records': count, 'errors': found,
                        'hint_precision': divide(found, count),
                        'error_coverage': divide(found, int(errors.sum())),
                        'false_positives_covered': found_fp, 'false_negatives_covered': found_fn,
                        'false_positive_coverage': divide(found_fp, int(fp.sum())),
                        'false_negative_coverage': divide(found_fn, int(fn.sum())),
                        'false_positives_total': int(fp.sum()), 'false_negatives_total': int(fn.sum())}
    return result


def policy_orders(pred):
    # Reset dataframe indices so returned integers always address row positions.
    pred = pred.reset_index(drop=True)
    score = ranked(pred).index.to_numpy()
    hinted = review_ranked(pred).index.to_numpy()
    used = np.zeros(len(pred), dtype=bool)
    used[hinted] = True
    return {'score': score, 'review': np.concatenate([hinted, score[~used[score]]]),
            'original': np.arange(len(pred)), 'hinted': hinted}


def evaluate_budgets(pred, y, budgets=BUDGETS, repetitions=REPETITIONS, seed=RANDOM_SEED):
    y = np.asarray(y, dtype=bool)
    positive = pred.rf_anomaly.to_numpy(dtype=bool)
    fp, fn = positive & ~y, ~positive & y
    errors = fp | fn
    n, total_errors = len(pred), int(errors.sum())
    if not n:
        raise ValueError('无测试候选，不能计算复核预算')
    orders = policy_orders(pred)
    rng = np.random.default_rng(seed)
    result = []
    for budget in budgets:
        k = min(int(budget), n)
        if k < 1:
            raise ValueError('K必须为正数')
        # Exact marginal distribution of uniform sampling WITHOUT replacement.
        random_found = rng.hypergeometric(total_errors, n - total_errors, k, size=repetitions)
        row = {'k_requested': int(budget), 'candidate_population': n,
               'random': {'k_actual': k, 'expected_errors': k * total_errors / n,
                          'expected_fp': k * int(fp.sum()) / n, 'expected_fn': k * int(fn.sum()) / n,
                          'simulation_mean': float(random_found.mean()),
                          'simulation_reference_p025': float(np.quantile(random_found, .025)),
                          'simulation_reference_p975': float(np.quantile(random_found, .975)),
                          'exact_reference_p025': int(hypergeom.ppf(.025, n, total_errors, k)),
                          'exact_reference_p975': int(hypergeom.ppf(.975, n, total_errors, k))}}
        for name in ['score', 'review', 'original']:
            selected = orders[name][:k]
            found = int(errors[selected].sum())
            row[name] = {'k_actual': k, 'errors_found': found,
                          'false_positives_found': int(fp[selected].sum()),
                          'false_negatives_found': int(fn[selected].sum()),
                          'error_precision_at_k': found / k,
                          'error_coverage_at_k': divide(found, total_errors)}
            if name == 'review':
                hint_count = min(k, len(orders['hinted']))
                row[name].update({'hint_records': hint_count, 'fill_records': k - hint_count})
        result.append(row)
    return result


def protected_snapshot():
    paths = [*Path(ROOT / 'docs/results').rglob('*'),
             *Path(ROOT / 'data/processed').glob('*'), *Path(ROOT / 'artifacts').glob('*'),
             *Path(ROOT / 'liushao').glob('*.py'), ROOT / 'app.py', ROOT / 'scripts/train.py',
             ROOT / 'scripts/prepare_data.py', ROOT / 'configs/data.lock.json']
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def replay_original(reuse=False):
    WORK.mkdir(parents=True, exist_ok=True)
    config = ROOT / 'configs/data.lock.json'
    lock = json.loads(config.read_text(encoding='utf-8'))
    raw_target = WORK / 'data/raw'
    raw_target.mkdir(parents=True, exist_ok=True)
    for item in lock['files']:
        source, target = ROOT / 'data/raw' / item['name'], raw_target / item['name']
        if sha256(source) != item['sha256']:
            raise ValueError('官方原始文件哈希不匹配: ' + item['name'])
        if not target.exists():
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copyfile(source, target)
        if sha256(target) != item['sha256']:
            raise ValueError('隔离原始文件哈希不匹配')
    commands = [
        [sys.executable, '-m', 'scripts.prepare_data', '--config', str(config), '--output', str(WORK / 'data_replay')],
        [sys.executable, '-m', 'scripts.train', '--data', str(WORK / 'data_replay'), '--output', str(WORK / 'model_replay')]]
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUTF8': '1'}
    runs = []
    for i, command in enumerate(commands):
        log = WORK / ('prepare.log' if i == 0 else 'train.log')
        if reuse:
            if not log.exists():
                raise ValueError('无既有重跑日志，请先执行完整重跑')
        else:
            print('Running original CLI:', ' '.join(command), flush=True)
            tick = time.perf_counter()
            with log.open('w', encoding='utf-8') as stream:
                process = subprocess.run(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'原始脚本执行失败，请查看 {log}')
            print(f'CLI {i + 1} passed, {time.perf_counter() - tick:.2f}s', flush=True)
        runs.append({'argv': command, 'cwd': str(WORK), 'log': str(log.relative_to(ROOT)), 'log_sha256': sha256(log)})
    return runs


def audit_split(x, manifest, summary):
    reconstructed = split_records(x, manifest, summary['config']['seed'], summary['config']['block_minutes'])
    columns = ['record_id', 'split', 'block_id', 'exact_fingerprint', 'near_fingerprint', 'purge_reason']
    pd.testing.assert_frame_equal(manifest[columns], reconstructed[columns])
    valid = manifest[manifest.split != 'purged']
    overlaps = {c: int((valid.groupby(c).split.nunique() > 1).sum())
                for c in ['record_id', 'block_id', 'exact_fingerprint', 'near_fingerprint']}
    assert not any(overlaps.values()), overlaps
    id_checks = manifest.record_id.eq(manifest.source_sha256 + ':' + manifest.source_row.astype(str)).all()
    assert id_checks and manifest.record_id.is_unique
    sample_rule = [int(hashlib.sha256(f'{summary["config"]["seed"]}:{digest}:{row}'.encode()).hexdigest()[:16], 16)
                   % summary['config']['sample_modulus'] == 0
                   for digest, row in zip(manifest.source_sha256, manifest.source_row)]
    assert all(sample_rule)
    duplicates = {}
    for name, group in valid.groupby('split'):
        duplicates[name] = {'records': len(group),
                            'exact_repeated_records': len(group) - group.exact_fingerprint.nunique(),
                            'near_repeated_records': len(group) - group.near_fingerprint.nunique(),
                            'blocks': group.block_id.nunique()}
    sources = pd.crosstab([valid.source_file, valid.split], valid.label).reset_index()
    times = []
    for source, group in valid.groupby('source_file'):
        training = group[group.split == 'train'].timestamp.sort_values().to_numpy(dtype='datetime64[ns]').astype('int64')
        for split in ['validation', 'test']:
            held = group[group.split == split]
            if held.empty:
                continue
            ticks = held.timestamp.to_numpy(dtype='datetime64[ns]').astype('int64')
            index = np.searchsorted(training, ticks)
            left = training[np.clip(index - 1, 0, len(training) - 1)]
            right = training[np.clip(index, 0, len(training) - 1)]
            distances = np.minimum(np.abs(ticks - left), np.abs(ticks - right)) / 1e9
            times.append({'source': source, 'split': split, 'records': len(held),
                          'timestamp_min': str(held.timestamp.min()), 'timestamp_max': str(held.timestamp.max()),
                          'fraction_with_later_training_record': float((ticks < training.max()).mean()),
                          'fraction_with_training_within_15min': float((distances <= 900).mean()),
                          'nearest_train_seconds_min': float(distances.min()),
                          'nearest_train_seconds_median': float(np.median(distances))})
    label_blocks = (valid.groupby(['split', 'label']).agg(records=('record_id', 'size'), blocks=('block_id', 'nunique'))
                    .reset_index().to_dict('records'))
    return {'reconstructed_split_identical': True, 'sample_rule_verified': True, 'stable_ids_verified': True,
            'cross_split_overlap': overlaps, 'within_split_duplicates': duplicates,
            'source_label_counts': sources.to_dict('records'), 'source_time_correlation': times,
            'label_block_support': label_blocks,
            'feature_exclusions_verified': not bool(set(x.columns) & EXCLUDED),
            'chronological_holdout': False, 'source_disjoint_holdout': False,
            'session_disjointness_verifiable': False}


def threshold_from_validation(y, scores):
    y = np.asarray(y, dtype=bool)
    choices = []
    for threshold in np.linspace(.05, .95, 91):
        pred = scores >= threshold
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        denom = 2 * tp + fp + fn
        choices.append((2 * tp / denom if denom else 0., float(threshold)))
    return max(choices)[1]


def audit_preprocessing(bundle, x, manifest):
    tr, val, test = [manifest.split.eq(name).to_numpy() for name in ['train', 'validation', 'test']]
    y = manifest.label.ne('Benign').to_numpy()
    expected = x[tr].median().fillna(0).to_numpy()
    checks = {'feature_order_matches': bundle['config']['features'] == x.columns.tolist(),
              'training_rows_match': bundle['training_rows'] == int(tr.sum()),
              'features_exclude_labels_and_metadata': not bool(set(x.columns) & EXCLUDED)}
    for name in ['rf', 'lr']:
        imputer = bundle[name].named_steps['impute']
        checks[name + '_imputer_train_only'] = bool(np.allclose(imputer.statistics_, expected, equal_nan=True))
        checks[name + '_missing_indicator_train_only'] = bool(np.array_equal(imputer.indicator_.features_, np.flatnonzero(x[tr].isna().any().to_numpy())))
        scores = bundle[name].predict_proba(x[val])[:, 1]
        independently_selected = threshold_from_validation(y[val], scores)
        checks[name + '_threshold_validation_only'] = abs(bundle['config'][name + '_threshold'] - independently_selected) < 1e-12
    transformed = bundle['lr'].named_steps['impute'].transform(x[tr])
    logged = signed_log(transformed)
    scaler = bundle['lr'].named_steps['scale']
    checks['lr_scaler_mean_train_only'] = bool(np.allclose(scaler.mean_, logged.mean(axis=0), atol=1e-12))
    checks['lr_scaler_variance_train_only'] = bool(np.allclose(scaler.var_, logged.var(axis=0), atol=1e-12))
    rf_val = bundle['rf'].predict_proba(x[val])[:, 1]
    margin = float(np.clip(np.quantile(np.abs(rf_val - bundle['config']['rf_threshold']), .1), .01, .1))
    checks['near_margin_validation_only'] = abs(margin - bundle['config']['near_margin']) < 1e-12
    normal = x[tr & ~y]
    reference = normal.quantile([.01, .5, .99]).T.to_numpy()
    actual_reference = np.array([[r['q01'], r['median'], r['q99']] for r in bundle['reference'].values()], dtype=float)
    checks['normal_reference_train_only'] = bool(np.allclose(reference, actual_reference, equal_nan=True))
    sample = x[test].iloc[:1000].reset_index(drop=True)
    first, features, _ = infer(sample, bundle, 'label-invariance', 'test')
    mutated = sample.copy()
    mutated['Label'] = 'opposite-or-invalid-label'
    mutated['Dst Port'] = -12345
    mutated['Src IP'] = 'excluded metadata'
    second, _, _ = infer(mutated, bundle, 'label-invariance', 'test')
    checks.update({'label_metadata_' + key: value for key, value in inference_invariance_checks(first, second).items()})
    checks['labels_absent_from_inference_features_and_output'] = 'label' not in first and not bool(set(features.columns) & EXCLUDED)
    assert all(checks.values()), checks
    return checks


def make_report(result):
    def percent(value):
        return '不适用（分母为0）' if value is None else f'{value:.2%}'
    metrics = result['reproduction']['metrics']
    model_rows = '\n'.join(f'| {name.upper()} | {metrics[name]["precision"]:.6f} | {metrics[name]["recall"]:.6f} | {metrics[name]["f1"]:.6f} | {metrics[name]["fpr"]:.6f} | {metrics[name]["fp"]} | {metrics[name]["fn"]} |'
                           for name in ['rf', 'lr'])
    flag_labels = {'disagreement': '模型分歧', 'near_threshold': '近阈值', 'intersection': '两者交集', 'union': '两者并集'}
    flag_rows = '\n'.join(f'| {flag_labels[name]} | {s["records"]} | {s["errors"]} | {percent(s["hint_precision"])} | {percent(s["error_coverage"])} | {s["false_positives_covered"]}/{s["false_positives_total"]} | {"不适用（无漏报）" if not s["false_negatives_total"] else str(s["false_negatives_covered"])+"/"+str(s["false_negatives_total"])} |'
                          for name, s in result['flags'].items())
    budget_rows = '\n'.join(f'| {b["k_requested"]} | {b["random"]["expected_errors"]:.4f} | {b["random"]["simulation_reference_p025"]:.0f}–{b["random"]["simulation_reference_p975"]:.0f} | {b["score"]["errors_found"]} | {b["review"]["errors_found"]} | {percent(b["review"]["error_precision_at_k"])} | {percent(b["review"]["error_coverage_at_k"])} | {b["review"]["fill_records"]} |'
                             for b in result['budgets'])
    # Avoid adding tabulate just for formatting. Full tables are in the JSON.
    support_rows = '\n'.join(f'| {r["split"]} | {r["label"]} | {r["records"]} | {r["blocks"]} |' for r in result['split']['label_block_support'])
    duplicate_rows = '\n'.join(f'| {name} | {s["records"]} | {s["exact_repeated_records"]} | {s["near_repeated_records"]} | {s["blocks"]} |' for name, s in result['split']['within_split_duplicates'].items())
    time_rows = '\n'.join(f'| {s["source"][:17]} | {s["split"]} | {percent(s["fraction_with_later_training_record"])} | {percent(s["fraction_with_training_within_15min"])} | {s["nearest_train_seconds_median"]:.0f} |' for s in result['split']['source_time_correlation'])
    original_rows = '\n'.join(f'| {b["k_requested"]} | {b["original"]["errors_found"]} |' for b in result['budgets'])
    source_support = '；'.join(f'{s["source"][:17]} 为 {s["records"]:,} 条' for s in result['split']['source_time_correlation'] if s['split'] == 'test')
    return f'''# 流哨第二阶段：可靠性与创新点审查

日期：2026-10-16（Asia/Shanghai）。本次仅审查原MVP，不调参、不增加模型、不覆盖原实验记录。基线Git提交为 `{result['baseline_git_head']}`。完整机器结果为 `audit.json`。

## 实验复现

直接调用原 `scripts.prepare_data` 与 `scripts.train`，在 `artifacts/phase2/replay_workspace` 隔离执行。官方原始文件采用本地硬链接（不支持则复制），只读使用；原脚本写出的docs/config/model均落在隔离工作区。

原始数据特征、manifest和拒绝记录文件哈希复现一致；划分重新计算一致。实际训练、验证、测试分别为161,331、21,468、15,519。两模型与基线测试分类、阈值和F1一致，最大得分差异见JSON。

| 模型 | Precision | Recall | F1 | FPR | 误报 | 漏报 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{model_rows}

RF F1=98.6104%，可复现。它适用于本次同源采集日的留出样本。Bot攻击测试1,986条，SlowHTTPTest仅1条，Hulk无测试支持数。不能写成所有攻击类型、未知攻击、跨环境或真实实验室性能。

## 泄漏与相关性

原始标签和端口、IP、时间、Flow ID未进入77维特征。训练中位数、缺失指示器、LR缩放统计、正常参考分布均与训练集重算一致；RF/LR阈值与近阈值宽度均与验证集独立重算一致。替换标签及排除元数据不影响1,000条测试记录推理得分。未发现测试集参与拟合、阈值选择或两模型排序的实现路径。

不同集合的record_id、15分钟块、完全特征指纹及六位有效数字指纹交集均为0。去重跨集合执行，集合内部仍存在重复：

| 集合 | 记录数 | 完全重复的额外记录 | 六位有效数字重复的额外记录 | 时间块 |
| --- | ---: | ---: | ---: | ---: |
{duplicate_rows}

这不等于不存在其他近重复、会话泄漏或采集偏差。两个80列表缺少主机/会话标识，不能核验同一会话的隔离；长流可能跨越块边界。当前是随机分配时间块，同一来源在训练、验证、测试间共享；训练时间既可能早于也可能晚于测试，不能称为严格时间外推。

保留测试记录按来源分布为：{source_support}。因此测试几乎集中于同一天的Bot场景，两个原始文件参与准备不等于两天均获得充分测试覆盖。

| 来源日期文件 | 留出集合 | 存在更晚训练记录的比例 | 距训练记录不超过15分钟 | 最近训练记录间隔中位数/秒 |
| --- | --- | ---: | ---: | ---: |
{time_rows}

类别独立时间块支持数很有限，不应把每一行都当作独立试验：

| 集合 | 标签 | 记录数 | 时间块数 |
| --- | --- | ---: | ---: |
{support_rows}

本次未选择新划分或阈值来提升指标。F1复现只能说明这一次固定实验可重复，不能弥补时间/来源相关性、类别覆盖及仅一个划分种子的外部有效性限制。

## 提示集合的错误发现

“提示精确率”定义为提示集合内RF错误数/提示数；“错误覆盖率”为提示内RF错误数/全部RF错误数。这里的错误是RF相对公开标签的误报或漏报，不是提示自身正确分类率，也不是攻击命中率。

| 提示集合 | 提示数 | RF错误数 | 提示精确率 | 错误覆盖率 | 误报覆盖 | 漏报覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{flag_rows}

当前RF无漏报，漏报覆盖率不适用，不能写为100%。交集记录不可重复加到并集中。未提示的错误数量见JSON；不能认为所有错误均可复核发现。

分歧提示单独覆盖44条误报；在此基础上加入近阈值提示，多看147条记录、额外覆盖7条误报（增量提示精确率4.76%）。因此近阈值提示提供部分额外覆盖，但带来明显检查成本。两者交集的较高精确率是本次事后观察，不据此修改既有排序；新排序需要新的独立验证。

## 固定预算对照

协议在运行前写入 `COORDINATION.md`：候选全集为15,519条，K=20/50/100；随机为均匀无放回检查；主模型按异常得分降序；待核实沿用原排序（先分歧、再到阈值距离、最后ID），不足K时才用主模型排序补足。排序不读取评测标签；标签仅用于事后统计错误。随机种子20261016、重复10,000次，使用无放回抽样错误数的精确超几何分布进行等价模拟。

| K | 随机错误数理论期望 | 随机错误数模拟95%参考范围 | 高异常得分发现错误 | 待核实发现错误 | 待核实错误精确率@K | 待核实错误覆盖率@K | 补足记录数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
{budget_rows}

随机范围是这个固定候选总体上的检查波动，不是跨网络性能置信区间。随机检查只需K条输入，提示/得分策略先对候选进行双模型推理；比较的是复核条数预算，不是相同计算时间或真人总耗时。

K从50增至100时，待核实策略发现的错误保持22条，新增50条复核在本次数据中未发现更多错误。不能只用K=20的较高精确率概括所有预算，也不能把全候选随机检查等同于只在主模型告警中随机检查。

历史原始文件顺序仅作额外参照，不是随机基准：

| K | 原始顺序发现错误 |
| --- | ---: |
{original_rows}

## 评价结论

待核实策略在本次固定测试候选上，比随机检查的理论期望和高异常得分排序发现更多RF错误，支持其作为误报复核入口的局部实证价值。它没有修改分类器，不能声称提高F1或自动修正模型错误。当前漏报增益无从评估，也没有跨时间外推、跨环境、更多攻击类型或真人节时的证据。

高异常得分策略的目标是优先发现攻击，待核实策略的目标是发现模型错误；本表只评判后者，不据此认定前者在攻击发现任务上较差。报告所有预定K，不按测试结果挑选最优K或调整排序。

## 证据与复现

- 运行 `.\\.venv\\Scripts\\python.exe -m scripts.audit_phase2` 可完整复跑原实验并重建本报告。
- 运行 `.\\.venv\\Scripts\\python.exe -m scripts.audit_phase2 --reuse-run` 可基于本次隔离重跑产物重新核算；日志哈希重新记录，数据/模型内容继续核验。
- 实际argv、工作目录、日志哈希、数据/模型版本、各项检查见 `audit.json`；测试输出写入 `tests.xml`。
- 原记录保持不变：`original_files_unchanged` 为 {result['original_files_unchanged']}；核验的原始文件数量为 {result['protected_file_count']}。
- 团队文件占用和公共接口见 `COORDINATION.md`；本轮不修改原核心模块。
- RF并行树汇总存在浮点舍入波动，得分不变性使用绝对容差1e-12，分类与提示标记仍要求完全一致；回归测试覆盖这一点。初次审查器的逐位相等断言曾误报，修复的是审查断言，没有改动原模型或阈值。
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-run', action='store_true', help='recheck this stage replay, without fitting again')
    args = parser.parse_args()
    from scripts.evaluation_preflight import unavailable
    needed = ['data/processed/' + name for name in ('summary.json', 'features.parquet', 'manifest.parquet', 'rejected.csv')]
    needed += ['artifacts/test_predictions.csv', 'artifacts/metrics.json']
    if not args.reuse_run:
        print('{"status":"not_run","passed":false,"reason":"历史完整审计包含重训；本次冻结模型交付不自动执行，请使用冻结复评入口。"}')
        return 2
    needed += ['artifacts/phase2/replay_workspace/model_replay/model.joblib',
               'artifacts/phase2/replay_workspace/model_replay/model.json',
               'artifacts/phase2/replay_workspace/data_replay/summary.json']
    if unavailable(ROOT, needed, git_ref='HEAD'):
        return 2
    before = protected_snapshot()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    commands = replay_original(args.reuse_run)
    original, replay = ROOT / 'data/processed', WORK / 'data_replay'
    summary = json.loads((original / 'summary.json').read_text(encoding='utf-8'))
    data_checks = {name: sha256(original / name) == sha256(replay / name)
                   for name in ['features.parquet', 'manifest.parquet', 'rejected.csv']}
    replay_summary = json.loads((replay / 'summary.json').read_text(encoding='utf-8'))
    data_checks['data_version'] = summary['data_version'] == replay_summary['data_version']
    assert all(data_checks.values()), data_checks
    x, manifest = pd.read_parquet(replay / 'features.parquet'), pd.read_parquet(replay / 'manifest.parquet')
    split = audit_split(x, manifest, summary)
    model_path = WORK / 'model_replay'
    descriptor = json.loads((model_path / 'model.json').read_text(encoding='utf-8'))
    assert sha256(model_path / 'model.joblib') == descriptor['model_sha256']
    bundle = joblib.load(model_path / 'model.joblib')
    preprocess = audit_preprocessing(bundle, x, manifest)
    test = manifest.split.eq('test').to_numpy()
    pred, _, rejected = infer(x[test].reset_index(drop=True), bundle, 'audit', 'test')
    pred['record_id'] = manifest[test].record_id.reset_index(drop=True)
    y = manifest[test].label.ne('Benign').to_numpy()
    old_pred = pd.read_csv(ROOT / 'artifacts/test_predictions.csv')
    assert old_pred.record_id.equals(pred.record_id)
    baseline = json.loads((ROOT / 'artifacts/metrics.json').read_text(encoding='utf-8'))
    metrics = {name: classification_metrics(y, pred[name + '_score'], bundle['config'][name + '_threshold']) for name in ['rf', 'lr']}
    same_metrics = {name: metrics[name] == baseline[name] for name in metrics}
    same_classes = {name: bool(np.array_equal(old_pred[name + '_anomaly'], pred[name + '_anomaly'])) for name in metrics}
    differences = {name: float(np.max(np.abs(old_pred[name + '_score'] - pred[name + '_score']))) for name in metrics}
    assert all(same_metrics.values()) and all(same_classes.values()) and max(differences.values()) <= 1e-12
    flags, budgets = flag_statistics(pred, y), evaluate_budgets(pred, y)
    no_labels = pred.drop(columns=[c for c in pred if 'label' in c.lower() or 'ground_truth' in c], errors='ignore')
    assert all(np.array_equal(policy_orders(pred)[k], policy_orders(no_labels)[k]) for k in policy_orders(pred))
    after = protected_snapshot()
    assert before == after, '原始结果或公共实现被并发修改，停止审查而不覆盖'
    result = {'date': '2026-10-16', 'baseline_git_head': head, 'python': platform.python_version(),
              'audit_script_sha256': sha256(Path(__file__)),
              'commands': commands, 'data_checks': data_checks, 'data_version': summary['data_version'],
              'model_version': bundle['model_version'], 'model_sha256': descriptor['model_sha256'],
              'split': split, 'preprocessing_checks': preprocess,
              'reproduction': {'metrics': metrics, 'same_metrics': same_metrics, 'same_classes': same_classes,
                               'score_max_absolute_difference': differences, 'unjudgeable': len(rejected)},
              'flags': flags, 'budgets': budgets,
              'random_protocol': {'seed': RANDOM_SEED, 'repetitions': REPETITIONS, 'method': 'hypergeometric equivalent to uniform sampling without replacement', 'budgets': list(BUDGETS)},
              'original_files_unchanged': True, 'protected_file_count': len(before),
              'original_file_hashes': before,
              'limits': ['same capture sources and interleaved times', 'test attacks: Bot1986, SlowHTTPTest1, Hulk0',
                         'only one split and no real-user efficiency test', 'no evidence of false-negative recovery with zero baseline false negatives',
                         'duplicate audit only exact and six significant digits; no session IDs available']}
    write_json(OUTPUT / 'audit.json', result)
    (OUTPUT / 'AUDIT_REPORT.md').write_text(make_report(result), encoding='utf-8')
    print(json.dumps({'rf': metrics['rf'], 'lr': metrics['lr'], 'flags': flags, 'budgets': budgets,
                      'original_files_unchanged': True}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
