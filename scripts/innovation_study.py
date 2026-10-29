"""Frozen-prediction ablation and paired time-block sensitivity analysis."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from liushao.data import sha256, write_json
from liushao.model import review_ranked

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/innovation'
LABELS = {'margin': 'RF阈值距离', 'vote_uncertainty': 'RF 0.5中心度（熵排序）',
          'disagreement': '仅分歧', 'near': '仅近阈值', 'combined': '原组合', 'attack': '攻击优先（任务参照）'}
BUDGETS = (20, 50, 100)


def ratio(n, d):
    return float(n / d) if d else None


def orders(pred):
    ids = pred.record_id.astype(str).to_numpy()
    score = pred.rf_score.to_numpy()
    distance = np.abs(score - pred.rf_threshold.to_numpy())
    disagreement, near = pred.disagreement.to_numpy(bool), pred.near_threshold.to_numpy(bool)
    high = np.lexsort((ids, -score))
    margin = np.lexsort((ids, distance))
    def fill(front):
        included = np.zeros(len(pred), dtype=bool)
        included[front] = True
        return np.concatenate([front, high[~included[high]]])
    d = margin[disagreement[margin]]
    t = margin[near[margin]]
    union = np.lexsort((ids, distance, ~disagreement))
    union = union[(disagreement | near)[union]]
    return {'margin': margin, 'vote_uncertainty': np.lexsort((ids, np.abs(score - .5))),
            'disagreement': fill(d), 'near': fill(t), 'combined': fill(union), 'attack': high}


def masks(pred):
    d, t = pred.disagreement.to_numpy(bool), pred.near_threshold.to_numpy(bool)
    return {'disagreement': d, 'near': t, 'intersection': d & t, 'combined': d | t,
            'disagreement_only': d & ~t, 'near_only': t & ~d, 'unhinted': ~(d | t)}


def workload(pred, y):
    y = np.asarray(y, dtype=bool)
    positive = pred.rf_anomaly.to_numpy(bool)
    fp, fn = positive & ~y, ~positive & y
    error = fp | fn
    result = {}
    for name, selected in masks(pred).items():
        n, e, f, missed = map(int, [selected.sum(), (selected & error).sum(),
                                  (selected & fp).sum(), (selected & fn).sum()])
        result[name] = {'population': len(pred), 'records': n, 'errors': e,
                        'error_precision': ratio(e, n), 'error_coverage': ratio(e, int(error.sum())),
                        'false_positives': f, 'false_positives_total': int(fp.sum()),
                        'false_positive_coverage': ratio(f, int(fp.sum())),
                        'false_negatives': missed, 'false_negatives_total': int(fn.sum()),
                        'false_negative_coverage': ratio(missed, int(fn.sum())),
                        'workload_fraction': ratio(n, len(pred)), 'checks_per_error': ratio(n, e),
                        'reviews_without_error': n - e}
    return result


def random_reference(pred, y, k):
    error = pred.rf_anomaly.to_numpy(bool) != np.asarray(y, dtype=bool)
    result = {}
    for name, selected in [('all', np.ones(len(pred), bool)), ('alert', pred.rf_anomaly.to_numpy(bool))]:
        n, e = int(selected.sum()), int(error[selected].sum())
        actual = min(k, n)
        result[name] = {'population': n, 'errors_in_population': e, 'k_actual': actual,
                        'expected_errors': ratio(actual * e, n),
                        'reference_p025': int(hypergeom.ppf(.025, n, e, actual)) if n else None,
                        'reference_p975': int(hypergeom.ppf(.975, n, e, actual)) if n else None,
                        'scope': 'all candidates' if name == 'all' else 'RF alarms only; not eligible for missed attacks'}
    return result


def budget_table(pred, y, budgets=BUDGETS):
    y = np.asarray(y, dtype=bool)
    positive = pred.rf_anomaly.to_numpy(bool)
    fp, fn = positive & ~y, ~positive & y
    errors, total = fp | fn, int((fp | fn).sum())
    policy_orders, hint_masks = orders(pred), masks(pred)
    rows = []
    for name, order in policy_orders.items():
        for k in budgets:
            take = order[:min(k, len(order))]
            found = int(errors[take].sum())
            hinted = min(len(take), int(hint_masks[name].sum())) if name in ['disagreement', 'near', 'combined'] else None
            rows.append({'strategy': name, 'label': LABELS[name], 'k': k,
                         'population': len(pred), 'k_actual': len(take),
                         'errors_found': found, 'total_errors': total,
                         'false_positives_found': int(fp[take].sum()), 'false_negatives_found': int(fn[take].sum()),
                         'error_precision_at_k': ratio(found, len(take)), 'error_coverage_at_k': ratio(found, total),
                         'workload_fraction': ratio(len(take), len(pred)),
                         'checks_per_error': ratio(len(take), found), 'reviews_without_error': len(take) - found,
                         'hint_records': hinted, 'fill_records': len(take) - hinted if hinted is not None else None})
    return rows


def block_sample_indices(groups, sampled_groups):
    groups = np.asarray(groups)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])


def interval(values):
    valid = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(valid):
        return {'n_valid': 0, 'p025': None, 'median': None, 'p975': None}
    lo, median, hi = np.quantile(valid, [.025, .5, .975])
    return {'n_valid': len(valid), 'p025': float(lo), 'median': float(median), 'p975': float(hi)}


def bootstrap(pred, y, groups, budgets=BUDGETS, repeats=2000, seed=20261017):
    y = np.asarray(y, bool)
    error = pred.rf_anomaly.to_numpy(bool) != y
    unique, codes = np.unique(groups, return_inverse=True)
    policy_orders, hint_masks = orders(pred), masks(pred)
    samples = {str(k): {name: [] for name in [*policy_orders, 'combined_minus_margin', 'combined_minus_vote_uncertainty', 'combined_minus_disagreement']} for k in budgets}
    precision_samples, coverage_samples = [], []
    populations, totals = [], []
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = multiplicity[codes]
        populations.append(int(weights.sum()))
        total_errors = int(np.dot(weights, error))
        totals.append(total_errors)
        union_n = int(weights[hint_masks['combined']].sum())
        union_e = int(weights[hint_masks['combined'] & error].sum())
        precision_samples.append(ratio(union_e, union_n))
        coverage_samples.append(ratio(union_e, total_errors))
        counts = {}
        for name, order in policy_orders.items():
            ordered_weights = weights[order]
            cumulative = np.cumsum(ordered_weights)
            previous = cumulative - ordered_weights
            for k in budgets:
                selected_weights = np.clip(k - previous, 0, ordered_weights)
                found = int(np.dot(selected_weights, error[order]))
                samples[str(k)][name].append(found)
                counts[name, k] = found
        for k in budgets:
            samples[str(k)]['combined_minus_margin'].append(counts['combined', k] - counts['margin', k])
            samples[str(k)]['combined_minus_vote_uncertainty'].append(counts['combined', k] - counts['vote_uncertainty', k])
            samples[str(k)]['combined_minus_disagreement'].append(counts['combined', k] - counts['disagreement', k])
    return {'replicates': repeats, 'seed': seed, 'cluster_count': len(unique),
            'method': 'paired resampling of observed 15-minute capture blocks; frozen predictions and multiplicity retained',
            'zero_error_replicates': sum(e == 0 for e in totals),
            'resampled_population': interval(populations), 'resampled_errors': interval(totals),
            'combined_precision': interval(precision_samples), 'combined_coverage': interval(coverage_samples),
            'budgets': {key: {name: interval(values) for name, values in strategies.items()} for key, strategies in samples.items()},
            'interpretation': 'conditional percentile sensitivity range from only 11 dependent observed blocks; not a guaranteed cross-environment confidence interval or retraining stability'}


def ranking_diagnostic(pred, y):
    order = orders(pred)['combined']
    error = pred.rf_anomaly.to_numpy(bool) != np.asarray(y, bool)
    selected = pred.iloc[order].copy().reset_index(drop=True)
    selected['rank'] = np.arange(1, len(selected) + 1)
    selected['rf_error_evaluation_only'] = error[order]
    selected['cumulative_errors'] = np.cumsum(error[order])
    selected['threshold_distance'] = np.abs(selected.rf_score - selected.rf_threshold)
    segments = []
    for lo, hi in [(1,20), (21,50), (51,100), (101,175), (176,322)]:
        segment = selected.iloc[lo - 1:hi]
        segments.append({'rank_start': lo, 'rank_end': hi, 'records': len(segment),
                         'errors': int(segment.rf_error_evaluation_only.sum()),
                         'disagreement': int(segment.disagreement.sum()),
                         'near_threshold': int(segment.near_threshold.sum()),
                         'rf_anomalies': int(segment.rf_anomaly.sum()),
                         'distance_min': float(segment.threshold_distance.min()),
                         'distance_max': float(segment.threshold_distance.max())})
    error_ranks = selected.loc[selected.rf_error_evaluation_only, 'rank'].tolist()
    result = {'segments': segments, 'error_ranks': error_ranks,
              'rank_of_23rd_error': error_ranks[22] if len(error_ranks) > 22 else None,
              'disagreement_prefix': int(pred.disagreement.sum()),
              'combined_equals_disagreement_through_k100': all(np.array_equal(order[:k], orders(pred)['disagreement'][:k]) for k in BUDGETS)}
    return result, selected


def random_seed_stability(pred, y):
    rows = []
    for k in BUDGETS:
        for population, reference in random_reference(pred, y, k).items():
            n, e, actual = reference['population'], reference['errors_in_population'], reference['k_actual']
            means = []
            for seed in range(20261016, 20261116):
                rng = np.random.default_rng(seed)
                means.append(float(rng.hypergeometric(e, n - e, actual, size=1000).mean()))
            rows.append({'k': k, 'population_name': population, 'population': n, 'seeds': 100,
                         'draws_per_seed': 1000, 'seed_first': 20261016, 'seed_last': 20261115,
                         'theoretical_expectation': reference['expected_errors'],
                         'mean_of_seed_means': float(np.mean(means)), 'min_seed_mean': min(means),
                         'max_seed_mean': max(means), 'seed_mean_sd': float(np.std(means, ddof=1))})
    return rows


def snapshot():
    patterns = ['docs/results/*', 'docs/phase2/*', 'liushao/*.py', 'configs/*', 'artifacts/*', 'data/processed/*']
    paths = [p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()]
    paths.extend([ROOT / name for name in ['app.py', 'scripts/train.py', 'scripts/prepare_data.py', 'requirements.txt', 'requirements-lock.txt']])
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def write_report(r):
    def pct(value):
        return '不适用' if value is None else f'{value:.2%}'
    def cell(value):
        return '不适用' if value is None else f'{value:.2f}'
    def get(name,k):
        return next(row for row in r['budgets'] if row['strategy']==name and row['k']==k)
    names={'disagreement':'分歧D','near':'近阈值T','intersection':'交集D∩T','combined':'并集D∪T',
           'disagreement_only':'分歧独有D\\T','near_only':'近阈值独有T\\D','unhinted':'未提示'}
    workload_rows='\n'.join(f'| {names[name]} | {s["records"]}/{s["population"]} | {s["errors"]}/{s["records"]} | {pct(s["error_precision"])} | {s["false_positives"]}/56 | {pct(s["false_positive_coverage"])} | {pct(s["workload_fraction"])} | {cell(s["checks_per_error"])} |'
                             for name,s in r['workload'].items())
    comparisons=[]
    for name in ['margin','vote_uncertainty','disagreement','near','combined','attack']:
        rows=[get(name,k) for k in BUDGETS]
        comparisons.append('| '+LABELS[name]+' | '+' | '.join(f'{row["errors_found"]}/{row["k_actual"]} ({pct(row["error_precision_at_k"])})' for row in rows)+' |')
    for population,label in [('all','全候选随机'),('alert','告警内随机（受限总体）')]:
        values=[]
        for k in BUDGETS:
            s=r['random_references'][str(k)][population]
            values.append(f'{s["expected_errors"]:.4f}条，95%参考[{s["reference_p025"]},{s["reference_p975"]}]')
        comparisons.append('| '+label+' | '+' | '.join(values)+' |')
    detail_rows='\n'.join(f'| {row["label"]} | {row["k"]} | {row["errors_found"]}/{row["total_errors"]} | {pct(row["error_coverage_at_k"])} | {row["reviews_without_error"]} | {cell(row["checks_per_error"])} | {row["fill_records"] if row["fill_records"] is not None else "—"} |'
                          for row in r['budgets'] if row['strategy']!='attack')
    segment_rows='\n'.join(f'| {s["rank_start"]}—{s["rank_end"]} | {s["records"]} | {s["disagreement"]} | {s["near_threshold"]} | {s["errors"]} | {s["distance_min"]:.6f}—{s["distance_max"]:.6f} |' for s in r['ranking']['segments'])
    interval_rows=[]
    for k in BUDGETS:
        for key,label,baseline in [('combined_minus_margin','RF阈值距离','margin'),
                                  ('combined_minus_vote_uncertainty','RF 0.5中心度','vote_uncertainty'),
                                  ('combined_minus_disagreement','仅分歧','disagreement')]:
            s=r['bootstrap']['budgets'][str(k)][key]
            delta=get('combined',k)['errors_found']-get(baseline,k)['errors_found']
            interval_rows.append(f'| {k} | {label} | {delta:+d} | [{s["p025"]:.0f}, {s["p975"]:.0f}] | {s["n_valid"]} |')
    seed_rows='\n'.join(f'| {s["k"]} | {"全候选" if s["population_name"]=="all" else "告警内"} | {s["population"]} | {s["theoretical_expectation"]:.4f} | {s["mean_of_seed_means"]:.4f} | {s["min_seed_mean"]:.4f}—{s["max_seed_mean"]:.4f} |'
                        for s in r['random_seeds'])
    precision=r['bootstrap']['combined_precision'];coverage=r['bootstrap']['combined_coverage']
    text=f'''# 流哨参赛创新点论证：消融、合理基线与边界

日期：2026-10-16。原测试预测、模型、阈值和排序保持冻结。本文为第二阶段审计后的探索性补充分析，不是新独立测试，也没有根据本结果改进产品排序。完整聚合结果见 `study.json`；全部原始实验记录保持不变。

## 结论先行

当前证据支持“分歧与近阈值能够集中部分RF误报，且具有可追溯的复核入口”这一工程价值，**不支持“原组合排序优于合理单模型错误发现基线”**。预先列入本轮对照的RF 0.5中心度基线在K=20/50/100时分别找到19/43/54个错误，原组合为13/22/22。近阈值提示在这三个小预算下没有为原组合增加错误发现数。

因此，不能把总体排序优势、算法性能领先或真人节时作为已验证创新。适合参赛报告的贡献定位是：实现攻击优先与模型复核分离、来源和统计证据可核对、人工复核可追溯、实验脚本可复用的轻量离线流程，并公开消融结果及排序局限。该定位也不等于证明其研究首创性。

## 冻结输入和分母

- 测试候选N=15,519，RF错误E=56，均为误报；漏报FN=0，所有漏报覆盖率均不适用。
- 真正常13,532，真攻击1,987（Bot1,986、SlowHTTPTest1；Hulk0）。RF告警2,043，其中56条是误报。
- RF阈值0.12，近阈值宽度0.10；两者均由原验证阶段冻结，本轮不改变。
- 源预测SHA-256：`{r['source_sha256']}`；manifest SHA-256：`{r['manifest_sha256']}`。
- 已复核标签隔离：所有策略先用预测得分/标记/记录ID排序，随后才使用公开评测标签计算RF错误。标签不参与选择。
- 不再复训模型；100个随机种子用于随机复核模拟，2,000次时间块重采样用于条件敏感性分析，不得写成100个训练种子或2,000次独立实验。

## 为什么K=50与100都只有22个错误

原产品规则为：筛出D∪T，按`分歧=True优先 → |RF分数−0.12|升序 → record_id升序`排列。D共有175条，T有189条，交集42条。因此前175名全属于D，T独有的147条从第176名才开始。

| 原队列名次 | 条数 | 属于D | 属于T | RF错误 | 阈值距离范围 |
| --- | ---: | ---: | ---: | ---: | --- |
{segment_rows}

前35名已包含22个错误；第36—146名没有新的RF错误，第23个错误直到第147名才出现。尤其第51—100名全是RF正确判为正常、LR与之分歧的记录。它们仍被优先展示，是因为“分歧”不是“RF错误”的等价标签，而冻结规则把分歧置于其他线索之前。

所以K=50与100的22/56覆盖、以及原组合与仅分歧完全相同，都有逐条排序证据，而非显示故障或统计遗漏。本轮没有按测试标签把第147名提前。

![原队列累计错误发现与平台期](figures/fig3_original_queue_plateau.png)

图1说明：横轴为原提示队列名次，纵轴为累计RF错误数；阶梯值来自冻结记录，无平滑。浅色区分175条分歧优先区与147条近阈值独有区；灰带覆盖第35—146名平台期。图中只展示322条提示队列，不展示补足后的全候选顺序。

## 目标一致的策略比较

攻击优先检查以“尽早找到真实攻击”为目标；模型错误复核以“尽早发现RF误报/漏报”为目标。高异常分排序在后一个任务中找到0个错误，不足以证明新策略优秀，因为它本来就服务另一个目标。

主要错误复核基线为：全候选均匀无放回随机；距RF冻结阈值最近；RF分数距0.5最近（与二元分数熵降序对应的常规启发式）。告警内随机是辅助对照，分母为2,043个RF告警，只适用于其中的误报，无法发现告警外漏报。

RF分数在本实现中是树叶类别概率的平均，并非硬投票次数比例，也未经错误概率校准。0.5中心度作为通用不确定性启发式，与实际0.12决策边界距离是不同基线；两者均不使用标签排序。其计算语义见 [scikit-learn官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba)。

仅D、仅T均保留各自集合后按RF阈值距离和ID排序；仅T不使用已消融掉的分歧信息。原组合完全沿用产品规则。集合不足K时才沿用第二阶段的RF高分规则补足；本次所有预定预算下补足均为0。

表2：单元格为“错误数/复核条数（错误发现精确率）”；随机行为理论期望与同一有限总体的95%检查波动参考范围，不是实测整数或泛化置信区间。

| 策略 | K=20 | K=50 | K=100 |
| --- | --- | --- | --- |
{chr(10).join(comparisons)}

![固定预算错误发现对照](figures/fig1_error_discovery.png)

图2说明：x为固定预算K（条），y为发现的RF错误（条）；每个点对应完整表格的一项。连接线仅帮助阅读三个预定预算，不做平滑或最佳K推断。重合策略在图例中合并命名，表2保留所有消融行。随机线为期望；高异常分的任务参照只在表中保留。

## 提示集合的工作量、覆盖与收益

错误比例=集合内RF错误/集合记录；误报覆盖=发现误报/56；工作量比例=集合记录/15,519。每错误检查数是假设标签核对完全正确的离线条数成本，不能解释为真人操作时间。漏报总数为0，因此漏报覆盖率的分母为0，记为不适用，不写成100%。

| 集合 | 工作量n/N | 错误数e/n | 错误比例 | 误报覆盖数 | 误报覆盖率 | 工作量比例 | 每找到一个错误需检查 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{workload_rows}

三个互斥部分是D独有133条/22错、交集42条/22错、T独有147条/7错，其数量可相加为并集322条/51错；D、T本身有重叠，不能直接相加。并集剩余271条记录中RF判断正确，另有5个RF错误未被提示。

全看并集的离线成本为322/51=6.31条/错误；在D基础上加T需要再看147条，只额外发现7错，边际成本21条/错误。不能用“2.07%的候选覆盖91.07%的误报”直接声称97.93%的真人工作量节省，因为未计双模型推理、证据核对成本及漏掉5错的后续代价。

![互斥提示部分的工作量与错误](figures/fig2_workload_partitions.png)

图3说明：x为互斥提示部分，y为记录条数；红色为RF错误，浅色为RF正确。这里是全量确定性计数，不是重复实验均值，故不添加误差棒。

表4：完整固定预算收益分母。错误发现覆盖率的分母始终是56；RF漏报发现均为0且无可评价漏报总数。

| 策略 | K | 发现错误/56 | 错误覆盖率 | 检查但未发现错误 | 每错误检查条数 | 补足条数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{detail_rows}

## 消融与组合是否得到验证

1. **去掉近阈值**：在K≤100时原组合与仅D相同，因此这一预算范围内，近阈值对组合的增量收益为0，不能写成协同提升。
2. **去掉分歧**：仅T得到5/17/26，与全候选阈值距离基线相同，因为前三个预算均未耗尽189条T。较大预算K=100时，它还高于原组合的22。
3. **比较强的合理基线**：常规0.5中心度得到19/43/54，三个K下均超过原组合；只看100条可发现54/56个错误，多于全看322条并集得到的51/56。不能选择性删除这一基线。
4. **组合的价值边界**：候选并集覆盖率高是一个观察事实，但不等于它的排序更高效。当前更有证据的贡献是错误复核任务分离、证据与复核记录可追溯，以及透明可复现实验，而非组合算法比较优势。

## 条件区间与随机种子稳定性

按11个原测试15分钟时间块有放回抽11块，重复2,000次、种子20261017。所有策略使用相同抽样权重；重复块中的记录保留多重性，并重新按固定规则取前K。未将时间块拆成独立行抽样，也未重新训练。差值直接在每次配对样本中计算，不以两个独立区间是否重叠代替配对分析。百分位和配对重采样的基础语义参见 [SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。

只有11个块且跨块时间相关，以下称为“条件95%重采样区间”，不宣称是对新网络有可靠名义覆盖率的严格置信区间。该计算描述观察样本组成变化的敏感性，不能补足来源独立性或新测试集。2,000次重采样中零错误样本数为{r['bootstrap']['zero_error_replicates']}。

并集错误比例的条件区间为[{pct(precision['p025'])}, {pct(precision['p975'])}]；误报覆盖的条件区间为[{pct(coverage['p025'])}, {pct(coverage['p975'])}]。比值分母在每次重采样中相应更新，而非固定56。

表5：原组合减去基线的错误发现数；正值表示组合较多，负值表示较少。实际点差值仍取原冻结15,519条候选。

| K | 基线 | 原样本差值 | 配对条件95%区间 | 有效重采样 |
| --- | --- | ---: | --- | ---: |
{chr(10).join(interval_rows)}

相对阈值距离，K=20的条件区间为正，K=50和100包含0；相对0.5中心度，K=50和100的区间均为负。不同基线与不同预算不支持统一的组合优势结论。这里是已观察测试集后的多项探索性比较，不将其转化为确认性显著性声明，也不据此改写产品规则。

![配对时间块重采样差值](figures/fig4_paired_block_ranges.png)

图4说明：点为原样本“组合减基线”错误数，线段为2,000次配对块重采样的2.5%—97.5%分位范围，竖线0为无差异；不是训练随机种子置信区间。

随机基线另外固定100个种子（20261016—20261115），每种子1,000次均匀无放回抽样的等价超几何模拟。下表“种子均值范围”仅反映Monte Carlo误差，不应称为模型稳定性。

| K | 随机总体 | 总体分母 | 错误数理论期望 | 100个种子均值的平均 | 种子均值最小—最大 |
| --- | --- | ---: | ---: | ---: | --- |
{seed_rows}

## 可直接用于参赛报告的文字

> 本作品实现了面向离线网络流量的双队列研判流程，将攻击优先检查与模型错误复核分离，并通过来源记录、模型阈值、训练参考分布及追加式人工复核历史提供可核对的依据。在CSE-CIC-IDS2018两个采集日形成的固定留出测试中，RF共出现56条误报、无漏报，分歧与近阈值提示的并集包含322/15,519条记录，覆盖51/56条误报，提示错误比例为51/322。该结果说明提示能够集中部分模型错误，但不代表错误被自动修正。
>
> 目标一致的消融对照表明，现有分歧优先排序在K=20/50/100时发现13/22/22条错误，均低于RF 0.5中心度排序的19/43/54条。由于前175条提示均来自分歧集合，近阈值提示在三个预算下未产生额外收益。因而当前证据尚不足以证明组合策略的排序优势，作品贡献主要体现为轻量复核闭环、证据可追溯与可复现实验。后续需要在未用于设计或调参的独立时间/来源数据及真实用户任务上验证排序与实际操作收益。

上述文字应与表2、表5一并呈现，不应只截取91.07%覆盖率作为算法领先证据。可以写“工程改进”“可复现评测及机制分析”；不宜写“首创算法”“显著降低真实运维成本”“提升检测F1”或“已证明组合优于单模型”。

## 尚存风险与后续验证边界

- 真实攻击主要为Bot且只覆盖5个测试时间块；SlowHTTPTest仅1条、Hulk无测试支持。无漏报使漏报发现价值不可评价。
- 同源日期的训练与测试时间交错，不能据此推断新场景、时间外推或未知攻击性能；块间相关使上述区间也可能过于乐观。
- 这是审计后补充比较，存在多项比较和测试集适应风险。0.5中心度更好是本批数据中的观察，不应直接升级为对所有数据最优的新规则。
- “发现错误”以公开标签为真值并假定人工能完整核对。没有真人实验、核对耗时或误判率，因此当前收益只按离线检查条数报告。
- 原组合排序没有因本结果改变。任何后续排序迭代均应与现有版本分开，在新的独立评价数据上检验。

## 文件与复现

- 协议与占用边界：`PROTOCOL.md`；原 `docs/phase2/COORDINATION.md` 保持原样。
- 机器结果：`study.json`；完整表：`workload.tsv`、`budget_comparisons.tsv`、`ranking_segments.tsv`、`random_seed_stability.tsv`。
- 行级排序证据只保存在本地忽略目录 `artifacts/innovation/ranking_trace.csv`，不提交到Git。
- 分析命令：`.\\.venv\\Scripts\\python.exe -m scripts.innovation_study`。
- 图表：`figures/`中四图各含SVG/PDF/300dpi PNG；轴、单位、来源映射及依赖记录见 `figures_manifest.json`。
- 独立绘图环境安装：`python -m venv artifacts/innovation/figures-env`，随后用该环境安装 `requirements-figures-lock.txt`并运行 `python -m scripts.plot_innovation`。需要微软雅黑或Noto Sans CJK字体；原项目环境和依赖锁未变动。
- 所有受保护输入哈希前后保持一致：`{r['originals_unchanged']}`。本分析耗时{r['elapsed_seconds']:.2f}秒，仅为本机脚本执行快照，不是产品操作耗时。
'''
    (OUT/'INNOVATION_REPORT.md').write_text(text,encoding='utf-8')


def main():
    started = time.perf_counter()
    before = snapshot()
    audit = json.loads((ROOT / 'docs/phase2/audit.json').read_text(encoding='utf-8'))
    for name, expected in audit['original_file_hashes'].items():
        assert sha256(ROOT / name) == expected, f'Frozen original changed: {name}'
    source = ROOT / 'artifacts/test_predictions.csv'
    pred = pd.read_csv(source, float_precision='round_trip')
    manifest = pd.read_parquet(ROOT / 'data/processed/manifest.parquet')
    test_meta = manifest[manifest.split == 'test'].reset_index(drop=True)
    assert pred.record_id.equals(test_meta.record_id)
    assert pred.model_version.nunique() == 1 and pred.model_version.iloc[0] == audit['model_version']
    y = pred.ground_truth_evaluation_only.to_numpy(dtype=bool)
    assert np.array_equal(y, test_meta.label.ne('Benign'))
    assert np.array_equal(pred.rf_anomaly, pred.rf_score >= pred.rf_threshold)
    assert np.array_equal(pred.disagreement, pred.rf_anomaly != pred.lr_anomaly)
    assert np.array_equal(pred.near_threshold, np.abs(pred.rf_score - pred.rf_threshold) <= pred.near_margin)
    expected_order = review_ranked(pred).index.to_numpy()
    actual_order = orders(pred)['combined'][:len(expected_order)]
    assert np.array_equal(expected_order, actual_order)
    diagnostic, trace = ranking_diagnostic(pred, y)
    (ROOT / 'artifacts/innovation').mkdir(parents=True, exist_ok=True)
    trace.to_csv(ROOT / 'artifacts/innovation/ranking_trace.csv', index=False)
    print('Fixed predictions verified; running paired block bootstrap...', flush=True)
    uncertainty = bootstrap(pred, y, test_meta.block_id.to_numpy())
    result = {'date': '2026-10-16', 'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': sha256(source), 'manifest_sha256': sha256(ROOT / 'data/processed/manifest.parquet'),
              'audit_script_sha256': sha256(Path(__file__)), 'model_version': audit['model_version'],
              'data_version': audit['data_version'], 'population': len(pred),
              'total_errors': int((pred.rf_anomaly.to_numpy(bool) != y).sum()),
              'false_positives': int((pred.rf_anomaly.to_numpy(bool) & ~y).sum()),
              'false_negatives': int((~pred.rf_anomaly.to_numpy(bool) & y).sum()),
              'workload': workload(pred, y), 'budgets': budget_table(pred, y),
              'random_references': {str(k): random_reference(pred, y, k) for k in BUDGETS},
              'ranking': diagnostic, 'bootstrap': uncertainty, 'random_seeds': random_seed_stability(pred, y),
              'originals_unchanged': before == snapshot(), 'protected_hashes': before,
              'elapsed_seconds': time.perf_counter() - started,
              'analysis_status': 'post-audit exploratory comparison; no product ranking change, retraining, or new independent test set'}
    assert result['originals_unchanged'], 'Original or teammate files changed; no overwrite attempted'
    write_json(OUT / 'study.json', result)
    write_report(result)
    for name, rows in [('workload', [{'set': k, **v} for k,v in result['workload'].items()]),
                       ('budget_comparisons', result['budgets']), ('random_seed_stability', result['random_seeds']),
                       ('ranking_segments', diagnostic['segments'])]:
        pd.DataFrame(rows).to_csv(OUT / (name + '.tsv'), index=False, sep='\t', na_rep='NA')
    print(json.dumps({'ranking': diagnostic, 'budgets': result['budgets'], 'bootstrap_differences':
                      {k:v['combined_minus_margin'] for k,v in uncertainty['budgets'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
