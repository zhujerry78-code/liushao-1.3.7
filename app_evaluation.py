"""Read-only, identity-bound historical evaluation; no inference or training."""
from __future__ import annotations

import copy
from datetime import date, datetime, timezone
from functools import lru_cache
import math
from pathlib import Path
import re

from delivery.safe_files import read_json

SUMMARY_PATH = 'resources/evaluation/summary.json'
STRATEGIES = {'combined': '原组合', 'disagreement': '仅分歧', 'near': '仅近阈值',
              'margin': 'RF 阈值距离', 'vote_uncertainty': 'RF 0.5 中心度', 'attack': 'RF 高分优先'}
METHOD = ('来源文件与 15 分钟时间块分组，固定种子分配；跨集合去除完全和六位有效数字近重复。'
          '填补、缩放及正常参考仅拟合训练集，判断阈值在验证集选择并冻结。')
LIMITS = ('同来源、同采集日分组留出，不是严格时间外推；来源仍有关联，主机或会话独立性未获证明。'
          '原留出集攻击主要为 Bot，不能据此推断全部攻击或真实部署效果。')


def evaluation_time(value):
    """Readable historical time, with its recorded timezone kept explicit."""
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return date.fromisoformat(value).strftime('%Y年%m月%d日')
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            return stamp.strftime('%Y年%m月%d日 %H:%M:%S') + '（原记录未注明时区）'
        return stamp.astimezone(timezone.utc).strftime('%Y年%m月%d日 %H:%M:%S') + '（UTC）'
    except (TypeError, ValueError):
        return '时间格式待核对'


def require(condition, message):
    if not condition:
        raise ValueError('评测资料无效：' + message)


def count(value):
    return type(value) is int and value >= 0


def ratio(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def close(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-10)


def validate_metrics(metrics, population):
    require(isinstance(metrics, dict), '缺少分类指标')
    for name in ('rf', 'lr'):
        m = metrics[name]
        for key in ('tn', 'fp', 'fn', 'tp', 'normal_support', 'attack_support'):
            require(count(m[key]), '混淆矩阵或支持数错误')
        tn, fp, fn, tp = (m[key] for key in ('tn', 'fp', 'fn', 'tp'))
        require(tn + fp == m['normal_support'] and fn + tp == m['attack_support']
                and tn + fp + fn + tp == population, '分类分母不一致')
        values = {'precision': tp / (tp + fp) if tp + fp else 0,
                  'recall': tp / (tp + fn) if tp + fn else 0,
                  'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
        require(all(ratio(m[key]) and close(m[key], value) for key, value in values.items()),
                '分类指标与混淆矩阵不一致')
        require(ratio(m['threshold']), '判断阈值错误')


def validate_summary(data, model_sha=None, descriptor_sha=None, data_version=None):
    """Validate arithmetic and identity before any historical number is displayed."""
    try:
        require(isinstance(data, dict) and data['format'] == 1, '不支持的摘要格式')
        identity = data['identity']
        for key in ('model_sha256', 'descriptor_sha256', 'data_version'):
            require(isinstance(identity[key], str) and re.fullmatch('[0-9a-f]{64}', identity[key]),
                    '缺少完整内容身份')
        for key, expected in (('model_sha256', model_sha), ('descriptor_sha256', descriptor_sha),
                              ('data_version', data_version)):
            if expected is not None:
                require(identity[key] == expected, '该评测不适用于当前模型或数据版本')
        info = data['model_info']
        require(count(info['training_rows']) and info['training_rows'] > 0
                and count(info['normal_reference_rows'])
                and 0 < info['normal_reference_rows'] <= info['training_rows'], '训练/参考条数错误')
        require(isinstance(info['features'], list) and len(set(info['features'])) == len(info['features'])
                and bool(info['features']), '特征清单错误')
        require(all(ratio(info[key]) for key in ('rf_threshold', 'lr_threshold', 'near_margin')),
                '模型分界信息错误')
        held = data['held_out']
        require(count(held['records']) and held['records'] > 0, '留出集分母错误')
        require(all(count(v) for v in held['class_support'].values())
                and sum(held['class_support'].values()) == held['records'], '类别支持数错误')
        validate_metrics(held['metrics'], held['records'])
        require(held['class_support']['Benign'] == held['metrics']['rf']['normal_support'],
                '正常类别支持数错误')
        for key in ('rf', 'lr'):
            require(close(held['metrics'][key]['threshold'], info[key + '_threshold']), '模型阈值不一致')
        top = data['attack_ranking']
        require(count(top['k_actual']) and 0 < top['k_actual'] <= held['records']
                and count(top['attacks_found']) and top['attacks_found'] <= top['k_actual']
                and top['total_attacks'] == held['metrics']['rf']['attack_support']
                and ratio(top['precision_at_k']) and ratio(top['recall_at_k'])
                and close(top['precision_at_k'], top['attacks_found'] / top['k_actual'])
                and close(top['recall_at_k'], top['attacks_found'] / top['total_attacks']), '攻击排序分母错误')
        ext = data['external']
        for key in ('all', 'any_overlap', 'no_overlap'):
            group = ext[key]
            require(count(group['records']) and sum(group['class_support'].values()) == group['records'],
                    '跨日分母错误')
            validate_metrics(group['metrics'], group['records'])
        require(ext['all']['records'] == ext['any_overlap']['records'] + ext['no_overlap']['records'],
                '跨日分层不完整')
        slow = ext['slowloris']
        require(count(slow['detected']) and count(slow['support'])
                and 0 <= slow['detected'] <= slow['support'] and slow['support'] > 0, 'Slowloris 分母错误')
        rows = data['error_review']['budgets']
        require({(r['strategy'], r['k']) for r in rows} ==
                {(s, k) for s in STRATEGIES for k in (20, 50, 100)} and len(rows) == len(STRATEGIES) * 3,
                '复核对照或强基线不完整')
        errors = held['metrics']['rf']['fp'] + held['metrics']['rf']['fn']
        require(data['error_review']['total_errors'] == errors, '复核错误分母不一致')
        for row in rows:
            require(count(row['errors_found']) and 0 <= row['errors_found'] <= row['k_actual'] <= row['k']
                    and row['total_errors'] == errors and row['population'] == held['records']
                    and ratio(row['error_precision_at_k'])
                    and close(row['error_precision_at_k'], row['errors_found'] / row['k_actual']),
                    '复核预算或命中率错误')
        split = data['split']
        require(split['counts']['train'] == info['training_rows'] and split['counts']['test'] == held['records'],
                '划分记录数不一致')
        require(all(split[key] is False for key in ('chronological_holdout', 'source_disjoint_holdout',
                                                   'session_disjointness_verifiable')), '划分限制被改变')
        require(len(data['sources']) >= 6, '缺少评测来源')
        for source in data['sources']:
            require(isinstance(source['path'], str) and bool(source['path'])
                    and re.fullmatch('[0-9a-f]{64}', source['sha256']), '来源哈希错误')
        require(isinstance(held['evaluated_at'], str) and bool(held['evaluated_at'])
                and isinstance(ext['evaluated_at'], str) and bool(ext['evaluated_at']), '缺少原评测日期')
    except (KeyError, TypeError, ZeroDivisionError, AttributeError) as error:
        raise ValueError('评测资料缺少字段或字段类型不正确') from error
    return data


@lru_cache(maxsize=2)
def _parse_summary(content):
    return validate_summary(read_json(content))


def read_summary(path, model_sha=None, descriptor_sha=None, data_version=None):
    # Read bytes every time: replacement, corruption and deletion cannot hit an old cache entry.
    content = Path(path).read_bytes()
    require(len(content) <= 1024 * 1024, '摘要超过 1 MiB')
    result = copy.deepcopy(_parse_summary(content))
    for key, expected in (('model_sha256', model_sha), ('descriptor_sha256', descriptor_sha),
                          ('data_version', data_version)):
        if expected is not None:
            require(result['identity'][key] == expected, '该评测不适用于当前模型或数据版本')
    return result


def attack_recall_from_source(summary, summary_path, attack):
    """Read per-attack counts only from the hash-bound historical source."""
    import hashlib
    source = next((item for item in summary['sources'] if Path(item['path']).name == 'MODEL_COMPARISON_20261016.json'), None)
    if source is None:
        return None
    root = Path(summary_path).resolve().parents[2]
    relative = Path(source['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('评测来源路径不安全')
    content = (root / relative).read_bytes()
    require(hashlib.sha256(content).hexdigest() == source['sha256'], '类别细分来源哈希不匹配')
    data = read_json(content)
    counts = data['external']['strata']['per_attack_vs_benign'][attack]['models']['original']['metrics']['rf']
    return counts['tp'], counts['attack_support']


def render_evaluation(path, verified, bundle):
    import pandas as pd
    import streamlit as st

    st.caption('同一冻结模型的历史评测 · 不代表当前上传文件的准确率')
    st.write(METHOD)
    st.info(LIMITS)
    if verified is None or bundle is None:
        st.warning('当前模型未核验，关联指标暂不展示。请到“关于流哨”核对模型状态。')
        return
    try:
        import json
        from delivery.evaluation_resources import check_identity
        descriptor = json.loads(verified.descriptor)
        historical_sha = descriptor.get('training_descriptor_sha256', verified.descriptor_sha256)
        summary = read_summary(path, verified.model_sha256, historical_sha,
                               bundle['config']['data_version'])
        original = Path(path).parent / 'model.original.json'
        check_identity(summary, descriptor, verified.descriptor_sha256,
                       original.read_bytes() if descriptor.get('training_descriptor_sha256') else None)
        expected_info = {key: bundle[key] for key in ('training_rows', 'normal_reference_rows')}
        expected_info.update({key: bundle['config'][key] for key in ('features', 'rf_threshold', 'lr_threshold', 'near_margin')})
        require(summary['model_info'] == expected_info
                and summary['identity']['model_version'] == bundle['model_version'], '资料与当前模型描述不一致')
    except FileNotFoundError:
        st.warning('未附评测资料。请恢复本地比赛包中的配套评测摘要。')
        return
    except (OSError, ValueError) as error:
        st.warning('评测资料暂不可用；当前模型的分析与复核仍按自身检查结果运行。')
        with st.expander('资料状态'):
            st.write(str(error))
        return
    info, held, external = (summary[key] for key in ('model_info', 'held_out', 'external'))
    def table(metrics):
        return pd.DataFrame([{'模型': key.upper(), 'Precision': f"{m['precision']:.2%}",
                              'Recall': f"{m['recall']:.2%}", 'F1': f"{m['f1']:.2%}",
                              '正常支持数': m['normal_support'], '攻击支持数': m['attack_support'],
                              'TN': m['tn'], 'FP': m['fp'], 'FN': m['fn'], 'TP': m['tp']}
                             for key, m in metrics.items()])

    tabs = st.tabs(['概览', '数据与划分', '模型表现', '人工复核'])
    with tabs[0]:
        a, b, c = st.columns(3)
        a.metric('训练记录', f"{info['training_rows']:,}")
        b.metric('正常参考记录', f"{info['normal_reference_rows']:,}")
        c.metric('输入特征', str(len(info['features'])))
        st.write(f"RF 分界 {info['rf_threshold']:.2f} · LR 分界 {info['lr_threshold']:.2f} · 近阈值带宽 {info['near_margin']:.2f}")
        st.caption('RF 是主要判断模型；LR 提供对照判断。分界只读，页面不会调整阈值。')

    with tabs[1]:
        with st.expander('划分、训练正常参考与来源', expanded=True):
            st.write(summary['split'])
            st.write('P01/P50/P99 是各特征的训练正常参考分布，不是模型分数分布，也不是攻击原因。')
            st.dataframe(pd.DataFrame(bundle['reference']).T.rename(columns={'q01': 'P01', 'median': 'P50', 'q99': 'P99'}), width='stretch')
            st.write('阈值由验证集选择并冻结；当前未附阈值曲线，不从测试集反向选择阈值。')
            st.dataframe(pd.DataFrame(summary['sources']), hide_index=True, width='stretch')
            st.json(summary['identity'])
            st.caption('SHA 用于内容一致性核对，不等同于数字签名或独立审查结论。')
    with tabs[2]:
        st.subheader('原留出集：分类表现')
        st.caption(f"{held['records']:,} 条 · 原评测时间：{evaluation_time(held['evaluated_at'])} · 攻击主要为 Bot")
        attack_count = held['metrics']['rf']['attack_support']
        usable = sum(summary['split']['counts'].get(k, 0) for k in ('train', 'validation', 'test'))
        st.caption(f"测试占有效非剔除数据 {held['records']/usable:.2%}；测试攻击占比 {attack_count/held['records']:.2%}；"
                   f"Bot占测试攻击 {held['class_support'].get('Bot', 0)/attack_count if attack_count else 0:.2%}；"
                   f"训练攻击占比 {(info['training_rows']-info['normal_reference_rows'])/info['training_rows']:.2%}。")
        st.dataframe(table(held['metrics']), hide_index=True, width='stretch')
        replay_path = Path(path).parent / 'reevaluation.json'
        if replay_path.exists():
            with st.expander('本轮冻结模型复评与条件区间'):
                try:
                    import hashlib
                    raw = replay_path.read_bytes()
                    resource = read_json((Path(path).parents[2] / 'EVALUATION_RESOURCES.json').read_bytes())
                    expected = resource['files']['resources/evaluation/reevaluation.json']
                    require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'], '复评报告哈希不匹配')
                    replay = read_json(raw)
                    require(replay['status'] == 'passed' and replay['training_executed'] is False
                            and replay['identity']['model_sha256'] == verified.model_sha256
                            and replay['identity']['descriptor_sha256'] == verified.descriptor_sha256
                            and replay['identity']['data_version'] == bundle['config']['data_version'], '复评身份不匹配')
                    st.caption('本轮重新推理原测试集；不训练、不调整阈值。' + evaluation_time(replay['evaluated_at']))
                    rows = []
                    for model in ('rf', 'lr'):
                        ci = replay['intervals']['metrics'][model]['f1']
                        rows.append({'模型': model.upper(), 'F1': f"{replay['metrics'][model]['f1']:.2%}",
                                     '条件95%区间': f"{ci['lower']:.2%} ～ {ci['upper']:.2%}",
                                     '有效重采样次数': ci['valid_replicates']})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
                    st.caption(f"{replay['records']:,} 条，{replay['intervals']['blocks']} 个观察时间块，{replay['attack_blocks']} 个块含攻击，{replay['intervals']['repetitions']} 次整块重采样。")
                    st.caption('这是观察时间块的条件区间；跨块相关仍可能存在，零漏报形成的退化区间不保证未来100%召回。')
                except (OSError, ValueError, KeyError, TypeError) as error:
                    st.warning('本轮复评资料不可用：' + str(error))
        st.caption('TN/FP/FN/TP：正确正常、误报、漏报、正确异常。人工复核结论与评测真值分开保存。')
        with st.expander('类别支持数与混淆矩阵'):
            st.dataframe(pd.DataFrame([{'类别': k, '记录数': v} for k, v in held['class_support'].items()]), hide_index=True)
            for key, m in held['metrics'].items():
                st.write(key.upper() + '：行是真实类别，列是模型判断')
                st.dataframe(pd.DataFrame([[m['tn'], m['fp']], [m['fn'], m['tp']]],
                                         index=['真实正常', '真实异常'], columns=['判断正常', '判断异常']))

        st.subheader('跨采集日：2018-02-15')
        st.caption(f"原评测时间：{evaluation_time(external['evaluated_at'])}。采集日早于训练日，不能称未来时间验证。")
        st.dataframe(table(external['all']['metrics']), hide_index=True, width='stretch')
        slow = external['slowloris']
        prefix = ''
        try:
            golden = attack_recall_from_source(summary, path, 'DoS attacks-GoldenEye')
            if golden:
                detected, support = golden
                prefix = f'对 GoldenEye 检出 {detected:,}/{support:,}，召回率 {detected/support:.2%}；'
        except (OSError, ValueError, KeyError, TypeError):
            st.caption('GoldenEye细分来源未通过核验，暂不展示该类别召回率。')
        st.warning(f"RF {prefix}对 Slowloris 检出 {slow['detected']:,}/{slow['support']:,}，召回率 {slow['detected']/slow['support']:.2%}。")
        st.caption('此场景同时改变采集日、两类未见攻击及攻击占比（12.80% → 约5%）；不能把落差单独归因于指纹重叠。')
        with st.expander('重叠分层与数据质量'):
            for key, label in (('all', '全量'), ('any_overlap', '存在指纹重叠'), ('no_overlap', '无指纹重叠')):
                st.write(f"{label}：{external[key]['records']:,} 条")
                st.dataframe(table(external[key]['metrics']), hide_index=True, width='stretch')
            st.write(external['quality'])

        st.subheader('攻击优先排序：Top 100')
        top = summary['attack_ranking']
        st.write(f"前 {top['k_actual']} 条命中 {top['attacks_found']} 个攻击，P@100 为 {top['precision_at_k']:.2%}；"
                 f"仅覆盖全部 {top['total_attacks']:,} 个攻击中的 {top['recall_at_k']:.2%}。")
        st.caption('这里衡量寻找攻击；下方对照衡量寻找模型错误，两个目标和分母不同。')

    with tabs[3]:
        st.subheader('固定预算：寻找模型错误')
        review = summary['error_review']
        rows = [{'策略': STRATEGIES[r['strategy']], '预算 K': r['k'], '实际检查': r['k_actual'],
                 '找到错误': r['errors_found'], '错误命中率': f"{r['error_precision_at_k']:.2%}",
                 '全部错误': r['total_errors']} for r in review['budgets']]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.info('原组合未优于所有合理基线。此对照不代表分类 F1 提升，也不能证明真人节时。原留出集没有漏报，漏报覆盖率不适用。')
