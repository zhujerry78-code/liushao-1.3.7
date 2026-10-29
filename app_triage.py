"""Optional read-only diagnostics, separate from frozen inference and review storage.

AI-assisted implementation after review of an externally supplied candidate.
RF tree dispersion describes probabilities, not calibrated uncertainty or causality.
The LR margin reuses the RF margin for exploration; it is not independently tuned.
"""
from __future__ import annotations

import hashlib
from app_types import needs_review

import numpy as np
import pandas as pd


COLUMNS = ['record_id', 'source_sha256', 'source_row', 'model_version',
           'rf_near_threshold', 'lr_near_threshold', 'dual_near_threshold',
           'diagnostic_review_hint', 'rf_tie_count', 'rf_tree_count',
           'rf_tree_probability_std']


def quality_summary(features: pd.DataFrame, rejected_rows: int) -> str:
    missing = features.isna()
    affected = features.columns[missing.any()].tolist()
    fields = '、'.join(affected) if affected else '无'
    return (f'本次可判断记录中 {int(missing.any(axis=1).sum())} 条含缺失，'
            f'共 {int(missing.sum().sum())} 个单元格；整行拒绝 {rejected_rows} 条。'
            f'受影响特征：{fields}。部分缺失使用训练期已拟合的中位数填补器，'
            '本次不会重新拟合。全部必需特征无效时整行拒绝；'
            '缺少必需列或表头不合法时整批导入失败。统计区间越界不等于输入无效。')


def diagnostics(predictions: pd.DataFrame, features: pd.DataFrame, bundle: dict,
                *, chunk_size: int = 512) -> pd.DataFrame:
    """Return a separate table without editing scores, labels, order or input frames.

    Tree statistics use Welford accumulation, O(chunk_size * feature_count) memory,
    without an N-by-tree matrix. Rows must retain the inference index and ordering.
    """
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
        raise ValueError('诊断分块必须是1至4096的整数')
    p, x = predictions, features
    required = {'record_id', 'source_sha256', 'source_row', 'model_version',
                'rf_score', 'lr_score', 'near_threshold', 'disagreement'}
    if (not p.columns.is_unique or not x.columns.is_unique
            or not required <= set(p) or not p.index.is_unique
            or not p.index.equals(x.index) or not p.record_id.is_unique
            or list(x.columns) != bundle['config']['features']):
        raise ValueError('诊断记录、索引或模型特征顺序不匹配')
    if not np.isfinite(p[['rf_score', 'lr_score']].to_numpy(dtype=float)).all():
        raise ValueError('诊断模型得分必须有限')
    if not p.model_version.eq(bundle['model_version']).all():
        raise ValueError('诊断模型版本不匹配')
    rf = getattr(bundle['rf'], 'predictor', bundle['rf'])
    trees = rf.named_steps['classifier'].estimators_
    if not len(trees) or len(trees) != bundle['config']['n_estimators']:
        raise ValueError('真实树数与模型配置不匹配')
    dispersion = np.empty(len(x), dtype=float)
    for start in range(0, len(x), chunk_size):
        z = rf.named_steps['impute'].transform(x.iloc[start:start + chunk_size])
        mean, m2 = np.zeros(len(z)), np.zeros(len(z))
        for count, tree in enumerate(trees, 1):
            values = tree.predict_proba(z)[:, 1]
            delta = values - mean
            mean += delta / count
            m2 += delta * (values - mean)
        dispersion[start:start + len(z)] = np.sqrt(np.maximum(m2 / len(trees), 0))
    d = p[['record_id', 'source_sha256', 'source_row', 'model_version']].copy()
    d['rf_near_threshold'] = p.near_threshold.astype(bool)
    d['lr_near_threshold'] = ((p.lr_score - bundle['config']['lr_threshold']).abs()
                              <= bundle['config']['near_margin'])
    d['dual_near_threshold'] = d.rf_near_threshold | d.lr_near_threshold
    d['diagnostic_review_hint'] = needs_review(p) | d.lr_near_threshold
    d['rf_tie_count'] = p.groupby('rf_score')['record_id'].transform('size').astype('int64')
    d['rf_tree_count'] = len(trees)
    d['rf_tree_probability_std'] = dispersion
    return d[COLUMNS]


def diagnostic_view(predictions: pd.DataFrame, diagnostic: pd.DataFrame,
                    view: str) -> pd.DataFrame:
    """Optional views; the original ranked/review_ranked functions are untouched."""
    if (not predictions.index.equals(diagnostic.index)
            or predictions.record_id.tolist() != diagnostic.record_id.tolist()):
        raise ValueError('旁路视图与预测身份不匹配')
    if view == 'lr_tie_break':
        return predictions.sort_values(['rf_score', 'lr_score', 'record_id'],
                                       ascending=[False, False, True], kind='stable')
    if view == 'dual_near':
        selected = predictions.loc[diagnostic.diagnostic_review_hint].copy()
        selected['_diagnostic_distance'] = np.minimum(
            (selected.rf_score - selected.rf_threshold).abs(),
            (selected.lr_score - selected.lr_threshold).abs())
        return selected.sort_values(['disagreement', '_diagnostic_distance', 'record_id'],
                                    ascending=[False, True, True], kind='stable').drop(
                                        columns='_diagnostic_distance')
    raise ValueError('未知旁路视图')


def _one_feature_frame(row: pd.Series, bundle: dict) -> pd.DataFrame:
    names = bundle['config']['features']
    if (not isinstance(row, pd.Series) or not row.index.is_unique
            or not set(names) <= set(row.index)):
        raise ValueError('特征线索需要一条具有完整且唯一字段的已清洗记录')
    try:
        frame = pd.DataFrame([row[names]], columns=names).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError('特征线索包含无法读取的数值') from error
    values = frame.to_numpy()
    if np.isinf(values).any() or np.isnan(values).all():
        raise ValueError('特征线索记录含无穷值或全部字段缺失，无法试算')
    return frame


def _one_score(predictor, frame: pd.DataFrame) -> float:
    result = np.asarray(predictor.predict_proba(frame))
    if (result.shape != (1, 2) or not np.isfinite(result).all()
            or (result < 0).any() or (result > 1).any()):
        raise ValueError('单条扰动模型输出无效，无法试算')
    return float(result[0, 1])


def feature_clues(row: pd.Series, bundle: dict) -> dict:
    """Selected-row heuristics and three independent, single-field perturbations.

    LR terms retain their sign in transformed training coordinates, including the
    missing indicators. RF importance is global; the mixed score is a heuristic.
    Nothing is fitted, persisted, or applied to the original inference result.
    """
    from liushao.model import evidence_for

    frame = _one_feature_frame(row, bundle)
    names = bundle['config']['features']
    width = len(names)
    lr = getattr(bundle['lr'], 'predictor', bundle['lr'])
    rf = getattr(bundle['rf'], 'predictor', bundle['rf'])
    imputer = lr.named_steps['impute']
    transformed = imputer.transform(frame)
    transformed = lr.named_steps['log'].transform(transformed)
    transformed = lr.named_steps['scale'].transform(transformed)[0]
    classifier = lr.named_steps['classifier']
    coefficients = np.asarray(classifier.coef_[0])
    indicators = imputer.indicator_.features_
    if len(transformed) != width + len(indicators) or coefficients.shape != transformed.shape:
        raise ValueError('LR预处理维度不匹配，无法计算单条线索')
    terms = transformed * coefficients
    value_terms = terms[:width].copy()
    missing_terms = np.zeros(width, dtype=float)
    for position, feature_index in enumerate(indicators):
        missing_terms[feature_index] += terms[width + position]
    grouped_terms = value_terms + missing_terms
    intercept = float(classifier.intercept_[0])
    logit = float(lr.decision_function(frame)[0])
    reconstructed = float(grouped_terms.sum() + intercept)
    rf_before = _one_score(bundle['rf'], frame)
    lr_before = _one_score(bundle['lr'], frame)
    # This form avoids overflow for extreme logits.
    probability = (1 / (1 + np.exp(-reconstructed)) if reconstructed >= 0
                   else np.exp(reconstructed) / (1 + np.exp(reconstructed)))
    reconstruction_error = abs(reconstructed - logit)
    probability_error = abs(float(probability) - lr_before)
    if (not np.isfinite(terms).all() or not np.isfinite([intercept, logit, reconstructed]).all()
            or reconstruction_error > 1e-10 or probability_error > 1e-10):
        raise ValueError('LR线性项与流水线输出未能重构，无法显示线索')

    importance = rf.named_steps['classifier'].feature_importances_.copy()
    rf_indicators = rf.named_steps['impute'].indicator_.features_
    if len(importance) != width + len(rf_indicators):
        raise ValueError('RF重要性与预处理维度不匹配')
    weights = importance[:width].copy()
    for position, feature_index in enumerate(rf_indicators):
        weights[feature_index] += importance[width + position]
    reference = bundle['reference']
    weighted = np.full(width, np.nan)
    for index, name in enumerate(names):
        ref = reference[name]
        value = frame[name].iloc[0]
        median = ref['median']
        if median is None or not np.isfinite(median):
            continue
        low, high = ref['q01'], ref['q99']
        span = (high - low if low is not None and high is not None and high > low
                else max(abs(median), 1.))
        deviation = 1. if pd.isna(value) else abs(float(value) - median) / max(span, 1e-6)
        weighted[index] = weights[index] * np.log1p(deviation)
    finite_weighted = np.where(np.isfinite(weighted), weighted, 0.)
    rf_scale = max(float(finite_weighted.max()), 1e-9)
    lr_scale = max(float(np.abs(grouped_terms).max()), 1e-9)
    strength = .5 * finite_weighted / rf_scale + .5 * np.abs(grouped_terms) / lr_scale
    table = evidence_for(frame.iloc[0], bundle)
    table['LR值项'] = value_terms
    table['LR缺失指示项'] = missing_terms
    table['LR对数几率项'] = grouped_terms
    table['RF全局重要性加权偏离（启发式）'] = weighted
    table['线索强度（启发式）'] = strength
    table = table.sort_values('线索强度（启发式）', ascending=False,
                              kind='stable').reset_index(drop=True)
    top3 = table.head(3).copy()
    perturbations = []
    for name in top3['字段']:
        median = reference[name]['median']
        item = {'feature': name, 'status': '无法试算：正常中位数缺失',
                'median': median, 'rf_before': rf_before, 'rf_after': np.nan,
                'rf_delta': np.nan, 'lr_before': lr_before, 'lr_after': np.nan,
                'lr_delta': np.nan}
        if median is not None and np.isfinite(median):
            changed = frame.copy()
            changed[name] = float(median)
            rf_after = _one_score(bundle['rf'], changed)
            lr_after = _one_score(bundle['lr'], changed)
            item.update(status='已试算（单项独立替换）', rf_after=rf_after,
                        rf_delta=rf_after - rf_before, lr_after=lr_after,
                        lr_delta=lr_after - lr_before)
        perturbations.append(item)
    text = ('仅核对当前选中记录。RF为全局重要性加权偏离启发式；'
            'LR为训练标准化坐标下的带符号线性项，含训练缺失指示器并另列截距。'
            '线索强度等权混合各自最大值归一化后的RF启发式与绝对LR项，'
            '因此抑制方向的LR项也可能排在前列；正常参考不足时RF项不参与。'
            '扰动对每个字段分别从原记录独立替换一次，差值=试算后减原分数，'
            '可上升或下降。联合约束未经验证，可能形成不真实组合，不能据此指导处置。')
    return {'table': table, 'top3': top3, 'perturbations': pd.DataFrame(perturbations),
            'summary': {'text': text, 'rf_score': rf_before, 'lr_score': lr_before,
                        'lr_intercept': intercept, 'lr_term_sum': float(grouped_terms.sum()),
                        'lr_logit': logit, 'lr_reconstructed_logit': reconstructed,
                        'lr_reconstruction_error': reconstruction_error,
                        'lr_probability_reconstruction_error': probability_error}}


def batch_reference(predictions: pd.DataFrame, features: pd.DataFrame, bundle: dict) -> dict:
    """Descriptive normal-reference counts and three model-only review layers.

    Approximate groups are six-significant-digit fingerprints, not learned flow
    clusters. No review state is read or written; all group sizes are batch-local.
    """
    from liushao.data import fingerprints

    p, x = predictions, features
    required = {'record_id', 'source_sha256', 'model_version', 'rf_score', 'lr_score',
                'rf_anomaly', 'lr_anomaly', 'disagreement', 'near_threshold'}
    if (not p.columns.is_unique or not x.columns.is_unique or not required <= set(p)
            or not p.index.is_unique or not p.index.equals(x.index)
            or not p.record_id.is_unique or p.record_id.isna().any()
            or p.source_sha256.isna().any()
            or list(x.columns) != bundle['config']['features']
            or not p.model_version.eq(bundle['model_version']).all()):
        raise ValueError('批次参考的记录、索引或模型特征顺序不匹配')
    values = x.to_numpy(dtype=float)
    if np.isinf(values).any() or not np.isfinite(p[['rf_score', 'lr_score']]).all().all():
        raise ValueError('批次参考含非有限模型得分或无穷特征值')
    observed = np.isfinite(values)
    outside_counts = np.zeros(len(x), dtype=np.int64)
    reference_observed = 0
    unavailable_fields = []
    for index, name in enumerate(bundle['config']['features']):
        ref = bundle['reference'][name]
        low, high = (_reference_number(ref.get(key)) for key in ('q01', 'q99'))
        if low is None or high is None or low > high:
            unavailable_fields.append(name)
            continue
        valid = observed[:, index]
        reference_observed += int(valid.sum())
        outside_counts += valid & ((values[:, index] < low) | (values[:, index] > high))
    outside_cells = int(outside_counts.sum())
    outside_rate = outside_cells / reference_observed if reference_observed else None
    lr_near = (p.lr_score - bundle['config']['lr_threshold']).abs() <= bundle['config']['near_margin']
    priority = needs_review(p) | lr_near
    positive = p.rf_anomaly & p.lr_anomaly
    layers = np.where(priority, '优先核实', np.where(positive, '双模型提示异常', '双模型未提示异常'))
    groups = p[['record_id']].copy()
    groups['model_triage'] = layers
    near = fingerprints(x, near=True)
    groups['approx_cluster_id'] = [
        'AFP1-' + hashlib.sha256(('approx-six-significant-v1|' + str(source) + '|' + value).encode()).hexdigest()
        for source, value in zip(p.source_sha256, near)]
    groups['approx_cluster_size'] = groups.groupby('approx_cluster_id')['record_id'].transform('size').astype('int64')
    triage_counts = {name: int((groups.model_triage == name).sum())
                     for name in ['优先核实', '双模型提示异常', '双模型未提示异常']}
    finite_observed = int(observed.sum())
    ratio = f'{outside_rate:.2%}' if outside_rate is not None else '无法计算（无可用参考观测值）'
    text = (f'本批可判断 {len(p)} 条；有正常参考区间的有限观测值 {reference_observed} 个，'
            f'其中区间外 {outside_cells} 个，占比 {ratio}；缺失 {int(np.isnan(values).sum())} 个单元格。'
            '这只是训练正常单变量参考的批次描述，规则未校准，不能据此确认或排除环境变化。'
            '三层仅为模型提示分组，与人工四种复核状态分别记录。'
            '近似组按来源身份和六位有效数字指纹统计归并，组大小仅限本批；'
            '不是学习聚类，不代表实际事件数或人工工作量减少。')
    interval_df = batch_interval_profile(x, bundle)
    return {'groups': groups, 'interval_profile': interval_df, 'summary': {
        'text': text, 'valid_rows': len(p), 'finite_observed_cells': finite_observed,
        'reference_observed_cells': reference_observed, 'missing_cells': int(np.isnan(values).sum()),
        'missing_rows': int(np.isnan(values).any(axis=1).sum()), 'outside_cells': outside_cells,
        'outside_rate': outside_rate, 'rows_outside_ge3': int((outside_counts >= 3).sum()),
        'reference_unavailable_cells': finite_observed - reference_observed,
        'reference_unavailable_fields': unavailable_fields, 'triage_counts': triage_counts,
        'approx_group_count': int(groups.approx_cluster_id.nunique()),
        'approx_duplicate_extra_rows': len(p) - int(groups.approx_cluster_id.nunique()),
        'rf_near_rows': int(p.near_threshold.sum()), 'lr_near_rows': int(lr_near.sum()),
        'dual_near_rows': int((p.near_threshold | lr_near).sum()),
        'disagreement_rows': int(p.disagreement.sum())}}


def _reference_number(value):
    """Accept finite numeric references without applying numpy ufuncs to strings."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def batch_interval_profile(features: pd.DataFrame, bundle: dict, field_name_fn=None) -> pd.DataFrame:
    """Describe observed values against training references, without risk grading.

    The legacy count columns remain for existing exports. Neutral display columns
    and separate unavailable/discrete/fixed counts explain those aggregates;
    they must not be added to the legacy aggregates a second time.
    """
    from app_evidence_rules import continuous, HIGH_RISK_SPAN_MULTIPLIER, PARAMETERS

    if field_name_fn is None:
        from app_ui_state import field_name as field_name_fn
    names = list(bundle['config']['features'])
    if not features.columns.is_unique or list(features.columns) != names:
        raise ValueError('整批区间统计的特征顺序与模型配置不一致')
    rows = []
    near_frac = PARAMETERS.get('near_boundary', 0.05)
    for name in names:
        ref = (bundle.get('reference') or {}).get(name, {}) or {}
        if not isinstance(ref, dict):
            ref = {}
        low, mid, high = (_reference_number(ref.get(key)) for key in ('q01', 'median', 'q99'))
        col = pd.to_numeric(features[name], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
        missing_mask = ~np.isfinite(col)
        valid_vals = col[~missing_mask]
        n_missing, n_valid = int(missing_mask.sum()), len(valid_vals)
        valid_ref = (all(v is not None for v in (low, mid, high)) and low <= mid <= high)
        if valid_ref and not np.isfinite(high - low):
            valid_ref = False
        n_large = n_moderate = n_near = n_core = 0
        n_unavailable = n_discrete = n_fixed_equal = n_fixed_different = 0
        outside_pct = attention_pct = None
        if not valid_ref:
            n_unavailable = n_valid
            reference_type, range_str = '参考不足', '参考不足'
            hint = '训练正常参考不足，未比较区间'
        elif not continuous(name):
            n_discrete = n_valid
            reference_type = '离散字段'
            range_str = f'{low:,.6g} ～ {high:,.6g}（离散编号/计数）'
            hint = '离散协议/端口/标志，按字段语义核对'
        elif low == high:
            reference_type, range_str = '固定参考', f'固定参考值 {low:,.6g}'
            n_fixed_different = int(np.count_nonzero(valid_vals != low))
            n_fixed_equal = n_valid - n_fixed_different
            # Kept in the legacy outside count, but displayed separately from a
            # continuous interval because a zero-width range has no span.
            n_moderate = n_fixed_different
            outside_pct = round(100.0 * n_fixed_different / n_valid, 2) if n_valid else None
            attention_pct = outside_pct
            hint = ('存在与固定参考不同的观测值' if n_fixed_different
                    else '有效观测值均与固定参考一致')
        else:
            reference_type, range_str = '连续区间', f'{low:,.6g} ～ {high:,.6g}'
            span = high - low
            below, above = valid_vals < low, valid_vals > high
            outside = below | above
            with np.errstate(over='ignore', invalid='ignore'):
                dist_out = np.where(below, (low - valid_vals) / span,
                                    np.where(above, (valid_vals - high) / span, 0.0))
                large = outside & (dist_out > HIGH_RISK_SPAN_MULTIPLIER)
                lower = valid_vals < mid
                side_span = np.where(lower, mid - low, high - mid)
                dist_edge = np.divide(np.where(lower, valid_vals - low, high - valid_vals),
                                      side_span, out=np.ones_like(valid_vals), where=side_span > 0)
            near = ~outside & (valid_vals != mid) & (side_span > 0) & (dist_edge <= near_frac)
            core = ~outside & ~near
            n_large, n_moderate = int(large.sum()), int((outside & ~large).sum())
            n_near, n_core = int(near.sum()), int(core.sum())
            n_outside = n_large + n_moderate
            outside_pct = round(100.0 * n_outside / n_valid, 2) if n_valid else None
            attention_pct = round(100.0 * (n_outside + n_near) / n_valid, 2) if n_valid else None
            if n_large:
                hint = '存在偏离参考超过一个区间跨度的观测值'
            elif n_outside:
                hint = '存在超出训练参考区间的观测值'
            elif n_near:
                hint = '有效观测值均在参考区间内，包含接近边界的记录'
            else:
                hint = '有效观测值均在参考区间内部'
        if not len(col):
            hint = '本批没有记录，无法比较'
        elif not n_valid:
            hint = '本批全部缺失或无效，无法比较'
        n_uncomparable = n_unavailable + n_discrete + n_fixed_equal
        rows.append({
            '数据项': field_name_fn(name), '字段': name,
            '参考类型': reference_type,
            '训练正常区间 (P01～P99)': range_str,
            '正常中间值 (P50)': mid,
            '有效观测条数': n_valid,
            '较大偏离条数': n_large,
            '一般偏离条数': n_moderate - n_fixed_different,
            '参考不足条数': n_unavailable,
            '离散字段未比较条数': n_discrete,
            '与固定参考一致条数': n_fixed_equal,
            '固定参考不同条数': n_fixed_different,
            '高风险极值条数': n_large,
            '一般超限条数': n_moderate,
            '超出区间合计': n_large + n_moderate,
            '接近边界条数': n_near,
            '区间核心内条数': n_core,
            '参考不足/未比较条数': n_uncomparable,
            '原始缺失条数': n_missing,
            '本批越界率 (%)': outside_pct,
            '越界与近边界占比 (%)': attention_pct,
            '区间取值态势提示': hint,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ['较大偏离条数', '超出区间合计', '接近边界条数', '参考不足/未比较条数', '原始缺失条数'],
            ascending=[False] * 5, kind='stable').reset_index(drop=True)
    return df
