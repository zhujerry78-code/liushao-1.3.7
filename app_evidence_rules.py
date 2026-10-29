"""Deterministic evidence selection and reference semantics, independent of UI."""
import math

from liushao.reference import (EFFECT, JOINT_EFFECT_GATE, DIRECTION_EPSILON,
    HIGH_RISK_SPAN_MULTIPLIER, PARAMETERS, number, continuous, reference_position)


def _resolved_position(item):
    """Resolve reference_position safely without ever defaulting unknown reference to normal."""
    field = str(item.get('field', ''))
    val = None if item.get('missing') else number(item.get('value'))
    ref = item.get('reference') or {}
    pos = item.get('reference_position')
    if isinstance(pos, dict) and 'valid' in pos and 'continuous' in pos and 'outside' in pos and 'near' in pos:
        # Re-derive severity_band / high_risk if missing on legacy fixture dicts
        if 'severity_band' not in pos or 'high_risk' not in pos:
            low, mid, high = (number(ref.get(k)) for k in ('p01', 'p50', 'p99'))
            if low is None and 'q01' in ref:
                low, mid, high = (number(ref.get(k)) for k in ('q01', 'median', 'q99'))
            if low is not None and mid is not None and high is not None:
                return reference_position(field, val, {'q01': low, 'median': mid, 'q99': high})
            inferred = dict(pos)
            dist = number(inferred.get('distance')) or 0.0
            inferred['high_risk'] = bool(inferred.get('outside') and not inferred.get('fixed') and dist > HIGH_RISK_SPAN_MULTIPLIER)
            if val is None:
                inferred['severity_band'] = '原始缺失'
            elif not inferred.get('valid'):
                inferred['severity_band'] = '训练正常参考不足（未知）'
            elif not inferred.get('continuous', True):
                inferred['severity_band'] = '离散协议/端口/标志编码'
            elif inferred.get('fixed'):
                inferred['severity_band'] = '偏离固定参考值' if inferred.get('outside') else '固定参考值'
            elif inferred.get('outside'):
                inferred['severity_band'] = '显著偏离参考（超限>1倍跨度）' if inferred['high_risk'] else '超出参考范围'
            elif inferred.get('near'):
                inferred['severity_band'] = '接近参考边界（区间内）'
            else:
                inferred['severity_band'] = '参考区间核心内'
            return inferred
        return pos
    low, mid, high = (number(ref.get(k)) for k in ('p01', 'p50', 'p99'))
    if low is None and 'q01' in ref:
        low, mid, high = (number(ref.get(k)) for k in ('q01', 'median', 'q99'))
    return reference_position(field, val, {'q01': low, 'median': mid, 'q99': high})


def summarize_interval_bands(all_fields):
    """Count how fields distribute across interval categories; never count unknown/missing reference as normal."""
    counts = {
        'high_risk': 0,
        'outside_moderate': 0,
        'outside_total': 0,
        'near_boundary': 0,
        'core_normal': 0,
        'insufficient_reference': 0,
        'missing': 0,
        'discrete_or_fixed': 0,
        'total': len(all_fields or []),
    }
    high_risk_names, outside_names, near_names, insufficient_names = [], [], [], []
    for item in (all_fields or []):
        if not isinstance(item, dict):
            continue
        field = str(item.get('field', ''))
        name = item.get('name') or field
        val = None if item.get('missing') else number(item.get('value'))
        pos = _resolved_position(item)
        if item.get('missing') or val is None:
            counts['missing'] += 1
        elif not pos.get('valid'):
            counts['insufficient_reference'] += 1
            insufficient_names.append(name)
        elif not pos.get('continuous', True):
            counts['discrete_or_fixed'] += 1
        elif pos.get('fixed'):
            if pos.get('outside'):
                counts['outside_moderate'] += 1
                counts['outside_total'] += 1
                outside_names.append(name)
            else:
                counts['discrete_or_fixed'] += 1
        elif pos.get('outside'):
            counts['outside_total'] += 1
            if pos.get('high_risk') or ((number(pos.get('distance')) or 0.) > HIGH_RISK_SPAN_MULTIPLIER):
                counts['high_risk'] += 1
                high_risk_names.append(name)
            else:
                counts['outside_moderate'] += 1
                outside_names.append(name)
        elif pos.get('near'):
            counts['near_boundary'] += 1
            near_names.append(name)
        else:
            counts['core_normal'] += 1
    counts['high_risk_names'] = high_risk_names
    counts['outside_names'] = outside_names
    counts['near_names'] = near_names
    counts['insufficient_names'] = insufficient_names
    return counts


def classify_attribution_archetype(payload):
    """Conservatively summarize score decomposition & interval state; never overclaim synergy when decomposition or reference is lacking."""
    payload = payload or {}
    scores = payload.get('scores') if isinstance(payload.get('scores'), dict) else {}
    decomp = payload.get('decomposition') if isinstance(payload.get('decomposition'), dict) else {}
    selection = payload.get('selection') if isinstance(payload.get('selection'), dict) else {}
    all_fields = payload.get('all_fields') if isinstance(payload.get('all_fields'), list) else []
    bands = summarize_interval_bands(all_fields)

    rf_score = number(scores.get('rf'))
    lr_score = number(scores.get('lr'))
    rf_th = number(scores.get('rf_threshold'))
    lr_th = number(scores.get('lr_threshold'))
    # Missing scores cannot be described as either normal or a disagreement.
    scores_known = all(value is not None and 0 <= value <= 1
                       for value in (rf_score, lr_score, rf_th, lr_th))
    rf_anomaly = rf_score >= rf_th if scores_known else None
    lr_anomaly = lr_score >= lr_th if scores_known else None

    # Gate 1: Decomposition or selection unavailable -> Never fabricate a mechanism claim
    if (selection.get('status') == 'unavailable'
            or decomp.get('status') != 'available'
            or not isinstance(decomp.get('rf'), dict)
            or not decomp['rf'].get('fields')
            or not all_fields):
        reason = selection.get('reason') or decomp.get('reason') or '缺少可用的评分分解或字段参考明细'
        return {
            'code': 'explanation_unavailable',
            'label': '评分分解或证据暂不可用（待人工核查）',
            'headline': f'评分分解未完成（{reason}），暂无法归纳评分线索。',
            'bands': bands,
            'positive_driver_count': 0,
            'negative_driver_count': 0,
            'positive_sum': 0.0,
            'negative_sum': 0.0,
            'missing_indicator_abs_contribution': 0.0,
        }

    by_field = {item.get('field'): item for item in all_fields if isinstance(item, dict)}
    rf_fields = decomp['rf'].get('fields', [])
    if (not isinstance(rf_fields, list)
            or any(not isinstance(row, dict) or not row.get('field')
                   or number(row.get('contribution')) is None for row in rf_fields)
            or len({row['field'] for row in rf_fields}) != len(rf_fields)):
        return {'code': 'explanation_unavailable', 'label': '评分分解数据不完整（待核查）',
                'headline': '字段路径贡献缺失、重复或不是有限数值，暂无法归纳评分线索。',
                'bands': bands, 'positive_driver_count': 0, 'negative_driver_count': 0,
                'positive_sum': None, 'negative_sum': None,
                'missing_indicator_abs_contribution': None}
    pos_drivers = sorted([r for r in rf_fields if (number(r.get('contribution')) or 0.) >= EFFECT],
                         key=lambda r: -(number(r.get('contribution')) or 0.))
    neg_drivers = sorted([r for r in rf_fields if (number(r.get('contribution')) or 0.) <= -EFFECT],
                         key=lambda r: (number(r.get('contribution')) or 0.))
    pos_sum = math.fsum(number(r.get('contribution')) or 0. for r in rf_fields if (number(r.get('contribution')) or 0.) > 0)
    neg_sum = math.fsum(number(r.get('contribution')) or 0. for r in rf_fields if (number(r.get('contribution')) or 0.) < 0)
    driver_sum = math.fsum(number(row['contribution']) for row in pos_drivers)

    missing_contrib = 0.0
    for r in rf_fields:
        for part in r.get('parts', []):
            if part.get('kind') == 'missing_indicator':
                missing_contrib += abs(number(part.get('contribution')) or 0.)

    top_pos_outside = []
    top_pos_in_range_continuous = []
    top_pos_uncomparable = []
    for r in pos_drivers:
        item = by_field.get(r['field'])
        if not item:
            top_pos_uncomparable.append(r['field'])
            continue
        pos = _resolved_position(item)
        item_name = item.get('name') or r['field']
        if item.get('missing') or number(item.get('value')) is None or not pos.get('valid') or not pos.get('continuous', True) or pos.get('fixed'):
            if pos.get('valid') and pos.get('fixed') and pos.get('outside'):
                top_pos_outside.append(item_name)
            else:
                top_pos_uncomparable.append(item_name)
        elif pos.get('outside'):
            top_pos_outside.append(item_name)
        else:
            top_pos_in_range_continuous.append(item_name)

    # Gate 2: Missing values affect the record
    if not scores_known:
        code = 'score_unavailable'
        label = '模型分数或分界值不完整（待核查）'
        headline = '缺少有效分数或分界值，暂不判断模型是否提示异常；已知的字段贡献可单独核对。'
    elif bands['missing'] > 0 and (missing_contrib >= EFFECT or any(c.get('missing') for c in payload.get('cards', []))):
        code = 'missing_imputation_sensitive'
        label = '存在原始缺失（先核对源数据）'
        headline = f'含 {bands["missing"]} 项原始缺失；缺失标记路径贡献的绝对值合计 {missing_contrib:.4f}。这不等于填补的总影响，请先核对原始采集记录。'
    # Gate 3: Dual-model threshold disagreement
    elif rf_anomaly != lr_anomaly:
        code = 'dual_model_disagreement'
        label = '双模型分界判断不一致（待核查）'
        headline = f'RF（{rf_score:.4f}，分界 {rf_th:.4f}）与 LR（{lr_score:.4f}，分界 {lr_th:.4f}）的提示不同。'
    # Gate 4: Primary model flags anomaly
    elif rf_anomaly:
        if len(pos_drivers) == 0:
            code = 'insufficient_driver_evidence'
            label = '达到展示条件的推高项不足（待补证核查）'
            max_c = max((number(r.get('contribution')) or 0. for r in rf_fields), default=0.0)
            headline = f'无单项特征 RF 路径正贡献达到 {EFFECT:.2f}（最大单项 {max_c:+.4f}），需结合日志补证。'
        elif top_pos_outside:
            code = 'outlier_feature_driven'
            label = '越界指标推高提示（待核对日志）'
            names_txt = '、'.join(top_pos_outside[:3])
            headline = f'推高项中，{names_txt} 超出训练正常参考（全表超出参考 {bands["outside_total"]} 项，其中显著偏离 {bands["high_risk"]} 项）。这是数值与路径贡献的对应，不说明攻击原因。'
        elif len(pos_drivers) == 1:
            code = 'single_feature_push'
            only_field = pos_drivers[0]['field']
            only_item = by_field.get(only_field, {})
            only_name = only_item.get('name') or only_field
            only_c = number(pos_drivers[0].get('contribution')) or 0.0
            if top_pos_in_range_continuous:
                label = '单字段区间内推高提示（非多维协同，待核查）'
                headline = f'仅 1 项特征（{only_name}，路径 {only_c:+.4f}）达到 {EFFECT:.2f} 的展示条件且在参考区间内。'
            else:
                label = '单字段推高提示（含参考不足或离散项，待核查）'
                headline = f'仅 1 项特征（{only_name}，路径 {only_c:+.4f}）达到 {EFFECT:.2f} 的展示条件；其参考为离散、固定或不足，需另行核查。'
        elif len(top_pos_in_range_continuous) >= 2 and not top_pos_uncomparable and driver_sum >= JOINT_EFFECT_GATE:
            code = 'multi_in_range_joint_push'
            label = '多项区间内特征共同推高提示（单指标未越界，待核查）'
            top_names = '、'.join(top_pos_in_range_continuous[:3])
            headline = f'{top_names} 等 {len(pos_drivers)} 项达到展示条件的正贡献字段均在训练正常 P01–P99 内，这些字段的路径贡献合计 {driver_sum:+.4f}。相加关系不证明特征协同或攻击成因。'
        else:
            code = 'unverified_multi_driver_push'
            label = '多项特征推高提示（含离散/参考不足或幅度有限，待核查）'
            headline = f'共 {len(pos_drivers)} 项达到展示条件，路径贡献合计 {driver_sum:+.4f}；其中 {len(top_pos_uncomparable)} 项缺少可比较的连续参考。未达到区间内多项合计的完整展示条件。'
    # Gate 5: Neither model flags anomaly
    else:
        code = 'normal_background_consistent'
        label = '模型未提示异常（仍需结合业务核查）'
        headline = f'两模型均未提示异常（RF={rf_score:.4f}，LR={lr_score:.4f}；负路径贡献合计 {neg_sum:+.4f}）。这不等于已确认正常。'

    return {
        'code': code,
        'label': label,
        'headline': headline,
        'bands': bands,
        'positive_driver_count': len(pos_drivers),
        'negative_driver_count': len(neg_drivers),
        'positive_sum': pos_sum,
        'positive_driver_sum': driver_sum,
        'negative_sum': neg_sum,
        'missing_indicator_abs_contribution': missing_contrib,
    }


def direction_difference(rf, lr):
    return rf is not None and lr is not None and abs(rf) >= EFFECT and abs(lr) >= DIRECTION_EPSILON and rf*lr < 0


def select_fields(signals, target, limit=6):
    """Reserve quality, direction and counterevidence, deduplicate before filling."""
    reasons, chosen = {}, []
    def add(item, reason):
        i = item['index']
        if i in chosen:
            if reason not in reasons[i]: reasons[i].append(reason)
        elif len(chosen) < limit:
            chosen.append(i); reasons[i] = [reason]
    magnitude = lambda s: abs(s['rf_delta'] or 0.)
    rank = lambda s: (-magnitude(s), s['index'])
    quality = [s for s in signals if s['missing'] or not s['position']['valid']]
    if quality:
        add(min(quality, key=lambda s: (not s['missing'], *rank(s))), '原始缺失' if any(s['missing'] for s in quality) else '训练正常参考不足')
    differences = sorted([s for s in signals if s['direction_difference']], key=rank)
    reverse = sorted([s for s in signals if s['significant'] and target*s['rf_delta'] > 0], key=rank)
    if differences: add(differences[0], '两模型替换方向相反')
    if reverse: add(reverse[0], '强反向线索')
    significant = [s for s in signals if s['significant']]
    if not significant:
        outside = [s for s in signals if s['position']['outside']]
        if outside:
            add(min(outside, key=lambda s: (not s['position']['fixed'], -s['position']['distance'], s['index'])), '统计偏离补充（无显著 RF 响应）')
    for s in sorted(significant, key=lambda s: (target*s['rf_delta'] > 0, -magnitude(s), -s['position']['distance'], s['index'])):
        add(s, '显著替换响应：' + ('支持当前 RF 提示' if -target*s['rf_delta'] >= EFFECT else '反向线索'))
    return chosen, reasons
