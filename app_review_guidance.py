"""Plain-language review guidance derived only from the frozen evidence payload.

These helpers format existing evidence. They never score, select model features,
edit the payload, or turn a model response into a claim about attack causes.
"""
from __future__ import annotations

import math
import re

from app_evidence_rules import reference_position
from app_ui_state import field_name


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value):
    number = _number(value)
    if number is None:
        return '未提供'
    if number == 0:
        return '0'
    # Six significant digits, written in ordinary grouped decimal notation.
    # Keeping enough decimal places also preserves very small nonzero values.
    places = 5 - math.floor(math.log10(abs(number)))
    rounded = round(number, places)
    text = f'{rounded:,.{max(0, places)}f}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _name(card):
    name = str(card.get('name', ''))
    return name if re.search(r'[\u3400-\u9fff]', name) else field_name(card.get('field', ''))


def _position(card):
    position = card.get('reference_position')
    if isinstance(position, dict):
        return position
    reference = card.get('reference', {})
    return reference_position(card.get('field', ''), _number(card.get('value')),
                              dict(q01=reference.get('p01'), median=reference.get('p50'),
                                   q99=reference.get('p99')))


def _missing(card):
    return bool(card.get('missing')) or _number(card.get('value')) is None


def _main_cards(payload):
    # Keep the payload's selected set; only change its presentation order.
    cards = list(payload.get('cards', []))[:6]
    return sorted(cards, key=lambda card: (not _missing(card), _position(card).get('valid', False)))


def _current(card):
    if _missing(card):
        return '原始缺失'
    return f'{_fmt(card.get("value"))} {card.get("unit", "单位未提供")}'.strip()


def _status(card):
    if _missing(card):
        return '原始缺失，先核对输入'
    position = _position(card)
    if not position.get('valid'):
        return '训练正常参考不足，暂不能比较'
    if not position.get('continuous', True):
        return '编号或标记，需结合协议、端口或连接语义核对'
    if position.get('fixed'):
        return '与固定参考值不同' if position.get('outside') else '与固定参考值相同'
    if position.get('outside'):
        low = _number(card.get('reference', {}).get('p01'))
        value = _number(card.get('value'))
        return '低于参考范围' if low is not None and value < low else '高于参考范围'
    if position.get('near'):
        side = '低端' if position.get('near_side') == '下界' else '高端'
        return f'在参考范围内，但靠近{side}'
    return '在参考范围内'


def _reference(card):
    reference = card.get('reference', {})
    unit = card.get('unit', '单位未提供')
    low, middle, high = (reference.get(key) for key in ('p01', 'p50', 'p99'))
    if not _position(card).get('valid'):
        return ('参考不足；范围低端 ' + _fmt(low) + '、高端 ' + _fmt(high)
                + '、中间值 ' + _fmt(middle) + ' ' + unit)
    if _position(card).get('fixed'):
        return f'训练正常记录的参考值固定为 {_fmt(middle)} {unit}'
    if not _position(card).get('continuous', True):
        return (f'训练正常记录的编号/计数参考：{_fmt(low)}～{_fmt(high)}；'
                f'中间值 {_fmt(middle)} {unit}')
    return (f'训练正常记录参考范围：{_fmt(low)}～{_fmt(high)} {unit}；'
            f'中间值 {_fmt(middle)} {unit}')


def _check(card):
    field = str(card.get('field', ''))
    if 'Flag' in field or field == 'Protocol':
        check = '查协议、连接建立/重置/重试、服务状态和维护工单。'
    elif 'Port' in field:
        check = '查端口服务、连接数和来源，核对防火墙日志与授权任务。'
    elif ('Duration' in field or 'IAT' in field or field.startswith(('Idle ', 'Active '))):
        check = '查邻近连接、同时段连接数、超时记录和压测/维护工单。'
    elif ('Byts' in field or 'TotLen' in field or 'Blk Rate' in field):
        check = '查收发方向、目的资产、业务时段及代理、外发、备份/同步日志。'
    else:
        check = '查同端口连接数、包速率、服务响应及防火墙/服务日志、压测工单。'
    if _missing(card):
        check = '先查原始文件/采集日志中的缺失值。' + check
    if not _position(card).get('valid'):
        check = '参考不足，补查数据字典与训练参考。' + check
    if card.get('role') == '反向线索':
        check = '此项与当前模型提示相反。' + check
    if card.get('direction_difference'):
        check = '两模型反应方向不同。' + check
    return check


def _issue_names(payload, fields, limit=2):
    fields = list(dict.fromkeys(fields))
    lookup = {card.get('field'): card for card in
              list(payload.get('all_fields', [])) + list(payload.get('cards', []))}
    names = [_name(lookup.get(field, dict(field=field))) for field in fields]
    text = '、'.join(names[:limit])
    return text + (f'等{len(fields)}项' if len(fields) > limit else '')


def _flags(payload):
    selection = payload.get('selection', {})
    result = {}
    for key, predicate in (
        ('missing_fields', _missing),
        ('insufficient_references', lambda card: not _position(card).get('valid')),
        ('direction_differences', lambda card: card.get('direction_difference', False)),
        ('counterevidence', lambda card: card.get('role') == '反向线索'),
    ):
        derived = [card.get('field') for card in payload.get('all_fields', []) if predicate(card)]
        result[key] = list(dict.fromkeys(list(selection.get(key, [])) + derived))
    return result


def _anomaly(scores, model):
    if model + '_anomaly' in scores:
        return bool(scores[model + '_anomaly'])
    value, threshold = _number(scores.get(model)), _number(scores.get(model + '_threshold'))
    return None if value is None or threshold is None else value >= threshold


def review_guidance(payload):
    """Return three short review steps and Chinese alerts, without mutating data.

    The returned ``facts`` contains at most three selected observations. Alerts
    still cover all fields, including issues outside the six displayed cards.
    """
    selection, scores = payload.get('selection', {}), payload.get('scores', {})
    cards, flags = _main_cards(payload), _flags(payload)
    rf_anomaly, lr_anomaly = _anomaly(scores, 'rf'), _anomaly(scores, 'lr')
    alerts = []
    if rf_anomaly is not None and lr_anomaly is not None and rf_anomaly != lr_anomaly:
        alerts.append('RF 与 LR 提示不同')
    for key, label in (('missing_fields', '原始缺失'), ('insufficient_references', '参考不足'),
                       ('direction_differences', '模型反应方向不同'),
                       ('counterevidence', '与当前提示相反')):
        if flags[key]:
            alerts.append(f'{label} {len(flags[key])} 项：{_issue_names(payload, flags[key])}')
    if payload.get('stability', {}).get('joint_direction_conflict'):
        alerts.append('同时对照主要支持项：两模型分数变化方向不同')
    if selection.get('status') == 'unavailable':
        return dict(facts=[], observed='当前核查说明暂不可用。',
                    reason='解释计算未完成。',
                    next='核对输入与失败原因，再查日志和授权任务。',
                    alerts=alerts, unavailable=selection.get('reason', '原因未提供'))
    facts = [f'{_name(card)}：{_current(card)}，{_status(card)}' for card in cards[:3]]
    observed = '；'.join(facts) + '。' if facts else '本次没有筛出合格的主要核查项。'
    support = [card for card in cards if card.get('role') == '支持当前模型提示']
    if support and rf_anomaly is not None:
        movement = '下降' if rf_anomaly else '上升'
        prompt = '提示异常' if rf_anomaly else '未提示异常'
        reason = (f'主要模型（RF）{prompt}；逐项用训练正常记录的常见值作对照，'
                  f'部分项使分数{movement}，优先核查。')
    elif rf_anomaly:
        reason = '主要模型（RF）提示异常，尚缺主要支持项，需补日志。'
    else:
        reason = '无明显主要支持项；主要模型（RF）未提示异常或信息不足。'
    if flags['missing_fields']:
        start = '先核对原始缺失值，再'
    elif flags['insufficient_references']:
        start = '先补足训练正常参考，再'
    else:
        start = '先'
    next_step = start + '核对原始记录、连接数、服务状态与授权工单。'
    return dict(facts=facts, observed=observed, reason=reason, next=next_step,
                alerts=alerts, unavailable=None)


def review_check_items(payload):
    """Return ready-to-display Chinese rows for the selected data and follow-up.

    At most six rows are selected feature cards. Two further rows distinguish
    actual imported-neighbor counts from business facts absent from the file.
    Counts are never relabelled as concurrent connections or attack events.
    """
    if payload.get('selection', {}).get('status') == 'unavailable':
        return []
    rows = [dict(数据项=_name(card), 当前值=_current(card), 训练正常参考=_reference(card),
                 当前状态=_status(card), 具体核查=_check(card)) for card in _main_cards(payload)]
    neighbor = payload.get('neighbors', {})
    counts = []
    for key, label, unit in (('same_source', '同源记录', '条'),
                             ('same_destination', '同目的记录', '条'),
                             ('source_fanout', '同源涉及的目的地址', '个'),
                             ('repeat_five_tuple', '同五元组重复记录', '条')):
        if _number(neighbor.get(key)) is not None:
            counts.append(f'{label} {_fmt(neighbor[key])} {unit}')
    window = neighbor.get('time_window') or {}
    if _number(window.get('all')) is not None:
        counts.append(f'前后各 {_fmt(neighbor.get("window_seconds", 60))} 秒内其他记录 {_fmt(window["all"])} 条')
        for key, label in (('same_source', '其中同源'), ('same_destination', '其中同目的')):
            if _number(window.get(key)) is not None:
                counts.append(f'{label} {_fmt(window[key])} 条')
    rows.append(dict(数据项='本批邻近记录', 当前值='；'.join(counts) if counts else '文件未提供可用地址/时间信息，需查原始流量或日志',
                     训练正常参考='本次导入记录的计数',
                     当前状态='仅本批统计，不含当前记录' if counts else '缺少核查上下文',
                     具体核查='核对地址、时间与相邻记录是否属于同一业务。'))
    rows.append(dict(数据项='同端口连接数、服务状态和授权任务', 当前值='文件未提供，需查日志/工单',
                     训练正常参考='本地业务和授权记录', 当前状态='待补充外部证据',
                     具体核查='按地址和时间查连接数、服务响应、超时/拒绝/重置，再核对压测、维护、备份等授权任务。'))
    return rows


def _severity_badge(card):
    """Return a concise severity badge combining reference status and high-risk span multiple."""
    if _missing(card):
        return '原始缺失（需核对来源）'
    pos = _position(card)
    if not pos.get('valid'):
        return '训练正常参考不足（未知）'
    if not pos.get('continuous', True):
        return '离散编码（核对语义）'
    if pos.get('fixed'):
        return '偏离固定参考值' if pos.get('outside') else '与固定参考值一致'
    if pos.get('outside'):
        dist = _number(pos.get('distance')) or 0.0
        side = '低于P01' if (_number(card.get('value')) or 0.) < (_number(card.get('reference', {}).get('p01')) or 0.) else '高于P99'
        if pos.get('high_risk') or dist > 1.0:
            return f'显著偏离参考（{side} · 超限 {dist:.1f} 倍跨度）'
        return f'超出参考范围（{side} · 超限 {dist:.2f} 倍跨度）'
    if pos.get('near'):
        frac = (_number(pos.get('boundary_fraction')) or 0.0) * 100.0
        side = '低端P01' if pos.get('near_side') == '下界' else '高端P99'
        return f'接近边界（距{side}仅 {frac:.1f}% 跨度）'
    return '参考区间核心内'


def review_check_contributions(payload):
    """Return 4th-column score contribution & interval severity metadata aligned with review_check_items."""
    if payload.get('selection', {}).get('status') == 'unavailable':
        return []
    decomp = payload.get('decomposition', {})
    rf_map = {}
    if isinstance(decomp, dict) and decomp.get('status') == 'available':
        rf_map = {r['field']: _number(r.get('contribution')) for r in decomp.get('rf', {}).get('fields', [])}
    items = []
    for card in _main_cards(payload):
        field = card.get('field', '')
        path_c = rf_map.get(field)
        rf_delta = _number(card.get('rf', {}).get('delta'))
        parts = []
        if path_c is not None:
            direction = '推高' if path_c > 1e-6 else '压低' if path_c < -1e-6 else '持平'
            parts.append(f'RF路径{direction} {path_c:+.4f}')
        if rf_delta is not None:
            parts.append(f'常见值对照后 {rf_delta:+.4f}')
            if _missing(card):
                parts.append('替换试算包含缺失状态变化')
        elif _missing(card):
            parts.append('缺失项（由填补器与指示器参与）')
        effect_text = '；'.join(parts) if parts else '对照影响未单列'
        items.append({
            'field': field,
            '数据项': _name(card),
            '区间分级': _severity_badge(card),
            '对异常分的影响': effect_text,
            'rf_path_contribution': path_c,
            'rf_replace_delta': rf_delta,
            'effect_note': '含缺失状态变化' if _missing(card) else '',
        })
    items.append({
        'field': '__neighbors__',
        '数据项': '本批邻近记录',
        '区间分级': '批内上下文统计',
        '对异常分的影响': '同源/同目的聚集度',
        'rf_path_contribution': None,
        'rf_replace_delta': None,
    })
    items.append({
        'field': '__external__',
        '数据项': '同端口连接数、服务状态和授权任务',
        '区间分级': '外部日志补证项',
        '对异常分的影响': '复核结论的外部依据',
        'rf_path_contribution': None,
        'rf_replace_delta': None,
    })
    return items


def unified_interval_contribution_rows(payload, *, top_only=False, limit=12):
    """Build the unified table combining training normal interval, 5-tier status, batch percentile, RF path contribution, and replacement delta."""
    all_fields = payload.get('all_fields', [])
    decomp = payload.get('decomposition', {})
    rf_map = {}
    lr_map = {}
    if isinstance(decomp, dict) and decomp.get('status') == 'available':
        rf_map = {r['field']: _number(r.get('contribution')) for r in decomp.get('rf', {}).get('fields', [])}
        lr_map = {r['field']: _number(r.get('term')) for r in decomp.get('lr', {}).get('fields', [])}

    visible_set = set()
    if top_only:
        if isinstance(decomp, dict) and decomp.get('status') == 'available':
            visible_set.update(decomp.get('default_fields', decomp.get('rf', {}).get('top_fields', []))[:limit])
        for c in payload.get('cards', []):
            visible_set.add(c.get('field'))

    rows = []
    for item in all_fields:
        field = item.get('field', '')
        if top_only and visible_set and field not in visible_set:
            continue
        ref = item.get('reference', {})
        val = None if item.get('missing') else _number(item.get('value'))
        pos = reference_position(field, val, dict(q01=ref.get('p01'), median=ref.get('p50'), q99=ref.get('p99')))
        unit = item.get('unit', '')
        val_str = '原始缺失' if item.get('missing') or val is None else f'{_fmt(val)} {unit}'.strip()
        p01, p50, p99 = ref.get('p01'), ref.get('p50'), ref.get('p99')
        if pos.get('valid'):
            ref_str = f'{_fmt(p01)} ～ {_fmt(p99)}（中间值 {_fmt(p50)}）'
        else:
            ref_str = f'参考不足（中间值 {_fmt(p50)}）'
        pct = _number(item.get('batch_percentile'))
        pct_str = '不可用' if pct is None else f'{pct:.1f}%'
        rf_c = rf_map.get(field)
        rf_d = _number(item.get('rf', {}).get('delta'))
        lr_t = lr_map.get(field)

        if rf_c is not None:
            if rf_c >= 0.01:
                role_txt = '↑ 推高 RF 分数（≥0.01）'
            elif rf_c > 1e-5:
                role_txt = '↗ 推高 RF 分数（<0.01）'
            elif rf_c <= -0.01:
                role_txt = '↓ 降低 RF 分数（≥0.01）'
            elif rf_c < -1e-5:
                role_txt = '↘ 降低 RF 分数（<0.01）'
            else:
                role_txt = '— 路径影响接近 0'
        elif rf_d is not None:
            role_txt = ('含缺失状态变化的替换试算' if item.get('missing') else
                        '↑ 原值推高异常分' if rf_d <= -0.01 else '↓ 原值压低异常分' if rf_d >= 0.01 else '— 替换变化很小')
        else:
            role_txt = item.get('role', '未单列')
        if item.get('missing') and rf_c is not None:
            role_txt += '；替换含缺失状态变化'

        rows.append({
            '_sort_rank': pos.get('severity_rank', 3),
            '_abs_contrib': abs(rf_c or 0.0) + abs(rf_d or 0.0),
            '数据项': _name(item),
            '当前原值': val_str,
            '训练正常区间 (P01～P99)': ref_str,
            '五级区间状态': _severity_badge(item),
            '本批百分位': pct_str,
            'RF路径贡献 (概率点)': round(rf_c, 6) if rf_c is not None else None,
            '常见值对照变化 (ΔRF)': round(rf_d, 6) if rf_d is not None else None,
            'LR线性项 (对数几率)': round(lr_t, 6) if lr_t is not None else None,
            '对异常指数的作用说明': role_txt,
            '字段': field,
        })

    rows.sort(key=lambda r: (-abs(r['RF路径贡献 (概率点)'] or 0.0), r['_sort_rank'], -r['_abs_contrib']))
    if top_only:
        rows = rows[:limit]
    for r in rows:
        r.pop('_sort_rank', None)
        r.pop('_abs_contrib', None)
    return rows


def build_audit_note_draft(payload):
    """Build a structured, traceable plain-Chinese review note draft from current evidence without auto-saving."""
    try:
        from app_evidence_rules import classify_attribution_archetype
        arch = classify_attribution_archetype(payload)
    except ImportError:
        arch = {'label': '待结合日志核查', 'bands': {'high_risk': 0, 'outside_total': 0, 'near_boundary': 0, 'insufficient_reference': 0, 'missing': 0}}
    scores = payload.get('scores', {})
    rf_s = _number(scores.get('rf'))
    lr_s = _number(scores.get('lr'))
    rf_th = _number(scores.get('rf_threshold'))
    lr_th = _number(scores.get('lr_threshold'))
    bands = arch['bands']

    # The two helpers have the same field order. Chinese display names need not
    # be unique, so they cannot serve as keys when joining contribution values.
    contributions = review_check_contributions(payload)
    check_rows = review_check_items(payload)
    feature_rows = check_rows[:-2] if len(check_rows) >= 2 else check_rows

    highlights = []
    for r, c_info in zip(feature_rows[:3], contributions):
        name = r['数据项']
        c_txt = c_info.get('对异常分的影响', '')
        highlights.append(f"{len(highlights) + 1}. {name}：{r['当前值']}；{r['当前状态']}。"
                          + (f"\n   {c_txt}" if c_txt else ""))

    score_part = (f"RF {rf_s:.6g}（分界 {rf_th:.6g}）；LR {lr_s:.6g}（分界 {lr_th:.6g}）"
                  if None not in (rf_s, lr_s, rf_th, lr_th) else "模型评分或分界值不完整")
    band_part = (f"超出参考 {bands.get('outside_total', 0)} 项，其中显著偏离参考 {bands.get('high_risk', 0)} 项\n"
                 f"靠近边界 {bands.get('near_boundary', 0)} 项 · 参考不足 {bands.get('insufficient_reference', 0)} 项 · 原始缺失 {bands.get('missing', 0)} 项")
    hl_part = "\n".join(highlights) if highlights else "未获得可展示的核查明细"
    guide = review_guidance(payload)
    # Record ownership lives in the editor callback, not in this display text.
    # Keep each section separately readable and leave the human verdict empty.
    return (f"【评分线索】\n{arch['label']}\n{score_part}\n\n"
            f"【区间统计】\n{band_part}\n\n"
            f"【关键数据】\n{hl_part}\n\n"
            f"【后续核查】\n{guide.get('next', '结合日志与资产工单核对')}\n\n"
            f"【复核结论】\n")
