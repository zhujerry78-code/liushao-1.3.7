"""Compact review reasoning and a separate, complete evidence detail view."""
import math
from html import escape

import pandas as pd
import streamlit as st
from app_ui_state import literal_markdown, field_name
from app_review_guidance import review_guidance, review_check_items

try:
    from app_evidence_rules import classify_attribution_archetype, summarize_interval_bands
    _INTERVAL_RULES_AVAILABLE = True
except ImportError:
    _INTERVAL_RULES_AVAILABLE = False
    def summarize_interval_bands(all_fields):
        return {'high_risk': 0, 'outside_moderate': 0, 'outside_total': 0, 'near_boundary': 0,
                'core_normal': 0, 'insufficient_reference': 0, 'missing': 0, 'discrete_or_fixed': 0,
                'total': len(all_fields or [])}

    def classify_attribution_archetype(payload):
        return {'code': 'explanation_unavailable', 'label': '规则模块版本未对齐（待核查）',
                'headline': '请重新解压完整运行包，再查看评分线索。',
                'bands': summarize_interval_bands((payload or {}).get('all_fields', []))}

try:
    from app_review_guidance import (
        review_check_contributions,
        unified_interval_contribution_rows,
        build_audit_note_draft,
    )
    _CONTRIBUTION_GUIDANCE_AVAILABLE = True
except ImportError:
    _CONTRIBUTION_GUIDANCE_AVAILABLE = False
    def review_check_contributions(payload):
        return []

    def unified_interval_contribution_rows(payload, *, top_only=False, limit=12):
        return []

    def build_audit_note_draft(payload):
        return '【客观证据核对】请结合日志核查后填写结论：'


def _fmt(value):
    return '不可用' if value is None else f'{value:,.6g}'


def _short_names(names, limit=2):
    names = list(dict.fromkeys(names))
    return '、'.join(names[:limit]) + (f'等{len(names)}项' if len(names) > limit else '')


def _next_checks(cards, missing=False):
    """Merge the existing field-specific checks without repeating whole actions."""
    actions = list(dict.fromkeys(c.get('action', {}).get('next', '') for c in cards))
    checks, logs = [], []
    for action in actions:
        if '协议、连接建立/重置日志' in action:
            checks.append('协议与连接建立/重置记录'); logs.extend(['连接/服务日志', '维护工单'])
        elif '服务端超时和连接数日志' in action:
            checks.append('邻近连接与超时记录'); logs.extend(['服务日志', '压测/维护工单'])
        elif '代理/DLP/备份日志' in action:
            checks.append('传输方向、目的资产与业务时段'); logs.extend(['代理/DLP/备份日志'])
        elif '同目的端口的短连接与包速率' in action:
            checks.append('同端口连接与包速率'); logs.extend(['防火墙/服务日志', '压测工单'])
    checks = list(dict.fromkeys(checks)); logs = list(dict.fromkeys(logs))
    if '防火墙/服务日志' in logs and '服务日志' in logs:
        logs.remove('服务日志')
    if '压测/维护工单' in logs:
        logs = [item for item in logs if item not in ('压测工单', '维护工单')]
    start = '先核对缺失原值；' if missing else ''
    if checks:
        return start + '核对' + '、'.join(checks) + '，查' + '、'.join(logs) + '；对照业务与授权记录后判断，证据不足选“待补证”。'
    return start + '对齐采集时间与资产，查服务/防火墙/代理日志及授权任务；证据不足选“待补证”。'


def compact_evidence_summary(payload):
    """Build presentation-only prose; never change the persisted evidence payload."""
    selection = payload.get('selection', {})
    scores = payload.get('scores', {})
    rf_anomaly = scores.get('rf_anomaly', scores.get('rf', 0) >= scores.get('rf_threshold', 1))
    lr_anomaly = scores.get('lr_anomaly', scores.get('lr', 0) >= scores.get('lr_threshold', 1))
    cards = payload.get('cards', [])
    facts = []
    for card in cards[:3]:
        value = '原始缺失' if card.get('missing') else _fmt(card.get('value')) + ' ' + card.get('unit', '')
        status = card.get('reference_status', '')
        facts.append(card['name'] + ' ' + value + (('，' + status) if status and status != '原始缺失' else ''))
    alerts = []
    for key, label in (('missing_fields', '原始缺失'), ('insufficient_references', '参考不足'),
                       ('direction_differences', '替换方向不同'), ('counterevidence', '反向线索')):
        names = selection.get(key, [])
        if names:
            alerts.append(f'{label}{len(names)}项（{_short_names(names)}）')
    if rf_anomaly != lr_anomaly:
        alerts.insert(0, 'RF 与 LR 的阈值判断不同')
    if selection.get('status') == 'unavailable':
        return dict(facts=[], observed='当前证据暂不可用。',
                    reason='解释计算未完成。',
                    next='核对输入或在详情查看失败原因；查外部日志后再判断，证据不足选“待补证”。',
                    alerts=alerts, unavailable=selection.get('reason', '原因未知'))
    support = [c for c in cards if c.get('role') == '支持当前模型提示']
    if support:
        if rf_anomaly:
            reason = f'主要证据中{len(support)}项分别改回训练正常中位值后，RF 分数下降，支持当前提示；仍需日志核对原因。'
        else:
            reason = f'主要证据中{len(support)}项分别改为参考中位值后，RF 分数上升，说明当前输入使分数更低。'
    else:
        reason = ('当前高分尚缺可核查的主要支持项，请先补证。' if rf_anomaly
                  else '尚未找到明显的主要支持项；未提示异常仍需结合业务核对。')
    if selection.get('counterevidence'):
        reason += '反向线索也需一并核对。'
    observed = '；'.join(facts) + '。' if facts else '未筛出明显问题项。'
    return dict(facts=facts, observed=observed, reason=reason,
                next=_next_checks(cards, bool(selection.get('missing_fields'))), alerts=alerts,
                unavailable=None)


def render_record_evidence(payload):
    summary = review_guidance(payload)
    st.markdown('**建议核查**  \n' + literal_markdown(summary['next']))
    if summary['alerts']:
        st.caption('；'.join(summary['alerts']))
    if summary['unavailable']:
        st.caption('失败原因：' + summary['unavailable'])


def _signed(value):
    return '不可用' if value is None else f'{value:+.4f}'


def _brief_archetype(arch):
    labels = {
        'dual_model_disagreement': '两模型提示不同',
        'missing_imputation_sensitive': '先核对缺失数据',
        'outlier_feature_driven': '超出参考的项目推高评分',
        'single_feature_push': '单项推高评分',
        'multi_in_range_joint_push': '范围内多项推高评分',
        'unverified_multi_driver_push': '多项推高评分',
        'insufficient_driver_evidence': '需补充核查依据',
        'normal_background_consistent': '两模型均未提示异常',
        'score_unavailable': '评分不完整',
        'explanation_unavailable': '评分依据不可用',
    }
    return labels.get(arch.get('code'), arch.get('label', '评分线索').split('（', 1)[0])


def _score_text(value):
    try:
        number = float(value)
        return f'{number:.4f}' if not isinstance(value, bool) and math.isfinite(number) else '不可用'
    except (TypeError, ValueError, OverflowError):
        return '不可用'


def _review_styles():
    return """<style>
.st-key-record_checklist [data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"]{height:auto;flex:none;gap:.75rem;}
.st-key-record_checklist [data-testid="stExpanderDetails"] [data-testid="stElementContainer"]{flex-shrink:0;}
.st-key-record_checklist h4{margin:0!important;padding:0!important;font-size:1.08rem;}
.review-check-table{width:100%;margin:0!important;border-collapse:collapse;font-size:.9rem;line-height:1.55;table-layout:fixed;}
.review-check-table th,.review-check-table td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ee;overflow-wrap:anywhere;}
.review-check-table th{background:#edf3f8;font-weight:600;color:#294760;}
.review-check-table th:nth-child(1){width:26%}.review-check-table th:nth-child(2){width:47%}.review-check-table th:nth-child(3){width:27%}
.review-check-table .check-state{display:block;color:#516e83;font-size:.8rem;margin-top:2px;}
.review-check-table .score-line{display:block;white-space:normal;font-variant-numeric:tabular-nums;}
.review-note-preview{padding:1rem 1.15rem;border:1px solid #dce4ee;border-radius:8px;background:#fff;line-height:1.65;overflow-wrap:anywhere;font-size:.9rem;}
.review-note-preview section+section{margin-top:1rem;}
.review-note-preview strong{display:block;margin-bottom:.3rem;}
.review-check-actions{margin:0;padding-left:1.3rem;line-height:1.65;font-size:.9rem;overflow-wrap:anywhere;}
.review-check-actions li+li{margin-top:.65rem;}
.review-check-actions strong{display:block;margin-bottom:.15rem;}
@media(max-width:700px){.review-check-table thead{display:none}.review-check-table,.review-check-table tbody,.review-check-table tr,.review-check-table td{display:block;width:100%;box-sizing:border-box;}
.review-check-table tr{border:1px solid #dce4ee;border-radius:8px;margin-bottom:8px;padding:5px 8px;}
.review-check-table td{border:0;padding:3px 0}.review-check-table td:before{content:attr(data-label) "：";color:#516e83;font-size:.82rem;}
.review-check-table td:first-child:before{content:none}}
</style>"""


def _check_table(rows, contrib_lookup=None, *, show_checks=False):
    """Keep observations, references and the two score operations distinct."""
    headings = ('数据项', '当前值 / 训练参考', 'RF 评分影响')
    body = []
    contrib_lookup = contrib_lookup or {}
    for row in rows:
        reference = row['训练正常参考'].replace('训练正常记录参考范围：', '').replace('中间值（排序后居中的值）：', '中间值 ')
        c_info = row.get('_contribution', contrib_lookup.get(row['数据项'], {}))
        band_text = c_info.get('区间分级', '')
        state_display = band_text or row['当前状态']
        values = (
            '<strong>' + escape(str(row['数据项'])) + '</strong><span class="check-state" title="'
            + escape(state_display, quote=True) + '">' + escape(state_display.split('（', 1)[0]) + '</span>',
            '<strong>' + escape(str(row['当前值'])) + '</strong><span class="check-state">参考：' + escape(reference) + '</span>',
            '<span class="score-line">路径贡献 <strong>' + _signed(c_info.get('rf_path_contribution')) + '</strong></span>'
            + '<span class="score-line">替换后变化 <strong>' + _signed(c_info.get('rf_replace_delta')) + '</strong></span>'
            + ('<span class="check-state">含缺失状态变化</span>' if c_info.get('effect_note') else ''),
        )
        cells = ['<td data-label="' + label + '">' + value + '</td>' for label, value in zip(headings, values)]
        body.append('<tr>' + ''.join(cells) + '</tr>')
    st.markdown(_review_styles() + '<table class="review-check-table"><thead><tr>'
        + ''.join('<th>' + label + '</th>' for label in headings)
        + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>', unsafe_allow_html=True)
    if show_checks:
        sentences = list(dict.fromkeys(part for row in rows for part in row['具体核查'].split('。') if part))
        st.caption('具体核查：' + '。'.join(sentences) + '。')


def _render_draft_preview(draft_text):
    sections = []
    for paragraph in draft_text.strip().split('\n\n'):
        title, _, body = paragraph.partition('\n')
        sections.append('<section><strong>' + escape(title.strip('【】')) + '</strong>'
                        + escape(body).replace('\n', '<br>') + '</section>')
    st.markdown('<div class="review-note-preview">' + ''.join(sections) + '</div>', unsafe_allow_html=True)


def _render_check_actions(primary, supplement):
    """Group identical follow-up actions once, keeping their affected fields."""
    grouped = {}
    for row in primary:
        for sentence in row['具体核查'].split('。'):
            if sentence:
                grouped.setdefault(sentence, []).append(row['数据项'])
    items = []
    for action, names in grouped.items():
        fields = '、'.join(dict.fromkeys(names))
        title = '共同检查' if len(set(names)) == len(primary) and len(primary) > 1 else fields
        items.append('<li><strong title="' + escape(fields, quote=True) + '">' + escape(title)
                     + '</strong>' + escape(action) + '。</li>')
    for row in supplement:
        value = row.get('当前值', '')
        items.append('<li><strong>' + escape(row['数据项']) + '</strong>'
                     + (escape(value) + '<br>' if value else '') + escape(row['具体核查']) + '</li>')
    st.markdown('<ul class="review-check-actions">' + ''.join(items) + '</ul>', unsafe_allow_html=True)


def _merged_note(existing, draft_text):
    """Append once; never discard an existing human note or truncate it."""
    if draft_text in existing:
        return existing
    merged = existing + ('\n\n' if existing else '') + draft_text
    return merged if len(merged) <= 4000 else None


def _apply_draft_note(draft_text, input_key=None, record_id=None):
    """Append to the current unlocked editor without saving or setting a verdict."""
    ui = st.session_state.get('ui')
    editor = getattr(ui, 'editor', None)
    if editor is None or getattr(ui, 'exports', None) is not None or getattr(ui, 'pending', None) is not None:
        return
    if (input_key is not None and editor.input_key != input_key) or (record_id is not None and editor.record_id != record_id):
        return
    note = _merged_note(st.session_state.get('_note', editor.note), draft_text)
    if note is not None:
        st.session_state['_note'] = editor.note = note


def _render_band_summary(bands):
    """Non-overlapping categories; distance is a reference comparison, not risk."""
    categories = [('high_risk', '显著偏离参考'), ('outside_moderate', '一般超出参考'),
                  ('near_boundary', '范围内靠近边界'), ('core_normal', '范围内其他'),
                  ('discrete_or_fixed', '编号 / 固定值一致'), ('insufficient_reference', '参考不足'), ('missing', '原始缺失')]
    chips = ''.join('<span class="interval-chip">' + label + ' <strong>' + str(bands.get(key, 0)) + '</strong></span>'
                    for key, label in categories if bands.get(key, 0))
    st.markdown('<style>.interval-summary{display:flex;flex-wrap:wrap;gap:6px;margin:5px 0 8px}'
                '.interval-chip{padding:4px 9px;border-radius:6px;background:#f0f4f8;color:#36516a;font-size:.82rem;}'
                '.interval-chip strong{font-variant-numeric:tabular-nums}</style>'
                '<div class="interval-summary">' + chips + '</div>', unsafe_allow_html=True)
    st.caption(f'共 {bands.get("total", 0)} 项')


def render_review_checklist(payload):
    """Put concrete checks, interval summary, and contribution impact before the judgment form."""
    st.markdown(_review_styles(), unsafe_allow_html=True)
    with st.container(border=True, key='record_checklist'):
        st.markdown('#### 核查重点')
        if not (_INTERVAL_RULES_AVAILABLE and _CONTRIBUTION_GUIDANCE_AVAILABLE):
            st.warning('文件版本不一致，请重新解压运行包。')
            return
        rows = review_check_items(payload)
        if not rows:
            st.info('暂无核查明细，请检查输入。')
            return
        arch = classify_attribution_archetype(payload)
        bands = arch['bands']
        counts = [f"{bands['total']}项", f"超出参考 {bands['outside_total']}"]
        counts.extend(f'{label} {bands[key]}' for key, label in
                      [('near_boundary', '近边界'), ('missing', '缺失'), ('insufficient_reference', '参考不足')]
                      if bands.get(key, 0))
        st.caption(_brief_archetype(arch) + ' · ' + ' · '.join(counts))
        contributions = review_check_contributions(payload)
        rows = [dict(row, _contribution=contributions[index] if index < len(contributions) else {})
                for index, row in enumerate(rows)]
        primary, supplement = rows[:-2], rows[-2:]
        if primary:
            _check_table(primary[:3])
        else:
            st.info('暂无重点项目，可查看核查方法。')
        if 'ui' in st.session_state and getattr(st.session_state['ui'], 'editor', None) is not None:
            editor = st.session_state['ui'].editor
            locked = getattr(st.session_state['ui'], 'exports', None) is not None or getattr(st.session_state['ui'], 'pending', None) is not None
            draft_text = build_audit_note_draft(payload)
            with st.expander('备注草稿'):
                _render_draft_preview(draft_text)
                current_note = st.session_state.get('_note', editor.note)
                merged = _merged_note(current_note, draft_text)
                already_added = draft_text in current_note
                if merged is None:
                    st.warning('追加后超过4,000字，请先精简备注。')
                st.button('已追加' if already_added else '追加到备注',
                          key='fill_evidence_draft_' + str(getattr(editor, 'record_id', '')),
                          disabled=locked or already_added or merged is None,
                          on_click=_apply_draft_note,
                          args=(draft_text, editor.input_key, editor.record_id))
        more = '核查方法' + (f'（共{len(primary)}项）' if primary else '')
        with st.expander(more):
            if len(primary) > 3:
                _check_table(primary[3:])
            _render_check_actions(primary, supplement)


def field_frame(payload):
    return pd.DataFrame([{'数据项': r['name'], '字段': r['field'], '原值': r['value'], '单位': r['unit'],
        '缺失': r['missing'], 'RF填补值': r['imputed_rf'], 'LR填补值': r['imputed_lr'],
        'P01': r['reference']['p01'], 'P50': r['reference']['p50'], 'P99': r['reference']['p99'],
        '参考状态': r['reference_status'], '本批百分位': r['batch_percentile'], '有效观测数': r['batch_observed_count']}
        for r in sorted(payload.get('all_fields', []), key=lambda r: r['missing'])])


def rf_contribution_rows(payload):
    """Chart shown fields plus the exact sum of every omitted field."""
    result = payload['decomposition']
    fields = {row['field']: row for row in payload.get('all_fields', [])}
    visible = list(dict.fromkeys(result.get('default_fields', result['rf']['top_fields'])))[:12]
    contributions = result['rf']['fields']
    shown = sorted((row for row in contributions if row['field'] in visible),
                   key=lambda row: -abs(row['contribution']))
    rows = [{'field': row['field'], 'name': fields.get(row['field'], {}).get('name', row['field']),
             'contribution': row['contribution'], 'kind': '字段'} for row in shown]
    omitted = [row for row in contributions if row['field'] not in visible]
    if omitted:
        rows.append({'field': '__remaining__', 'name': f'其余{len(omitted)}项合计',
                     'contribution': math.fsum(row['contribution'] for row in omitted), 'kind': '其余合计'})
    return rows


def _contribution_spec(rows):
    names = [row['name'] for row in rows]
    values = [dict(row, label=(row['name'] + f'（{i+1}）') if names.count(row['name']) > 1 else row['name'],
                   direction='抬高分数' if row['contribution'] > 0 else
                   '降低分数' if row['contribution'] < 0 else '无加减') for i, row in enumerate(rows)]
    return {'height': max(90, len(rows) * 28) + 150, 'data': {'values': values},
            'layer': [
                {'mark': {'type': 'bar', 'cornerRadiusEnd': 2}, 'encoding': {
                    'x': {'field': 'contribution', 'type': 'quantitative', 'title': '对主要模型（RF）分数的加减',
                          'scale': {'zero': True}, 'axis': {'tickCount': 4, 'labelFlush': False}},
                    'y': {'field': 'label', 'type': 'nominal', 'sort': None, 'title': None,
                          'axis': {'labelLimit': 165}},
                    'color': {'field': 'direction', 'type': 'nominal', 'title': None,
                              'scale': {'domain': ['抬高分数', '降低分数', '无加减'],
                                        'range': ['#B55335', '#27758B', '#82909C']},
                              'legend': {'orient': 'top', 'direction': 'vertical', 'columns': 1}},
                    'tooltip': [{'field': 'name', 'title': '数据项'}, {'field': 'field', 'title': '字段'},
                                {'field': 'contribution', 'title': 'RF加减', 'format': '+.6f'}]}},
                {'mark': {'type': 'rule', 'color': '#718091'}, 'encoding': {'x': {'datum': 0}}}],
            'config': {'view': {'stroke': None}}}


def render_score_accounting(payload):
    if not (_INTERVAL_RULES_AVAILABLE and _CONTRIBUTION_GUIDANCE_AVAILABLE):
        st.warning('文件版本不一致，请重新解压运行包。')
        return
    result = payload.get('decomposition', {})
    if result.get('status') != 'available':
        st.info('评分分解暂不可用：' + result.get('reason', '当前无法还原分数。'))
        return
    fields = {r['field']: r for r in payload.get('all_fields', [])}
    rf, lr = result['rf'], result['lr']
    chart_rows = rf_contribution_rows(payload)
    arch = classify_attribution_archetype(payload)
    bands = arch['bands']

    st.markdown('**' + literal_markdown(_brief_archetype(arch)) + '**')
    scores = payload.get('scores', {})
    st.caption(f'RF {_score_text(rf["score"])} / 分界 {_score_text(scores.get("rf_threshold"))} · '
               f'LR {_score_text(lr["reconstructed"])} / 分界 {_score_text(scores.get("lr_threshold"))}')
    _render_band_summary(bands)

    st.markdown('**RF 评分构成**')
    st.write(f'起始 {rf["base"]:.4f} {rf["sum"]:+.4f} ＝ {rf["reconstructed"]:.4f}')
    st.vega_lite_chart(_contribution_spec(chart_rows), width='stretch')

    unified_top = unified_interval_contribution_rows(payload, top_only=True, limit=12)
    if unified_top:
        st.markdown('**关键数据**')
        visible_rows = [dict(数据项=row['数据项'], 当前值=row['当前原值'],
                             训练正常参考=row['训练正常区间 (P01～P99)'], 当前状态=row['五级区间状态'],
                             具体核查='', _contribution=dict(区间分级=row['五级区间状态'],
                                 rf_path_contribution=row['RF路径贡献 (概率点)'],
                                 rf_replace_delta=row['常见值对照变化 (ΔRF)'])) for row in unified_top]
        _check_table(visible_rows[:6])
        if len(visible_rows) > 6:
            with st.expander(f'其余 {len(visible_rows)-6} 项重点数据'):
                _check_table(visible_rows[6:])

    unified_all_map = {r['字段']: r for r in unified_interval_contribution_rows(payload, top_only=False)}

    def row(item):
        field = fields.get(item['field'], {})
        u_row = unified_all_map.get(item['field'], {})
        return {
            '数据项': field.get('name', item['field']),
            '字段': item['field'],
            '原值': field.get('value'),
            '单位': field.get('unit', ''),
            '缺失': field.get('missing'),
            '训练正常区间 (P01～P99)': u_row.get('训练正常区间 (P01～P99)', ''),
            '参考区间状态': u_row.get('五级区间状态', field.get('reference_status', '')),
            '参考状态': field.get('reference_status', ''),
            '本批百分位': u_row.get('本批百分位', ''),
            'RF分数加减': item['contribution'],
            '常见值对照变化 (ΔRF)': u_row.get('常见值对照变化 (ΔRF)'),
            '路径影响说明': u_row.get('对异常指数的作用说明', ''),
        }

    with st.expander('RF 完整计算'):
        st.dataframe(pd.DataFrame([row(r) for r in rf['fields']]), hide_index=True, height=320, width='stretch')
        rest = next((r['contribution'] for r in chart_rows if r['kind'] == '其余合计'), 0.)
        st.write(f'基准 {rf["base"]:.12g}；其余合计 {rest:+.12g}；全部合计 {rf["sum"]:+.12g}；'
                 f'重构 {rf["reconstructed"]:.12g}；原分数 {rf["score"]:.12g}。')
        st.caption(f'实际 {rf["tree_count"]} 棵树；重构误差 {result["errors"]["rf_score"]:.3g}；'
                   f'未单列字段最大绝对贡献 {result.get("omitted_max_abs_path_contribution", 0):.9g}。')
        st.json(rf, expanded=False)
    st.markdown('**辅助模型（LR）**')
    st.write(f'当前 LR 分数：{lr["reconstructed"]:.4f}')
    with st.expander('LR 完整计算'):
        st.write(f'截距 {lr["intercept"]:.4f} → 内部线性值 {lr["logit"]:.4f} → LR 分数 {lr["reconstructed"]:.4f}')
        rf_terms = {r['field']: r['contribution'] for r in rf['fields']}
        st.dataframe(pd.DataFrame([{'数据项': fields.get(r['field'], {}).get('name', r['field']),
            '字段': r['field'], 'LR线性项（对数几率）': r['term'],
            '贡献方向说明': '各自基准下贡献方向不同' if rf_terms.get(r['field'], 0)*r['term'] < 0 else ''}
            for r in lr['fields']]), hide_index=True, height=320, width='stretch')
        st.caption(f'线性值误差 {result["errors"]["lr_logit"]:.3g}；分数误差 {result["errors"]["lr_score"]:.3g}。')
        st.json(lr, expanded=False)


def _render_card(card):
    plain = review_check_items({'cards': [card]})[0]
    st.markdown('**' + literal_markdown(card['name']) + '** · ' + card['role'])
    if card['missing']:
        st.write('原值：原始缺失')
        st.caption('替换试算包含缺失状态变化，不能解释为缺失项的原值贡献。')
    else:
        st.write('原值 ' + plain['当前值'] + '；' + plain['当前状态'] + '。')
    ref, pos = card['reference'], card['reference_position']
    st.write(plain['训练正常参考'])
    if pos.get('near'):
        side = '低端' if pos['near_side'] == '下界' else '高端'
        st.caption(f'距参考{side} {pos["boundary_fraction"]:.2%} 中位数分段跨度')
    if card['missing']:
        st.write(f'模型填补值：RF {_fmt(card["imputed_rf"])}；LR {_fmt(card["imputed_lr"])}')
    st.markdown('**参考值对照**')
    st.dataframe(pd.DataFrame([{'模型': '主要模型（RF）' if key == 'rf' else '辅助模型（LR）', '当前分数': card[key]['before'],
        '按常见值对照后': card[key]['after'], '分数变化': card[key]['delta'],
        '怎么看': '暂时无法比较' if card[key]['delta'] is None else '变化很小' if card[key]['weak'] else card[key]['direction']}
        for key in ('rf', 'lr')]), hide_index=True, width='stretch')
    if card['direction_difference']:
        st.caption('两模型变化方向不同' + ('，LR变化较小。' if card['lr']['weak'] else '。'))
    st.markdown('**核查方法**')
    st.write(plain['具体核查'])
    with st.expander('字段详情'):
        st.caption(card['field'] + ' · ' + '；'.join(card['selection_reasons']))
        st.caption(f'P01（低端）/ P50（中间值）/ P99（高端）：{_fmt(ref["p01"])} / {_fmt(ref["p50"])} / {_fmt(ref["p99"])} {card["unit"]}')
        percentile = '不可用' if card['batch_percentile'] is None else f'{card["batch_percentile"]:.1f}%'
        st.caption(f'本批位置 {percentile} · 有效观测 {card["batch_observed_count"]:,} 条')
        st.caption(card['meaning'])


def _render_evidence_cards(payload):
    cards = payload.get('cards', [])
    selection = payload.get('selection', {})
    if selection.get('status') == 'unavailable':
        st.warning('主要证据暂不可用：' + selection.get('reason', '原因未知'))
        return
    st.caption(f'主要核查数据 {len(cards)} 项')
    if cards:
        frame = pd.DataFrame([{'数据项': c['name'], '原值': '缺失' if c['missing'] else _fmt(c['value']), '单位': c['unit'],
            '参考状态': review_check_items({'cards':[c]})[0]['当前状态'], '为何列出': c['role']} for c in cards])
        st.dataframe(frame, hide_index=True, height=min(260, 38 + len(cards)*35), width='stretch')
        fields = [c['field'] for c in cards]
        names = {c['field']: c['name'] for c in cards}
        chosen = st.selectbox('查看一项证据', fields, format_func=names.get,
                              key='_detail_evidence_field_' + str(payload.get('record_id', '')))
        with st.container(border=True):
            _render_card(next(c for c in cards if c['field'] == chosen))
    else:
        st.info('未筛出明显问题项。' +
                ('当前高分仍缺可核查解释，建议待补证。' if payload.get('scores', {}).get('rf_anomaly') else ''))
    if selection.get('lr_only'):
        with st.expander(f'辅助模型另外提醒了 {len(selection["lr_only"])} 项'):
            st.caption('LR 对照变化较大、RF 对照变化较小的项目')
            st.dataframe(pd.DataFrame(selection['lr_only']), hide_index=True, width='stretch')


def _render_all_field_issues(payload):
    selection = payload.get('selection', {})
    st.markdown('**其他线索**')
    alerts = review_guidance(payload)['alerts']
    for alert in alerts:
        st.write(alert)
    if not alerts:
        st.caption('无其他线索。')
    rows = [{'数据项': field_name(r['field']), '怎么看': r['direction_label'], '与当前提示的关系': r['role'],
             '主要模型对照后的变化': r['rf']['delta'], '辅助模型对照后的变化': r['lr']['delta']}
            for r in payload.get('all_fields', []) if r['direction_difference'] or r['role'] == '反向线索']
    if rows:
        with st.expander(f'方向差异与反向线索（{len(rows)}项）'):
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=260, width='stretch')
    if selection.get('omitted', 0):
        st.caption(f'其他候选 {selection["omitted"]} 项，见全部字段。')


def _render_raw_methods(payload):
    _render_all_field_issues(payload)
    full = field_frame(payload)
    if not full.empty:
        with st.expander('全部字段与填补值'):
            st.caption('训练正常参考：P01 低端 · P50 中间值 · P99 高端')
            st.dataframe(full, hide_index=True, height=320, width='stretch')
    if payload.get('input_quality_details'):
        with st.expander('原值与数据问题'):
            st.dataframe(pd.DataFrame(payload['input_quality_details']), hide_index=True, width='stretch')
    with st.expander('多项参考值对照'):
        joint, stability = payload.get('joint_test', {}), payload.get('stability', {})
        st.caption('将主要项目一起换为参考值，查看分数变化。')
        st.write(stability.get('status', '稳定性不可评估'))
        st.json({'joint': joint, 'stability': stability}, expanded=False)
    with st.expander('邻近记录与日志'):
        neighbor = payload.get('neighbors', {})
        st.caption(neighbor.get('scope', '未保留可比较的IP或时间上下文。'))
        if neighbor.get('available'):
            st.write(f'同一来源地址的其他记录 {_fmt(neighbor.get("same_source"))} 条；同一目标地址的其他记录 {_fmt(neighbor.get("same_destination"))} 条；'
                     f'该来源还涉及 {_fmt(neighbor.get("source_fanout"))} 个目标地址；地址、端口、协议均相同的其他记录 {_fmt(neighbor.get("repeat_five_tuple"))} 条。')
        if neighbor.get('time_window'):
            window = neighbor['time_window']
            st.write(f'前后各{neighbor.get("window_seconds", 60)}秒：{window["all"]} 条；'
                     f'可比较时间 {window["valid_timestamps"]} 条（含当前）；{window["timezone"]}。')
        if not neighbor.get('available') and not neighbor.get('time_window'):
            st.caption('请从原始流量或外部日志补证。')
    with st.expander('完整计算快照'):
        st.write(payload.get('assessment', {}).get('status', '当前解释不可用。'))
        for reason in payload.get('assessment', {}).get('reasons', []):
            st.caption(reason)
        st.json(payload, expanded=False)


def render_evidence_detail(payload):
    """Return tabs so the host can append existing tools to the raw-data tab."""
    tabs = st.tabs(['评分图解', '证据明细', '原始数据与方法'])
    with tabs[0]:
        render_score_accounting(payload)
    with tabs[1]:
        _render_evidence_cards(payload)
    with tabs[2]:
        _render_raw_methods(payload)
    return tabs


def render_engine_panel(snapshot, *, historical=False):
    """Render the same immutable values used by save/export, including past rows."""
    if not snapshot or snapshot.get('status') == 'not_used':
        return
    if snapshot.get('status') != 'verified':
        st.warning('研判引擎证据暂不可用：' + snapshot.get('reason', '证据不完整'))
        return
    record, params = snapshot['results'], snapshot['parameters']
    with st.container(border=True, key='historical_engine_panel' if historical else 'engine_panel'):
        st.markdown('**保存时的研判引擎证据**' if historical else '**研判引擎证据**')
        st.markdown(f"研判结论 **{record['cp_set']}** · 信任度 Bel **{record['ds_belief']:.3f}** · "
                    f"似然度 Pl **{record['ds_plausibility']:.3f}** · 冲突度 K **{record['ds_conflict']:.3f}**")
        st.caption('结论来自 Mondrian 共形预测集：只含一个标签时确定，否则为不确定；'
                   '[Bel, Pl] 是 Dempster 融合后的攻击信任区间，K 是三个证据源相互矛盾的质量。')
        st.caption(f"共形 p 值：攻击 {record['cp_p_attack']:.3f} · 正常 {record['cp_p_normal']:.3f}"
                   f"（显著性 ε = {params['epsilon']:.2f}）；"
                   f"Venn–Abers 概率端点 [{record['va_low']:.3f}, {record['va_high']:.3f}]")
        st.caption('概率端点按冻结校准样本逐分数精确计算；这不是固定置信水平的置信区间。'
                   if params.get('va_method') == 'exact-isotonic-two-label-refits' else
                   '此历史快照使用旧版网格近似概率端点，仅供参考。')
        st.caption(f"复核提示：{'需要再检查' if record['engine_review'] else '未进入引擎复核队列'}；"
                   f"冲突阈值 {params['conflict_cut']:.4f}。结论仍需结合外部信息核实。")
        with st.expander('三个证据源的基本概率分配'):
            st.dataframe(pd.DataFrame(snapshot['source_masses']), hide_index=True, width='stretch',
                         column_config={k: st.column_config.NumberColumn(format='%.4f')
                                        for k in ['原始分数', '可靠度 r', 'm(攻击)', 'm(正常)', 'm(未知Θ)']})
            decomposition = snapshot['ecod_decomposition']
            table = pd.DataFrame(decomposition['top_fields'])
            table.insert(0, '数据项', table['field'].map(field_name))
            table['direction'] = table['direction'].map({'left':'偏低（左尾）','right':'偏高（右尾）'})
            table['input_missing'] = table['input_missing'].map({True:'已使用训练中位数',False:'原始有效值'})
            table = table.rename(columns={'field':'字段','contribution':'尾部贡献','direction':'方向','input_missing':'取值状态'})
            branch = {'left':'左尾','right':'右尾','auto':'偏度自动方向'}[decomposition['selected_branch']]
            st.caption(f"ECOD 采用{branch}分支；全部 {len(decomposition['fields'])} 项贡献合计 "
                       f"{decomposition['contribution_sum']:.3f}，还原原始分数 {record['ecod_score']:.3f}。"
                       '下表仅列前六项；完整贡献随操作记录保存。')
            st.dataframe(table, hide_index=True, width='stretch',
                         column_config={'尾部贡献': st.column_config.NumberColumn(format='%.3f')})
            st.caption('贡献用于解释 ECOD 原始分数，不代表该字段对最终攻击结论的因果影响。')
