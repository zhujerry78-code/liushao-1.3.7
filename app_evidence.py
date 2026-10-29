"""Read-only, record-local model sensitivity and evidence for human review.

No retraining, global-importance proxy, causal attribution or label correction.
The result contains plain JSON data so the exact displayed evidence can be frozen.
"""
from __future__ import annotations

from datetime import datetime
import ipaddress

import numpy as np
import pandas as pd

from app_record_tools import batch_positions
from app_ui_state import field_name
from liushao.feature_metadata import feature_definition as _feature_definition


METHOD_VERSION = 'review-evidence-v2.3'
MEANINGFUL_EFFECT = .01
JOINT_EFFECT_GATE = .1
DIRECTION_EPSILON = 1e-6


def feature_definition(field):
    """Apply the UI name to the shared, presentation-independent field metadata."""
    return _feature_definition(field, field_name(field))


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _row_number(value):
    number = _number(value)
    return int(number) if number is not None and number >= 2 and number.is_integer() else None


def _scores(bundle, frame):
    result = {}
    for key in ('rf', 'lr'):
        values = np.asarray(bundle[key].predict_proba(frame))
        if (values.shape != (len(frame), 2) or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0)):
            raise ValueError('局部解释的模型输出无效，不能生成判断依据。')
        result[key] = values[:, 1].astype(float)
    return result


def _effect(before, after):
    delta = float(after - before)
    direction = ('原值相对参考值推高异常分数' if delta < -1e-8 else
                 '原值相对参考值压低异常分数' if delta > 1e-8 else '本次替换几乎无影响')
    return {'before': float(before), 'after': float(after), 'delta': delta,
            'direction': direction, 'weak': abs(delta) < MEANINGFUL_EFFECT}


def _ranking(names, impacts, target_sign):
    def key(index):
        impact = impacts[index]
        if not np.isfinite(impact):
            return (3, 0, index)
        aligned = target_sign * impact
        role = 0 if aligned > 1e-8 else 1 if aligned < -1e-8 else 2
        return (role, -abs(impact), index)
    return sorted(range(len(names)), key=key)


def _imputed(bundle, key, position):
    predictor = getattr(bundle[key], 'predictor', bundle[key])
    try:
        return _number(predictor.named_steps['impute'].statistics_[position])
    except (AttributeError, KeyError, IndexError):
        return None


def _action(field, reference_status):
    if 'Flag' in field or field == 'Protocol':
        action = '核对协议、连接建立/重置日志及目标服务是否可用。'
        gate = '若日志记录异常重试/扫描且无授权操作，可支持异常；正常重连或服务维护可支持正常。'
    elif 'Duration' in field or 'IAT' in field or field.startswith(('Idle ', 'Active ')):
        action = '核对同源邻近连接、服务端超时和连接数日志，并查询压测/维护工单。'
        gate = '重复超时或长占用并影响服务且无授权说明，才支持异常；单流时长不能定性。'
    elif 'Byts' in field or 'TotLen' in field:
        action = '核对发出/返回方向、目的资产、业务时段以及代理/DLP/备份日志。'
        gate = '未授权目的地与敏感数据传输日志相互印证可支持异常；已批准备份/同步可支持正常。'
    else:
        action = '核对同目的端口的短连接与包速率，查询防火墙/服务日志和压测工单。'
        gate = '短时重复请求伴随服务异常且排除授权压测，才支持异常；正常业务可解释则支持正常。'
    if reference_status == '参考区间内':
        action = '该值在单变量参考内；' + action
    return {'next': action, 'decision_gate': gate + '证据不足时选“需要补充信息（待补证）”。'}


def _context_column(frame, aliases):
    lookup = {str(column).strip().casefold(): column for column in frame.columns}
    return next((lookup[name.casefold()] for name in aliases if name.casefold() in lookup), None)


def _ip(value):
    if pd.isna(value):
        return None
    try:
        return str(ipaddress.ip_address(str(value).strip()))
    except ValueError:
        return None


def _time(value):
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    if not isinstance(value, str):
        return None  # A numeric value has no declared epoch/unit; do not invent one.
    try:
        return pd.Timestamp(datetime.fromisoformat(value.strip().replace('Z', '+00:00')))
    except ValueError:
        for fmt in ('%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S.%f'):
            try:
                return pd.Timestamp(datetime.strptime(value.strip(), fmt))
            except ValueError:
                continue
    return None


def neighbor_summary(context, row_index, *, window_seconds=60):
    """Count only supplied batch metadata; never infer addresses from flow values."""
    result = {'available': False, 'window_seconds': window_seconds,
              'scope': '仅本次导入记录；所有邻居数均不含当前记录。',
              'same_source': None, 'same_destination': None, 'time_window': None,
              'source_fanout': None, 'repeat_five_tuple': None, 'notes': []}
    if not isinstance(window_seconds, int) or not 1 <= window_seconds <= 3600:
        raise ValueError('邻居时间窗必须为 1 至 3600 秒。')
    if (not isinstance(context, pd.DataFrame) or not context.index.is_unique
            or not context.columns.is_unique or row_index not in context.index):
        result['notes'].append('输入未保留可对齐的 IP/时间字段，邻居核查不可用；请从原始流量或日志补证。')
        return result
    src = _context_column(context, ('Src IP', 'Source IP', 'src_ip'))
    dst = _context_column(context, ('Dst IP', 'Destination IP', 'dst_ip'))
    stamp = _context_column(context, ('Timestamp', 'timestamp', 'time'))
    valid_src = context[src].map(_ip) if src else pd.Series(None, index=context.index, dtype=object)
    valid_dst = context[dst].map(_ip) if dst else pd.Series(None, index=context.index, dtype=object)
    source, destination = valid_src.at[row_index], valid_dst.at[row_index]
    source = None if pd.isna(source) else source
    destination = None if pd.isna(destination) else destination
    others = pd.Series(context.index != row_index, index=context.index)
    same_source = valid_src.eq(source) & others if source else None
    same_destination = valid_dst.eq(destination) & others if destination else None
    if same_source is not None:
        result['same_source'] = int(same_source.sum())
        result['source_fanout'] = int(valid_dst[same_source].dropna().nunique()) if dst else None
    else:
        result['notes'].append('当前记录缺少有效源 IP，同源数量与扇出不可用。')
    if same_destination is not None:
        result['same_destination'] = int(same_destination.sum())
    else:
        result['notes'].append('当前记录缺少有效目的 IP，同目的数量不可用。')
    result['available'] = source is not None or destination is not None
    times = context[stamp].map(_time) if stamp else None
    current = times.at[row_index] if times is not None else None
    if current is not None and not pd.isna(current):
        aware = current.tzinfo is not None
        comparable = times.map(lambda item: item is not None and not pd.isna(item)
                               and (item.tzinfo is not None) == aware)
        in_window = pd.Series(False, index=context.index)
        for index, item in times[comparable].items():
            in_window.at[index] = abs((item - current).total_seconds()) <= window_seconds
        in_window &= others
        result['time_window'] = {'all': int(in_window.sum()),
            'same_source': int((in_window & same_source).sum()) if source else None,
            'same_destination': int((in_window & same_destination).sum()) if destination else None,
            'valid_timestamps': int(comparable.sum()),
            'current': current.isoformat(),
            'timezone': '输入提供的时区' if aware else '原始本地时间（输入未声明时区）'}
        if not comparable.all():
            result['notes'].append('缺失、不可解析或时区类型不一致的时间未纳入时间窗统计。')
    else:
        result['notes'].append('缺少可解析的当前时间，时间窗不可用；仅展示本批同源/同目的数量。')
    ports = [_context_column(context, aliases) for aliases in
             [('Src Port', 'Source Port', 'src_port'), ('Dst Port', 'Destination Port', 'dst_port'),
              ('Protocol', 'protocol')]]
    if source and destination and all(ports):
        valid = [pd.to_numeric(context[column], errors='coerce') for column in ports]
        ranges = [(0, 65535), (0, 65535), (0, 255)]
        if all(_number(values.at[row_index]) is not None and
               low <= values.at[row_index] <= high and values.at[row_index] % 1 == 0
               for values, (low, high) in zip(valid, ranges)):
            repeats = same_source & same_destination
            for values in valid:
                repeats &= values.eq(values.at[row_index])
            result['repeat_five_tuple'] = int(repeats.sum())
    if result['repeat_five_tuple'] is None:
        result['notes'].append('五元组不完整或无效，无法统计同五元组重复。')
    result['notes'].append('扇出=其他同源记录中的不同有效目的 IP 数；重复数不等于并发连接、攻击次数或独立事件。')
    return result


def build_record_evidence(record, features, bundle, *, context=None, meta=None, top_k=6):
    from copy import deepcopy
    from datetime import timezone
    from app_decomposition import decompose_scores
    from app_evidence_rules import PARAMETERS, continuous, direction_difference, reference_position, select_fields
    names = bundle['config']['features']
    if type(top_k) is not int or not 0 <= top_k <= 6:
        raise ValueError('主要证据数量必须为 0 至 6。')
    if (not isinstance(record, pd.Series) or not record.index.is_unique
            or not features.index.is_unique or not features.columns.is_unique
            or list(features.columns) != names or record.name not in features.index
            or record.get('model_version') != bundle['model_version'] or not record.get('record_id')):
        raise ValueError('关键证据的记录、特征顺序或模型身份不匹配。')
    frame = features.loc[[record.name]].astype(float).copy()
    original = frame.iloc[0].to_numpy(dtype=float)
    if np.isinf(original).any() or np.isnan(original).all():
        raise ValueError('当前记录含无穷值或全部缺失，不能生成关键证据。')
    before = {key: float(values[0]) for key, values in _scores(bundle, frame).items()}
    if any(_number(record.get(key+'_score')) is None or abs(float(record[key+'_score'])-value)>1e-10 for key,value in before.items()):
        raise ValueError('当前记录分数与冻结模型复算不一致，不能关联证据。')
    cfg = bundle['config']; anomaly = before['rf'] >= cfg['rf_threshold']; target = 1 if anomaly else -1
    references = [bundle['reference'].get(name, {}) for name in names]
    medians = [_number(r.get('median')) for r in references]
    usable = [i for i,m in enumerate(medians) if m is not None]
    positions = [reference_position(name, _number(original[i]), references[i]) for i,name in enumerate(names)]
    adjustable = {i for i in usable if continuous(names[i]) and positions[i]['valid'] and not positions[i]['fixed']}
    variants = [list(medians) for _ in range(3)]
    for v,quantile in ((1,'q01'),(2,'q99')):
        for i in adjustable: variants[v][i] = .9*medians[i]+.1*float(references[i][quantile])
    trials = []
    for values in variants:
        for i in usable:
            changed=original.copy(); changed[i]=values[i]; trials.append(changed)
    deltas = {key:np.full((3,len(names)),np.nan) for key in before}
    if trials:
        scores = _scores(bundle,pd.DataFrame(trials,columns=names))
        for key in before: deltas[key][:,usable] = scores[key].reshape(3,len(usable))-before[key]
    signals_by_variant = []
    for v in range(3):
        signals=[]
        for i,name in enumerate(names):
            rf,lr = _number(deltas['rf'][v,i]),_number(deltas['lr'][v,i])
            signals.append({'index':i,'field':name,'missing':_number(original[i]) is None,
                'position':positions[i],'rf_delta':rf,'lr_delta':lr,
                'significant':rf is not None and abs(rf)>=MEANINGFUL_EFFECT,
                'direction_difference':direction_difference(rf,lr)})
        signals_by_variant.append(signals)
    signals=signals_by_variant[0]
    chosen,reasons=select_fields(signals,target,top_k)
    batch=batch_positions(features,record.name)
    all_fields=[]
    for i,name in enumerate(names):
        value=_number(original[i]); label,unit,meaning=feature_definition(name); pos=positions[i]
        delta=_number(deltas['rf'][0,i])
        role=('无法试算' if delta is None else '影响不足' if abs(delta)<MEANINGFUL_EFFECT else
              '支持当前模型提示' if -target*delta>=MEANINGFUL_EFFECT else '反向线索')
        card={'field':name,'name':label,'unit':unit,'meaning':meaning,'value':value,
              'reference':dict(zip(('p01','p50','p99'),(_number(references[i].get(k)) for k in ('q01','median','q99')))),
              'reference_status':pos['status'],'reference_position':pos,
              'batch_percentile':_number(batch[name][0]),'batch_observed_count':int(batch[name][1]),
              'missing':value is None,
              'effect_kind':'value_and_missing_state' if value is None else 'value_replacement',
              'effect_note':'替换试算包含缺失状态变化' if value is None else '数值替换试算',
              'imputed_rf':_imputed(bundle,'rf',i) if value is None else None,
              'imputed_lr':_imputed(bundle,'lr',i) if value is None else None,
              'role':role,'action':_action(name,pos['status']),
              'selection_reasons':reasons.get(i,[]),'selected':i in chosen,
              'direction_difference':signals[i]['direction_difference'],
              'direction_label':('方向相反（LR 分数变化很小）' if signals[i]['direction_difference'] and abs(signals[i]['lr_delta'])<.01 else
                                 '方向相反，实际影响待评估' if signals[i]['direction_difference'] else '未达到方向差异展示条件')}
        for key in before:
            d=_number(deltas[key][0,i])
            card[key]=_effect(before[key],before[key]+d) if d is not None else {
                'before':before[key],'after':None,'delta':None,'direction':'正常 P50 缺失，无法试算','weak':True}
        all_fields.append(card)
    cards=[all_fields[i] for i in chosen]
    support=[i for i in chosen if i in usable and -target*deltas['rf'][0,i]>=MEANINGFUL_EFFECT]
    joint={'fields':[names[i] for i in support],'count':len(support),
           'definition':'固定主要证据中支持当前 RF 提示的集合 S；三个参考版本只改变参考值，不重选 S。',
           'status':'不可评估：支持集合为空' if not support else '已计算'}
    joint_effects=[]; joint_directions=[]; joint_versions=[]
    if support:
        rows=[]
        for values in variants:
            changed=original.copy()
            for i in support: changed[i]=values[i]
            rows.append(changed)
        scores=_scores(bundle,pd.DataFrame(rows,columns=names))
        for v in range(3):
            effects={key:_effect(before[key],scores[key][v]) for key in before}
            joint_versions.append({'reference':('P50','P50向P01移动10%','P50向P99移动10%')[v],**effects})
            joint_effects.append(-target*effects['rf']['delta'])
            joint_directions.append(direction_difference(effects['rf']['delta'],effects['lr']['delta']))
        for key in before: joint[key]=joint_versions[0][key]
    else:
        for key in before: joint[key]={'before':before[key],'after':None,'delta':None,'direction':'不可评估','weak':True}
    baseline_set={i for i in chosen if i in adjustable and signals[i]['significant']}
    overlap=[]; reversals=[]
    for v in (1,2):
        other,_=select_fields(signals_by_variant[v],target,top_k)
        overlap.append(len(baseline_set&set(other))/len(baseline_set) if baseline_set else None)
        reversals.append([names[i] for i in support if target*deltas['rf'][v,i]>.01])
    coverage=sum(i in adjustable for i in support)
    evaluable=bool(support and coverage and baseline_set)
    passed=(min(v for v in overlap if v is not None)>=.5 and not any(reversals) and min(joint_effects)>=.1) if evaluable else None
    full=coverage==len(support) and bool(support)
    stability={'passed':passed,'coverage':coverage,'support_count':len(support),'full_coverage':full,
        'status':('稳定性不可评估' if passed is None else ('全部可评估支持项' if full else f'可评估部分，覆盖 {coverage}/{len(support)} 项')+('通过' if passed else '未通过')),
        'top_overlap':overlap,'direction_reversals':reversals,'rf_aligned_joint_effects':joint_effects,
        'joint_direction_conflict':joint_directions[0] if joint_directions else None,
        'contradictory_fields':[s['field'] for s in signals if s['direction_difference']],
        'distinct_reference_variants':bool(adjustable),
        'definition':'连续非恒定字段的 P50 向 P01/P99 各移动10%；固定 S 做联合试算，变体筛卡仅用于入选重合率。工程条件：重合≥50%、无超过0.01的反向变化、三组联合影响≥0.10；未作实用标定。'}
    has_significant=any(s['significant'] for s in signals)
    candidate=[s for s in signals if s['missing'] or not s['position']['valid'] or s['significant'] or (not has_significant and s['position']['outside'])]
    omitted=[s for s in candidate if s['index'] not in chosen]
    lr_only=[s for s in signals if s['lr_delta'] is not None and abs(s['lr_delta'])>=.01 and not s['significant']]
    selection={'scanned':len(names),'valid_references':sum(p['valid'] for p in positions),'tested':len(usable),
        'eligible':len(candidate),'selected':len(cards),'omitted':len(omitted),
        'omitted_max_abs_rf_delta':max((abs(s['rf_delta'] or 0) for s in omitted),default=0.),
        'missing_fields':[s['field'] for s in signals if s['missing']],
        'insufficient_references':[s['field'] for s in signals if not s['position']['valid']],
        'direction_differences':[s['field'] for s in signals if s['direction_difference']],
        'counterevidence':[s['field'] for s in signals if s['significant'] and target*s['rf_delta']>0],
        'lr_only':[{'field':s['field'],'rf_delta':s['rf_delta'],'lr_delta':s['lr_delta']} for s in lr_only],
        'signals':[{'field':s['field'],'rf_delta':s['rf_delta'],'lr_delta':s['lr_delta'],
                    'selected':s['index'] in chosen,'reasons':reasons.get(s['index'],[]),
                    'not_selected_reason':None if s['index'] in chosen else '合格候选未入选' if s in candidate else '未达到主要证据展示条件'} for s in signals],
        'support_fields':[names[i] for i in support],'parameters':deepcopy(PARAMETERS)}
    warnings=[]
    if not support: warnings.append('支持字段集合为空，联合检查不可评估。')
    elif joint_effects[0]<.1: warnings.append('联合方向影响不足 0.10。')
    if passed is not True or not full: warnings.append(stability['status'])
    if selection['missing_fields']: warnings.append(f'全部字段中有 {len(selection["missing_fields"])} 项缺失，先核对输入；替换可能同时改变缺失状态。')
    if len(usable)!=len(names): warnings.append('部分字段缺少 P50，无法完成全字段试算。')
    if (before['lr']>=cfg['lr_threshold'])!=anomaly: warnings.append('两模型阈值判断不一致。')
    warnings.append('实用响应冲突尚未标定，实际影响待评估。')
    decomposition=decompose_scores(frame,bundle)
    if decomposition['status']=='available':
        visible=list(dict.fromkeys([c['field'] for c in cards]+decomposition['rf']['top_fields']))
        decomposition['default_fields']=visible
        decomposition['omitted_max_abs_path_contribution']=max((abs(r['contribution']) for r in decomposition['rf']['fields'] if r['field'] not in visible),default=0.)
    metadata=meta or {}; source_sha=str(record.get('source_sha256',''))
    return {'schema_version':2,'method_version':METHOD_VERSION,'generated_at':datetime.now(timezone.utc).isoformat(),
        'record_id':str(record.record_id),'model_version':str(bundle['model_version']),
        'input_name':metadata.get('input_name'),'input_csv_row':_row_number(metadata.get('input_csv_row')),
        'input_sha256':metadata.get('input_sha256',metadata.get('input_sha',source_sha)),
        'model_sha256':metadata.get('model_sha256'),'descriptor_sha256':metadata.get('descriptor_sha256'),
        'identity':{'source_sha256':source_sha,'source_file':str(record.get('source_file','')),'source_row':_row_number(record.get('source_row'))},
        'scores':{**before,'rf_threshold':float(cfg['rf_threshold']),'lr_threshold':float(cfg['lr_threshold']),
                  'near_margin':float(cfg['near_margin']),'rf_anomaly':bool(anomaly),'lr_anomaly':bool(before['lr']>=cfg['lr_threshold'])},
        'definition':'Δ＝替换后分数－原分数；单项替换不可相加。它与评分拆账单是两件事。',
        'reference_definition':'P01/P50/P99 来自训练正常记录，非本地网络基线；范围内不等于正常，越界不证明攻击。',
        'ranking_definition':'先筛合格候选，再保留质量、方向差异和反向线索，按字段去重；主要证据 0–6 项，不凑数。',
        'cards':cards,'all_fields':all_fields,'tested_fields':len(usable),'total_fields':len(names),
        'selection':selection,'decomposition':decomposition,'joint_test':dict(joint,versions=joint_versions,direction_differences=joint_directions),
        'stability':stability,'assessment':{'status':'待补证：数值解释不能替代外部证据','sufficient':False,'reasons':warnings,
            'recommendation':'查资产、服务/防火墙/代理日志与授权任务；证据不足时保存“待补证”。',
            'meaning':'分数是核查排队号，拆账单只解释分数组成；两模型单位不同，贡献不能相加。'},
        'neighbors':neighbor_summary(context,record.name),
        'limitations':'工程解释规则尚待验证；实用冲突未标定；没有外部日志/资产不能确认攻击。'}
