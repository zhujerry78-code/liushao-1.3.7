"""Presentation-only helpers for the local r8 UI revision."""
import hashlib
import json
import re

import pandas as pd

try:
    from liushao_engine import engine_ranked, engine_review_ranked
except ImportError:  # engine package absent: keep the 1.3.6 queues
    from liushao.model import ranked as engine_ranked, review_ranked as engine_review_ranked
from app_types import Filters, ExportSnapshot, line_number, needs_review

UI_REVISION = 'r11-final-readable'
STATUS_NAMES = {'待复核': '未检查', '待补证': '需要补充信息（待补证）',
                '确认异常': '确认异常', '确认正常': '确认正常', '复核不可用': '暂不可用'}
FIELD_NAMES = {
    'Flow Duration': '连接持续时间', 'Flow Byts/s': '每秒传输字节数',
    'Flow Pkts/s': '每秒数据包数', 'Tot Fwd Pkts': '发出数据包数',
    'Tot Bwd Pkts': '返回数据包数', 'TotLen Fwd Pkts': '发出数据总量',
    'TotLen Bwd Pkts': '返回数据总量', 'Fwd Pkts/s': '每秒发出数据包数',
    'Bwd Pkts/s': '每秒返回数据包数', 'Pkt Size Avg': '平均数据包大小',
    'Down/Up Ratio': '收发数量比', 'Fwd Seg Size Avg': '平均发出数据段大小',
    'Bwd Seg Size Avg': '平均返回数据段大小', 'Fwd Byts/b Avg': '发出数据块平均大小',
    'Fwd Pkts/b Avg': '每块平均发出包数', 'Fwd Blk Rate Avg': '平均发出数据块速率',
    'Bwd Byts/b Avg': '返回数据块平均大小', 'Bwd Pkts/b Avg': '每块平均返回包数',
    'Bwd Blk Rate Avg': '平均返回数据块速率', 'Init Fwd Win Byts': '发出方向起始窗口大小',
    'Init Bwd Win Byts': '返回方向起始窗口大小', 'Fwd Act Data Pkts': '含数据的发出包数',
    'Fwd Seg Size Min': '发出数据段最小大小', 'Protocol': '通信协议',
    'Dst Port': '目标端口', 'Fwd Header Len': '发出包头总长度',
    'Bwd Header Len': '返回包头总长度',
}
for prefix, name in [('Fwd Pkt Len', '发出包长度'), ('Bwd Pkt Len', '返回包长度'),
                     ('Pkt Len', '数据包长度'), ('Flow IAT', '数据包间隔'),
                     ('Fwd IAT', '发出包间隔'), ('Bwd IAT', '返回包间隔'),
                     ('Active', '活跃时长'), ('Idle', '空闲时长')]:
    for suffix, meaning in [('Max', '最大'), ('Min', '最小'), ('Mean', '平均'),
                            ('Std', '波动'), ('Tot', '总计'), ('Var', '差异')]:
        FIELD_NAMES[f'{prefix} {suffix}'] = meaning + name
for direction, meaning in [('Fwd', '发出'), ('Bwd', '返回')]:
    for flag in ['PSH', 'URG']:
        FIELD_NAMES[f'{direction} {flag} Flags'] = f'{meaning}{flag}标记数'
    FIELD_NAMES[f'Subflow {direction} Pkts'] = f'分段{meaning}包数'
    FIELD_NAMES[f'Subflow {direction} Byts'] = f'分段{meaning}字节数'
for flag in ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'CWE', 'ECE']:
    flag_name = {'FIN': '连接结束', 'SYN': '连接请求', 'RST': '连接重置', 'PSH': '数据推送',
                 'ACK': '确认接收', 'URG': '紧急数据', 'CWE': '拥塞窗口调整', 'ECE': '拥塞提示'}[flag]
    FIELD_NAMES[f'{flag} Flag Cnt'] = flag_name + '标记数'
FIELD_NAMES['CWE Flag Count'] = '拥塞窗口调整标记数'
for direction, meaning in [('Fwd', '发出'), ('Bwd', '返回')]:
    FIELD_NAMES[f'{direction} PSH Flags'] = meaning + '数据推送标记数'
    FIELD_NAMES[f'{direction} URG Flags'] = meaning + '紧急数据标记数'


def field_name(name):
    return FIELD_NAMES.get(name, '其他数据项')


def literal_markdown(value):
    """Display input names as text, without Markdown formatting or remote embeds."""
    return re.sub(r'([\\`*_{}\[\]()#+.!|<>-])', r'\\\1', str(value))


def safe_csv_text(value):
    if not isinstance(value, str): return value
    at = 0
    while at < len(value) and (value[at].isspace() or ord(value[at]) < 33 or value[at] == '\ufeff'):
        at += 1
    if value.startswith(('\t','\r','\n')) or value[at:].startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def csv_bytes(frame):
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object', 'string']).columns:
        safe[col] = safe[col].map(safe_csv_text)
    return safe.to_csv(index=False).encode('utf-8-sig')


def default_filters():
    return Filters()


def filtered(pred, filters, bundle):
    queue = filters['queue']
    if filters['scope'] == '需要再检查':
        queue = '待核实'
    table = engine_ranked(pred) if queue == '高异常得分' else engine_review_ranked(pred) if queue == '待核实' else pred
    if (queue == '高异常得分' and filters['lr_tie_break']) or (queue == '待核实' and filters['include_lr_near']):
        from app_triage import diagnostic_view
        d = pred[['record_id']].copy()
        d['diagnostic_review_hint'] = (needs_review(pred) |
            ((pred.lr_score - bundle['config']['lr_threshold']).abs() <= bundle['config']['near_margin']))
        table = diagnostic_view(pred, d, 'lr_tie_break' if queue == '高异常得分' else 'dual_near')
    if filters['scope'] == '异常提示':
        table = table[table.rf_anomaly]
    if filters['status'] != '全部':
        table = table[table.review_status.eq(filters['status'])]
    table = table[table.rf_score >= filters['minimum']]
    if filters['disagreement']:
        table = table[table.disagreement]
    if filters['near']:
        table = table[table.near_threshold]
    line = filters['line'].strip()
    if line:
        number = line_number(line)
        table = table[table.source_row.eq(number)] if number is not None else table.iloc[:0]
    return table


def dirty(editor):
    return bool(editor and (editor['status'], editor['note']) != tuple(editor['original']))


def base_meta(active, result, bundle, verified):
    pred, rejected = result.predictions, result.rejected
    return dict(model_version=bundle['model_version'], data_version=bundle['config']['data_version'],
        model_sha256=verified.model_sha256,
        descriptor_sha256=verified.descriptor_sha256,
        input_sha256=active['digest'], input_name=active['name'], input_source=active['source'],
        valid_rows=len(pred), unjudgeable_rows=len(rejected), rf_threshold=bundle['config']['rf_threshold'],
        lr_threshold=bundle['config']['lr_threshold'], near_margin=bundle['config']['near_margin'],
        ui_revision=UI_REVISION)


def export_snapshot(active, pred, audit, meta, available, *, features=None, physical_rows=None, bundle=None):
    from liushao.reviews import review_with_evidence
    # Read only persisted explanations; an export must never recreate the past
    # using today's model, input, or explanation implementation.
    audit = [review_with_evidence(row) for row in audit]
    meta = dict(meta, review_storage_available=available, has_unsaved_review=False,
                saved_review_records=len({r['record_id'] for r in audit}) if available else None,
                review_history_entries=len(audit) if available else None,
                review_evidence_schema='review-evidence-2',
                review_evidence_saved=sum(row['evidence_snapshot_status'] == 'saved' for row in audit) if available else None,
                review_evidence_missing=sum(row['evidence_snapshot_status'] == 'missing' for row in audit) if available else None,
                review_evidence_invalid=sum(row['evidence_snapshot_status'] == 'invalid' for row in audit) if available else None)
    confirmed, missing = None, 0
    if available:
        from app_record_tools import confirmed_frame
        confirmed, missing = confirmed_frame(active, pred, audit, meta, features, physical_rows, bundle, field_name)
    exported=pred.copy()
    if (isinstance(physical_rows,pd.Series) and physical_rows.index.is_unique
            and physical_rows.index.equals(pred.index)):
        valid=physical_rows.map(lambda value: type(value) is not bool and pd.notna(value) and float(value)>=2 and float(value).is_integer())
        if not valid.all(): raise ValueError('输入 CSV 行号映射无效，不能生成检查结果。')
        exported['input_csv_row']=physical_rows.astype('int64')
    else:
        exported['input_csv_row']=pd.Series(pd.NA,index=pred.index,dtype='Int64')
    return ExportSnapshot(input_key=active['key'], meta=meta,
                predictions=csv_bytes(exported) if available else None,
                history=json.dumps(audit, ensure_ascii=False, indent=2).encode('utf-8') if available else None,
                manifest=json.dumps(meta, ensure_ascii=False, indent=2).encode('utf-8'),
                confirmed=csv_bytes(confirmed) if confirmed is not None else None,
                confirmed_count=len(confirmed) if confirmed is not None else 0, missing_input_rows=missing)
