"""Application-only review deliverables and descriptive statistics."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from liushao.reviews import review_with_evidence

CONFIRMED_COLUMNS = ['输入文件', '输入CSV行号', '来源文件', '来源行号', '保存时系统提示', '我的判断', '备注',
    '复核记录ID', '复核保存时间UTC', '保存时证据摘要', '结论单生成时间UTC', '记录ID', '模型版本',
    '保存时输入SHA256', '保存时模型SHA256', '保存时描述SHA256', '证据快照状态', '证据快照SHA256', '解释方法版本']

def latest_reviews(audit, model_version):
    latest = {}
    for row in audit:
        previous = latest.get(row['record_id'])
        if row['model_version'] == model_version and (previous is None or row['id'] > previous['id']):
            latest[row['record_id']] = row
    return latest

def batch_positions(features, row_index):
    if not features.index.is_unique or row_index not in features.index:
        raise ValueError('本批记录索引不一致，请重新检查文件。')
    result = {}
    for field in features:
        values = features[field].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        value = float(features.at[row_index, field])
        percentile = (100 * (np.count_nonzero(valid < value) + .5 * np.count_nonzero(valid == value)) / len(valid)) if np.isfinite(value) and len(valid) else None
        result[field] = (percentile, len(valid))
    return result

def evidence_summaries(predictions, features, bundle, field_name):
    """Same feature order/reference tests as evidence_for; at most three messages."""
    if not predictions.index.is_unique or not features.index.equals(predictions.index):
        raise ValueError('证据与记录索引不一致，请重新检查文件。')
    count = len(predictions)
    parts = [[] for _ in range(count)]
    engine = 'engine_review' in predictions
    for i, (disagreement, near) in enumerate(zip(predictions.disagreement, predictions.near_threshold)):
        if engine and predictions.cp_set.iat[i] == '不确定': parts[i].append('研判引擎无法给出确定结论，建议再检查。')
        elif engine and predictions.engine_review.iat[i]: parts[i].append('多源证据冲突较高，建议再检查。')
        elif disagreement: parts[i].append('两种判断不一致，建议再检查。')
        elif near and not engine: parts[i].append('接近判断分界，建议再检查。')
    for field in bundle['config']['features']:
        values = features[field].to_numpy(dtype=float)
        missing = np.isnan(values)
        reference = bundle['reference'][field]
        low, high = reference['q01'], reference['q99']
        outside = ((values < low) | (values > high)) if low is not None and high is not None else np.zeros(count, bool)
        for i in np.flatnonzero(missing | outside):
            if len(parts[i]) < 3:
                parts[i].append(field_name(field) + ('：数据缺失，已补齐后分析。' if missing[i] else '：与参考数据差异较大。'))
    return ['；'.join(items) if items else '两种判断一致，可按需查看详细数据。' for items in parts]


def saved_evidence_summary(snapshot):
    """Format only values frozen with the review; never rerun an explanation."""
    scores = snapshot['scores']
    def number(value):
        return f'{value:.6g}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    parts = [f"RF 分数 {number(scores.get('rf'))} / 分界 {number(scores.get('rf_threshold'))}",
             f"LR 对照 {number(scores.get('lr'))} / 分界 {number(scores.get('lr_threshold'))}"]
    for card in snapshot['cards']:
        reference, rf = card.get('reference', {}), card.get('rf', {})
        parts.append(f"{card.get('name', card.get('field', '数据项'))}={number(card.get('value'))} {card.get('unit', '')}；"
                     f"训练正常参考 P01—P99：{number(reference.get('p01'))}—{number(reference.get('p99'))}；"
                     f"单项替换参考中位数后 RF：{number(rf.get('before'))}→{number(rf.get('after'))}")
    assessment = snapshot.get('assessment')
    if isinstance(assessment, dict):
        parts.append(assessment.get('status', ''))
        parts.extend(assessment.get('reasons', []))
    parts.append('敏感性试算只解释当前模型响应，不能单独确认攻击。')
    return '；'.join(str(part) for part in parts)

def confirmed_frame(active, pred, audit, meta, features, physical_rows, bundle, field_name, generated_at=None):
    latest = latest_reviews(audit, meta['model_version'])
    selected = pred.loc[pred.record_id.map(lambda rid: latest.get(rid, {}).get('status') in ('确认正常', '确认异常'))]
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    mapped = (isinstance(physical_rows, pd.Series) and physical_rows.index.is_unique
              and physical_rows.index.equals(pred.index) and len(physical_rows) == len(pred))
    rows, missing_rows = [], 0
    for index, row in selected.iterrows():
        review = review_with_evidence(latest[row.record_id])
        snapshot = review['evidence_snapshot']
        identity = snapshot.get('identity', {}) if snapshot else {}
        physical = snapshot.get('input_csv_row') if snapshot else None
        # An older frozen payload may omit the physical row mapping. Keep the
        # current-file locator separate from historical model evidence.
        if physical is None and mapped:
            physical = physical_rows.at[index]
        if isinstance(physical, (float, np.floating)) and np.isfinite(physical) and physical.is_integer():
            physical = int(physical)
        if not isinstance(physical, (int, np.integer)) or isinstance(physical, bool) or physical < 2:
            physical = ''; missing_rows += 1
        stamp = datetime.fromisoformat(review['created_at']).astimezone(timezone.utc).isoformat()
        scores = snapshot['scores'] if snapshot else {}
        rf_anomaly = scores.get('rf_anomaly')
        if rf_anomaly is None and isinstance(scores.get('rf'), (int, float)) and isinstance(scores.get('rf_threshold'), (int, float)):
            rf_anomaly = scores['rf'] >= scores['rf_threshold']
        rows.append([(snapshot.get('input_name') if snapshot else None) or active['name'], physical,
            identity.get('source_file', row.source_file), identity.get('source_row', row.source_row),
            ('提示异常' if rf_anomaly else '未提示异常') if rf_anomaly is not None else '无当时快照',
            review['status'], review['note'], review['id'], stamp,
            saved_evidence_summary(snapshot) if snapshot else review['evidence_snapshot_label'],
            generated_at, row.record_id, review['model_version'],
            snapshot['input_sha256'] if snapshot else '', snapshot['model_sha256'] if snapshot else '',
            snapshot['descriptor_sha256'] if snapshot else '', review['evidence_snapshot_label'],
            review['evidence_sha256'] or '', snapshot['method_version'] if snapshot else ''])
    return pd.DataFrame(rows, columns=CONFIRMED_COLUMNS), missing_rows
