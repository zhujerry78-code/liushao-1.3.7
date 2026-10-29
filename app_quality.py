"""Read-only application feedback from the existing adapter's returned outcomes.

This view does not clean, score or change the three-value inference contract.
Only problem rows and required feature cells are retained for on-demand details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd

from liushao.data import EXCLUDED


KNOWN_REJECTION = '全部必需特征无效，无法判断'
DETAIL_COLUMNS = ['字段', 'CSV读取值', '适配结果', '处理状态']
IDENTITY_COLUMNS = ['record_id', 'source_file', 'source_sha256', 'source_row']
REMAPPED_COLUMNS = set(IDENTITY_COLUMNS) | {'timestamp'}
LABEL_COLUMNS = {name.casefold() for name in EXCLUDED} | {
    'ground_truth', 'ground_truth_evaluation_only', 'attack_type_evaluation_only',
}
RECORD_COLUMNS = ['record_id', 'source_file', 'source_row', 'input_file', 'input_row',
                  'issue_count', 'disposition', 'reason']


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('字段反馈关联失败：' + message)


def physical_row(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 2


def aligned_context(raw, physical_rows, feature_index):
    """Align accepted records to physical CSV rows, before demo source remapping."""
    rows = list(raw.attrs.get('source_rows', []))
    require(len(rows) == len(raw) and len(set(rows)) == len(rows)
            and all(physical_row(row) for row in rows), 'CSV物理行映射无效')
    require(isinstance(physical_rows, pd.Series) and physical_rows.index.equals(feature_index)
            and physical_rows.is_unique, '有效记录物理行映射无效')
    positions = {int(row): pos for pos, row in enumerate(rows)}
    require(all(physical_row(row) and int(row) in positions for row in physical_rows),
            '有效记录未完整关联当前CSV')
    result = raw.iloc[[positions[int(row)] for row in physical_rows]].copy()
    result.index = feature_index
    return result


def check_identity(frame: pd.DataFrame, name: str) -> None:
    require(frame.columns.is_unique and set(IDENTITY_COLUMNS) <= set(frame.columns),
            f'{name}缺少唯一的来源字段')
    require(frame.index.is_unique, f'{name}索引不唯一')
    require(all(physical_row(value) for value in frame.source_row), f'{name}物理行号无效')
    for column in ('record_id', 'source_file', 'source_sha256'):
        require(all(isinstance(value, str) and bool(value) for value in frame[column]),
                f'{name}来源身份无效')
    require(frame.record_id.is_unique, f'{name}记录编号不唯一')
    require(all(record_id == f'{digest}:{row}' for record_id, digest, row in zip(
        frame.record_id, frame.source_sha256, frame.source_row)), f'{name}记录与来源行不匹配')


@dataclass
class QualityView:
    """Application-only counts and source identities; unknown rejection cells are uncounted."""

    summary: pd.DataFrame
    records: pd.DataFrame
    problem_cells: int
    affected_rows: int
    rejected_rows: int
    unknown_rejected_rows: int
    input_rows: int
    accepted_problem_rows: int
    known_rejected_rows: int
    _read_values: pd.DataFrame = field(repr=False)
    _mask: pd.DataFrame = field(repr=False)

    def details(self, record_id: str) -> pd.DataFrame:
        """Build at most one row per model feature, without interpreting CSV text."""
        require(record_id in self.records.index, '所选问题记录不在本次输入中')
        if record_id not in self._read_values.index:
            return pd.DataFrame(columns=DETAIL_COLUMNS)
        record = self.records.loc[record_id]
        mask = self._mask.loc[record_id]
        values = self._read_values.loc[record_id]
        result = []
        for feature in mask.index[mask]:
            value = values[feature]
            empty = isinstance(value, str) and value == ''
            prefix = '读取值为空；' if empty else ''
            outcome = ('全部必需字段无效；具体原因未细分'
                       if record.reason == KNOWN_REJECTION
                       else '适配后缺失；具体原因未细分')
            result.append({'字段': feature, 'CSV读取值': value,
                           '适配结果': prefix + outcome, '处理状态': record.disposition})
        return pd.DataFrame(result, columns=DETAIL_COLUMNS)


def validate_final_mapping(predictions, final):
    require(isinstance(final, pd.DataFrame), '最终来源需要数据表')
    check_identity(final, '最终记录')
    require(final.index.equals(predictions.index) and final.columns.equals(predictions.columns),
            '最终来源与预测行顺序或字段不一致')
    require(all(final[column].equals(predictions[column]) for column in predictions.columns
                if column not in REMAPPED_COLUMNS), '最终来源不能改变既有预测结果')


def build_quality_view(raw: pd.DataFrame, predictions: pd.DataFrame, features: pd.DataFrame,
                       rejected: pd.DataFrame,
                       final_predictions: pd.DataFrame | None = None) -> QualityView:
    """Associate existing masks/rejections with CSV cells, before any source remapping.

    ``predictions`` and ``features`` must keep their shared inference row index.
    Optional ``final_predictions`` may only replace demo source identities/timestamps.
    Invalid associations raise ValueError for the UI to degrade without changing predictions.
    """
    require(all(isinstance(frame, pd.DataFrame)
                for frame in (raw, predictions, features, rejected)), '需要数据表输入')
    require(raw.columns.is_unique and features.columns.is_unique, '字段名不唯一')
    fields = features.columns.tolist()
    require(bool(fields) and all(isinstance(name, str) and bool(name) for name in fields),
            '模型字段列表为空或无效')
    require(not LABEL_COLUMNS.intersection(name.casefold() for name in fields),
            '标签或来源字段不得进入字段明细')
    require(set(fields) <= set(raw.columns), '读取表缺少模型字段')
    source_rows = raw.attrs.get('source_rows')
    require(isinstance(source_rows, (list, tuple, np.ndarray, pd.Index, pd.Series)),
            '缺少CSV物理行号向量')
    source_rows = list(source_rows)
    require(len(source_rows) == len(raw), '物理行号向量长度与输入不一致')
    require(all(physical_row(value) for value in source_rows), 'CSV物理行号无效')
    require(len(set(source_rows)) == len(source_rows), 'CSV物理行号不唯一')
    check_identity(predictions, '可判断记录')
    check_identity(rejected, '拒绝记录')
    require(features.index.is_unique and features.index.equals(predictions.index),
            '特征索引与可判断记录不一致')
    require('missing_features' in predictions, '缺少既有缺失字段计数')
    mask = features.isna()
    counts = mask.sum(axis=1)
    require(all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
                for value in predictions.missing_features), '缺失字段计数类型无效')
    require(np.array_equal(counts.to_numpy(), predictions.missing_features.to_numpy()),
            '返回的缺失字段计数与特征不一致')
    require(not mask.all(axis=1).any(), '全特征缺失的记录不能列为可判断')
    combined = pd.concat([predictions[IDENTITY_COLUMNS], rejected[IDENTITY_COLUMNS]],
                         ignore_index=True)
    require(len(combined) == len(raw) and combined.record_id.is_unique
            and combined.source_row.is_unique
            and set(combined.source_row) == set(source_rows), '可判断与拒绝记录未完整分割输入')
    require(combined.source_file.nunique() <= 1 and combined.source_sha256.nunique() <= 1,
            '输入记录的文件来源不一致')
    require('reason' in rejected and all(isinstance(value, str) and bool(value)
                                        for value in rejected.reason), '拒绝记录缺少原因')
    final = predictions if final_predictions is None else final_predictions
    require(isinstance(final, pd.DataFrame), '最终来源需要数据表')
    check_identity(final, '最终记录')
    require(final.index.equals(predictions.index) and final.columns.equals(predictions.columns),
            '最终来源与预测行顺序或字段不一致')
    require(all(final[column].equals(predictions[column]) for column in predictions.columns
                if column not in REMAPPED_COLUMNS), '最终来源不能改变既有预测结果')
    require(not set(final.record_id).intersection(rejected.record_id), '最终记录与拒绝记录编号冲突')

    known = rejected.reason.eq(KNOWN_REJECTION)
    known_count = int(known.sum())
    accepted_counts = mask.sum(axis=0).astype('int64')
    summary = pd.DataFrame({'字段': fields,
                            '可判断记录缺失数': accepted_counts.to_numpy(),
                            '整行拒绝涉及数': known_count,
                            '问题单元格数': accepted_counts.to_numpy() + known_count})
    input_positions = {int(row): position for position, row in enumerate(source_rows)}
    record_items, value_positions, detail_ids, detail_masks = [], [], [], []
    accepted_positions = np.flatnonzero(counts.to_numpy() > 0)
    for position in accepted_positions:
        original = predictions.iloc[position]
        identity = final.iloc[position]
        record_items.append({
            'record_id': identity.record_id, 'source_file': identity.source_file,
            'source_row': int(identity.source_row), 'input_file': original.source_file,
            'input_row': int(original.source_row), 'issue_count': int(counts.iloc[position]),
            'disposition': '可判断，使用既有训练填补器',
            'reason': '适配结果存在缺失；具体原因未细分',
        })
        value_positions.append(input_positions[int(original.source_row)])
        detail_ids.append(identity.record_id)
        detail_masks.append(mask.iloc[position].to_numpy())
    for position in range(len(rejected)):
        identity = rejected.iloc[position]
        is_known = bool(known.iloc[position])
        record_items.append({
            'record_id': identity.record_id, 'source_file': identity.source_file,
            'source_row': int(identity.source_row), 'input_file': identity.source_file,
            'input_row': int(identity.source_row), 'issue_count': len(fields) if is_known else pd.NA,
            'disposition': ('整行无法判断，未进入模型' if is_known
                            else '无法判断，处理阶段未提供'), 'reason': identity.reason,
        })
        if is_known:
            value_positions.append(input_positions[int(identity.source_row)])
            detail_ids.append(identity.record_id)
            detail_masks.append(np.ones(len(fields), dtype=bool))
    records = pd.DataFrame(record_items, columns=RECORD_COLUMNS)
    records['issue_count'] = records.issue_count.astype('Int64')
    records = records.sort_values('input_row', kind='stable').set_index('record_id', drop=False)
    read_values = raw.iloc[value_positions][fields].copy()
    read_values.attrs = {}
    read_values.index = pd.Index(detail_ids, name='record_id')
    detail_mask = pd.DataFrame(detail_masks, index=read_values.index, columns=fields, dtype=bool)
    return QualityView(
        summary=summary, records=records,
        problem_cells=int(accepted_counts.sum()) + known_count * len(fields),
        affected_rows=len(records), rejected_rows=len(rejected),
        unknown_rejected_rows=len(rejected) - known_count, input_rows=len(raw),
        accepted_problem_rows=len(accepted_positions), known_rejected_rows=known_count,
        _read_values=read_values, _mask=detail_mask,
    )
