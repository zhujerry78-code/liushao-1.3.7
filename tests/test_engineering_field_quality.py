"""Application-only field feedback follows returned masks, never a second cleaning pass."""
import io

import numpy as np
import pandas as pd
import pytest

from app_quality import build_quality_view
from liushao.data import clean, read_csv_checked


KNOWN_REJECTION = '全部必需特征无效，无法判断'
DETAIL_COLUMNS = ['字段', 'CSV读取值', '适配结果', '处理状态']


def inputs(content=b'a,b,Label,extra\n1,2,hidden,private\n,3,hidden,private\n\nNaN,bad,hidden,private\n4,inf,hidden,private\n'):
    raw = read_csv_checked(io.BytesIO(content))
    values, predictions, rejected = clean(raw, ['a', 'b'], 'input-hash', 'upload.csv')
    return raw, predictions, values, rejected


def test_mixed_rows_preserve_physical_lines_read_values_and_existing_outcomes():
    raw, predictions, features, rejected = inputs()
    before = [frame.copy(deep=True) for frame in (raw, predictions, features, rejected)]
    view = build_quality_view(raw, predictions, features, rejected)
    assert view.input_rows == 4
    assert (view.problem_cells, view.affected_rows, view.rejected_rows) == (4, 3, 1)
    assert (view.accepted_problem_rows, view.known_rejected_rows, view.unknown_rejected_rows) == (2, 1, 0)
    assert view.records.index.tolist() == ['input-hash:3', 'input-hash:5', 'input-hash:6']
    assert view.records.input_row.tolist() == [3, 5, 6]
    assert view.records.source_row.tolist() == [3, 5, 6]
    assert view.records.issue_count.tolist() == [1, 2, 1]
    summary = view.summary.set_index('字段')
    assert summary['可判断记录缺失数'].to_dict() == {'a': 1, 'b': 1}
    assert summary['整行拒绝涉及数'].to_dict() == {'a': 1, 'b': 1}
    assert summary['问题单元格数'].to_dict() == {'a': 2, 'b': 2}
    detail = view.details('input-hash:3')
    assert detail.columns.tolist() == DETAIL_COLUMNS
    assert detail['字段'].tolist() == ['a'] and detail['CSV读取值'].tolist() == ['']
    assert '读取值为空' in detail.iloc[0]['适配结果']
    rejected_detail = view.details('input-hash:5')
    assert rejected_detail['CSV读取值'].tolist() == ['NaN', 'bad']
    assert rejected_detail['处理状态'].str.contains('未进入模型').all()
    assert '未细分' in view.details('input-hash:6').iloc[0]['适配结果']
    assert view.details('input-hash:6')['CSV读取值'].tolist() == ['inf']
    for frame, original in zip((raw, predictions, features, rejected), before):
        pd.testing.assert_frame_equal(frame, original)
    assert 'hidden' not in repr(view) and 'private' not in repr(view)
    assert not {'Label', 'extra'} & set(view.summary['字段'])


def test_feedback_trusts_returned_mask_even_when_read_text_looks_numeric(monkeypatch):
    raw, predictions, features, rejected = inputs(b'a,b\n007,2\n3,  bad  \n')
    features.loc[0, 'a'] = np.nan
    predictions.loc[0, 'missing_features'] = 1
    # Conversely a caller-owned mask, rather than raw text, decides which cells need feedback.
    raw.loc[0, 'b'] = 'not numeric but not marked by the adapter'
    monkeypatch.setattr(pd, 'to_numeric', lambda *args, **kwargs: pytest.fail('second cleaning'))
    view = build_quality_view(raw, predictions, features, rejected)
    assert view.problem_cells == 2
    assert view.details('input-hash:2')['CSV读取值'].tolist() == ['007']
    assert view.details('input-hash:3')['CSV读取值'].tolist() == ['  bad  ']
    assert not view.details('input-hash:2')['字段'].eq('b').any()


def test_raw_dataframe_labels_do_not_replace_source_line_identity():
    raw, predictions, features, rejected = inputs()
    raw.index = [90, 80, 70, 60]
    view = build_quality_view(raw, predictions, features, rejected)
    assert view.details('input-hash:5')['CSV读取值'].tolist() == ['NaN', 'bad']


def test_joint_prediction_feature_index_reorder_preserves_pairing():
    raw, predictions, features, rejected = inputs()
    predictions = predictions.iloc[::-1]
    features = features.loc[predictions.index]
    view = build_quality_view(raw, predictions, features, rejected)
    assert view.details('input-hash:3')['CSV读取值'].tolist() == ['']
    assert view.details('input-hash:6')['字段'].tolist() == ['b']


def test_demo_mapping_keeps_both_csv_and_original_sources():
    raw, predictions, features, rejected = inputs(b'a,b,Label\n,2,secret\n3,4,secret\n')
    final = predictions.copy()
    final['record_id'] = ['original-a:90', 'original-b:800']
    final['source_sha256'] = ['original-a', 'original-b']
    final['source_file'] = ['official-a.csv', 'official-b.csv']
    final['source_row'] = [90, 800]
    view = build_quality_view(raw, predictions, features, rejected, final)
    record = view.records.loc['original-a:90']
    assert (record.input_file, record.input_row) == ('upload.csv', 2)
    assert (record.source_file, record.source_row) == ('official-a.csv', 90)
    assert view.details('original-a:90')['CSV读取值'].tolist() == ['']
    with pytest.raises(ValueError):
        view.details('input-hash:2')


def test_unknown_rejection_has_only_row_information_and_does_not_invent_fields():
    raw, predictions, features, rejected = inputs(b'a,b\nNaN,bad\n1,2\n')
    rejected.loc[0, 'reason'] = '未约定的新拒绝原因'
    view = build_quality_view(raw, predictions, features, rejected)
    assert (view.problem_cells, view.affected_rows, view.unknown_rejected_rows) == (0, 1, 1)
    assert pd.isna(view.records.iloc[0].issue_count)
    assert view.records.iloc[0].reason == '未约定的新拒绝原因'
    assert view.records.iloc[0].disposition == '无法判断，处理阶段未提供'
    assert '未进入模型' not in view.records.iloc[0].disposition
    assert view.details('input-hash:2').empty
    assert view.details('input-hash:2').columns.tolist() == DETAIL_COLUMNS
    assert view.summary['问题单元格数'].sum() == 0


def test_no_problem_records_returns_typed_empty_views():
    view = build_quality_view(*inputs(b'a,b\n1,2\n3,4\n'))
    assert view.records.empty and view.problem_cells == view.affected_rows == 0
    assert view.summary['问题单元格数'].sum() == 0


@pytest.mark.parametrize('fault', [
    'missing_rows', 'short_rows', 'duplicate_rows', 'noninteger_rows', 'small_rows',
    'feature_index', 'duplicate_index', 'missing_count', 'count_type', 'missing_feature',
    'duplicate_feature', 'missing_partition', 'overlap_partition', 'wrong_record_id',
    'wrong_source_row', 'duplicate_record_id', 'wrong_source_hash', 'all_missing_accepted',
])
def test_invalid_input_associations_raise_valueerror(fault):
    raw, predictions, features, rejected = inputs()
    if fault == 'missing_rows':
        raw.attrs.clear()
    elif fault == 'short_rows':
        raw.attrs['source_rows'] = [2]
    elif fault == 'duplicate_rows':
        raw.attrs['source_rows'] = [2, 3, 3, 6]
    elif fault == 'noninteger_rows':
        raw.attrs['source_rows'] = ['2', 3, 5, 6]
    elif fault == 'small_rows':
        raw.attrs['source_rows'] = [1, 3, 5, 6]
    elif fault == 'feature_index':
        features = features.iloc[::-1]
    elif fault == 'duplicate_index':
        features.index = [0, 0, 2]
    elif fault == 'missing_count':
        predictions.loc[0, 'missing_features'] = 1
    elif fault == 'count_type':
        predictions['missing_features'] = ['0', '1', '1']
    elif fault == 'missing_feature':
        raw = raw.drop(columns='a')
    elif fault == 'duplicate_feature':
        features.columns = ['a', 'a']
    elif fault == 'missing_partition':
        rejected = rejected.iloc[:0]
    elif fault == 'overlap_partition':
        rejected.loc[0, ['source_row', 'record_id']] = [2, 'input-hash:2']
    elif fault == 'wrong_record_id':
        predictions.loc[0, 'record_id'] = 'input-hash:999'
    elif fault == 'wrong_source_row':
        predictions.loc[0, ['source_row', 'record_id']] = [999, 'input-hash:999']
    elif fault == 'duplicate_record_id':
        predictions.loc[1, 'record_id'] = predictions.loc[0, 'record_id']
    elif fault == 'wrong_source_hash':
        predictions.loc[0, ['source_sha256', 'record_id']] = ['other', 'other:2']
    else:
        features.loc[0, :] = np.nan
        predictions.loc[0, 'missing_features'] = 2
    with pytest.raises(ValueError):
        build_quality_view(raw, predictions, features, rejected)


@pytest.mark.parametrize('fault', ['count', 'index', 'duplicate_id', 'metadata', 'outcome'])
def test_invalid_final_demo_associations_raise_valueerror(fault):
    raw, predictions, features, rejected = inputs()
    final = predictions.copy()
    if fault == 'count':
        final = final.iloc[:1]
    elif fault == 'index':
        final = final.iloc[::-1]
    elif fault == 'duplicate_id':
        final.loc[1, ['record_id', 'source_sha256', 'source_row']] = ['input-hash:2', 'input-hash', 2]
    elif fault == 'metadata':
        final.loc[0, 'record_id'] = 'wrong:2'
    else:
        final.loc[0, 'missing_features'] = 1
    with pytest.raises(ValueError):
        build_quality_view(raw, predictions, features, rejected, final)


def test_large_all_rejected_batch_has_complete_counts_and_details_generated_per_record():
    count, fields = 10000, [f'feature_{i}' for i in range(77)]
    raw = pd.DataFrame('', index=range(count), columns=fields)
    raw.attrs['source_rows'] = list(range(2, count + 2))
    rejected = pd.DataFrame({
        'record_id': [f'large:{i}' for i in raw.attrs['source_rows']],
        'source_row': raw.attrs['source_rows'], 'source_file': 'large.csv',
        'source_sha256': 'large', 'reason': KNOWN_REJECTION,
    })
    predictions = rejected.drop(columns='reason').iloc[:0].copy()
    predictions['missing_features'] = pd.Series(dtype='int64')
    features = pd.DataFrame(columns=fields, dtype='float64')
    view = build_quality_view(raw, predictions, features, rejected)
    assert view.problem_cells == count * len(fields)
    assert view.affected_rows == view.rejected_rows == count
    assert len(view.summary) == len(fields) and len(view.records) == count
    assert view.summary['整行拒绝涉及数'].eq(count).all()
    assert len(view.details(f'large:{count + 1}')) == len(fields)
