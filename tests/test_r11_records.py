import io
import numpy as np
import pandas as pd
from app_record_tools import batch_positions, confirmed_frame, evidence_summaries
from app_ui_state import csv_bytes

def context():
    pred = pd.DataFrame(dict(record_id=['a', 'b'], source_file=['original.csv']*2,
        source_row=[101, 105], rf_anomaly=[True, False], disagreement=[False]*2, near_threshold=[False]*2))
    features = pd.DataFrame({'x': [0., np.nan], 'y': [2., 2.]})
    bundle = {'config': {'features': ['x', 'y']}, 'reference': {f: dict(q01=1., q99=3., median=2.) for f in features}}
    active = dict(name='input.csv', digest='input', key='key')
    meta = dict(model_version='m', model_sha256='model', descriptor_sha256='descriptor')
    audit = [dict(id=8, record_id='a', model_version='m', status='确认正常', note='=danger', created_at='2020-01-01T00:00:00+00:00'),
             dict(id=2, record_id='a', model_version='m', status='待补证', note='old', created_at='2030-01-01T00:00:00+00:00')]
    return active, pred, audit, meta, features, pd.Series([2, 5]), bundle

def test_confirmed_uses_max_id_and_separates_rows_and_times():
    args = context()
    frame, missing = confirmed_frame(*args, str, generated_at='2040-01-01T00:00:00+00:00')
    assert len(frame) == 1 and missing == 0
    assert frame.iloc[0]['输入CSV行号'] == 2 and frame.iloc[0]['来源行号'] == 101
    assert frame.iloc[0]['复核记录ID'] == 8 and frame.iloc[0]['复核保存时间UTC'].startswith('2020')
    assert frame.iloc[0]['结论单生成时间UTC'].startswith('2040')
    assert frame.iloc[0]['保存时证据摘要']=='无当时快照'
    assert pd.read_csv(io.BytesIO(csv_bytes(frame))).iloc[0]['备注'] == "'=danger"
    assert args[2][0]['note'] == '=danger'

def test_later_retraction_and_no_reviews_make_header_only():
    args = list(context()); args[2].append(dict(args[2][0], id=9, status='待复核'))
    assert confirmed_frame(*args, str)[0].empty
    args[2] = []
    frame, _ = confirmed_frame(*args, str)
    assert frame.empty and len(frame.columns) == 19

def test_missing_physical_mapping_is_blank_not_guessed():
    args = list(context()); args[5] = pd.Series([99, 100], index=[5, 6])
    frame, missing = confirmed_frame(*args, str)
    assert missing == 1 and frame.iloc[0]['输入CSV行号'] == ''

def test_midrank_excludes_missing_and_preserves_input():
    features = pd.DataFrame({'x': [1., 1., 3., np.nan, np.inf], 'constant': [2.]*5, 'none': [np.nan]*5})
    before = features.copy(deep=True)
    assert batch_positions(features, 0)['x'] == (100/3, 3)
    assert batch_positions(features, 0)['constant'] == (50., 5)
    assert batch_positions(features, 3)['x'] == (None, 3)
    assert batch_positions(features, 0)['none'] == (None, 0)
    assert batch_positions(pd.DataFrame({'x': [9.]}), 0)['x'] == (50., 1)
    pd.testing.assert_frame_equal(features, before)

def test_batch_summary_matches_feature_order_and_missing_rules():
    _, pred, _, _, features, _, bundle = context()
    summaries = evidence_summaries(pred, features, bundle, str)
    assert summaries == ['x：与参考数据差异较大。', 'x：数据缺失，已补齐后分析。']
