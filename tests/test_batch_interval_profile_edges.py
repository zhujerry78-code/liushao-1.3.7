"""Boundary cases for batch reference presentation, independent of model scores."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from app_triage import batch_interval_profile


def profile(values, reference, name='Flow Duration'):
    frame = pd.DataFrame({name: values})
    bundle = {'config': {'features': [name]}, 'reference': {name: reference}}
    before_frame, before_bundle = frame.copy(deep=True), deepcopy(bundle)
    row = batch_interval_profile(frame, bundle, field_name_fn=str).iloc[0]
    pd.testing.assert_frame_equal(frame, before_frame)
    assert bundle == before_bundle
    return row


def test_numeric_reference_strings_are_parsed():
    row = profile([0, 50, 100, 201, np.nan], {'q01': '0', 'median': '50', 'q99': '100'})
    assert row['参考类型'] == '连续区间'
    assert row['较大偏离条数'] == 1
    assert row['接近边界条数'] == 2
    assert row['区间核心内条数'] == 1
    assert row['原始缺失条数'] == 1
    assert row['本批越界率 (%)'] == 25


@pytest.mark.parametrize('reference', [
    {'q01': 'invalid', 'median': 'unknown', 'q99': 100},
    {'q01': 0, 'median': float('inf'), 'q99': 100},
    {'q01': 100, 'median': 50, 'q99': 0},
    {'q01': -1e308, 'median': 0, 'q99': 1e308},
    {'q01': True, 'median': 50, 'q99': 100},
])
def test_unusable_references_stay_unknown_without_crashing(reference):
    row = profile([1, 2, np.nan], reference)
    assert row['参考类型'] == '参考不足'
    assert row['参考不足条数'] == 2
    assert row['区间核心内条数'] == 0
    assert pd.isna(row['本批越界率 (%)'])


@pytest.mark.parametrize('values, message', [([np.nan, np.nan], '全部缺失'), ([], '没有记录')])
def test_no_observations_never_claim_fixed_value_agreement(values, message):
    row = profile(values, {'q01': 7, 'median': 7, 'q99': 7})
    assert message in row['区间取值态势提示']
    assert '一致' not in row['区间取值态势提示']
    assert row['与固定参考一致条数'] == 0
    assert pd.isna(row['本批越界率 (%)'])


def test_fixed_and_discrete_observations_have_separate_counts():
    fixed = profile([7, 7, 8, np.nan], {'q01': 7, 'median': 7, 'q99': 7})
    assert fixed['与固定参考一致条数'] == 2
    assert fixed['固定参考不同条数'] == 1
    assert fixed['一般偏离条数'] == 0
    assert fixed['参考不足条数'] == 0
    assert fixed['区间核心内条数'] == 0
    discrete = profile([6, 17, np.nan], {'q01': 6, 'median': 6, 'q99': 17}, 'Protocol')
    assert discrete['离散字段未比较条数'] == 2
    assert discrete['参考不足条数'] == 0
    assert discrete['与固定参考一致条数'] == 0
    assert pd.isna(discrete['本批越界率 (%)'])


def test_neutral_categories_partition_every_observation():
    row = profile([-101, -1, 0, 4, 50, 96, 100, 101, 201, np.nan],
                  {'q01': 0, 'median': 50, 'q99': 100})
    columns = ['较大偏离条数', '一般偏离条数', '接近边界条数', '区间核心内条数',
               '参考不足条数', '离散字段未比较条数', '与固定参考一致条数',
               '固定参考不同条数', '原始缺失条数']
    assert sum(row[c] for c in columns) == 10
    assert row['有效观测条数'] == 9
    assert not any(word in row['区间取值态势提示'] for word in ['高风险', '严重', '分布偏移'])
