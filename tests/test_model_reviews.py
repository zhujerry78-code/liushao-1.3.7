import importlib
import numpy as np
import pandas as pd
import pytest


def module(name):
    assert importlib.util.find_spec('liushao.' + name) is not None, name + ' 尚未实现'
    return importlib.import_module('liushao.' + name)


def test_prediction_ignores_labels_and_preserves_invalid_rows():
    m = module('model')
    x = pd.DataFrame({'a': [0., 0.1, 0.2, 0.3, 5., 6., 7., 8.], 'b': [1.] * 8})
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bundle = m.fit_bundle(x, y, x, y, 'fixture', n_estimators=8)
    raw = pd.DataFrame({'a': [0., 8., 'bad'], 'b': [1., 1., 'bad'], 'Label': ['Bot'] * 3})
    first, evidence, rejected = m.infer(raw, bundle, 'hash', 'sample.csv')
    raw['Label'] = 'Benign'
    second, _, _ = m.infer(raw, bundle, 'hash', 'sample.csv')
    pd.testing.assert_frame_equal(first, second)
    assert first.rf_score.iloc[1] > first.rf_score.iloc[0]
    assert len(rejected) == 1
    assert 'Label' not in evidence
    with pytest.raises(ValueError, match='缺少'):
        m.infer(raw.drop(columns='a'), bundle, 'hash', 'sample.csv')


def test_ordering_and_metrics_use_actual_budget_and_tie_ids():
    m = module('model')
    frame = pd.DataFrame({'record_id': ['b', 'a', 'c'], 'rf_score': [.9, .9, .1]})
    assert m.ranked(frame).record_id.tolist() == ['a', 'b', 'c']
    out = m.ranking_metrics([1, 0, 1], [1, 0, 2], budgets=[2, 20])
    assert out[0]['precision_at_k'] == .5
    assert out[1]['k_actual'] == 3
    assert out[1]['recall_at_k'] == 1


def test_review_history_survives_reopen_and_is_scoped_by_model(tmp_path):
    r = module('reviews')
    path = tmp_path / 'reviews.sqlite'
    r.save_review(path, 'record', 'v1', '确认异常', 'first')
    r.save_review(path, 'record', 'v1', '待补证', 'second')
    assert r.current_reviews(path, 'v1')['record']['status'] == '待补证'
    assert r.current_reviews(path, 'v2') == {}
    assert len(r.history(path, 'record', 'v1')) == 2
    with pytest.raises(ValueError):
        r.save_review(path, 'record', 'v1', 'invalid', '')


def test_imputation_and_reference_are_fit_only_on_training():
    m = module('model')
    x = pd.DataFrame({'a': [0., 2., 4., 6.], 'b': [1., 2., 3., 4.]})
    val = pd.DataFrame({'a': [1000., 2000.], 'b': [1000., 2000.]})
    bundle = m.fit_bundle(x, [0, 0, 1, 1], val, [0, 1], 'fixture', n_estimators=8)
    assert bundle['rf'].named_steps['impute'].statistics_[0] == 3
    assert bundle['reference']['a']['median'] == 1
    assert bundle['reference']['a']['q99'] < 2
    reference_before = dict(bundle['reference']['a'])
    all_invalid = pd.DataFrame({'a': ['bad'], 'b': [np.inf]})
    pred, values, rejected = m.infer(all_invalid, bundle, 'hash', 'empty.csv')
    assert pred.empty and values.empty and len(rejected) == 1
    assert bundle['reference']['a'] == reference_before


def test_evidence_uses_actual_values_and_explicit_missing_state():
    m = module('model')
    assert hasattr(m, 'evidence_for')
    bundle = {'config': {'features': ['a', 'b']}, 'reference': {
        'a': {'q01': 0, 'median': 1, 'q99': 2}, 'b': {'q01': 0, 'median': 1, 'q99': 2}}}
    result = m.evidence_for(pd.Series({'a': 10., 'b': np.nan}), bundle)
    assert result.iloc[0]['原值'] == 10.
    assert result.iloc[0]['统计提示'] == '高于训练正常 P99'
    assert result.iloc[1]['统计提示'] == '原始缺失'
