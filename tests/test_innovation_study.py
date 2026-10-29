import importlib.util
import numpy as np
import pandas as pd


def api():
    assert importlib.util.find_spec('scripts.innovation_study') is not None, '创新点分析模块尚未实现'
    from scripts import innovation_study
    return innovation_study


def fixture():
    return pd.DataFrame({'record_id': list('abcde'), 'rf_score': [.9, .51, .2, .12, .11],
                         'rf_threshold': [.12] * 5, 'rf_anomaly': [True, True, True, True, False],
                         'disagreement': [False, True, False, True, False],
                         'near_threshold': [False, False, True, True, True]}), np.array([1, 1, 0, 0, 0])


def test_ablation_removes_signal_and_preserves_frozen_combined_priority():
    a = api()
    pred, _ = fixture()
    orders = a.orders(pred)
    assert orders['combined'].tolist() == [3, 1, 4, 2, 0]
    assert orders['disagreement'].tolist() == [3, 1, 0, 2, 4]
    assert orders['near'].tolist() == [3, 4, 2, 0, 1]
    assert orders['margin'].tolist() == [3, 4, 2, 1, 0]
    assert orders['vote_uncertainty'][0] == 1
    changed = pred.assign(ground_truth_evaluation_only=1)
    for name, order in orders.items():
        np.testing.assert_array_equal(order, a.orders(changed)[name])


def test_workload_denominators_and_random_reference_are_explicit():
    a = api()
    pred, y = fixture()
    result = a.workload(pred, y)
    assert result['combined']['records'] == 4
    assert result['combined']['errors'] == 2
    assert result['combined']['error_precision'] == .5
    assert result['combined']['workload_fraction'] == .8
    assert result['combined']['checks_per_error'] == 2
    stats = a.random_reference(pred, y, 2)
    assert stats['all']['expected_errors'] == .8
    assert stats['alert']['population'] == 4
    assert stats['alert']['expected_errors'] == 1.


def test_block_bootstrap_preserves_multiplicity_and_pairing():
    a = api()
    pred, y = fixture()
    groups = np.array(['x','x','y','y','y'])
    indices = a.block_sample_indices(groups, ['x','x'])
    assert indices.tolist() == [0, 1, 0, 1]
    result = a.bootstrap(pred, y, groups, budgets=[2], repeats=20, seed=17)
    assert result == a.bootstrap(pred, y, groups, budgets=[2], repeats=20, seed=17)
    assert result['replicates'] == 20
    assert result['cluster_count'] == 2
    assert result['budgets']['2']['combined_minus_margin']['n_valid'] == 20
    assert 'combined_minus_vote_uncertainty' in result['budgets']['2'], '必须配对报告已纳入的常规不确定性基线'
    assert result['budgets']['2']['combined_minus_vote_uncertainty']['p025'] == 0
    assert result['budgets']['2']['combined_minus_vote_uncertainty']['p975'] == 0
