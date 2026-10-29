import importlib.util

import numpy as np
import pandas as pd


def api():
    assert importlib.util.find_spec('scripts.audit_phase2') is not None, '第二阶段实验审查尚未实现'
    from scripts import audit_phase2
    return audit_phase2


def fixture():
    # RF errors at indices1 (false alarm) and3 (miss); 2 errors / 5 candidates.
    return pd.DataFrame({'record_id': list('abcde'), 'rf_score': [.9, .8, .6, .2, .1],
                         'rf_threshold': [.5] * 5, 'rf_anomaly': [True, True, True, False, False],
                         'disagreement': [False, True, False, True, False],
                         'near_threshold': [False, True, True, False, False]}), np.array([1, 0, 1, 1, 0])


def test_flags_report_overlap_fp_fn_and_null_zero_denominators():
    a = api()
    pred, y = fixture()
    out = a.flag_statistics(pred, y)
    assert out['disagreement']['records'] == 2
    assert out['disagreement']['errors'] == 2
    assert out['union']['records'] == 3 and out['union']['errors'] == 2
    assert out['union']['hint_precision'] == 2 / 3
    assert out['intersection']['errors'] == 1
    assert out['union']['false_positives_covered'] == 1
    assert out['union']['false_negatives_covered'] == 1
    out = a.flag_statistics(pred, pred.rf_anomaly.astype(int).to_numpy())
    assert out['union']['error_coverage'] is None
    assert out['union']['false_negative_coverage'] is None


def test_budget_uses_full_population_and_fills_short_review_queue():
    a = api()
    pred, y = fixture()
    out = a.evaluate_budgets(pred, y, budgets=[2, 5], repetitions=1000, seed=20261016)
    assert out[0]['score']['errors_found'] == 1
    assert out[0]['review']['errors_found'] == 2
    assert out[0]['random']['expected_errors'] == .8
    assert out[1]['review']['k_actual'] == 5
    assert out[1]['review']['hint_records'] == 3
    assert out[1]['review']['fill_records'] == 2
    assert out[1]['review']['errors_found'] == 2
    assert out == a.evaluate_budgets(pred, y, budgets=[2, 5], repetitions=1000, seed=20261016)


def test_orders_do_not_read_test_labels():
    a = api()
    pred, y = fixture()
    before = a.policy_orders(pred)
    changed = pred.copy()
    changed['ground_truth_evaluation_only'] = 1 - y
    after = a.policy_orders(changed)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_label_invariance_tolerates_roundoff_but_never_changed_decisions():
    a = api()
    assert hasattr(a, 'inference_invariance_checks'), '需要显式区分浮点容差与决策一致性'
    first = pd.DataFrame({'rf_score': [.2], 'lr_score': [.3], 'rf_anomaly': [False],
                          'lr_anomaly': [False], 'disagreement': [False], 'near_threshold': [False]})
    second = first.copy()
    second['rf_score'] = np.nextafter(.2, 1.)
    assert all(a.inference_invariance_checks(first, second).values())
    second['rf_anomaly'] = True
    assert not a.inference_invariance_checks(first, second)['decisions_identical']
    second = first.copy()
    second['rf_score'] = .21
    assert not a.inference_invariance_checks(first, second)['scores_within_1e_12']
