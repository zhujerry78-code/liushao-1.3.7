import numpy as np
import pandas as pd
import pytest


def test_demo_reader_passes_a_binary_stream_and_keeps_physical_rows(tmp_path, monkeypatch):
    from scripts import compare_release_models as comparison
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts/demo.csv').write_bytes(b'a,b\n1,2\n\n3,4\n')
    frame = comparison.read_demo_input()
    assert list(frame.attrs['source_rows']) == [2, 4]
    assert frame.a.tolist() == ['1', '3']


def test_truth_alignment_uses_physical_source_row_not_dataframe_position():
    from scripts.compare_release_models import aligned_truth
    pred = pd.DataFrame({'source_row': [5, 2]})
    assert aligned_truth(pred, {2: 'Benign', 5: 'Bot'}).tolist() == [1, 0]
    with pytest.raises(ValueError, match='真值'):
        aligned_truth(pred, {2: 'Benign'})


def test_test_predictions_keep_frozen_manifest_identity():
    from scripts.compare_release_models import restore_test_identity
    pred = pd.DataFrame({'record_id': ['synthetic:2', 'synthetic:3'], 'rf_score': [.1, .3]})
    meta = pd.DataFrame({'record_id': ['sha-b:42', 'sha-a:8'], 'source_file': ['b', 'a'],
        'source_sha256': ['sha-b', 'sha-a'], 'source_row': [42, 8], 'timestamp': pd.to_datetime(['2018-03-02']*2)})
    result = restore_test_identity(pred, meta)
    assert result.record_id.tolist() == meta.record_id.tolist()
    assert result.source_row.tolist() == [42, 8]
    assert result.rf_score.tolist() == [.1, .3]
    with pytest.raises(ValueError, match='分母'):
        restore_test_identity(pred.iloc[:1], meta)


def test_strata_label_support_includes_zero_categories():
    from scripts.compare_release_models import label_support
    assert label_support(np.array(['Benign']), ['Benign', 'Bot']) == {'Benign': 1, 'Bot': 0}
    assert label_support(np.array([], dtype=str), ['Benign', 'Bot']) == {'Benign': 0, 'Bot': 0}


def test_runtime_source_equivalence_accepts_only_line_endings():
    from scripts.replay_release_pair import same_source
    assert same_source(b'x = 1\r\n', b'x = 1\n')
    assert not same_source(b'x = 2\r\n', b'x = 1\n')


def test_error_review_preserves_denominators_and_false_negative_coverage():
    from scripts.compare_release_models import review_summary
    pred = pd.DataFrame({'record_id': ['a', 'b', 'c', 'd'], 'rf_score': [.2, .9, .01, .04],
        'rf_anomaly': [True, True, False, False], 'disagreement': [True, False, True, False],
        'near_threshold': [True, False, False, True], 'rf_threshold': [.12]*4})
    result = review_summary(pred, np.array([0, 1, 1, 0]), budgets=(1, 3))
    assert result['population'] == 4 and result['rf_errors'] == 2
    union = result['sets']['union']
    assert (union['records'], union['errors'], union['fp'], union['fn']) == (3, 2, 1, 1)
    assert union['error_fraction'] == 2/3 and union['fp_coverage'] == 1 and union['fn_coverage'] == 1
    assert result['budgets']['union'][1]['errors_found'] == 2
    assert result['random_expected_errors']['3'] == 1.5
