import hashlib
import io
import json
import numpy as np
import pandas as pd
import pytest
from app_quality import aligned_context
from app_evidence import neighbor_summary
from app_evidence_rules import reference_position
from app_triage import batch_interval_profile
from delivery.evaluation_resources import verify_training_descriptor
from liushao.data import read_csv_checked, clean
from liushao.model import choose_threshold, review_ranked, evidence_for
from scripts.reevaluate_frozen import bootstrap, metrics_from_counts

def test_rejected_and_skipped_rows_keep_correct_neighbors():
    raw = read_csv_checked(io.BytesIO(b'x,Src IP,Dst IP,Timestamp\n1,10.0.0.1,10.0.0.2,2026-01-01T00:00:00\n\n,10.0.0.9,10.0.0.8,2026-01-01T00:00:00\n2,10.0.0.1,10.0.0.3,2026-01-01T00:00:01\n3,10.0.0.1,10.0.0.4,2026-01-01T00:00:02\n'))
    values, meta, rejected = clean(raw, ['x'], 'f', 'test.csv')
    assert meta.source_row.tolist() == [2, 5, 6] and rejected.source_row.tolist() == [4]
    physical = meta.source_row.copy()
    meta['source_row'] = [1002, 1005, 1006]
    context = aligned_context(raw, physical, values.index)
    assert neighbor_summary(context, 1)['same_source'] == 2
    assert neighbor_summary(context, 1)['time_window']['same_source'] == 2
    with pytest.raises(ValueError):
        aligned_context(raw, meta.source_row, values.index)

@pytest.mark.parametrize('mid', [0., .001, 5., 10.])
def test_skew_and_zero_width_sides_do_not_mark_median_near(mid):
    ref = {'q01': 0., 'median': mid, 'q99': 10.}
    assert not reference_position('Flow Duration', mid, ref)['near']
    values = pd.DataFrame({'Flow Duration': [mid, 0., 10., -.1, 10.1, np.nan]})
    bundle = {'config': {'features': ['Flow Duration']}, 'reference': {'Flow Duration': ref}}
    result = batch_interval_profile(values, bundle).iloc[0]
    expected = sum(reference_position('Flow Duration', value, ref)['near'] for value in values.iloc[:, 0])
    assert result['接近边界条数'] == expected

def test_discrete_fixed_legacy_table_uses_unified_labels():
    row = pd.Series({'Protocol': 0., 'Flow Duration': 2.})
    bundle = {'config': {'features': row.index.tolist()},
              'reference': {'Protocol': {'q01': 6., 'median': 6., 'q99': 17.},
                            'Flow Duration': {'q01': 2., 'median': 2., 'q99': 2.}}}
    table = evidence_for(row, bundle).set_index('字段')
    assert table.loc['Protocol', '统计提示'] == '离散取值，需核对协议/标志/端口语义'
    assert table.loc['Flow Duration', '统计提示'] == '固定参考值'
    assert table.loc['Flow Duration', '单位'] == '微秒 (μs)'

def test_threshold_maximum_and_higher_tie_without_training():
    assert choose_threshold([0, 0], [.1, .2]) == pytest.approx(.95)
    assert choose_threshold([0, 1], [.2, .8]) == pytest.approx(.8)

def test_review_queue_disagreement_distance_id_and_empty():
    frame = pd.DataFrame({'record_id': ['c', 'b', 'a', 'z'], 'rf_score': [.5, .7, .7, .5],
                          'rf_threshold': [.5]*4, 'disagreement': [False, True, True, False],
                          'near_threshold': [True, False, False, False]})
    assert review_ranked(frame).record_id.tolist() == ['a', 'b', 'c']
    assert review_ranked(frame.assign(disagreement=False, near_threshold=False)).empty

def test_bridge_checks_original_bytes_and_model_semantics():
    original = {'model_sha256': 'm', 'model_version': 'v', 'config': {}, 'reference': {},
                'training_rows': 3, 'normal_reference_rows': 2}
    content = json.dumps(original).encode()
    release = dict(original, training_descriptor_sha256=hashlib.sha256(content).hexdigest())
    verify_training_descriptor(release, content)
    for changed in (dict(release, model_sha256='other'), dict(release, reference={'x': 1})):
        with pytest.raises(ValueError):
            verify_training_descriptor(changed, content)
    with pytest.raises(ValueError):
        verify_training_descriptor(release, content+b' ')

def test_bootstrap_preserves_block_multiplicity_and_undefined_metrics():
    assert metrics_from_counts([2, 1, 1, 3])['f1'] == pytest.approx(.75)
    assert metrics_from_counts([2, 0, 0, 0])['f1'] is None
    counts = {'rf': [[100, 0, 0, 0], [0, 0, 0, 1]], 'lr': [[100, 0, 0, 0], [0, 0, 1, 0]]}
    result = bootstrap(counts, repetitions=200)
    assert result == bootstrap(counts, repetitions=200)
    assert result['metrics']['rf']['recall']['undefined_replicates'] > 0
    assert result['metrics']['rf']['f1']['lower'] == 1.
    assert result['metrics']['lr']['recall']['upper'] == 0.


@pytest.mark.parametrize('fault', ['changed', 'missing'])
def test_all_descriptor_registrations_including_reviews_are_enforced(tmp_path, monkeypatch, fault):
    import app_model
    package = tmp_path / 'liushao'
    package.mkdir()
    content = b'# independently registered fixture\n'
    descriptor = {'source_code_sha256': {}}
    for name in ('__init__.py', 'model.py', 'data.py', 'reviews.py'):
        (package / name).write_bytes(content)
        descriptor['source_code_sha256']['liushao/'+name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(app_model.liushao, '__file__', str(package / '__init__.py'))
    assert len(app_model._runtime_context(descriptor)['source_sha256']) == 4
    if fault == 'changed':
        (package / 'reviews.py').write_bytes(b'changed')
    else:
        (package / 'reviews.py').unlink()
    with pytest.raises((ValueError, OSError)):
        app_model._runtime_context(descriptor)
