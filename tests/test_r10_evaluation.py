"""Engineered evaluation fixtures test binding and UI behavior, not model accuracy."""
import copy
import hashlib
import json

import pytest
from streamlit.testing.v1 import AppTest

from app_evaluation import STRATEGIES, read_summary, validate_summary, _parse_summary
from app_ui_helpers import messages


def metrics(tn, fp, fn, tp, threshold):
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, normal_support=tn+fp, attack_support=fn+tp,
                precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0, threshold=threshold)


@pytest.fixture
def evaluation_data(r9_app):
    bundle = r9_app[4]
    def group(tn, fp, tp):
        return dict(records=tn+fp+tp, class_support={'Benign': tn+fp, 'Bot': tp},
                    metrics={k: metrics(tn, fp, 0, tp, bundle['config'][k+'_threshold']) for k in ('rf', 'lr')})
    held = group(7, 1, 2)
    held['evaluated_at'] = '2026-01-01T00:00:00Z'
    info = {k: bundle[k] for k in ('training_rows', 'normal_reference_rows')}
    info.update({k: bundle['config'][k] for k in ('features', 'rf_threshold', 'lr_threshold', 'near_margin')})
    return {'format': 1,
            'identity': {'model_sha256': hashlib.sha256((r9_app[1]/'model.joblib').read_bytes()).hexdigest(),
                         'descriptor_sha256': hashlib.sha256((r9_app[1]/'model.json').read_bytes()).hexdigest(),
                         'data_version': 'a'*64, 'model_version': bundle['model_version']},
            'model_info': info, 'held_out': held,
            'attack_ranking': dict(k_actual=10, attacks_found=2, total_attacks=2, precision_at_k=.2, recall_at_k=1),
            'external': {'evaluated_at': '2026-01-02', 'all': group(7, 1, 2),
                         'any_overlap': group(2, 0, 0), 'no_overlap': group(5, 1, 2),
                         'slowloris': dict(detected=1, support=2), 'quality': {}},
            'error_review': {'total_errors': 1, 'budgets': [dict(strategy=s, k=k, k_actual=10,
                             population=10, errors_found=1, total_errors=1, error_precision_at_k=.1)
                             for s in STRATEGIES for k in (20, 50, 100)]},
            'split': dict(counts=dict(train=info['training_rows'], test=10), chronological_holdout=False,
                          source_disjoint_holdout=False, session_disjointness_verifiable=False),
            'sources': [dict(path=f'fixture/{i}.json', sha256='b'*64) for i in range(6)]}


def write_summary(path, data):
    path.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')


def test_summary_cache_cannot_hide_same_size_replacement_deletion_or_mutation(tmp_path, evaluation_data):
    path = tmp_path/'summary.json'
    write_summary(path, evaluation_data)
    original = path.read_bytes()
    result = read_summary(path)
    result['held_out']['records'] = 999
    assert read_summary(path)['held_out']['records'] == 10
    assert _parse_summary.cache_info().hits > 0
    stat = path.stat()
    path.write_bytes(original.replace(b'"records": 10', b'"records": 11', 1))
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError): read_summary(path)
    path.unlink()
    with pytest.raises(FileNotFoundError): read_summary(path)


@pytest.mark.parametrize('damage', ['identity', 'confusion', 'ranking', 'baseline', 'split', 'nan'])
def test_summary_rejects_wrong_identity_or_misleading_numbers(evaluation_data, damage):
    data = copy.deepcopy(evaluation_data)
    if damage == 'identity': data['identity']['model_sha256'] = 'c'*64
    if damage == 'confusion': data['held_out']['metrics']['rf']['fp'] = 7
    if damage == 'ranking': data['attack_ranking']['precision_at_k'] = 1.0
    if damage == 'baseline': data['error_review']['budgets'] = data['error_review']['budgets'][3:]
    if damage == 'split': data['split']['chronological_holdout'] = True
    if damage == 'nan': data['held_out']['metrics']['rf']['f1'] = float('nan')
    with pytest.raises(ValueError):
        validate_summary(data, evaluation_data['identity']['model_sha256'])


def test_evaluation_page_is_available_without_input_and_hides_unbound_metrics(r9_app, tmp_path, monkeypatch, evaluation_data):
    path = tmp_path/'summary.json'
    write_summary(path, evaluation_data)
    monkeypatch.setenv('LIUSHAO_EVALUATION_SUMMARY', str(path))
    app = AppTest.from_file(str(r9_app[0]), default_timeout=45).run()
    app.button(key='nav_操作说明').click().run()
    # Fixture data_version is deliberately not the current model: no performance tables.
    assert not app.exception and app.session_state.ui.result is None
    assert '评测资料暂不可用' in messages(app) and len(app.dataframe) == 0
    evaluation_data['identity']['data_version'] = r9_app[4]['config']['data_version']
    # The engineered model uses an arbitrary textual data_version, unlike production SHA.
    # Test the real-model page separately; missing/corrupt state must remain accessible here.
    path.write_text('{broken', encoding='utf-8')
    app.run()
    assert not app.exception and len(app.dataframe) == 0
    path.unlink()
    app.run()
    assert '未附评测资料' in messages(app)
    (r9_app[1]/'model.joblib').unlink()
    app.run()
    assert not app.exception and '当前模型未核验' in messages(app)


def test_evaluation_captions_format_both_original_times(r9_app, monkeypatch, evaluation_data):
    import app_evaluation
    from types import SimpleNamespace
    evaluation_data['identity']['data_version'] = r9_app[4]['config']['data_version']
    evaluation_data['held_out']['evaluated_at'] = '2026-10-16T09:40:10.905670+00:00'
    evaluation_data['external']['evaluated_at'] = '2026-10-17T01:40:10.123456+08:00'
    monkeypatch.setattr(app_evaluation, 'read_summary', lambda *args: copy.deepcopy(evaluation_data))
    app = AppTest.from_string('''
import streamlit as st
from types import SimpleNamespace
from app_evaluation import render_evaluation
render_evaluation('unused', st.session_state['verified'],
                  st.session_state['bundle'])
''', default_timeout=45)
    app.session_state['bundle'] = r9_app[4]
    app.session_state['verified'] = SimpleNamespace(
        model_sha256=evaluation_data['identity']['model_sha256'],
        descriptor_sha256=evaluation_data['identity']['descriptor_sha256'],
        descriptor=(r9_app[1]/'model.json').read_text(encoding='utf-8'))
    app.run()
    assert not app.exception
    captions = '\n'.join(element.value for element in app.caption)
    assert '2026年10月16日 09:40:10（UTC）' in captions
    assert '2026年10月16日 17:40:10（UTC）' in captions
    assert '采集日早于训练日' in captions
    assert '905670' not in captions and '123456' not in captions
    assert 'T09:40' not in captions and '+08:00' not in captions
