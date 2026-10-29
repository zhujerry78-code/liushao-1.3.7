from pathlib import Path
import os
from streamlit.testing.v1 import AppTest


def test_app_loads_demo_and_keeps_review_after_rerun(tmp_path, monkeypatch):
    assert Path('app.py').exists(), '应用尚未实现'
    artifacts = Path(os.environ.get('LIUSHAO_ARTIFACTS_DIR', 'artifacts'))
    if not (artifacts / 'model.joblib').exists():
        import pytest
        if 'LIUSHAO_ARTIFACTS_DIR' in os.environ:
            pytest.fail('显式指定的模型目录缺少 model.joblib，不能跳过验收')
        pytest.skip('请交接匹配模型并设置 LIUSHAO_ARTIFACTS_DIR 后运行真实模型应用测试')
    monkeypatch.setenv('LIUSHAO_REVIEW_DB', str(tmp_path / 'reviews.sqlite'))
    app = AppTest.from_file('app.py', default_timeout=45).run()
    assert not app.exception
    app.button(key='run').click().run()
    assert not app.exception
    app.button(key='nav_检查记录').click().run()
    assert app.session_state.ui.selected is not None
    app.selectbox(key='_status').select('待补证')
    app.text_area(key='_note').input('复核持久化验证')
    app.button(key='save_review').click().run()
    assert not app.exception
    app.run()
    assert app.selectbox(key='_status').value == '待补证'
    assert app.text_area(key='_note').value == '复核持久化验证'


def test_real_frozen_model_scores_ranking_exports_and_evidence():
    """Golden values computed independently with the retained 1.3.4 baseline."""
    import hashlib
    import json
    import io
    import joblib
    import pandas as pd
    import pytest
    from app_evidence import build_record_evidence
    from app_decomposition import decompose_scores
    from liushao.model import infer, ranked
    artifacts = Path(os.environ.get('LIUSHAO_ARTIFACTS_DIR', 'artifacts'))
    if not (artifacts / 'model.joblib').is_file():
        pytest.fail('真实冻结模型是本次验收的必需输入，不能跳过')
    assert hashlib.sha256((artifacts / 'model.joblib').read_bytes()).hexdigest() == '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7'
    bundle = joblib.load(artifacts / 'model.joblib')
    raw = pd.read_csv(artifacts / 'demo.csv')
    pred, features, rejected = infer(raw, bundle, 'demo', 'demo.csv')
    assert len(pred) == 1000 and rejected.empty
    assert pred.iloc[0].rf_score == pytest.approx(.9999942548086483, abs=1e-12)
    assert pred.iloc[0].lr_score == pytest.approx(.9992146187619277, abs=1e-12)
    assert pred.iloc[67].rf_score == 0.
    assert pred.iloc[777].lr_score == pytest.approx(.03114959023620625, abs=1e-12)
    assert hashlib.sha256(','.join(map(str, ranked(pred).index)).encode()).hexdigest() == 'bc5aec7f6d59e051b124a452e075010368d2c269a21644edf2f6fa07b3c69934'
    assert len(pd.read_csv(io.StringIO(pred.to_csv(index=False)))) == 1000
    for index in (0, 67, 777):
        payload = build_record_evidence(pred.loc[index], features, bundle)
        assert len(payload['cards']) <= 6 and len(payload['all_fields']) == 77
        assert payload['record_id'] == pred.loc[index, 'record_id']
        assert payload['scores']['rf'] == pytest.approx(pred.loc[index, 'rf_score'], abs=1e-12)
        assert payload['decomposition']['status'] == 'available'
        assert max(payload['decomposition']['errors'].values()) <= 1e-10
        missing_cards = [c for c in payload['all_fields'] if c['missing']]
        assert all(c['effect_kind'] == 'value_and_missing_state' and '缺失状态变化' in c['effect_note'] for c in missing_cards)
