import io
import pytest
import app_input
import app_service
from app_input import preflight_input, problem_sample, present_error
from app_ui_helpers import start, navigate, messages

def test_preflight_does_not_predict_and_analysis_parses_once(engineering_bundle, tmp_path, monkeypatch):
    bundle, _ = engineering_bundle
    content = b'a,b,Label\n1,2,PRIVATE_LABEL\n'
    parsed = []
    read = app_input.read_csv_checked
    def track(*args, **kw):
        parsed.append(1); return read(*args, **kw)
    monkeypatch.setattr(app_input, 'read_csv_checked', track)
    checked = preflight_input(content, bundle, 'model')
    copy = checked.preview_copy(); copy.iloc[0, 0] = 'bad'; copy.attrs['source_rows'][0] = 999
    monkeypatch.setattr(app_service, 'read_csv_checked', lambda *a, **k: pytest.fail('parsed twice'))
    active, result, physical, quality, _ = app_service.analyze_input(content, 'input.csv', '上传 CSV', bundle, 'model', tmp_path, preflight=checked)
    assert parsed == [1] and len(result.predictions) == 1 and physical.iloc[0] == 2
    assert checked._raw is None and result.features.iloc[0, 0] == 1
    assert 'PRIVATE_LABEL' not in str(checked.summary())

def test_preflight_rejects_different_content_or_model(engineering_bundle):
    bundle, _ = engineering_bundle
    checked = preflight_input(b'a,b\n1,2\n', bundle, 'm')
    with pytest.raises(ValueError): checked.take(b'a,b\n1,3\n', 'm')
    with pytest.raises(ValueError): checked.take(b'a,b\n1,2\n', 'different')

def test_problem_samples_have_exact_dispositions_and_physical_rows(engineering_bundle, tmp_path):
    bundle, _ = engineering_bundle
    with pytest.raises(ValueError, match='缺少必需特征: a'):
        preflight_input(problem_sample(['a', 'b'], '缺少必需列'), bundle, 'm')
    content = problem_sample(['a', 'b'], '缺值与无效记录')
    checked = preflight_input(content, bundle, 'm')
    _, result, _, quality, _ = app_service.analyze_input(content, '练习.csv', '格式练习', bundle, 'm', tmp_path, preflight=checked)
    assert len(result.predictions) == 3 and len(result.rejected) == 1
    assert quality.accepted_problem_rows == 2
    assert sorted(quality.records.input_row) == [3, 4, 5]

@pytest.mark.parametrize('content', [b'', b'a,a\n1,2\n', b'a,b\n1,2,3\n', b'a,b\n"a\nb",2\n'])
def test_structural_failures_have_action_and_original_details(content, engineering_bundle):
    with pytest.raises(Exception) as caught:
        preflight_input(content, engineering_bundle[0], 'm')
    shown = present_error(caught.value)
    assert shown.action and str(caught.value) in shown.detail

def test_sample_ui_disables_missing_columns_then_exposes_quality(r9_app):
    app = start(r9_app)
    app.button(key='change_file').click().run()
    app.radio(key='_source').set_value('格式练习').run()
    assert app.button(key='run').disabled and '缺少必需数据列' in messages(app)
    app.selectbox(key='_sample_kind').select('缺值与无效记录').run()
    app.button(key='run').click().run()
    assert not app.exception and len(app.session_state.ui.result.predictions) == 3
    navigate(app, '文件问题')
    assert '已填补后分析' in messages(app)
