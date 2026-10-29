"""Readable overview states preserve counts, predictions and failure boundaries."""
import altair as alt
import pandas as pd
import pytest

from app_score_charts import histogram_spec
from app_ui_helpers import messages, start, upload


@pytest.mark.parametrize('scale', ['symlog', 'linear'])
def test_skewed_large_distribution_keeps_exact_counts_on_both_scales(scale):
    scores = [0.] * 42650 + [.5] + [1.] * 7349
    chart = histogram_spec(scores, .12, .1, count_scale=scale)
    alt.LayerChart.from_dict(chart, validate=True)
    bars = chart['layer'][0]
    assert sum(row['count'] for row in bars['data']['values']) == 50000
    assert [row['count'] for row in bars['data']['values'] if row['count']] == [42650, 1, 7349]
    assert bars['encoding']['y']['scale']['type'] == scale
    assert bars['encoding']['y2'] == {'datum': 0}
    if scale == 'symlog':
        assert bars['encoding']['y']['axis']['values'] == [0, 1, 10, 100, 1000, 10000, 42650]
    assert chart['layer'][2]['data']['values'] == [{'threshold': .12}]


def test_zero_progress_explains_not_started_and_scale_changes_do_not_change_analysis(r9_app):
    app = start(r9_app)
    assert '尚未开始检查' in messages(app)
    assert app.get('progress')[0].proto.text == '人工检查：已确认 0 / 8 条'
    original = app.session_state.ui.result.predictions.copy(deep=True)
    assert app.radio(key='_histogram_scale').value == '对数数量'
    app.radio(key='_histogram_scale').set_value('线性数量').run()
    assert not app.exception
    pd.testing.assert_frame_equal(original, app.session_state.ui.result.predictions)


def test_quality_summary_uses_record_counts_and_suppresses_label_contents(r9_app, monkeypatch):
    app = upload(r9_app, monkeypatch, b'a,b,Label\n1,2,PRIVATE_LABEL\n,3,Bot\nbad,,Bot\ninf,4,Benign\n')
    assert '2 条填补后可判断' in messages(app)
    assert '1 条无法判断' in messages(app)
    assert 'PRIVATE_LABEL' not in messages(app)
    app.button(key='overview_problems').click().run()
    assert not app.exception and app.selectbox(key='_quality_record')


def test_all_unjudgeable_still_has_quality_summary_and_navigation(r9_app, monkeypatch):
    app = upload(r9_app, monkeypatch, b'a,b\nbad,inf\n,\n')
    assert not app.exception and '2 条无法判断' in messages(app)
    assert not app.get('arrow_vega_lite_chart')
    assert app.button(key='overview_problems')
    assert '尚未开始检查' not in messages(app)


def test_quality_failure_never_claims_zero_problems(r9_app, monkeypatch):
    import app_service
    def failed(*args, **kwargs): raise ValueError('injected quality failure')
    monkeypatch.setattr(app_service, 'build_quality_view', failed)
    app = upload(r9_app, monkeypatch, b'a,b\n1,2\n')
    assert not app.exception and len(app.session_state.ui.result.predictions) == 1
    assert '文件问题明细暂不可用' in messages(app)
    assert '未发现缺失数据' not in messages(app)


def test_clean_quality_summary_has_explicit_zero_without_hiding_review(r9_app, monkeypatch):
    app = upload(r9_app, monkeypatch, b'a,b\n1,2\n3,4\n')
    assert '0 条填补后可判断' in messages(app) and '0 条无法判断' in messages(app)
    assert any('未发现缺失数据或无法判断的记录' in item.value for item in app.success)
    assert '尚未开始检查' in messages(app)


@pytest.mark.parametrize('status', ['待补证', '待复核'])
def test_saved_work_without_confirmation_is_not_misreported_as_unstarted(r9_app, status):
    from liushao.reviews import save_review
    app = start(r9_app)
    rid = app.session_state.ui.result.predictions.record_id.iloc[0]
    save_review(r9_app[2], rid, r9_app[4]['model_version'], status, '工程检查依据')
    app.run()
    assert '检查已开始，尚无确认结果' in messages(app)
    assert '尚未开始检查' not in messages(app)
    assert '已确认 0 / 8' in app.get('progress')[0].proto.text


def test_unsaved_current_draft_also_explains_zero_confirmations(r9_app):
    from app_ui_helpers import navigate, edit
    app = start(r9_app)
    navigate(app, '检查记录')
    edit(app)
    navigate(app, '查看结果')
    assert '检查已开始，尚无确认结果' in messages(app)
    assert '尚未开始检查' not in messages(app)
