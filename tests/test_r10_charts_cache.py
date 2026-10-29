"""Score boundaries and hot-cache failure detection."""
import json
import os

import altair as alt
import pytest

import app_model
from app_score_charts import histogram_rows, histogram_spec, ruler_spec, _histogram
from app_ui_helpers import start, messages


@pytest.mark.parametrize('scores', [[], [0], [1], [0, .05, .5, .95, 1], [.12]*100])
def test_histogram_includes_all_valid_scores_and_one(scores):
    rows = histogram_rows(scores)
    assert len(rows) == 20 and sum(row['count'] for row in rows) == len(scores)
    if 1 in scores: assert rows[-1]['count'] >= scores.count(1)
    if 0 in scores: assert rows[0]['count'] >= scores.count(0)
    rows[0]['count'] = 999
    assert sum(row['count'] for row in histogram_rows(scores)) == len(scores)


@pytest.mark.parametrize('score', [-.01, 1.01, float('nan'), float('inf')])
def test_histogram_rejects_invalid_scores(score):
    with pytest.raises(ValueError): histogram_rows([score])


def test_specs_validate_locally_and_keep_independent_thresholds():
    spec = histogram_spec([0., .12, 1.], .12, .1)
    alt.LayerChart.from_dict(spec, validate=True)
    ruler = ruler_spec(.2, .8, dict(rf_threshold=.12, lr_threshold=.84, near_margin=.1))
    alt.LayerChart.from_dict(ruler, validate=True)
    assert [v['threshold'] for v in ruler['layer'][0]['data']['values']] == [.12, .84]


def test_score_hover_labels_never_fall_back_to_internal_field_names():
    charts = [histogram_spec([0., .12, 1.], .12, .1),
              ruler_spec(.2, .8, dict(rf_threshold=.12, lr_threshold=.84, near_margin=.1))]
    for chart in charts:
        for layer in chart['layer']:
            # An omitted tooltip enables Vega's automatic internal-field labels.
            tooltip = layer['encoding']['tooltip']
            if tooltip is not None:
                assert all(item.get('title') and item['title'] != item['field'] for item in tooltip)
        interval = chart['layer'][1]
        assert [item['title'] for item in interval['encoding']['tooltip']] == ['范围下限', '范围上限']
        assert interval['data']['values'][0]['lo'] == pytest.approx(.02)
        assert interval['data']['values'][0]['hi'] == pytest.approx(.22)


def test_descriptor_cache_returns_copies_and_still_rejects_same_size_model_replacement(r9_app):
    path = r9_app[1]/'model.json'
    content = path.read_text(encoding='utf-8')
    first = app_model.parse_descriptor(content)
    first['config']['features'].clear()
    assert app_model.parse_descriptor(content)['config']['features']
    assert app_model._parsed_descriptor.cache_info().hits > 0
    app = start(r9_app)
    model = r9_app[1]/'model.joblib'
    stat = model.stat()
    data = bytearray(model.read_bytes()); data[len(data)//2] ^= 1
    model.write_bytes(data)
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    app.run()
    assert not app.exception and app.session_state.ui.result is None
    assert '尚未就绪' in messages(app)


def test_hot_descriptor_cache_detects_same_size_descriptor_replacement(r9_app):
    app = start(r9_app)
    path = r9_app[1]/'model.json'
    stat = path.stat()
    content = path.read_bytes()
    assert b'"training_rows": 8' in content
    changed = content.replace(b'"training_rows": 8', b'"training_rows": 9')
    assert len(changed) == len(content)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    app.run()
    assert not app.exception and app.session_state.ui.result is None
