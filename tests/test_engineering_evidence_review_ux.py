"""Evidence, progress, navigation and stale event guards for r9."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from app_types import SessionState
from app_ui_helpers import start,navigate,edit,messages,open_record,select_record_id
from app_ui_state import filtered
from liushao import reviews as store
from scripts import engineering_ci


@pytest.mark.parametrize('queue',['高异常得分','待核实','原始顺序'])
@pytest.mark.parametrize('position',[0,1])
def test_save_next_follows_filtered_queue_and_loads_persisted_review(engineered_app,queue,position):
    app=start(engineered_app);navigate(app,'检查记录')
    app.selectbox(key='_filter_queue').select(queue).run()
    state=app.session_state.ui
    order=filtered(state.result.predictions.assign(review_status='待复核'),state.filters,engineered_app[4]).record_id.tolist()
    assert len(order)>position+1
    for previous in order[:position]:store.save_review(engineered_app[2],previous,engineered_app[4]['model_version'],'确认正常','先前核对')
    app.selectbox(key='_filter_status').select('待复核').run()
    chosen,successor=order[position:position+2];select_record_id(app,chosen)
    edit(app,'确认正常','当前核对');app.button(key='save_review_next').click().run()
    assert not app.exception and app.session_state.ui.selected==successor
    assert store.history(engineered_app[2])[-1]['record_id']==chosen
    assert app.text_area(key='_note').value==''


def test_failed_save_next_preserves_selection_draft_history_and_exports(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    edit(app,'确认正常',' ');app.button(key='save_review_next').click().run()
    assert app.session_state.ui.selected==rid and app.session_state.ui.editor.note==' '
    assert store.history(r9_app[2])==[] and app.session_state.ui.exports is None


def test_save_next_disabled_at_last_record_without_preventing_regular_save(r9_app):
    app=start(r9_app);navigate(app,'检查记录')
    state=app.session_state.ui
    table=filtered(state.result.predictions.assign(review_status='待复核'),state.filters,r9_app[4])
    select_record_id(app,table.record_id.iloc[-1])
    assert app.button(key='save_review_next').disabled and not app.button(key='save_review').disabled
    edit(app,'待补证','单条依据');app.button(key='save_review').click().run()
    assert len(store.history(r9_app[2]))==1


def test_regular_save_keeps_current_record_when_next_exists(r9_app):
    app=start(r9_app);navigate(app,'检查记录');rid=app.session_state.ui.selected
    assert not app.button(key='save_review_next').disabled
    edit(app);app.button(key='save_review').click().run()
    assert app.session_state.ui.selected==rid


@pytest.mark.parametrize('key,rows,expected',[('_records_1',[0],[]),('_records_2',[-1],[]),('_records_2',[2],[]),('_records_2',[0],[('record','r')])])
def test_stale_or_invalid_table_event_never_changes_current_record(key,rows,expected):
    source=(Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8')
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='row_selected')
    calls=[];state=SessionState(table_epoch=2)
    env={'u':state,'st':SimpleNamespace(session_state={key:SimpleNamespace(selection=SimpleNamespace(rows=rows,cells=[]))}),
         'request':lambda *args:calls.append(args)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'row-event-unit','exec'),env)
    env['row_selected'](key,['r']);assert calls==expected


def test_review_progress_counts_confirmations_separately_from_missing_evidence(r9_app):
    app=start(r9_app);ids=app.session_state.ui.result.predictions.record_id.tolist()
    for rid,status in zip(ids,['确认正常','确认异常','待补证','待复核']):
        store.save_review(r9_app[2],rid,r9_app[4]['model_version'],status,'工程依据')
    app.run();assert '已确认 2 / 8' in app.get('progress')[0].proto.text


def test_evidence_card_uses_selected_feature_evidence_and_does_not_expose_labels(r9_app,monkeypatch):
    app=start(r9_app);navigate(app,'检查记录')
    state=app.session_state.ui
    payload=next(iter(app.session_state['_record_evidence_cache'].values()))
    row=state.result.predictions.set_index('record_id').loc[state.selected]
    assert payload['record_id']==state.selected
    assert payload['scores']['rf']==pytest.approx(row.rf_score)
    assert payload['total_fields']==len(r9_app[4]['config']['features'])
    assert len(payload['cards'])<=6
    assert all('Label' not in d.value.columns for d in app.dataframe)


@pytest.mark.parametrize('row',[101,102,103,104])
def test_evidence_card_explains_flags_without_overriding_main_judgment(engineered_app,row):
    app=start(engineered_app);before=app.session_state.ui.result.predictions.copy();navigate(app,'检查记录')
    open_record(app,row)
    item=before[before.source_row.eq(row)].iloc[0]
    assert f'第 {row} 行' in messages(app)
    assert '分数' in messages(app)
    pd.testing.assert_frame_equal(before,app.session_state.ui.result.predictions)
