"""Acceptance of the v2 evidence semantics using independent engineering fixtures."""
import copy
import json
import sqlite3
import numpy as np
import pandas as pd
import pytest
from app_decomposition import decompose_scores
from app_evidence_rules import reference_position, select_fields, direction_difference
from app_quick_search import promote_matches
from app_evidence import build_record_evidence
from liushao.model import fit_bundle, infer
from liushao import reviews
from delivery.migrate_reviews import migrate


@pytest.mark.parametrize('tree_count',[3,11])
def test_accounting_uses_actual_tree_count_and_fitted_missing_indicators(tree_count):
    x=pd.DataFrame({'a':[0.,np.nan,.2,.3,5.,6.,7.,8.], 'b':[np.nan,1.,2.,3.,1.,2.,3.,4.]})
    y=np.array([0]*4+[1]*4)
    bundle=fit_bundle(x,y,x,y,'engineering accounting fixture',n_estimators=tree_count)
    before=copy.deepcopy(bundle['rf'].named_steps['classifier'].estimators_[0].tree_.value)
    for i in range(len(x)):
        d=decompose_scores(x.iloc[[i]],bundle)
        assert d['status']=='available' and d['rf']['tree_count']==tree_count
        assert max(d['errors'].values())<=1e-10
        assert d['rf']['base']+sum(r['contribution'] for r in d['rf']['fields'])==pytest.approx(bundle['rf'].predict_proba(x.iloc[[i]])[0,1],abs=1e-10)
        assert {p['kind'] for r in d['rf']['fields'] for p in r['parts']}=={'value','missing_indicator'}
        assert abs(d['lr']['intercept']+sum(r['term'] for r in d['lr']['fields'])-d['lr']['actual_logit'])<=1e-10
    np.testing.assert_array_equal(before,bundle['rf'].named_steps['classifier'].estimators_[0].tree_.value)


def test_incompatible_decomposition_reports_unavailable_without_fake_terms():
    d=decompose_scores(pd.DataFrame({'a':[1]}),{'config':{'features':['a']},'rf':object(),'lr':object()})
    assert d['status']=='unavailable' and 'rf' not in d


@pytest.mark.parametrize('value,status,near,outside',[(0,'接近参考边界（区间内）',True,False),(100,'接近参考边界（区间内）',True,False),(-1,'低于训练正常 P01',False,True),(50,'参考区间内',False,False)])
def test_reference_endpoints_are_near_and_not_outside(value,status,near,outside):
    result=reference_position('Flow Duration',value,dict(q01=0,median=50,q99=100))
    assert (result['status'],result['near'],result['outside'])==(status,near,outside)


def test_reference_constant_missing_and_discrete_do_not_fake_continuous_distances():
    assert reference_position('Flow Duration',3,dict(q01=2,median=2,q99=2))['status']=='偏离固定参考值'
    assert not reference_position('Flow Duration',2,dict(q01=2,median=2,q99=2))['near']
    assert not reference_position('Protocol',6,dict(q01=6,median=6,q99=17))['near']
    assert reference_position('Flow Duration',None,{})['status']=='原始缺失'
    assert not reference_position('Flow Duration',1,dict(q01=9,median=2,q99=3))['valid']


def test_small_opposite_response_is_visible_without_practical_conflict_claim():
    assert direction_difference(-.1,.000002)
    assert not direction_difference(-.1,.0000002)


def signal(index,rf=0.,lr=0.,missing=False,valid=True,outside=False):
    return dict(index=index,field=str(index),rf_delta=rf,lr_delta=lr,missing=missing,
        significant=abs(rf)>=.01,direction_difference=direction_difference(rf,lr),
        position=dict(valid=valid,outside=outside,fixed=False,distance=float(index)))


def test_selection_reserves_quality_direction_counterevidence_and_never_fills():
    signals=[signal(0,missing=True),signal(1,-.02,.000002),signal(2,.5,.1),
             signal(3,-.8,-.1),signal(4,-.4,-.1),signal(5,-.3,-.1),signal(6,-.2,-.1),signal(7)]
    chosen,reasons=select_fields(signals,1)
    assert len(chosen)==6 and len(set(chosen))==6
    assert chosen[:3]==[0,1,2] and 3 in chosen and 7 not in chosen
    assert select_fields([signal(0),signal(1)],1)==([], {})
    assert len(select_fields([signal(0,-.2,.1)],1)[0])==1


def test_search_promotes_cross_page_literal_matches_and_keeps_all_other_records():
    table=pd.DataFrame({'source_row':list(range(1,151))+[169,1690],'record_id':[str(i) for i in range(152)],
        'rf_anomaly':[True]*152,'review_status':['待复核']*152})
    for keyword in ('169','第169行','第 １６９ 行'):
        result,mask=promote_matches(table,keyword)
        assert len(result)==152 and result.iloc[0].source_row==169 and mask.sum()==1
        assert result.iloc[1:].record_id.tolist()==table.loc[~mask].record_id.tolist()
    result,mask=promote_matches(table,'[.*]')
    pd.testing.assert_frame_equal(result,table)
    assert not mask.any()
    pd.testing.assert_frame_equal(promote_matches(table,'')[0],table)


def test_v2_snapshot_limits_and_unknown_schema_are_rejected_before_write(tmp_path,engineering_bundle):
    bundle,raw=engineering_bundle
    pred,features,_=infer(raw,bundle,'a'*64,'engineering.csv')
    record=pred.iloc[0]
    payload=build_record_evidence(record,features,bundle,meta=dict(input_sha256='a'*64,model_sha256='b'*64,descriptor_sha256='c'*64))
    assert payload['schema_version']==2 and not payload['assessment']['sufficient']
    assert payload['selection']['parameters']['practical_conflict']['status']=='uncalibrated'
    db=tmp_path/'history.sqlite'
    reviews.save_review(db,record.record_id,bundle['model_version'],'待补证','工程接口记录',evidence_snapshot=payload)
    with sqlite3.connect(db) as conn:
        assert reviews.schema_kind(conn)=='v2'
        assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]
    bad=copy.deepcopy(payload);bad['cards']=[dict(field=str(i),reference={},rf={},lr={}) for i in range(7)]
    with pytest.raises(ValueError): reviews.save_review(db,record.record_id,bundle['model_version'],'待补证','test',evidence_snapshot=bad)
    with sqlite3.connect(tmp_path/'unknown.sqlite') as conn:conn.execute('CREATE TABLE surprise(x)')
    with pytest.raises(ValueError,match='未知'):reviews.connect(tmp_path/'unknown.sqlite')


def test_working_schema_migration_is_readonly_atomic_and_preserves_snapshot(tmp_path):
    from test_r13_snapshot import evidence
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir();state=old/'demo-state';state.mkdir()
    db=state/'reviews.sqlite';p=evidence();raw=reviews._snapshot_json(p,'record','synthetic-model')
    import hashlib
    digest=hashlib.sha256(raw.encode()).hexdigest()
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE reviews(id INTEGER PRIMARY KEY AUTOINCREMENT,record_id TEXT NOT NULL,model_version TEXT NOT NULL,status TEXT NOT NULL,note TEXT NOT NULL,created_at TEXT NOT NULL)')
        conn.execute("INSERT INTO reviews VALUES(9,'record','synthetic-model','待补证','工程记录','2020-01-01T00:00:00+00:00')")
        conn.execute('CREATE TABLE review_evidence(review_id INTEGER PRIMARY KEY,evidence_json TEXT NOT NULL,evidence_sha256 TEXT NOT NULL,FOREIGN KEY(review_id) REFERENCES reviews(id))')
        conn.execute('INSERT INTO review_evidence VALUES(?,?,?)',(9,raw,digest))
    before=db.read_bytes();migrate(old,new,checker=lambda _:False)
    assert db.read_bytes()==before
    saved=reviews.history(new/'demo-state/reviews.sqlite')[0]
    assert saved['id']==9 and saved['evidence_snapshot']==p and saved['evidence_sha256']==digest
    with pytest.raises(FileExistsError):migrate(old,new,checker=lambda _:False)


from test_r13_evidence import local_case, LinearFixture


def test_zero_candidate_and_lr_only_cases_never_fill_main_cards(local_case):
    record,features,bundle=local_case
    bundle['rf']=LinearFixture([0.]*6)
    bundle['lr']=LinearFixture([.05,0,0,0,0,0])
    record['rf_score']=.5;record['lr_score']=.55
    payload=build_record_evidence(record,features,bundle)
    assert payload['cards']==[] and payload['joint_test']['count']==0
    assert payload['stability']['passed'] is None
    assert payload['selection']['lr_only'][0]['field']=='Fwd Pkt Len Max'
    assert not payload['assessment']['sufficient']


def test_fixed_support_has_partial_stability_when_a_flag_is_in_support(local_case):
    record,features,bundle=local_case
    bundle['rf']=LinearFixture([.1,-.25,-.2,0,0,0])
    bundle['lr']=LinearFixture([.1,-.25,-.2,0,0,0])
    record[['rf_score','lr_score']]=.15
    payload=build_record_evidence(record,features,bundle)
    assert payload['joint_test']['fields']==['PSH Flag Cnt','Flow Duration']
    assert payload['stability']['coverage']==1 and payload['stability']['support_count']==2
    assert not payload['stability']['full_coverage'] and '部分' in payload['stability']['status']
    for version in payload['joint_test']['versions']:
        assert version['rf']['after']>record.rf_score



def test_predictions_export_keeps_physical_row_mapping_after_rejection_and_does_not_mutate():
    import io
    from app_ui_state import export_snapshot
    pred=pd.DataFrame({'record_id':['a','c'],'source_file':['original.csv']*2,'source_row':[217809,609344],
        'rf_anomaly':[True,False]},index=[0,2])
    before=pred.copy(deep=True)
    active=dict(name='input.csv',digest='a'*64,key='input')
    physical=pd.Series([2,4],index=pred.index)
    exported=export_snapshot(active,pred,[],dict(model_version='m'),True,physical_rows=physical)
    result=pd.read_csv(io.BytesIO(exported.predictions))
    assert result.input_csv_row.tolist()==[2,4] and result.source_row.tolist()==[217809,609344]
    pd.testing.assert_frame_equal(pred,before)
    with pytest.raises(ValueError,match='行号映射'):
        export_snapshot(active,pred,[],dict(model_version='m'),True,physical_rows=pd.Series([2,1],index=pred.index))
