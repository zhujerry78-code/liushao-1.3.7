"""Synthetic engineering records verify frozen evidence, not human/model validity."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import threading

import pandas as pd
import pytest

from app_record_tools import confirmed_frame
from app_ui_state import export_snapshot
from delivery.migrate_reviews import migrate
from liushao import reviews


def evidence(score=.8):
    return dict(method_version='synthetic-engineering-method-v1', record_id='record',
        model_version='synthetic-model', input_sha256='a'*64, model_sha256='b'*64,
        descriptor_sha256='c'*64, input_name='saved-input.csv', input_csv_row=3,
        identity=dict(source_file='source.csv', source_row=101, source_sha256='d'*64),
        scores=dict(rf=score, lr=.1, rf_threshold=.5, lr_threshold=.6,
                    rf_anomaly=score >= .5, lr_anomaly=False),
        cards=[dict(field='duration', name='工程持续时间', unit='微秒', value=100.,
                    reference=dict(p01=1., p50=2., p99=3.),
                    rf=dict(before=score, after=.2, delta=score-.2),
                    lr=dict(before=.1, after=.08, delta=.02))],
        checklist=['工程记录：核对原始输入，不代表真人复核'], assessment='synthetic')


def save(db, snapshot=None, **kwargs):
    reviews.save_review(db, 'record', 'synthetic-model', '确认正常', '合成工程判断依据',
                        evidence_snapshot=snapshot, **kwargs)


def export_args(audit):
    active = dict(name='current-input.csv', digest='e'*64, key='current-key')
    pred = pd.DataFrame(dict(record_id=['record'], source_file=['current.csv'],
                             source_row=[999], rf_anomaly=[False]))
    meta = dict(model_version='synthetic-model', model_sha256='f'*64, descriptor_sha256='0'*64)
    return active, pred, audit, meta


def test_saved_snapshot_is_immutable_json_plus_hash_and_reopens(tmp_path):
    db = tmp_path/'history.sqlite'
    snapshot = evidence()
    expected = copy.deepcopy(snapshot)
    save(db, snapshot, expected_review_id=0)
    snapshot['scores']['rf'] = .01
    snapshot['cards'].clear()
    audit = reviews.history(db, 'record', 'synthetic-model')
    assert len(audit) == 1
    assert audit[0]['evidence_snapshot'] == expected
    assert audit[0]['evidence_snapshot_status'] == 'saved'
    assert reviews.current_reviews(db, 'synthetic-model')['record'] == audit[0]
    with sqlite3.connect(db) as connection:
        raw, digest = connection.execute('SELECT payload_json,payload_sha256 FROM review_evidence').fetchone()
        assert hashlib.sha256(raw.encode('utf-8')).hexdigest() == digest == audit[0]['evidence_sha256']
        assert json.loads(raw) == expected
        assert [r[1] for r in connection.execute('PRAGMA table_info(reviews)')] == [
            'id', 'record_id', 'model_version', 'status', 'note', 'created_at']
    assert 'current-input.csv' not in raw


def test_original_database_is_upgraded_without_fabricating_snapshots(tmp_path):
    db = tmp_path/'old.sqlite'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, model_version TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)')
        connection.execute("INSERT INTO reviews VALUES(4,'record','synthetic-model','待补证','原有备注','2020-01-01T00:00:00+00:00')")
    old = reviews.history(db)[0]
    assert old['note'] == '原有备注' and old['id'] == 4
    assert old['evidence_snapshot'] is None and old['evidence_snapshot_label'] == '无当时快照'
    save(db, evidence(), expected_review_id=4)
    assert [r['id'] for r in reviews.history(db)] == [4, 5]
    assert reviews.history(db)[0] == old


@pytest.mark.parametrize('damage', ['nan', 'record', 'model', 'hash', 'large', 'type', 'many', 'score_bounds'])
def test_invalid_snapshot_rejected_before_creating_database(tmp_path, damage):
    snapshot = evidence()
    if damage == 'nan': snapshot['scores']['rf'] = float('nan')
    if damage == 'record': snapshot['record_id'] = 'another'
    if damage == 'model': snapshot['model_version'] = 'another'
    if damage == 'hash': snapshot['input_sha256'] = 'not-a-sha'
    if damage == 'large': snapshot['unexpected_raw_table'] = 'x' * (256*1024)
    if damage == 'type': snapshot['scores'] = object()
    if damage == 'many': snapshot['cards'] *= 13
    if damage == 'score_bounds': snapshot['scores']['rf'] = 1.01
    db = tmp_path/'absent'/'reviews.sqlite'
    with pytest.raises((ValueError, TypeError)):
        save(db, snapshot)
    assert not db.parent.exists()


def test_snapshot_failure_rolls_back_review_in_same_transaction(tmp_path):
    db = tmp_path/'atomic.sqlite'
    save(db, evidence())
    before = reviews.history(db)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TRIGGER fail_evidence BEFORE INSERT ON review_evidence BEGIN SELECT RAISE(ABORT, 'injected snapshot failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='injected snapshot failure'):
        save(db, evidence(.9), expected_review_id=before[0]['id'])
    assert reviews.history(db) == before
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT COUNT(*) FROM review_evidence').fetchone()[0] == 1


def test_concurrent_editors_only_append_one_review_and_matching_snapshot(tmp_path):
    db = tmp_path/'concurrent.sqlite'
    save(db, evidence(.5))
    baseline = reviews.history(db)[-1]['id']
    barrier = threading.Barrier(2)
    def writer(score):
        barrier.wait(timeout=10)
        try:
            save(db, evidence(score), expected_review_id=baseline)
            return score
        except reviews.ReviewConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(writer, [.6, .9]))
    saved = [value for value in outcomes if value is not None]
    assert len(saved) == 1
    audit = reviews.history(db)
    assert len(audit) == 2 and audit[-1]['evidence_snapshot']['scores']['rf'] == saved[0]
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT COUNT(*) FROM review_evidence').fetchone()[0] == 2


@pytest.mark.parametrize('damage', ['hash', 'json', 'identity'])
def test_corrupt_snapshot_marks_invalid_but_keeps_historical_review(tmp_path, damage):
    db = tmp_path/'corrupt.sqlite'
    save(db, evidence())
    with sqlite3.connect(db) as connection:
        if damage == 'hash': connection.execute("UPDATE review_evidence SET payload_sha256='wrong'")
        elif damage == 'json': connection.execute("UPDATE review_evidence SET payload_json='not json'")
        else:
            payload = evidence(); payload['record_id'] = 'other'
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            connection.execute('UPDATE review_evidence SET payload_json=?,payload_sha256=?',
                               (raw, hashlib.sha256(raw.encode()).hexdigest()))
    audit = reviews.history(db)
    assert audit[0]['note'] == '合成工程判断依据'
    assert audit[0]['evidence_snapshot'] is None and audit[0]['evidence_snapshot_status'] == 'invalid'
    exported = export_snapshot(*export_args(audit), True)
    assert json.loads(exported.history)[0]['evidence_snapshot_status'] == 'invalid'
    assert exported.meta['review_evidence_invalid'] == 1
    assert exported.meta['review_evidence_saved'] == 0


def test_export_uses_each_saved_snapshot_and_never_recomputes_history(tmp_path, monkeypatch):
    db = tmp_path/'export.sqlite'
    first, second = evidence(.8), evidence(.9)
    second['input_sha256'] = '1'*64
    save(db, first)
    save(db, second)
    audit = reviews.history(db)
    def unexpected(*args, **kwargs):
        raise AssertionError('export must not recompute evidence')
    monkeypatch.setattr('app_record_tools.evidence_summaries', unexpected)
    args = export_args(audit)
    exported = export_snapshot(*args, True, features=object(), bundle=object())
    payload = json.loads(exported.history)
    assert [row['evidence_snapshot'] for row in payload] == [first, second]
    assert exported.meta['review_evidence_saved'] == 2
    assert exported.meta['model_sha256'] == 'f'*64  # Current run has a distinct scope.
    frame, missing = confirmed_frame(*args, object(), pd.Series([2]), object(), str)
    row = frame.iloc[0]
    assert missing == 0 and row['输入文件'] == 'saved-input.csv' and row['输入CSV行号'] == 3
    assert row['来源文件'] == 'source.csv' and row['来源行号'] == 101
    assert row['保存时输入SHA256'] == '1'*64 and row['保存时模型SHA256'] == 'b'*64
    assert '0.9' in row['保存时证据摘要'] and '工程持续时间' in row['保存时证据摘要']
    assert row['解释方法版本'] == second['method_version']
    assert row['证据快照SHA256'] == audit[-1]['evidence_sha256']


def test_legacy_export_does_not_fill_historical_hashes_or_evidence_from_current_run(tmp_path):
    db = tmp_path/'legacy.sqlite'
    save(db)
    args = export_args(reviews.history(db))
    frame, _ = confirmed_frame(*args, None, pd.Series([2]), None, str)
    row = frame.iloc[0]
    assert row['保存时证据摘要'] == row['保存时系统提示'] == '无当时快照'
    assert row['保存时输入SHA256'] == row['保存时模型SHA256'] == row['保存时描述SHA256'] == ''
    exported = export_snapshot(*args, True)
    assert exported.meta['review_evidence_missing'] == 1


def test_snapshot_history_is_scoped_to_model_and_record(tmp_path):
    db = tmp_path/'scoped.sqlite'
    save(db, evidence())
    other = evidence(); other['model_version'] = 'other-model'
    reviews.save_review(db, 'record', 'other-model', '待补证', 'other model', evidence_snapshot=other)
    other = evidence(); other['record_id'] = 'other-record'
    reviews.save_review(db, 'other-record', 'synthetic-model', '待补证', 'other record', evidence_snapshot=other)
    assert len(reviews.history(db, 'record', 'synthetic-model')) == 1
    assert len(reviews.current_reviews(db, 'synthetic-model')) == 2
    assert reviews.current_reviews(db, 'other-model')['record']['evidence_snapshot']['model_version'] == 'other-model'


def test_actual_explanation_payload_passes_snapshot_contract(tmp_path, engineering_bundle):
    from app_evidence import build_record_evidence
    from liushao.model import infer
    bundle, raw = engineering_bundle
    predictions, features, _ = infer(raw, bundle, 'a'*64, 'synthetic.csv')
    row = predictions.iloc[0]
    metadata = dict(input_sha256='a'*64, model_sha256='b'*64, descriptor_sha256='c'*64,
                    input_name='synthetic.csv', input_csv_row=2)
    snapshot = build_record_evidence(row, features, bundle, meta=metadata)
    db = tmp_path/'actual-contract.sqlite'
    reviews.save_review(db, row.record_id, bundle['model_version'], '待补证',
                        '只作合成工程接口验证', evidence_snapshot=snapshot)
    assert reviews.history(db)[0]['evidence_snapshot'] == snapshot
    assert len(snapshot['cards']) <= 6
    reviews.save_review(db, row.record_id, bundle['model_version'], '确认正常',
                        '仅供软件接口测试的合成工程依据', evidence_snapshot=snapshot)
    active = dict(name='current.csv', digest='e'*64, key='key')
    frame, missing = confirmed_frame(active, predictions, reviews.history(db),
        dict(metadata, model_version=bundle['model_version']), features,
        predictions.source_row, bundle, str)
    assert missing == 0 and frame.iloc[0]['输入CSV行号'] == 2
    assert frame.iloc[0]['证据快照状态'] == '已保存当时证据'


@pytest.mark.parametrize('old_schema', [False, True])
def test_migration_preserves_old_database_or_exact_frozen_json(tmp_path, monkeypatch, old_schema):
    monkeypatch.delenv('LIUSHAO_REVIEW_DB', raising=False)
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir(); new.mkdir()
    db = old/'demo-state/reviews.sqlite'
    save(db, None if old_schema else evidence())
    if old_schema:
        with sqlite3.connect(db) as connection: connection.execute('DROP TABLE review_evidence')
    original = db.read_bytes()
    result = migrate(old, new, checker=lambda root: False)
    target = new/'demo-state/reviews.sqlite'
    assert result['records'] == 1 and db.read_bytes() == original
    assert reviews.history(db) == reviews.history(target)
    assert reviews.history(target)[0]['evidence_snapshot_status'] == ('missing' if old_schema else 'saved')
