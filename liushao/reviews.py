"""Append-only review history, scoped to record AND model version."""
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ['待复核', '确认异常', '确认正常', '待补证']
SNAPSHOT_STATUSES = {'saved': '已保存当时证据', 'missing': '无当时快照',
                     'invalid': '当时快照校验失败'}
MAX_SNAPSHOT_BYTES = 256 * 1024


def _snapshot_json(snapshot, record_id, model_version):
    """Validate a small, versioned explanation, never a complete input table."""
    if (not isinstance(snapshot, dict) or snapshot.get('record_id') != record_id
            or snapshot.get('model_version') != model_version
            or not isinstance(snapshot.get('method_version'), str)
            or not snapshot['method_version'].strip()
            or not isinstance(snapshot.get('scores'), dict)
            or not isinstance(snapshot.get('cards'), list) or len(snapshot['cards']) > (6 if snapshot.get('schema_version') == 2 else 12)):
        raise ValueError('保存时证据快照的记录、模型、方法或关键数据无效')
    if len({card.get('field') for card in snapshot['cards'] if isinstance(card, dict)}) != len(snapshot['cards']):
        raise ValueError('主要证据字段不能重复')
    if snapshot.get('schema_version') == 2 and (not isinstance(snapshot.get('selection'), dict) or not isinstance(snapshot.get('decomposition'), dict)):
        raise ValueError('新版证据缺少筛选或评分分解状态')
    if 'identity' in snapshot and not isinstance(snapshot['identity'], dict):
        raise ValueError('保存时证据快照的来源标识无效')
    for key in ('input_sha256', 'model_sha256', 'descriptor_sha256'):
        if not isinstance(snapshot.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', snapshot[key]):
            raise ValueError('保存时证据快照缺少有效的输入或模型 SHA256')
    for key in ('rf', 'lr', 'rf_threshold', 'lr_threshold'):
        value = snapshot['scores'].get(key)
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError('保存时证据快照的分数或分界无效')
    for card in snapshot['cards']:
        if (not isinstance(card, dict) or not isinstance(card.get('field'), str)
                or any(not isinstance(card.get(key), dict) for key in ('reference', 'rf', 'lr'))):
            raise ValueError('保存时证据快照的关键字段解释无效')
    serialized = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
    if len(serialized.encode('utf-8')) > MAX_SNAPSHOT_BYTES:
        raise ValueError('保存时证据快照过大，请仅保留关键证据')
    return serialized


def review_with_evidence(review):
    """Verify persisted bytes before exposing historical evidence to the UI/export.

    A legacy or damaged snapshot never causes the review's note/status to vanish.
    This helper also labels legacy audit dictionaries supplied by older clients.
    """
    result = dict(review)
    serialized = result.pop('evidence_json', None)
    snapshot = result.get('evidence_snapshot')
    digest = result.get('evidence_sha256')
    state = 'missing'
    if (serialized is not None or snapshot is not None or digest is not None
            or result.get('evidence_snapshot_status') == 'invalid'):
        state = 'invalid'
        try:
            if serialized is not None:
                if not isinstance(serialized, str) or hashlib.sha256(serialized.encode('utf-8')).hexdigest() != digest:
                    raise ValueError('证据快照哈希不一致')
                snapshot = json.loads(serialized)
            if not isinstance(snapshot,dict): raise ValueError('证据快照载荷无效')
            if result.get('evidence_schema_version',snapshot.get('schema_version',1)) != snapshot.get('schema_version',1):
                raise ValueError('证据快照版本不一致')
            canonical = _snapshot_json(snapshot, result['record_id'], result['model_version'])
            if hashlib.sha256(canonical.encode('utf-8')).hexdigest() != digest:
                raise ValueError('证据快照哈希不一致')
            state = 'saved'
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
            snapshot = None
    result.update(evidence_snapshot=snapshot if state == 'saved' else None,
                  evidence_sha256=digest,
                  evidence_snapshot_status=state,
                  evidence_snapshot_label=SNAPSHOT_STATUSES[state])
    return result


class ReviewConflictError(RuntimeError):
    """The persisted review changed after this editor loaded its baseline."""


REVIEW_COLUMNS = [('id','INTEGER',0,1),('record_id','TEXT',1,0),('model_version','TEXT',1,0),
                  ('status','TEXT',1,0),('note','TEXT',1,0),('created_at','TEXT',1,0)]
LEGACY_EVIDENCE_COLUMNS = [('review_id','INTEGER',0,1),('evidence_json','TEXT',1,0),('evidence_sha256','TEXT',1,0)]
EVIDENCE_COLUMNS = [('review_id','INTEGER',0,1),('schema_version','INTEGER',1,0),('payload_json','TEXT',1,0),('payload_sha256','TEXT',1,0)]


def columns(db, table):
    return [(r[1],r[2].upper(),r[3],r[5]) for r in db.execute('PRAGMA table_info('+table+')')]


def schema_kind(db):
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")}
    if not tables: return 'empty'
    if not tables <= {'reviews','review_evidence'} or columns(db,'reviews') != REVIEW_COLUMNS:
        raise ValueError('未知数据库结构；请保留原库并核对版本。')
    if 'review_evidence' not in tables: return 'six-column'
    fields=columns(db,'review_evidence')
    if fields == EVIDENCE_COLUMNS: return 'v2'
    if fields == LEGACY_EVIDENCE_COLUMNS: return 'working-v1'
    raise ValueError('未知证据侧表结构；请保留原库。')


def upgrade_schema(db):
    kind=schema_kind(db)
    if kind=='v2': return kind
    try:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, model_version TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_reviews_record ON reviews (model_version, record_id, id)')
        if kind=='working-v1': db.execute('ALTER TABLE review_evidence RENAME TO review_evidence_working')
        db.execute('CREATE TABLE review_evidence (review_id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, FOREIGN KEY(review_id) REFERENCES reviews(id))')
        if kind=='working-v1':
            for row in db.execute('SELECT review_id,evidence_json,evidence_sha256 FROM review_evidence_working').fetchall():
                # Preserve the old bytes and method. No recomputation or empty backfill.
                payload=json.loads(row[1])
                db.execute('INSERT INTO review_evidence VALUES(?,?,?,?)',(row[0],payload.get('schema_version',1),row[1],row[2]))
            db.execute('DROP TABLE review_evidence_working')
        db.execute('PRAGMA user_version=2'); db.commit()
    except BaseException:
        db.rollback(); raise
    return kind


def connect(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=10); db.row_factory=sqlite3.Row
    try:
        db.execute('PRAGMA foreign_keys=ON')
        kind=schema_kind(db)
        if kind not in ('empty','v2'):
            # A full SQLite backup retains the prior schema before an additive upgrade.
            backup=path.with_name(path.name+'.before-evidence-v2.bak')
            if not backup.exists():
                import os
                fd=os.open(backup,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
                try:
                    with sqlite3.connect(backup) as target: db.backup(target)
                except BaseException:
                    backup.unlink(missing_ok=True); raise
        upgrade_schema(db)
        return db
    except BaseException:
        db.close(); raise


def save_review(path, record_id, model_version, status, note, *,
                expected_review_id: int | None = None, evidence_snapshot=None):
    """Append a review, optionally requiring the editor's last persisted revision.

    ``None`` retains the unconditional legacy behavior; ``0`` requires no history.
    A stale baseline raises :class:`ReviewConflictError` without adding a row.
    Successful writes retain the legacy ``None`` return value. A supplied snapshot
    is serialized before opening the database and committed with the review in one
    transaction. Omitting it preserves legacy calls and marks them as missing.
    """
    if (expected_review_id is not None
            and (type(expected_review_id) is not int or expected_review_id < 0)):
        raise ValueError('复核修订必须是非负整数或 None')
    if status not in STATUSES or not record_id or not model_version:
        raise ValueError('复核状态或记录标识无效')
    if len(note) > 4000:
        raise ValueError('备注最多4000字')
    if status in ('确认异常', '确认正常') and not note.strip():
        raise ValueError('确认异常或确认正常时必须填写判断依据')
    serialized = (_snapshot_json(evidence_snapshot, record_id, model_version)
                  if evidence_snapshot is not None else None)
    db = connect(path)
    try:
        # Reserve the write transaction before reading the revision. A competing
        # writer cannot compare the same revision and then append after this one.
        db.execute('BEGIN IMMEDIATE')
        if expected_review_id is not None:
            latest = db.execute(
                'SELECT MAX(id) FROM reviews WHERE model_version=? AND record_id=?',
                (model_version, record_id),
            ).fetchone()[0] or 0
            if latest != expected_review_id:
                raise ReviewConflictError('复核记录已被其他窗口修改，请读取最新复核并替换草稿后再保存')
        cursor = db.execute('INSERT INTO reviews (record_id, model_version, status, note, created_at) VALUES (?, ?, ?, ?, ?)',
                            (record_id, model_version, status, note, datetime.now(timezone.utc).isoformat()))
        if serialized is not None:
            db.execute('INSERT INTO review_evidence (review_id, schema_version, payload_json, payload_sha256) VALUES (?, ?, ?, ?)',
                       (cursor.lastrowid, evidence_snapshot.get('schema_version',1), serialized, hashlib.sha256(serialized.encode('utf-8')).hexdigest()))
        db.commit()
    except BaseException as primary:
        try:
            db.rollback()
        except BaseException as cleanup:
            primary.add_note(f'回滚同时失败：{cleanup}')
        try:
            db.close()
        except BaseException as cleanup:
            primary.add_note(f'关闭连接同时失败：{cleanup}')
        raise
    else:
        db.close()


def _select(include_evidence):
    if include_evidence:
        return 'SELECT r.*, e.payload_json AS evidence_json, e.payload_sha256 AS evidence_sha256, e.schema_version AS evidence_schema_version FROM reviews r LEFT JOIN review_evidence e ON e.review_id=r.id'
    return 'SELECT r.*, e.payload_sha256 AS evidence_sha256 FROM reviews r LEFT JOIN review_evidence e ON e.review_id=r.id'


def history(path, record_id=None, model_version=None, *, record_ids=None, include_evidence=True):
    db=connect(path)
    try:
        db.execute('BEGIN')
        ids=sorted(set(record_ids)) if record_ids is not None else None
        groups=[ids[i:i+400] for i in range(0,len(ids),400)] if ids is not None else [None]
        result=[]
        for group in groups:
            sql=_select(include_evidence)+' WHERE 1=1'; params=[]
            if record_id is not None: sql+=' AND r.record_id=?'; params.append(record_id)
            if model_version is not None: sql+=' AND r.model_version=?'; params.append(model_version)
            if group is not None:
                sql+=' AND r.record_id IN ('+','.join('?' for _ in group)+')'; params.extend(group)
            for row in db.execute(sql+' ORDER BY r.id',params):
                result.append(review_with_evidence(row) if include_evidence else dict(row, evidence_snapshot_status='deferred' if row['evidence_sha256'] else 'missing',
                    evidence_snapshot_label='按需读取当时证据' if row['evidence_sha256'] else '无当时快照'))
        return sorted(result,key=lambda row:row['id'])
    finally: db.close()


def current_reviews(path, model_version):
    rows=history(path,model_version=model_version)
    return {r['record_id']:r for r in rows}


def evidence_for_review(path, review_id):
    db=connect(path)
    try:
        row=db.execute(_select(True)+' WHERE r.id=?',(review_id,)).fetchone()
        return review_with_evidence(row) if row else None
    finally: db.close()
