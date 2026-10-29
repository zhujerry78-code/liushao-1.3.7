"""Copy a selected old review database before the new package's first launch."""
from contextlib import ExitStack
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid
try:
    from .local_lock import file_lock, active_service
except ImportError:
    from local_lock import file_lock, active_service

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['id', 'record_id', 'model_version', 'status', 'note', 'created_at']

def plain_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
        raise ValueError('迁移路径包含文件链接，请选择实际本机目录。')
    return path.resolve()

def database_signature(db):
    sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
    from liushao.reviews import schema_kind, _snapshot_json
    kind=schema_kind(db)
    if kind=='empty': raise ValueError('数据库没有复核表。')
    if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
        raise ValueError('旧库完整性检查未通过，请保留原文件。')
    allowed={('table','reviews'),('table','review_evidence'),('index','idx_reviews_record')}
    if any(tuple(row) not in allowed for row in db.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")):
        raise ValueError('数据库包含额外结构，已停止复制。')
    digest,count=hashlib.sha256(),0
    for row in db.execute('SELECT id,record_id,model_version,status,note,created_at FROM reviews ORDER BY id'):
        if (type(row[0]) is not int or row[0]<=0 or not row[1] or not row[2]
                or row[3] not in ('待复核','确认正常','确认异常','待补证')
                or not isinstance(row[4],str) or len(row[4])>4000 or datetime.fromisoformat(row[5]).tzinfo is None):
            raise ValueError('数据库包含无效复核字段，已停止复制。')
        digest.update(json.dumps(tuple(row),ensure_ascii=False).encode('utf-8')+b'\n');count+=1
    if kind!='six-column':
        raw,digest_col=('evidence_json','evidence_sha256') if kind=='working-v1' else ('payload_json','payload_sha256')
        query=f'SELECT e.review_id,e.{raw},e.{digest_col},r.record_id,r.model_version FROM review_evidence e LEFT JOIN reviews r ON r.id=e.review_id ORDER BY e.review_id'
        for row in db.execute(query):
            if row[3] is None or hashlib.sha256(row[1].encode('utf-8')).hexdigest()!=row[2]:
                raise ValueError('快照关联或哈希校验失败，旧库未改动。')
            payload=json.loads(row[1]); canonical=_snapshot_json(payload,row[3],row[4])
            if canonical!=row[1]: raise ValueError('快照不是规范 JSON。')
            digest.update(b'evidence:'+json.dumps(tuple(row[:3]),ensure_ascii=False).encode('utf-8')+b'\n')
        if kind=='v2':
            for version,payload in db.execute('SELECT schema_version,payload_json FROM review_evidence'):
                if version != json.loads(payload).get('schema_version',1): raise ValueError('快照版本列不一致。')
    sequence=db.execute('SELECT name,seq FROM sqlite_sequence ORDER BY name').fetchall()
    return count,digest.hexdigest(),sequence


def migrate(source_root, target_root, source_database=None, *, checker=active_service):
    source_root, target_root = plain_path(source_root), plain_path(target_root)
    if source_root == target_root: raise ValueError('新旧目录不能相同。')
    source = plain_path(source_database if source_database else source_root/'demo-state/reviews.sqlite')
    target = plain_path(target_root/'demo-state/reviews.sqlite')
    configured = os.environ.get('LIUSHAO_REVIEW_DB')
    if configured and Path(configured).resolve() != target.resolve():
        raise ValueError('LIUSHAO_REVIEW_DB 指向其他位置，请清除或核对该设置后再迁移。')
    if not source_root.is_dir() or not target_root.is_dir() or not source.is_file():
        raise ValueError('旧目录、旧复核库或新目录不存在。')
    if source.is_symlink() or source.is_junction(): raise ValueError('旧复核库不能是文件链接。')
    if target.exists(): raise FileExistsError('新库已存在；请使用尚未启动过的全新解压目录。')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent/('migration-'+uuid.uuid4().hex+'.sqlite')
    try:
        with ExitStack() as stack:
            for root in sorted({source_root, target_root}, key=lambda p: str(p).casefold()):
                stack.enter_context(file_lock(root/'demo-state/install.lock', timeout=2))
            if checker(source_root) or checker(target_root):
                raise RuntimeError('请先关闭新旧流哨，再重新执行迁移。')
            if target.exists(): raise FileExistsError('新库已出现，已停止迁移。')
            src = sqlite3.connect(source.as_uri()+'?mode=ro', uri=True, timeout=2)
            stack.callback(src.close)
            src.execute('PRAGMA trusted_schema=OFF')
            src.execute('BEGIN')
            before = database_signature(src)
            dst = sqlite3.connect(temporary)
            stack.callback(dst.close)
            deadline = time.monotonic()+60
            def progress(status, remaining, total):
                if time.monotonic() > deadline: raise TimeoutError('数据库复制超时，旧库保持原样。')
            src.backup(dst, pages=1024, progress=progress, sleep=.05)
            from liushao.reviews import upgrade_schema
            upgrade_schema(dst)
            after = database_signature(dst)
            if before != after: raise ValueError('复制结果与旧库记录不一致。')
            dst.close(); src.close()
            # Windows rename refuses an existing destination; POSIX link is also no-clobber.
            if os.name == 'nt': os.rename(temporary, target)
            else: os.link(temporary, target); temporary.unlink()
            return dict(records=after[0], history_sha256=after[1], source=str(source), target=str(target))
    finally:
        for suffix in ('', '-wal', '-shm', '-journal'):
            path = Path(str(temporary)+suffix)
            if path.exists(): path.unlink()

def select_folder():
    code = "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='Select the old Liushao folder'; if($d.ShowDialog() -eq 'OK'){[Console]::OutputEncoding=[Text.Encoding]::UTF8; [Console]::Write($d.SelectedPath)}"
    return subprocess.check_output(['powershell.exe', '-NoProfile', '-STA', '-Command', code], creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8-sig').strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--source-db', type=Path)
    parser.add_argument('--target-root', type=Path, default=ROOT)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    try:
        try: from .verify_demo import verify
        except ImportError: from verify_demo import verify
        verify(args.target_root)
        root = args.source_root
        if args.interactive:
            print('请先关闭新旧流哨。迁移只复制复核历史，不复制输入、缓存或未保存填写。')
            selected = select_folder()
            if not selected: return 0
            root = Path(selected)
            custom = input('旧库默认为所选目录的 demo-state/reviews.sqlite；自定义库请输入完整路径，默认请直接回车：').strip().strip('"')
            if custom: args.source_db = Path(custom)
            print('旧库：', args.source_db or root/'demo-state/reviews.sqlite')
            if input('请确认该库没有其他写入程序。输入 YES 开始复制：').strip() != 'YES': return 0
        if root is None: parser.error('--source-root or --interactive is required')
        result = migrate(root, args.target_root, args.source_db)
        print(f'迁移完成：保留 {result["records"]:,} 次历史操作，旧库未覆盖。现在可一键安装启动。')
        print('请用原来的输入方式重新检查：内置演示与上传同一 demo.csv 的记录身份不同。')
        return 0
    except Exception as error:
        print('迁移未完成：', str(error)); return 1

if __name__ == '__main__': raise SystemExit(main())
