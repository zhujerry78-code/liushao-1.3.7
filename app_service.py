"""Atomic input analysis, separate from Streamlit rendering."""
import hashlib
import io
import json
import time

import pandas as pd

from app_quality import build_quality_view, validate_final_mapping
from app_types import ActiveInput, AnalysisResult, ReviewSnapshot, validate_history
from liushao import reviews as review_store
from liushao.data import read_csv_checked
from liushao.model import infer

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 50000
_ENGINE_CACHE = {}


def load_trusted_engine(artifacts, bundle):
    """artifacts/engine.joblib verified against engine.json; None means 1.3.6 behaviour."""
    try:
        import liushao_engine
        path, descriptor = artifacts / 'engine.joblib', artifacts / 'engine.json'
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        meta = json.loads(descriptor.read_text(encoding='utf-8'))
        if digest != meta['engine_sha256'] or meta['base_model_version'] != bundle['model_version']:
            return None
        key = (digest, bundle['model_version'])
        if key not in _ENGINE_CACHE:
            _ENGINE_CACHE.clear()
            _ENGINE_CACHE[key] = liushao_engine.load_engine(path, bundle)
        return _ENGINE_CACHE[key]
    except Exception:
        return None


def with_engine(pred, features, bundle, artifacts):
    """Append trusted-triage columns; any failure keeps the frozen 1.3.6 predictions."""
    engine = load_trusted_engine(artifacts, bundle)
    if engine is None:
        return pred
    try:
        import liushao_engine
        return liushao_engine.analyze(pred, features, bundle, engine)
    except Exception:
        return pred


def analyze_input(content, name, source, bundle, model_identity, artifacts, *, preflight=None):
    tick = time.perf_counter()
    if len(content) > MAX_BYTES:
        raise ValueError('文件超过 50 MiB，请分成较小的文件后再上传。')
    digest = hashlib.sha256(content).hexdigest()
    raw = preflight.take(content, model_identity) if preflight is not None else None
    if raw is None:
        raw = read_csv_checked(io.BytesIO(content), max_rows=MAX_ROWS)
    if raw.empty or len(raw) > MAX_ROWS:
        raise ValueError('文件需要包含 1 至 50,000 条记录。')
    pred, features, rejected = infer(raw, bundle, digest, name)
    pred = with_engine(pred, features, bundle, artifacts)
    physical = pred.source_row.copy()
    original = pred.copy()
    if source == '内置演示数据':
        metadata_bytes = (artifacts / 'demo.json').read_bytes()
        if model_identity.split('|')[0] == '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7':
            if hashlib.sha256(metadata_bytes).hexdigest() != 'f659acedda59cea6b4bdf8952146f7273c7e7bf4122454daa96dad703d353789':
                raise ValueError('演示描述不符合可信冻结身份，请恢复配套文件。')
        metadata = json.loads(metadata_bytes)
        origins_bytes = (artifacts / 'demo_origins.parquet').read_bytes()
        if (digest != metadata['csv_sha256'] or
                hashlib.sha256(origins_bytes).hexdigest() != metadata['origins_sha256']):
            raise ValueError('演示文件与来源信息不匹配，请恢复配套文件。')
        origins = pd.read_parquet(io.BytesIO(origins_bytes))
        if len(origins) != len(pred) or len(rejected):
            raise ValueError('演示记录与来源信息不匹配。')
        for col in ['record_id', 'source_file', 'source_sha256', 'source_row', 'timestamp']:
            pred[col] = origins[col]
        validate_final_mapping(original, pred)
    try:
        quality = build_quality_view(raw, original, features, rejected, final_predictions=pred)
        error = None
    except Exception as exc:
        quality, error = None, str(exc)
    active = ActiveInput(content, name, source, digest,
                         '|'.join((digest, source, name, model_identity)), model_identity)
    return active, AnalysisResult(pred, features, rejected, time.perf_counter()-tick, len(raw)), physical, quality, error



def read_review_snapshot(path, model_version, record_ids):
    """One live query; release unrelated history before returning to rendering."""
    rows = validate_history(review_store.history(path, model_version=model_version, record_ids=record_ids, include_evidence=False), model_version)
    audit = [item for item in rows if item['record_id'] in record_ids]
    from app_record_tools import latest_reviews
    return ReviewSnapshot(latest_reviews(audit, model_version), audit)
