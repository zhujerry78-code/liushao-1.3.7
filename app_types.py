"""Application state contracts; no model or persistence protocol changes."""
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from typing import Any


class Fields(MutableMapping):
    """Named state with mapping compatibility for presentation helpers."""
    def __getitem__(self, key):
        if key not in self: raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self: raise KeyError(key)
        setattr(self, key, value)

    def __delitem__(self, key):
        self[key] = None

    def __iter__(self):
        return iter(f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def __contains__(self, key):
        return any(f.name == key for f in fields(self))


@dataclass
class Filters(Fields):
    scope: str = '全部记录'
    status: str = '全部'
    line: str = ''
    queue: str = '高异常得分'
    minimum: float = 0.0
    disagreement: bool = False
    near: bool = False
    lr_tie_break: bool = False
    include_lr_near: bool = False


@dataclass
class ActiveInput(Fields):
    content: bytes
    name: str
    source: str
    digest: str
    key: str
    model: str


@dataclass
class AnalysisResult:
    predictions: Any
    features: Any
    rejected: Any
    elapsed: float
    total: int


@dataclass
class Draft(Fields):
    record_id: str
    input_key: str
    status: str
    note: str
    original: tuple[str, str]
    baseline: int
    next_id: str | None = None
    conflict: bool = False


@dataclass(frozen=True)
class PendingAction:
    kind: str
    value: Any = None


@dataclass(frozen=True)
class ReviewSnapshot:
    current: dict
    history: list


@dataclass
class ExportSnapshot(Fields):
    input_key: str
    meta: dict
    predictions: bytes | None
    history: bytes | None
    manifest: bytes
    confirmed: bytes | None = None
    confirmed_count: int = 0
    missing_input_rows: int = 0


@dataclass
class SessionState(Fields):
    page: str = 'overview'
    filters: Filters = field(default_factory=Filters)
    active: ActiveInput | None = None
    result: AnalysisResult | None = None
    quality: Any = None
    quality_error: str | None = None
    quality_selected: str | None = None
    quality_page: int = 0
    physical_rows: Any = None
    editor: Draft | None = None
    suspended_draft: Draft | None = None
    selected: str | None = None
    page_index: int = 0
    table_epoch: int = 0
    search_keyword: str = ''
    search_open: bool = False
    pending: PendingAction | None = None
    exports: ExportSnapshot | None = None
    show_import: bool = True
    stage_source: str = '内置演示数据'
    stage_upload: Any = None
    run_error: str | None = None
    flash: str | None = None
    diagnostics: Any = None
    batch: Any = None
    clues: Any = None
    tool_downloads: dict = field(default_factory=dict)
    percentiles: Any = None
    preflight: Any = None
    preflight_key: Any = None
    preflight_error: Any = None
    remember_key: Any = None
    save_generation: str | None = None
    cache_message: str | None = None


def invalidate_model(state: SessionState):
    """Never resurrect results after a fault; preserve an explicitly restorable draft."""
    if state.editor and (state.editor.status, state.editor.note) != state.editor.original:
        state.suspended_draft = state.editor
    state.result = state.active = state.editor = state.exports = state.pending = None
    state.quality = state.physical_rows = state.selected = None
    state.diagnostics = state.batch = state.clues = None
    state.percentiles = None
    state.preflight = state.preflight_key = state.preflight_error = None
    state.remember_key = state.save_generation = None
    state.cache_message = None
    state.tool_downloads.clear()


def needs_review(predictions):
    if 'engine_review' in predictions:  # 1.3.7 engine: conformal "uncertain" or high conflict
        return predictions.engine_review.astype(bool)
    return predictions.disagreement | predictions.near_threshold


def line_number(text):
    text = text.strip()
    if not text or len(text) > 12 or not text.isdecimal(): return None
    try: return int(text)
    except ValueError: return None


def validate_history(rows, model_version):
    from datetime import datetime
    from liushao.reviews import STATUSES
    for row in rows:
        if (type(row.get('id')) is not int or row['id'] <= 0
                or not isinstance(row.get('record_id'), str) or not row['record_id']
                or row.get('model_version') != model_version or row.get('status') not in STATUSES
                or not isinstance(row.get('note'), str) or len(row['note']) > 4000):
            raise ValueError('已保存操作含无效字段，请保留数据库并检查。')
        stamp = datetime.fromisoformat(row['created_at'])
        if stamp.tzinfo is None: raise ValueError('已保存操作时间缺少时区。')
    return rows
