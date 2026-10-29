"""Local competition interface. Run through delivery/run_demo.py."""
import hashlib
import io
import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app_score_charts import render_histogram, render_ruler
from app_evaluation import SUMMARY_PATH, render_evaluation
from app_navigation import PAGES, BY_ID, page_id, page_label
from app_navigation_scroll import queue_navigation_scroll, render_navigation_scroll
from app_model import load_model_bundle, verified_model
from app_record_tools import batch_positions
from app_evidence import build_record_evidence, METHOD_VERSION
from app_evidence_ui import render_record_evidence, render_evidence_detail, render_review_checklist, render_engine_panel
from app_quick_search import promote_matches
from app_input import preflight_input, problem_sample, present_error, UserError
from app_session import consent_generation, saved_upload, save_upload, clear_upload, load_upload
from app_service import analyze_input, load_trusted_engine, read_review_snapshot
from app_types import (SessionState, Filters, Draft, PendingAction, invalidate_model,
                       needs_review, line_number)
from app_quality import build_quality_view
from app_ui_state import (STATUS_NAMES, UI_REVISION, base_meta, csv_bytes, default_filters,
                          dirty, export_snapshot, field_name, filtered, literal_markdown)
from liushao.data import read_csv_checked, sha256
from liushao.model import evidence_for, infer
from liushao.reviews import STATUSES, ReviewConflictError, save_review, history, evidence_for_review

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get('LIUSHAO_ARTIFACTS_DIR', str(ROOT / 'artifacts')))
DB = Path(os.environ.get('LIUSHAO_REVIEW_DB', str(ROOT / 'data/reviews.sqlite')))
EVALUATION = Path(os.environ.get('LIUSHAO_EVALUATION_SUMMARY', str(ROOT / SUMMARY_PATH)))
VERSION = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
st.set_page_config(page_title='流哨 · 网络记录检查', page_icon=str(ROOT / 'liushao_icon.svg'), layout='wide')
st.markdown('''<style>
html,body,[data-testid="stApp"] {font-family:"Microsoft YaHei",sans-serif;}
.block-container {padding:3.75rem 2rem 1.5rem;max-width:1600px;}
/* stToolbar also hosts the sidebar restore button in Streamlit 1.50. */
[data-testid="stAppDeployButton"],[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stDecoration"] {display:none;}
.st-key-_records_main [data-testid="stElementToolbar"] {display:none;}
h1 {font-size:1.65rem!important;margin-bottom:.25rem!important;} h2 {font-size:1.2rem!important;}
h3 {font-size:1.05rem!important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.45rem;}
[data-testid="stSidebar"] button {justify-content:flex-start;}
[data-testid="stSidebar"] button[kind="primary"] {border-left:5px solid #102c47;font-weight:700;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce4ee;border-radius:10px;padding:.65rem 1rem;}
[data-testid="stMetricValue"] {font-size:2rem;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {gap:.7rem;}
.st-key-review_workspace [data-testid="stVerticalBlock"] {gap:.55rem;}
.st-key-record_header [data-testid="stVerticalBlock"] {gap:.2rem;}
.st-key-record_header h3 {margin:0!important;}
@media(max-width:850px){.block-container {padding:3.75rem 1rem 1rem;} }
@media(max-width:1000px){
 .st-key-review_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-direction:column;}
 .st-key-review_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width:100%;flex:1 1 100%;}
}
</style>''', unsafe_allow_html=True)

if 'ui' not in st.session_state:
    st.session_state.ui = SessionState()
u = st.session_state.ui
u.page = page_id(u.page)
if u.page == 'evaluation': u.page = 'help'
st.session_state['_sample_kind'] = st.session_state.get('_sample_kind', '缺少必需列')
st.session_state['_remember'] = st.session_state.get('_remember', False)
st.session_state['_source'] = st.session_state.get('_source', u.stage_source)


def nav(page):
    update_editor()
    destination = page_id(page)
    if u.page == 'evidence_detail' and destination == 'review':
        queue_navigation_scroll(st.session_state, destination)
    u.page = destination


def open_evidence_detail():
    update_editor()
    st.session_state['_evidence_detail_identity'] = (u.active.key, u.selected, model_identity)
    u.page = 'evidence_detail'
    queue_navigation_scroll(st.session_state, u.page)


def update_editor():
    editor = u.editor
    if editor:
        editor.status = st.session_state.get('_status', editor.status)
        editor.note = st.session_state.get('_note', editor.note)
        if dirty(editor):
            u.exports = None


def reload_saved_editor():
    # A callback completes before a navigation rerun can interrupt the discard.
    u.editor = None
    for key in ('_editor_identity', '_status', '_note'):
        st.session_state.pop(key, None)


def show_local_file_error(error, details='本机文件操作详情'):
    message = error if isinstance(error, UserError) else present_error(error, '本机文件操作')
    st.warning(message.reason)
    st.caption(message.action)
    with st.expander(details):
        st.text(message.detail)


def request(kind, value=None):
    if u.pending is not None:
        return
    if kind == 'record' and value == u.selected:
        return
    if dirty(u.editor) and kind in ('record', 'filters', 'run', 'review_scope', 'restore'):
        u.pending = PendingAction(kind, value)
    else:
        apply_action(kind, value)
    if kind != 'record': u.table_epoch += 1


def filter_change(name, key):
    request('filters', {**u.filters, name: st.session_state[key]})


def row_selected(key, record_ids):
    if (key not in ('_records_main','_records_'+str(u.table_epoch))) or key not in st.session_state:
        return
    selection = st.session_state[key].selection
    rows = selection.rows or ([selection.cells[0][0]] if selection.cells else [])
    if rows and type(rows[0]) is int and 0 <= rows[0] < len(record_ids):
        request('record', record_ids[rows[0]])


def page_to(index):
    if u.pending is not None: return
    u.page_index = max(0, index)
    u.table_epoch += 1


def apply_action(kind, value):
    if kind == 'record':
        if u.selected != value:
            u.selected, u.editor = value, None
    elif kind in ('filters', 'review_scope'):
        u.filters = Filters(**dict(value))
        for key in list(st.session_state):
            if key.startswith('_filter_'):
                del st.session_state[key]
        u.page_index = 0
        u.filters.line = ''
        if kind == 'review_scope':
            u.page = 'review'
    elif kind == 'run':
        run_input(value)
    elif kind == 'restore':
        try:
            payload = load_upload(DB, value, model_identity)
            u.preflight = u.preflight_key = u.preflight_error = None
            u.save_generation = u.remember_key = None
            st.session_state['_remember'] = False
            run_input(payload)
        except Exception as error:
            u.run_error = present_error(error, '恢复文件')
            u.show_import = True
    u.pending = None


@st.cache_resource(max_entries=2)
def load_model(identity, _content, _descriptor):
    return load_model_bundle(_content, _descriptor)


bundle = None
model_error = None
verified = None
try:
    verified = verified_model(ARTIFACTS / 'model.joblib', ARTIFACTS / 'model.json')
    model_identity = verified.identity
    bundle = load_model(model_identity, verified.content, verified.descriptor)
    if u.active and u.active.model != model_identity:
        invalidate_model(u)
        st.session_state['_remember'] = False
except Exception as error:
    model_error = str(error)
    model_identity = None
    invalidate_model(u)
    st.session_state['_remember'] = False


def run_input(payload):
    if bundle is None:
        u.run_error = '分析程序尚未就绪，请查看“关于流哨”中的详细信息。'
        return
    try:
        permission = (u.remember_key, u.save_generation)
        input_key = (hashlib.sha256(payload[0]).hexdigest(), payload[1], payload[2])
        with st.spinner('正在检查文件和分析记录…'):
            active, result, physical, quality, error = analyze_input(*payload, bundle, model_identity, ARTIFACTS,
                                                                  preflight=u.preflight)
        u.update(active=active, result=result, physical_rows=physical, quality=quality,
                 quality_error=error, quality_selected=None, quality_page=0,
                 editor=None, selected=None, filters=Filters(), page_index=0,
                 show_import=False, run_error=None, exports=None, diagnostics=None,
                 batch=None, clues=None, page='overview', tool_downloads={})
        u.table_epoch += 1
        for key in list(st.session_state):
            if key.startswith(('_filter_', '_quality_')) or key in ('_status', '_note'):
                del st.session_state[key]
    except Exception as error:
        u.run_error = present_error(error)
        return
    u.cache_message = None
    if (payload[2] == '上传 CSV' and st.session_state.get('_remember', False)
            and permission[0] == input_key and permission == (u.remember_key, u.save_generation)
            and permission[1] is not None):
        try:
            saved = save_upload(DB, payload[0], payload[1], model_identity, permission[1])
            u.cache_message = '已在本机保留此文件，下次可继续检查。' if saved else '另一窗口已清除缓存，本次未再保留文件。'
        except Exception as error:
            u.cache_message = present_error(error, '分析已完成，保留文件未完成')


if u.suspended_draft:
    st.warning('模型曾改变或校验失败，未保存填写已暂存。请重新检查原文件后明确恢复。')
    draft = u.suspended_draft
    can_restore = bool(u.active and u.active.key == draft.input_key and u.result is not None
                       and draft.record_id in set(u.result.predictions.record_id))
    st.caption('暂存判断：' + STATUS_NAMES.get(draft.status, draft.status))
    st.code(draft.note or '（无备注）')
    if st.button('恢复暂存填写', key='restore_suspended', disabled=not can_restore):
        u.editor, u.selected = draft, draft.record_id
        u.suspended_draft, u.exports, u.page = None, None, 'review'
        st.session_state.pop('_editor_identity', None)
        st.rerun()
    if st.button('放弃暂存填写', key='discard_suspended'):
        u.suspended_draft = None
        st.rerun()


def sidebar():
    with st.sidebar:
        mark, title = st.columns([1, 5])
        mark.image(str(ROOT / 'liushao_icon.svg'), width=28)
        title.markdown('## 流哨')
        st.caption('流量检查与复核')
        def buttons(group):
            for page in PAGES:
                if page.group == group:
                    st.button(page.label, icon=f':material/{page.icon}:', key=page.key,
                              width='stretch', type='primary' if u.page == page.id or (u.page == 'evidence_detail' and page.id == 'review') else 'secondary',
                              on_click=nav, args=(page.id,))
        st.markdown('**主要功能**')
        buttons('main')
        with st.expander('更多工具', expanded=BY_ID[u.page].group == 'tools'):
            buttons('tools')
        with st.expander('使用帮助', expanded=BY_ID[u.page].group == 'help'):
            buttons('help')
        st.caption('本机运行 · 文件保存在本地')


sidebar()
st.title(page_label(u.page))
if u.flash:
    st.success(u.flash)
    u.flash = None
if u.cache_message:
    if isinstance(u.cache_message, UserError):
        show_local_file_error(u.cache_message)
    else:
        st.info(u.cache_message)
if model_error and u.page != 'about':
    st.error('分析程序尚未就绪。你仍可查看帮助，或到“关于流哨”核对详情。')

valid_result = bool(bundle is not None and u.active and u.active.model == model_identity and u.result is not None)
available, reviews, audit = True, {}, []
if valid_result:
    result = u.result
    raw_pred, features, rejected = result.predictions, result.features, result.rejected
    elapsed, total = result.elapsed, result.total
    try:
        snapshot = read_review_snapshot(DB, bundle['model_version'], set(raw_pred.record_id))
        reviews, audit = snapshot.current, snapshot.history
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as error:
        available = False
        if u.exports and u.exports.predictions is not None:
            u.exports = None
        st.warning('暂时无法读取已保存的操作。分析结果仍可查看，保存和相关下载已暂停。')
        with st.expander('查看错误详情'):
            st.write(str(error))
        st.button('重试读取', key='retry_reviews')
    pred = raw_pred.copy()
    pred['review_status'] = pred.record_id.map(lambda rid: reviews.get(rid, {}).get('status', '待复核')) if available else '复核不可用'
    pred['review_note'] = pred.record_id.map(lambda rid: reviews.get(rid, {}).get('note', ''))
    meta = base_meta(u.active, u.result, bundle, verified)
    if available and u.exports and not u.exports.meta['review_storage_available']:
        u.exports = None
    if u.exports and u.exports['input_key'] != u.active.key:
        u.exports = None
else:
    pred = None


def save_editor(next_after=False):
    editor = u.editor
    if not valid_result or not available or not editor or u.exports:
        st.error('当前暂不能保存，请先恢复文件与保存记录，或退出下载准备状态。')
        return False
    if editor.input_key != u.active.key:
        st.error('文件已改变，请重新选择记录。')
        return False
    try:
        save_review(DB, editor.record_id, bundle['model_version'], editor.status, editor.note,
                    expected_review_id=editor.baseline, evidence_snapshot=current_evidence(editor.record_id))
    except ReviewConflictError:
        editor.conflict = True
        st.error('另一窗口已修改这条记录。你的填写已保留，请先查看操作历史，再读取最新记录。')
        return False
    except Exception as error:
        st.error(f'保存失败，填写内容已保留：{error}')
        return False
    next_id = None
    if next_after:
        try:
            fresh = read_review_snapshot(DB, bundle['model_version'], set(raw_pred.record_id))
            values = fresh.current
            refreshed = raw_pred.copy()
            refreshed['review_status'] = refreshed.record_id.map(lambda rid: values.get(rid, {}).get('status', '待复核'))
            canonical = promote_matches(filtered(pred, dict(u.filters, status='全部', line=''), bundle),u.search_keyword)[0].record_id.tolist()
            choices = set(filtered(refreshed, u.filters, bundle).record_id)
            position = canonical.index(editor.record_id) if editor.record_id in canonical else -1
            if position >= 0:
                next_id = next((rid for rid in canonical[position+1:] if rid in choices), None)
        except Exception as error:
            u.flash = f'已保存，但刷新列表失败：{error}。请重试读取；请勿重复保存。'
    u.selected = next_id or editor.record_id
    u.editor, u.exports = None, None
    u.flash = u.flash or ('已保存。' + ('已打开下一条。' if next_id else ''))
    return True


if u.pending:
    st.warning('有尚未保存的填写。请先保存，或明确放弃后再切换；取消会留在原记录。')
    a, b, c = st.columns(3)
    if a.button('保存后继续', key='pending_save', disabled=not available or not valid_result):
        pending = u.pending
        if save_editor():
            apply_action(pending.kind, pending.value)
            st.rerun()
    if b.button('放弃填写并继续', key='pending_discard'):
        pending = u.pending
        u.editor = None
        apply_action(pending.kind, pending.value)
        st.rerun()
    if c.button('取消切换', key='pending_cancel'):
        for key in list(st.session_state):
            if key.startswith('_filter_'):
                del st.session_state[key]
        u.pending = None
        u.table_epoch += 1
        st.rerun()


def empty_page():
    st.info('请先导入文件并完成检查。')
    st.button('前往导入数据', key='to_import', on_click=nav, args=('overview',))


def file_bar():
    left, right = st.columns([5, 1])
    label = '演示数据' if u.active.source == '内置演示数据' else u.active.name
    left.caption(f'{literal_markdown(label)} · 已检查 {total:,} 条')
    if right.button('更换文件', key='change_file', width='stretch'):
        u.show_import, u.page = True, 'overview'
        st.rerun()


def upload_changed():
    upload = st.session_state.get('_upload')
    u.stage_upload = None if upload is None else (upload.getvalue(), upload.name)
    u.preflight = u.preflight_key = u.preflight_error = None
    u.remember_key = u.save_generation = None
    st.session_state['_remember'] = False


def remember_changed():
    u.save_generation = None
    if st.session_state.get('_remember') and u.stage_upload:
        try:
            content, name = u.stage_upload
            u.remember_key = (hashlib.sha256(content).hexdigest(), name, '上传 CSV')
            u.save_generation = consent_generation(DB)
        except Exception as error:
            st.session_state['_remember'] = False
            u.cache_message = present_error(error, '启用文件保留')


def resume_card():
    try:
        saved = saved_upload(DB)
        if saved:
            if u.save_generation is not None and u.save_generation != saved['generation']:
                u.save_generation = u.remember_key = None
                st.session_state['_remember'] = False
            st.caption('上次保留：' + literal_markdown(saved['name']) + ' · ' + saved.get('saved_at', ''))
            if saved['model'] != model_identity:
                st.caption('当前模型身份不匹配，不能恢复；可重新上传原文件。')
            if st.button('继续上次检查', key='restore_last', disabled=bundle is None or saved['model'] != model_identity or u.pending is not None):
                request('restore', (saved['generation'], saved['id']))
                st.rerun()
        elif u.save_generation is not None:
            # A clear in another window must also revoke the old checkbox permission.
            from app_session import session_root
            if session_root(DB).exists() and consent_generation(DB) != u.save_generation:
                u.save_generation = u.remember_key = None
                st.session_state['_remember'] = False
    except Exception as error:
        show_local_file_error(error, '查看恢复文件详情')
    if st.button('清除已保留文件', key='clear_last'):
        u.save_generation = u.remember_key = None
        st.session_state['_remember'] = False
        try:
            clear_upload(DB)
            u.cache_message = '已清除保留的输入文件；复核记录仍在本机数据库。'
        except Exception as error:
            u.cache_message = present_error(error, '清除保留文件')
        st.rerun()


def import_card():
    with st.container(border=True):
        st.markdown('**导入数据**')
        resume_card()
        st.session_state.setdefault('_source', u.stage_source)
        source = st.radio('数据来源', ['上传 CSV', '内置演示数据', '格式练习'], horizontal=True, key='_source',
                          format_func=lambda x: {'上传 CSV': '上传文件', '内置演示数据': '使用演示数据', '格式练习': '格式练习'}[x],
                          on_change=lambda: u.update(stage_source=st.session_state['_source']))
        content, name = None, None
        if source == '上传 CSV':
            upload = st.file_uploader('选择 CSV 文件', type=['csv'], key='_upload', on_change=upload_changed)
            if upload is not None:
                u.stage_upload = (upload.getvalue(), upload.name)
            if u.stage_upload:
                content, name = u.stage_upload
                st.caption(f'待检查：{literal_markdown(name)}')
                st.checkbox('在本机保留此文件，便于下次继续', key='_remember', on_change=remember_changed)
                st.caption('取消勾选仅撤销本次保存；删除旧副本请点“清除已保留文件”。')
        elif source == '格式练习':
            st.caption('合成输入格式练习，不代表实际网络或模型评测。')
            kind = st.selectbox('练习内容', ['缺少必需列', '缺值与无效记录'], key='_sample_kind')
            if bundle:
                content = problem_sample(bundle['config']['features'], kind)
                name = '格式练习_' + kind + '.csv'
                st.download_button('下载问题样例', content, name, 'text/csv', key='sample_download')
        else:
            try:
                content = (ARTIFACTS / 'demo.csv').read_bytes()
                name = '演示样本（测试集固定抽样）'
            except OSError:
                st.warning('演示文件暂不可用，请上传文件。')
        blocked = False
        if content is not None and bundle is not None:
            key = (hashlib.sha256(content).hexdigest(), name, source, model_identity)
            if u.preflight_key != key:
                u.preflight, u.preflight_error, u.run_error = None, None, None
                u.preflight_key = key
                try:
                    u.preflight = preflight_input(content, bundle, model_identity)
                except Exception as error:
                    u.preflight_error = u.run_error = present_error(error)
            blocked = u.preflight_error is not None
            if u.preflight:
                info = u.preflight.summary()
                st.caption(f'格式通过：{info["rows"]:,} 条记录 · {info["columns"]} 列（必需 {info["required"]}，额外 {info["extra"]}）')
        a, b, c = st.columns([2, 2, 3])
        if a.button('开始检查', key='run', type='primary', width='stretch', disabled=blocked or content is None or bundle is None or u.pending is not None):
            request('run', (content, name, source))
            st.rerun()
        if valid_result and b.button('取消更换', key='cancel_import', width='stretch'):
            u.show_import, u.run_error = False, None
            st.rerun()
        with c.popover('格式说明', icon=':material/help:'):
            st.write('支持 CSV，最多 50 MiB、50,000 条；数据列须与演示文件一致，标签列不参与分析。')
            demo = ARTIFACTS / 'demo.csv'
            if demo.is_file():
                st.download_button('下载演示文件', demo.read_bytes(), 'liushao_demo.csv', 'text/csv', key='demo_download')
            if bundle:
                with st.expander('查看需要的数据列'):
                    st.write(bundle['config']['features'])
    if u.run_error:
        st.error('文件检查未完成，请查看原因后重试。')
        if hasattr(u.run_error, 'reason'):
            st.write(u.run_error.reason)
            st.write(u.run_error.action)
            with st.expander('技术详情'):
                st.text(u.run_error.detail)
        else:
            st.write(u.run_error)
        if valid_result:
            st.caption('下方仍是上一次成功检查的文件结果：' + literal_markdown(u.active.name))


def overview():
    if not valid_result or u.show_import:
        import_card()
    if not valid_result:
        if u.active and bundle:
            st.info('分析程序或模型信息已改变，请重新检查文件。原来已保存的操作仍在本地。')
        return
    file_bar()
    anomalies = int(pred.rf_anomaly.sum())
    hints = int(needs_review(pred).sum())
    st.markdown(f'**检查完成：{anomalies:,} 条出现异常提示，{len(rejected):,} 条无法判断。**')
    a, b, c = st.columns(3)
    a.metric('系统提示异常', f'{anomalies:,}')
    b.metric('未提示异常', f'{len(pred) - anomalies:,}')
    c.metric('无法判断', f'{len(rejected):,}')
    if 'cp_set' in pred:
        counts = pred.cp_set.value_counts()
        st.caption(f"可信研判引擎（RF/LR/ECOD 证据融合 + 共形预测）：攻击 {int(counts.get('攻击', 0)):,} 条 · "
                   f"正常 {int(counts.get('正常', 0)):,} 条 · 不确定 {int(counts.get('不确定', 0)):,} 条；"
                   "“需要再检查”收录不确定与高冲突记录。")
    a, b = st.columns(2)
    for col, label, scope, count in [(a, '查看异常提示', '异常提示', anomalies),
                                    (b, '查看需要再检查的记录', '需要再检查', hints)]:
        col.button(f'{label}（{count:,}）', key='open_' + scope, type='primary' if scope == '异常提示' else 'secondary',
                   width='stretch', disabled=count == 0, on_click=request,
                   args=('review_scope', dict(default_filters(), scope=scope)))
    if len(pred):
        count = int(pred.review_status.isin(['确认正常', '确认异常']).sum())
        if available:
            st.progress(count / len(pred), text=f'人工检查：已确认 {count:,} / {len(pred):,} 条')
            if count == 0:
                if audit or dirty(u.editor):
                    st.caption('检查已开始，尚无确认结果；待补证和未保存填写不计入进度。')
                else:
                    st.caption('尚未开始检查，点击上方按钮开始；保存判断后更新进度。')
        else:
            st.caption('人工检查进度暂不可用。')
    distribution, problems = st.columns([3, 2], gap='large')
    with distribution:
        if len(pred):
            render_histogram(raw_pred, len(rejected), bundle['config'])
        else:
            st.info('本次没有可判断的分数，请先查看文件问题并核对原文件。')
    with problems:
        with st.container(border=True):
            st.subheader('文件问题摘要')
            st.caption(f'本次导入 {u.result.total:,} 条记录')
            if u.quality is None:
                st.warning('文件问题明细暂不可用，请到“文件问题”查看原因。')
            else:
                st.write(f'{u.quality.accepted_problem_rows:,} 条填补后可判断')
                st.write(f'{u.quality.rejected_rows:,} 条无法判断')
                if not u.quality.affected_rows:
                    st.success('未发现缺失数据或无法判断的记录。')
                else:
                    st.info('请核对原文件中的缺值和无效内容。')
            st.button('查看文件问题', key='overview_problems', on_click=nav, args=('quality',), width='stretch')


def editor_for(rid, next_id):
    editor = u.editor
    if editor is None or editor.record_id != rid or editor.input_key != u.active.key:
        saved = reviews.get(rid, {})
        status, note = saved.get('status', '待复核'), saved.get('note', '')
        u.editor = Draft(record_id=rid, input_key=u.active.key, status=status, note=note,
            original=(status, note), baseline=saved.get('id', 0), next_id=next_id, conflict=False)
    else:
        editor.next_id = next_id
    return u.editor


def current_evidence(rid):
    row_index=pred.index[pred.record_id.eq(rid)][0];record=pred.loc[row_index]
    key=(u.active.key,rid,meta['model_sha256'],meta['descriptor_sha256'],METHOD_VERSION)
    cache=st.session_state.setdefault('_record_evidence_cache',{})
    if key not in cache:
        physical=int(u.physical_rows.at[row_index]) if u.physical_rows is not None else None
        try:
            context_key=st.session_state.get('_raw_context_key')
            if context_key!=u.active.key:
                st.session_state['_raw_context']=read_csv_checked(io.BytesIO(u.active.content))
                st.session_state['_raw_context_key']=u.active.key
            from app_quality import aligned_context
            context=aligned_context(st.session_state['_raw_context'],u.physical_rows,features.index)
            payload=build_record_evidence(record,features,bundle,context=context,meta=dict(meta,input_csv_row=physical))
            if u.quality is not None and rid in u.quality.records.index:
                details=u.quality.details(rid)
                payload['input_quality_details']=details.to_dict(orient='records')
            cache[key]=payload
        except Exception as error:
            cache[key]={'schema_version':2,'method_version':METHOD_VERSION,'record_id':rid,
                'model_version':bundle['model_version'],'input_name':u.active.name,'input_csv_row':physical,
                'input_sha256':meta['input_sha256'],'model_sha256':meta['model_sha256'],
                'descriptor_sha256':meta['descriptor_sha256'],
                'identity':dict(source_file=str(record.source_file),source_row=int(record.source_row),source_sha256=str(record.source_sha256)),
                'scores':dict(rf=float(record.rf_score),lr=float(record.lr_score),rf_threshold=bundle['config']['rf_threshold'],lr_threshold=bundle['config']['lr_threshold']),
                'cards':[],'all_fields':[],'selection':dict(status='unavailable',reason=str(error)),
                'decomposition':dict(status='unavailable',reason=str(error)),
                'assessment':dict(status='证据计算暂不可用，需外部补证',reasons=[str(error)])}
        while len(cache)>8: del cache[next(iter(cache))]
    return cache[key]


def record_panel(rid, order):
    row_index = pred.index[pred.record_id.eq(rid)][0]
    record = pred.loc[row_index]
    pos = order.index(rid) if rid in order else -1
    next_id = order[pos + 1] if pos >= 0 and pos + 1 < len(order) else None
    editor = editor_for(rid, next_id)
    with st.container(key='record_header'):
        engine_label = f" · 引擎研判：{record['cp_set']}" if 'cp_set' in record else ''
        st.markdown(f'### 来源第 {record.source_row} 行 · ' + ('系统提示异常' if record.rf_anomaly else '未提示异常') + engine_label)
        st.caption(f'输入 CSV 第 {int(u.physical_rows.at[row_index])} 行 · 来源文件：'+literal_markdown(str(record.source_file)))
        if rid not in order[u.page_index*50:(u.page_index+1)*50]:
            st.caption('当前编辑记录不在当前页，详情与填写仍保留。')
    payload = current_evidence(rid)
    render_review_checklist(payload)
    with st.container(key='review_workspace'):
        evidence, judgment = st.columns([1.45, 1], gap='large')
        with evidence:
            with st.container(key='record_evidence'):
                render_ruler(record, bundle['config'], compact=True)
                render_record_evidence(payload)
                try:
                    render_engine_panel(record, features.loc[[row_index]], load_trusted_engine(ARTIFACTS, bundle))
                except Exception:
                    st.caption('研判引擎详情暂不可用，已按基学习器结果显示。')
        with judgment:
            with st.container(border=True, key='record_editor'):
                st.markdown('### 我的判断')
                render_editor(editor)
    st.button('查看完整评分依据', key='open_evidence_detail', on_click=open_evidence_detail,
              disabled=u.pending is not None, icon=':material/insights:')
    opened_history = st.session_state.get('_opened_history')
    with st.expander('操作历史', expanded=bool(opened_history and opened_history['record_id'] == rid)):
        entries=[item for item in audit if item['record_id']==rid]
        if not available: st.info('暂时无法读取已保存操作。')
        elif not entries: st.info('这条记录还没有已保存操作。')
        else:
            frame=pd.DataFrame(entries)[['id','status','note','created_at','evidence_snapshot_label']].copy()
            frame.status=frame.status.map(STATUS_NAMES)
            frame.created_at=pd.to_datetime(frame.created_at,utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d %H:%M:%S')
            st.caption('北京时间 · 点击查看保存时的证据')
            st.dataframe(frame.rename(columns={'id':'操作ID','status':'我的判断','note':'备注','created_at':'保存时间','evidence_snapshot_label':'当时证据'}),hide_index=True,width='stretch')
            selected_id=st.selectbox('查看哪次保存', [item['id'] for item in entries],key='_history_choice_'+rid)
            if st.button('查看当时证据',key='read_history_'+rid):
                st.session_state['_opened_history']=evidence_for_review(DB,selected_id)
                st.rerun()
            opened=st.session_state.get('_opened_history')
            if opened and opened['record_id']==rid and opened['id']==selected_id:
                st.info(opened['evidence_snapshot_label'])
                if opened['evidence_snapshot'] is not None:
                    st.json(opened['evidence_snapshot'],expanded=False)


def render_editor(editor):
    rid = editor.record_id
    latest = reviews.get(rid, {}).get('id', 0)
    if editor.conflict or latest != editor.baseline:
        st.warning('这条记录已被其他窗口修改。你的填写已保留，请核对操作历史。')
    locked = u.exports is not None or u.pending is not None
    if u.exports is not None:
        st.info('下载文件已准备好。返回继续检查后可修改。')
        if st.button('返回继续检查', key='unlock_editor'):
            u.exports = None
            st.rerun()
    if u.pending is not None:
        st.info('请先处理上方的保存、放弃或取消切换。')
    widget_identity = (editor.input_key, editor.record_id, editor.baseline)
    if (st.session_state.get('_editor_identity') != widget_identity
            or '_status' not in st.session_state or '_note' not in st.session_state):
        st.session_state['_status'], st.session_state['_note'] = editor.status, editor.note
        st.session_state['_editor_identity'] = widget_identity
    st.selectbox('我的判断', STATUSES, key='_status', format_func=lambda s: STATUS_NAMES[s],
                 disabled=locked or not available, on_change=update_editor)
    st.text_area('备注', key='_note', height=86, max_chars=4000,
                 placeholder='确认正常或异常时，请写下判断依据',
                 disabled=locked or not available, on_change=update_editor)
    a, b = st.columns([2, 1])
    next_save = a.button('保存并下一条', key='save_review_next', type='primary', width='stretch',
                         disabled=locked or not available or editor.next_id is None)
    save = b.button('保存', key='save_review', width='stretch', disabled=locked or not available)
    if next_save or save:
        if save_editor(next_save):
            st.rerun()
    if dirty(editor):
        st.caption('有未保存的填写；切栏目会保留，下载前请先保存。')
    if editor.next_id is None:
        st.caption('当前已是最后一条，仍可点击“保存”。')
    if dirty(editor) or editor.conflict or latest != editor.baseline:
        st.button('放弃填写并读取已保存记录', key='reload_review', disabled=locked or not available,
                  on_click=reload_saved_editor)


def evidence_detail_page():
    st.button('返回复核', key='back_to_review', on_click=nav, args=('review',), icon=':material/arrow_back:')
    identity = st.session_state.get('_evidence_detail_identity')
    if (not valid_result or u.selected is None
            or identity != (u.active.key, u.selected, model_identity)
            or u.selected not in set(pred.record_id)):
        st.warning('当前记录的评分依据已失效，请返回复核重新选择记录。')
        return
    row_index = pred.index[pred.record_id.eq(u.selected)][0]
    record = pred.loc[row_index]
    st.markdown(f'### 来源第 {record.source_row} 行 · ' + ('系统提示异常' if record.rf_anomaly else '未提示异常'))
    st.caption(f'输入 CSV 第 {int(u.physical_rows.at[row_index])} 行 · 来源文件：'+literal_markdown(str(record.source_file)))
    if dirty(u.editor):
        st.caption('未保存的判断和备注已保留，返回复核后可继续填写。')
    tabs = render_evidence_detail(current_evidence(u.selected))
    if tabs is not None:
        with tabs[2]:
            with st.expander('单条原始记录与下载'):
                from app_ui_tools import single_record
                single_record(u, record, row_index, features, bundle, meta, csv_bytes)


def toggle_quick_search():
    u.search_open=not u.search_open


def apply_quick_search():
    u.search_keyword=st.session_state.get('_quick_search_input','')
    u.page_index=0


def review_page():
    if not len(pred):
        st.info('没有可供检查的记录，请到“文件问题”查看原因。')
        st.button('查看文件问题', on_click=nav, args=('quality',))
        return
    f = u.filters
    a, b, c, d = st.columns([2.1, 1.8, 1.4, 1.3])
    for col, name, options, label in [(a, 'scope', ['全部记录', '异常提示', '需要再检查'], '查看范围'),
            (b, 'status', ['全部'] + STATUSES, '检查状态')]:
        key = '_filter_' + name
        st.session_state.setdefault(key, f[name])
        col.selectbox(label, options, key=key, format_func=lambda x: STATUS_NAMES.get(x, x),
                      on_change=filter_change, args=(name, key), disabled=u.pending is not None)
    with c:
        st.markdown('<div style="height:1.75rem"></div>',unsafe_allow_html=True)
        st.button('快速检索',key='quick_search_toggle',on_click=toggle_quick_search,width='stretch')
    with d:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        with st.popover('更多筛选', width='stretch'):
            definitions = [('queue', '排列方式', ['高异常得分', '待核实', '原始顺序']),
                           ('minimum', '最低排序分数', None),
                           ('disagreement', '只看两种判断不一致', None),
                           ('near', '只看接近判断分界', None),
                           ('lr_tie_break', '相同分数时参考第二种判断排序', None),
                           ('include_lr_near', '纳入第二种判断接近分界的记录', None)]
            for name, label, options in definitions:
                key = '_filter_' + name
                st.session_state.setdefault(key, f[name])
                kwargs = dict(key=key, on_change=filter_change, args=(name, key), disabled=u.pending is not None)
                if name == 'queue':
                    st.selectbox(label, options, format_func=lambda x: {'高异常得分': '异常分数从高到低', '待核实': '优先需要再检查', '原始顺序': '按文件顺序'}[x], **kwargs)
                elif name == 'minimum':
                    st.slider(label, 0.0, 1.0, step=.01, **kwargs)
                else:
                    st.checkbox(label, **kwargs)
            st.caption('筛选条件同时生效；排序开关只调整排列顺序。')
    tags = []
    if f['scope'] != '全部记录': tags.append(f['scope'])
    if f['status'] != '全部': tags.append(STATUS_NAMES[f['status']])
    if f['line']: tags.append('第 ' + f['line'] + ' 行')
    if f['queue'] != '高异常得分': tags.append('已调整排列方式')
    if f['minimum']: tags.append(f'分数至少 {f["minimum"]:.2f}')
    for key, label in [('disagreement', '判断不一致'), ('near', '接近分界'),
                       ('lr_tie_break', '同分参考已开'), ('include_lr_near', '扩大检查范围已开')]:
        if f[key]: tags.append(label)
    if tags:
        a, b = st.columns([5, 1])
        a.caption('当前条件：' + ' · '.join(tags))
        b.button('清除筛选', key='clear_filters', on_click=request, args=('filters', default_filters()))
    if u.search_open:
        st.text_input('检索记录、系统提示或我的判断',key='_quick_search_input',value=u.search_keyword,
            placeholder='如 第217809行，输入后回车；清空恢复原顺序',on_change=apply_quick_search)
    table, matches = promote_matches(filtered(pred, dict(f,line=''), bundle),u.search_keyword)
    if u.search_keyword:
        st.caption(f'匹配 {int(matches.sum()):,} 条，已置顶；其他记录保留。' if matches.any() else '未找到匹配，原表保留。')
    order = table.record_id.tolist()
    if u.selected not in set(pred.record_id):
        u.selected = order[0] if order else None
    # Keep the current record across external writes and navigation; edits never jump target.
    pages = max(1, (len(table) + 49) // 50)
    u.page_index = max(0, min(u.page_index, pages - 1))
    start = u.page_index * 50
    chunk = table.iloc[start:start + 50]
    st.caption(f'{len(table):,} 条符合条件 · 点击一行打开')
    rows = pd.DataFrame({'记录': chunk.source_row.map(lambda n: f'第 {n} 行'),
        '系统提示': chunk.rf_anomaly.map({True: '提示异常', False: '未提示异常'}),
        **({'引擎研判': chunk.cp_set, '信任度': chunk.ds_belief.round(3)} if 'cp_set' in chunk else {}),
        '我的判断': chunk.review_status.map(STATUS_NAMES)})
    key = '_records_main'
    st.dataframe(rows, hide_index=True, width='stretch', height=240,
                 selection_mode=['single-row', 'single-cell'], key=key,
                 on_select='ignore' if u.pending else lambda: row_selected(key, chunk.record_id.tolist()))
    a, b, c = st.columns([1, 1.2, 1])
    a.button('上一页', key='prev_page', disabled=u.pending is not None or u.page_index == 0, on_click=page_to, args=(u.page_index - 1,))
    b.caption(f'第 {u["page_index"] + 1} / {pages} 页')
    c.button('下一页', key='next_page', disabled=u.pending is not None or u.page_index == pages - 1, on_click=page_to, args=(u.page_index + 1,))
    if not order and u.selected is None:
        st.info('没有符合条件的记录，请调整或清除筛选。')
    elif u.selected:
        record_panel(u.selected, order)


def quality_page():
    quality = u.quality
    if quality is None:
        st.warning('本次无法生成文件问题明细。分析结果和已保存的操作仍可查看。')
        with st.expander('查看详情'):
            st.write(u.quality_error)
        return
    if not quality.affected_rows:
        st.success('没有发现缺失数据或无法分析的记录。')
        return
    st.write(f'{quality.affected_rows:,} 条记录存在数据问题；其中 {quality.rejected_rows:,} 条无法分析。')
    st.caption(f'{quality.accepted_problem_rows:,} 条已填补后分析')
    records = quality.records
    locate = st.text_input('查找文件行号', key='_quality_line', placeholder='上传文件中的行号')
    if locate.strip():
        number = line_number(locate)
        records = records[records.input_row.eq(number)] if number is not None else records.iloc[:0]
    if records.empty:
        st.info('没有符合行号的问题记录。')
        return
    pages = max(1, (len(records) + 49) // 50)
    u.quality_page = max(0, min(u.quality_page, pages-1))
    previous, current, following = st.columns(3)
    if previous.button('上一页问题', disabled=u.quality_page == 0):
        u.quality_page -= 1
        st.rerun()
    current.caption(f'问题记录第 {u.quality_page+1} / {pages} 页')
    if following.button('下一页问题', disabled=u.quality_page == pages-1):
        u.quality_page += 1
        st.rerun()
    records = records.iloc[u.quality_page*50:(u.quality_page+1)*50]
    ids = records.index.tolist()
    chosen = u.get('quality_selected')
    if chosen not in ids:
        chosen = ids[0]
    a, b = st.columns([1, 1.7], gap='large')
    with a:
        labels = {rid: f'第 {records.loc[rid, "input_row"]} 行' for rid in ids}
        if st.session_state.get('_quality_record') not in ids:
            st.session_state['_quality_record'] = chosen
        chosen = st.selectbox('选择问题记录', ids, format_func=labels.get, key='_quality_record',
                              on_change=lambda: setattr(u, 'quality_selected', st.session_state['_quality_record']))
        u.quality_selected = chosen
        summary = quality.summary
        summary = summary[summary['问题单元格数'].gt(0) | summary['整行拒绝涉及数'].gt(0)].copy()
        summary['字段'] = summary['字段'].map(field_name)
        st.dataframe(summary.rename(columns={'字段': '数据项', '可判断记录缺失数': '缺失记录数', '整行拒绝涉及数': '无法分析记录数', '问题单元格数': '问题数量'}), hide_index=True, width='stretch')
    with b:
        item = records.loc[chosen]
        st.markdown(f'**文件第 {item.input_row} 行**')
        rejected_item = any(word in str(item.disposition) for word in ('拒绝', '无法判断'))
        st.write('这条记录无法分析，请补齐数据后重新上传。' if rejected_item else '部分数据缺失；系统已补齐后分析，建议核对原文件。')
        details = quality.details(chosen).copy()
        if len(details):
            details['字段'] = details['字段'].map(field_name)
            details = details.rename(columns={'字段': '数据项', 'CSV读取值': '文件中的内容', '适配结果': '发现的问题', '处理状态': '处理方式'})
            if '发现的问题' in details: details['发现的问题'] = details['发现的问题'].map(lambda _: '数据为空或无法读取')
            if '处理方式' in details: details['处理方式'] = '无法分析' if rejected_item else '补齐后分析'
            st.dataframe(details, hide_index=True, width='stretch')
        else:
            st.info('没有更具体的数据项原因，请查看原文件。')
        with st.expander('查看原始问题信息'):
            st.write(f'来源：{item.source_file}，第 {item.source_row} 行')
            st.write(item.reason)
            st.dataframe(quality.details(chosen), hide_index=True)
    if len(rejected):
        st.download_button('下载无法分析的记录', csv_bytes(rejected), 'unjudgeable.csv', 'text/csv', key='export_rejected')


DOWNLOAD_DESCRIPTIONS = {
    'predictions': '检查结果 CSV：全部可判断记录的分数、提示、来源行号及最新已保存的判断和备注。',
    'history': '操作记录 JSON：当前模型下历次已保存的判断、备注、证据和保存时间（UTC）。',
    'manifest': '运行信息 JSON：输入文件、模型版本、判断分界、记录数量和存储状态。',
}


def download_page():
    st.write(f'下载全部 {len(pred):,} 条可判断记录，不受筛选和分页影响。')
    st.caption('无法判断的记录可从“文件问题”单独下载。')
    if u.exports is None:
        for description in DOWNLOAD_DESCRIPTIONS.values():
            st.write(description)
    if dirty(u.editor):
        st.warning('有未保存的判断或备注，请先保存，或明确放弃填写，再生成下载文件。')
        st.button('返回填写位置', key='return_editor', on_click=nav, args=('review',))
        st.button('生成下载文件', key='prepare_exports', type='primary', disabled=True)
        return
    if u.exports is None:
        if available:
            st.write(f'已保存 {len({r["record_id"] for r in audit}):,} 条记录，共 {len(audit):,} 次操作。')
            if not audit:
                st.info('尚无已保存操作；可下载检查结果，操作记录为空。')
        else:
            st.warning('暂时无法读取已保存操作，检查结果和操作记录暂停下载；已保存数量暂不可用。仍可生成运行信息。')
        if st.button('生成下载文件', key='prepare_exports', type='primary', disabled=u.pending is not None):
            full_audit=history(DB,model_version=bundle['model_version'],record_ids=set(pred.record_id)) if available else []
            if [(r['id'],r['note'],r['status']) for r in full_audit] != [(r['id'],r['note'],r['status']) for r in audit]:
                st.warning('操作记录刚被其他窗口更新，请刷新后重新生成下载文件。'); return
            u.exports = export_snapshot(u.active, pred, full_audit, meta, available,
                                        features=features, physical_rows=u.physical_rows, bundle=bundle)
            st.rerun()
        return
    fixed = u.exports
    if fixed.meta['review_storage_available']:
        st.success('文件已生成；继续修改后请重新生成。')
    else:
        st.info('本次运行信息已准备好；复核存储暂不可用，未生成检查结果和操作记录。')
    st.markdown('**检查结果 CSV**')
    st.write(DOWNLOAD_DESCRIPTIONS['predictions'].removeprefix('检查结果 CSV：'))
    if fixed.predictions is not None:
        st.download_button('下载检查结果', fixed.predictions, 'liushao_predictions.csv', 'text/csv', key='export_predictions', type='primary', on_click='ignore')
        st.write(f'复核结论单：{fixed.confirmed_count:,} 条最新已保存的确认正常或确认异常记录。')
        st.caption('附保存时的证据与时间；无快照的旧记录会标明。')
        if fixed.missing_input_rows:
            st.warning(f'{fixed.missing_input_rows} 条输入行号不可用，结论单留空，原始来源行号另列。')
        if fixed.confirmed is not None:
            st.download_button('下载复核结论单', fixed.confirmed, 'liushao_confirmed.csv', 'text/csv', key='export_confirmed', on_click='ignore')
    else:
        st.warning('操作记录暂不可用，目前仅能下载运行信息。')
    with st.container(border=True):
        st.markdown('**操作记录 JSON**')
        st.write(DOWNLOAD_DESCRIPTIONS['history'].removeprefix('操作记录 JSON：'))
        if fixed.history is not None:
            st.download_button('下载已保存的操作记录', fixed.history, 'liushao_reviews.json', 'application/json', key='export_history', on_click='ignore')
    with st.container(border=True):
        st.markdown('**运行信息 JSON**')
        st.write(DOWNLOAD_DESCRIPTIONS['manifest'].removeprefix('运行信息 JSON：'))
        st.download_button('下载本次运行信息', fixed.manifest, 'liushao_run.json', 'application/json', key='export_manifest', on_click='ignore')
    if st.button('返回继续检查', key='resume_review_edit'):
        u.exports, u.page = None, 'review'
        st.rerun()
    st.caption('填写已暂时锁定；点“返回继续检查”可修改。')


def help_page():
    st.markdown('''1. 在“查看结果”上传文件或选择演示数据，点击“开始检查”。
2. 在“检查记录”选择一行，查看核查数据和分数。
3. 填写判断和备注，点击“保存”或“保存并下一条”；需要细看时点“查看完整评分依据”。
4. 在“下载结果”生成并下载文件。

确认正常或异常时，请填写依据；证据不足选“需要补充信息”。

切换栏目保留当前填写，重启不保留未保存内容。勾选保留上传文件后，重启可点“继续上次检查”；否则重新导入同一文件。''')
    with st.expander('文件和保存位置'):
        st.write('已保存记录在下方本机数据库中。换电脑或删除数据库后不会自动恢复；下载的操作记录用于留档，暂不支持导入。')
        st.code(str(DB))
    with st.expander('模型与评测'):
        render_evaluation(EVALUATION, verified, bundle)
    with st.expander('怎样理解提示'):
        st.write('双模型判断不一致或接近分界时，结合来源记录复核。“文件问题”列出缺失数据和格式问题。')


if u.page == 'overview':
    overview()
elif u.page == 'evaluation':
    render_evaluation(EVALUATION, verified, bundle)
elif u.page == 'help':
    help_page()
elif u.page == 'about':
    st.write('流哨 · 本机网络记录检查助手')
    st.caption(f'软件 {VERSION} · 本地界面修订 {UI_REVISION}')
    st.write('检查流量、查看依据、保存复核记录。')
    for name, label in [('MODEL_CARD.md', '模型卡'), ('LOCAL_DATA.md', '本地文件与使用边界'), ('MIGRATION.md', '旧版复核迁移')]:
        document = ROOT / 'docs/usage' / name
        if document.is_file():
            with st.expander(label):
                st.markdown(document.read_text(encoding='utf-8'))
    if model_error:
        st.error('分析程序未就绪。')
        st.write(model_error)
    with st.expander('版本与技术信息'):
        st.write('当前模型', bundle['model_version'] if bundle else '暂不可用')
        if bundle:
            st.json({key: bundle['config'][key] for key in ('rf_threshold', 'lr_threshold', 'near_margin', 'data_version')})
            st.write(bundle['config']['features'])
        st.caption('RF：主要判断模型。LR：对照判断模型。阈值：模型使用的判断分界。')
elif u.page == 'evidence_detail':
    evidence_detail_page()
elif not valid_result:
    empty_page()
else:
    file_bar()
    if u.page == 'review':
        review_page()
    elif u.page == 'quality':
        quality_page()
    elif u.page == 'download':
        download_page()
    else:
        from app_ui_tools import auxiliary, batch_page
        if u.page == 'auxiliary':
            auxiliary(u, pred, features, bundle, meta, csv_bytes)
        else:
            batch_page(u, pred, features, bundle, meta, csv_bytes)

from app_desktop_ready import render_desktop_ready
render_navigation_scroll(st.session_state, u.page)
render_desktop_ready(ROOT)
