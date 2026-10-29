"""On-demand r8 analysis tools; outputs stay separate from primary predictions."""
import json

import pandas as pd
import streamlit as st


def download_bytes(state, name, value, serializer):
    cache = state['tool_downloads']
    identity = (state['active']['key'], state['clues'][:2] if name == 'clues' else name)
    if name not in cache or cache[name][0] != identity:
        cache[name] = (identity, serializer(value))
    return cache[name][1]


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def auxiliary(u, pred, features, bundle, meta, csv_bytes):
    if pred.empty:
        st.info('没有可分析的记录，请先查看文件问题。')
        return
    with st.expander('计算方法'):
        st.write('RF 同分时按 LR 排序；扩大范围会纳入两种模型接近分界的记录。')
        st.write('RF 树概率标准差表示树间差异，不是校准后的不确定度。')
    if st.button('生成辅助检查', key='build_diagnostics', type='primary'):
        from app_triage import diagnostics
        try:
            import time
            tick = time.perf_counter()
            with st.spinner('正在生成辅助检查…'):
                result = diagnostics(pred, features, bundle)
                result['input_csv_row'] = u['physical_rows']
                u['tool_downloads'].pop('diagnostics', None)
                u['tool_downloads'].pop('diagnostic_manifest', None)
            u['diagnostics'] = (result, time.perf_counter() - tick)
        except Exception as error:
            u['diagnostics'] = None
            st.warning(f'暂时无法生成，普通检查仍可使用：{error}')
    if u['diagnostics'] is not None:
        from app_triage import diagnostic_view
        result, seconds = u['diagnostics']
        choice = st.radio('查看方式', ['参考第二种判断排序', '扩大需要再检查的范围'], horizontal=True, key='_diagnostic_view')
        extra = diagnostic_view(pred, result, 'lr_tie_break' if choice == '参考第二种判断排序' else 'dual_near')
        st.write(f'原提示 {int((pred.disagreement | pred.near_threshold).sum()):,} 条；扩大范围后 {int(result.diagnostic_review_hint.sum()):,} 条。')
        display = extra[['source_row', 'rf_score', 'lr_score', 'rf_anomaly', 'disagreement', 'near_threshold']].rename(columns={
            'source_row': '来源行号', 'rf_score': '主要排序分数', 'lr_score': '对照分数',
            'rf_anomaly': '系统提示异常', 'disagreement': '两种判断不一致', 'near_threshold': '接近判断分界'})
        st.dataframe(display.head(1000), hide_index=True, width='stretch', height=360)
        st.caption(f'显示前 1,000 条；下载包含全部记录。计算耗时 {seconds:.2f} 秒。')
        st.download_button('下载辅助检查数据', download_bytes(u, 'diagnostics', result, csv_bytes), 'liushao_diagnostics.csv', 'text/csv', key='export_diagnostics', on_click='ignore')
        manifest = dict(meta, diagnostic_schema='liushao-sidecar-1', candidate_version='1.3.0-rc.2',
            diagnostic_seconds=seconds, lr_margin_rule='reuse_frozen_near_margin_experimental',
            rf_std_definition='population_std_of_tree_class1_probabilities_ddof0')
        st.download_button('下载辅助检查信息', download_bytes(u, 'diagnostic_manifest', manifest, json_bytes), 'liushao_diagnostics_run.json', 'application/json', key='export_diagnostic_manifest', on_click='ignore')


def batch_page(u, pred, features, bundle, meta, csv_bytes):
    if pred.empty:
        st.info('没有可分析的记录，请先查看文件问题。')
        return
    with st.expander('计算方法'):
        st.write('逐项对照训练参考范围；分组来自系统提示。')
        st.write('相似记录按同来源和六位有效数字指纹分组，不等于网络会话。')
    if st.button('生成整批数据分析', key='batch_reference', type='primary'):
        from app_triage import batch_reference
        try:
            with st.spinner('正在生成整批数据分析…'):
                u['batch'] = batch_reference(pred, features, bundle)
                u['tool_downloads'].pop('batch_groups', None)
                u['tool_downloads'].pop('batch_reference', None)
                u['tool_downloads'].pop('batch_intervals', None)
        except Exception as error:
            u['batch'] = None
            st.warning(f'暂时无法生成，普通检查仍可使用：{error}')
    if u['batch'] is not None:
        batch = u['batch']
        st.dataframe(pd.DataFrame(list(batch['summary']['triage_counts'].items()),
                                 columns=['系统分组', '记录数']), hide_index=True)
        interval_df = batch.get('interval_profile')
        if isinstance(interval_df, pd.DataFrame) and not interval_df.empty:
            st.markdown(f'#### {len(interval_df)} 项参考比较')
            st.caption('按偏离记录数排序；比例仅含有效观测，缺失另列。')
            export_intervals = interval_df.drop(columns=['高风险极值条数', '一般超限条数', '参考不足/未比较条数'], errors='ignore').rename(columns={
                '较大偏离条数': '显著偏离参考条数', '区间核心内条数': '参考区间内部条数',
                '区间取值态势提示': '核查提示'})
            preview_columns = ['数据项', '参考类型', '有效观测条数', '本批越界率 (%)', '原始缺失条数', '核查提示']
            st.dataframe(export_intervals[[name for name in preview_columns if name in export_intervals]],
                         hide_index=True, width='stretch', height=340)
            with st.expander('全部数值'):
                st.caption('连续区间、离散编号、固定参考和参考不足分别统计；“超出区间合计”已含各分项。')
                st.dataframe(export_intervals, hide_index=True, width='stretch', height=340)
            st.download_button('下载整批区间取值状况表',
                               download_bytes(u, 'batch_intervals', export_intervals, csv_bytes),
                               'liushao_batch_intervals.csv', 'text/csv',
                               key='export_batch_intervals', on_click='ignore')
        with st.expander('详细统计'):
            st.write(batch['summary']['text'])
            st.dataframe(batch['groups'].head(1000), hide_index=True, width='stretch')
        st.download_button('下载分组与相似记录', download_bytes(u, 'batch_groups', batch['groups'], csv_bytes), 'liushao_batch_groups.csv', 'text/csv', key='export_batch_groups', on_click='ignore')
        st.download_button('下载整批分析信息', download_bytes(u, 'batch_reference', dict(meta, batch_reference=batch['summary']), json_bytes),
                           'liushao_batch_reference.json', 'application/json', key='export_batch_reference', on_click='ignore')


def single_record(u, record, row_index, features, bundle, meta, csv_bytes):
    with st.expander('单项替换试算'):
        st.caption('每次替换一项为训练中位数，查看分数变化。')
        with st.popover('计算说明'):
            st.write('Top-3：RF 按全局重要性与统计偏离选取，LR 保留正负方向。')
            st.write('这是启发式试算；替换组合不一定符合实际业务。')
        if st.button('计算这条记录', key='feature_clues'):
            from app_triage import feature_clues
            try:
                clues = feature_clues(features.loc[row_index], bundle)
                u['clues'] = (u['active']['key'], record.record_id, clues)
                u['tool_downloads'].pop('clues', None)
            except Exception as error:
                u['clues'] = None
                st.warning(f'暂时无法计算，普通检查仍可使用：{error}')
        cache = u['clues']
        if cache and cache[:2] == (u['active']['key'], record.record_id):
            clues = cache[2]
            st.dataframe(clues['top3'], hide_index=True, width='stretch')
            st.dataframe(clues['perturbations'], hide_index=True, width='stretch')
            st.write(clues['summary']['text'])
            payload = dict(model_sha256=meta['model_sha256'], input_sha256=meta['input_sha256'],
                record_id=record.record_id, source_row=int(record.source_row),
                input_csv_row=int(u['physical_rows'].loc[row_index]), summary=clues['summary'],
                top3=json.loads(clues['top3'].to_json(orient='records', force_ascii=False, double_precision=15)),
                perturbations=json.loads(clues['perturbations'].to_json(orient='records', force_ascii=False, double_precision=15)))
            st.download_button('下载这条记录的试算结果', download_bytes(u, 'clues', payload, json_bytes), 'liushao_feature_clues.json',
                               'application/json', key='export_feature_clues', on_click='ignore')
