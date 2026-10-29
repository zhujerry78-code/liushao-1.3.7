"""Bounded pure aggregation and local Vega-Lite score displays."""
from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=2)
def _histogram(scores):
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError('分数必须是 0 到 1 之间的有限数值')
    values, edges = np.histogram(scores, bins=20, range=(0, 1))
    return tuple((float(edges[i]), float(edges[i+1]), int(value)) for i, value in enumerate(values))


def histogram_rows(scores):
    return [{'start': start, 'end': end, 'count': count}
            for start, end, count in _histogram(tuple(float(value) for value in scores))]


def histogram_spec(scores, threshold, margin, *, count_scale='symlog'):
    if count_scale not in ('symlog', 'linear'):
        raise ValueError('不支持的数量刻度')
    rows = histogram_rows(scores)
    scale = {'type': count_scale, 'zero': True}
    axis = {'format': ',d', 'tickMinStep': 1}
    if count_scale == 'symlog':
        scale['constant'] = 1
        maximum = max(1, max(row['count'] for row in rows))
        scale.update(domain=[0, maximum], nice=False)
        ticks = [0]
        value = 1
        while value < maximum:
            ticks.append(value)
            value *= 10
        axis['values'] = ticks + [maximum]
    return {'height': 230, 'layer': [
        {'data': {'values': rows}, 'mark': {'type': 'rect', 'color': '#245C8D'},
         'encoding': {'x': {'field': 'start', 'type': 'quantitative', 'title': 'RF 排序分数', 'scale': {'domain': [0, 1], 'nice': False}},
                      'x2': {'field': 'end'}, 'y': {'field': 'count', 'type': 'quantitative',
                          'title': '记录数（对数刻度）' if count_scale == 'symlog' else '记录数',
                          'scale': scale, 'axis': axis}, 'y2': {'datum': 0},
                      'tooltip': [{'field': 'start', 'title': '区间起点', 'format': '.2f'},
                                  {'field': 'end', 'title': '区间终点', 'format': '.2f'},
                                  {'field': 'count', 'title': '记录数'}]}},
        {'data': {'values': [{'lo': max(0., threshold-margin), 'hi': min(1., threshold+margin)}]},
         'mark': {'type': 'rect', 'color': '#F4B860', 'opacity': .22},
         'encoding': {'x': {'field': 'lo', 'type': 'quantitative'}, 'x2': {'field': 'hi'},
                      'tooltip': [{'field': 'lo', 'title': '范围下限', 'format': '.2f'},
                                  {'field': 'hi', 'title': '范围上限', 'format': '.2f'}]}},
        {'data': {'values': [{'threshold': threshold}]}, 'mark': {'type': 'rule', 'color': '#B54C25', 'strokeWidth': 2},
         'encoding': {'x': {'field': 'threshold', 'type': 'quantitative'},
                      'tooltip': [{'field': 'threshold', 'title': 'RF 判断分界'}]}}
    ], 'config': {'view': {'stroke': None}}}


def ruler_spec(rf_score, lr_score, config):
    values = [{'model': name.upper(), 'score': float(score), 'threshold': config[name+'_threshold'], 'zero': 0., 'one': 1.}
              for name, score in (('rf', rf_score), ('lr', lr_score))]
    y = {'field': 'model', 'type': 'nominal', 'title': None, 'sort': ['RF', 'LR']}
    return {'height': 160, 'layer': [
        {'data': {'values': values}, 'mark': {'type': 'rule', 'strokeWidth': 8, 'color': '#DCE4EE'},
         'encoding': {'x': {'field': 'zero', 'type': 'quantitative', 'title': '排序分数', 'scale': {'domain': [0, 1], 'nice': False}},
                      'x2': {'field': 'one'}, 'y': y, 'tooltip': None}},
        {'data': {'values': [{'model': 'RF', 'lo': max(0., config['rf_threshold']-config['near_margin']),
                             'hi': min(1., config['rf_threshold']+config['near_margin'])}]},
         'mark': {'type': 'rule', 'strokeWidth': 10, 'color': '#F4B860'},
         'encoding': {'x': {'field': 'lo', 'type': 'quantitative'}, 'x2': {'field': 'hi'}, 'y': y,
                      'tooltip': [{'field': 'lo', 'title': '范围下限', 'format': '.2f'},
                                  {'field': 'hi', 'title': '范围上限', 'format': '.2f'}]}},
        {'data': {'values': values}, 'mark': {'type': 'tick', 'color': '#B54C25', 'thickness': 2, 'size': 24},
         'encoding': {'x': {'field': 'threshold', 'type': 'quantitative'}, 'y': y,
                      'tooltip': [{'field': 'model', 'title': '模型'}, {'field': 'threshold', 'title': '判断分界'}]}},
        {'data': {'values': values}, 'mark': {'type': 'point', 'filled': True, 'color': '#17314D', 'size': 110},
         'encoding': {'x': {'field': 'score', 'type': 'quantitative'}, 'y': y,
                      'tooltip': [{'field': 'model', 'title': '模型'}, {'field': 'score', 'title': '当前分数', 'format': '.6f'}]}}
    ], 'config': {'view': {'stroke': None}}}


def render_histogram(predictions, rejected_count, config):
    import streamlit as st
    if predictions.empty:
        return
    st.subheader('本次记录的分数分布')
    choice = st.radio('数量刻度', ['对数数量', '线性数量'], horizontal=True, key='_histogram_scale')
    st.vega_lite_chart(histogram_spec(predictions.rf_score, config['rf_threshold'], config['near_margin'],
                                    count_scale='symlog' if choice == '对数数量' else 'linear'), width='stretch')
    largest = max(histogram_rows(predictions.rf_score), key=lambda row: row['count'])
    st.caption(f"最多集中在 {largest['start']:.2f}–{largest['end']:.2f}：{largest['count']:,} 条（{largest['count']/len(predictions):.1%}）。")
    st.caption(f"图中 {len(predictions):,} 条，另有 {rejected_count:,} 条无法判断。"
               f"橙线：RF 分界 {config['rf_threshold']:.2f}；浅橙：接近分界。")


def render_ruler(record, config, *, compact=False):
    import streamlit as st
    spec = ruler_spec(record.rf_score, record.lr_score, config)
    if compact:
        spec['height'] = 160
        for layer in spec['layer']:
            data = layer.get('data', {}).get('values', [])
            for item in data:
                if 'score' in item:
                    item['label'] = f"{item['score']:.3f} / 分界 {item['threshold']:.2f}"
        spec['layer'].append({'data': spec['layer'][0]['data'],
            'mark': {'type': 'text', 'align': 'right', 'dy': -16, 'fontSize': 12, 'color': '#17314D'},
            'encoding': {'x': {'datum': 1}, 'y': spec['layer'][0]['encoding']['y'],
                         'text': {'field': 'label'}}})
    st.vega_lite_chart(spec, width='stretch')
    if compact:
        st.caption('实心点：当前分数 · 橙线：判断分界 · 浅橙段：接近分界')
        return
    st.caption(f"RF {record.rf_score:.6f} / 分界 {config['rf_threshold']:.2f} · "
               f"LR {record.lr_score:.6f} / 分界 {config['lr_threshold']:.2f}。实心点为分数，浅橙段为接近分界。")
