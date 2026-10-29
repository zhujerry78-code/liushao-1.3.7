"""Pure reference interval semantics shared by the library and application."""
import math

EFFECT = .01
JOINT_EFFECT_GATE = .10
DIRECTION_EPSILON = 1e-6
HIGH_RISK_SPAN_MULTIPLIER = 1.0
PARAMETERS = {'rf_display_effect': EFFECT, 'lr_direction_epsilon': DIRECTION_EPSILON,
              'lr_small_effect': .01, 'joint_effect': JOINT_EFFECT_GATE, 'stability_overlap': .5,
              'reference_move': .1, 'near_boundary': .05,
              'high_risk_span_multiplier': HIGH_RISK_SPAN_MULTIPLIER,
              'engineering_rules_status': '待验证的工程解释规则',
              'practical_conflict': {'status': 'uncalibrated', 'label': '实际影响待评估',
                                     'tau_rf': None, 'tau_lr': None, 'report_id': None}}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def continuous(name):
    return name != 'Protocol' and 'Flag' not in name and 'Port' not in name


def reference_position(name, value, reference):
    # Invalid numeric input is missing, never an in-range observation.
    value = number(value)
    low, mid, high = (number((reference or {}).get(key)) for key in ('q01', 'median', 'q99'))
    valid = all(v is not None for v in (low, mid, high)) and low <= mid <= high and math.isfinite(high - low)
    is_cont = continuous(name)
    result = {'valid': valid, 'continuous': is_cont, 'outside': False,
              'distance': 0., 'near': False, 'fixed': bool(valid and low == high),
              'high_risk': False, 'severity_band': '训练正常参考不足（未知）', 'severity_rank': 3}
    if value is None:
        label = '原始缺失'
        result.update(severity_band='原始缺失', severity_rank=3)
    elif not valid:
        label = '训练正常参考不足'
        result.update(severity_band='训练正常参考不足（未知）', severity_rank=3)
    elif not is_cont:
        label = '离散取值，需核对协议/标志/端口语义'
        result.update(severity_band='离散协议/端口/标志编码', severity_rank=3)
    elif low == high:
        result['outside'] = value != low
        label = '偏离固定参考值' if value != low else '固定参考值'
        result.update(severity_band='偏离固定参考值' if value != low else '固定参考值',
                      severity_rank=1 if value != low else 3)
    elif value < low or value > high:
        result['outside'] = True
        amount = low - value if value < low else value - high
        distance = amount / (high - low)
        high_risk = distance > HIGH_RISK_SPAN_MULTIPLIER
        result['distance'] = distance
        result['outside_amount'] = amount
        result['high_risk'] = high_risk
        label = '低于训练正常 P01' if value < low else '高于训练正常 P99'
        band = ('显著偏离参考（超限>1倍跨度）' if high_risk else
                ('低于训练正常 P01（超出区间）' if value < low else '高于训练正常 P99（超出区间）'))
        result.update(severity_band=band, severity_rank=0 if high_risk else 1)
    else:
        lower_side = value < mid
        span = mid - low if lower_side else high - mid
        distance = ((value - low if lower_side else high - value) / span
                    if value != mid and span > 0 else 1.)
        near = value != mid and span > 0 and distance <= PARAMETERS['near_boundary']
        result.update(near=near, boundary_fraction=distance,
                      near_side='下界' if lower_side else '上界', boundary_basis='中位数分段跨度')
        label = '接近参考边界（区间内）' if near else '参考区间内'
        result.update(severity_band='接近参考边界（区间内）' if near else '参考区间核心内',
                      severity_rank=2 if near else 4)
    result['status'] = label
    return result
