"""Versioned, serializable engine evidence; never changes frozen model parameters.

The ECOD score chooses one *aggregate* tail branch.  Its additive explanation
must use that same branch, not a separate maximum for each feature.
"""
import math

import numpy as np

ENGINE_EVIDENCE_VERSION = 'engine-evidence-v1'
IDENTITY_COLUMNS = ('engine_sha256', 'engine_descriptor_sha256', 'engine_inference_version')


def engine_identity(record):
    return tuple(str(record.get(name, '')) for name in IDENTITY_COLUMNS)


def ecod_contributions(features_row, engine):
    from liushao_engine.ecod import tail_logs
    if len(features_row) != 1 or not features_row.columns.is_unique:
        raise ValueError('ECOD evidence requires one row with unique fields')
    left, right = tail_logs(engine['ecod'], features_row.to_numpy(dtype=float))
    left, right = left[0], right[0]
    use_left = np.asarray(engine['ecod']['skew_sign']) < 0
    branches = [left, right, np.where(use_left, left, right)]
    totals = np.array([values.sum() for values in branches])
    selected = int(np.argmax(totals))  # deterministic tie: left, right, auto
    values = branches[selected]
    directions = (np.ones(len(values), dtype=bool) if selected == 0 else
                  np.zeros(len(values), dtype=bool) if selected == 1 else use_left)
    fields = [{'field': str(name), 'contribution': float(value),
               'direction': 'left' if directions[i] else 'right',
               'input_missing': bool(not np.isfinite(features_row.iloc[0, i]))}
              for i, (name, value) in enumerate(zip(features_row.columns, values))]
    score = float(totals[selected])
    reconstruction = math.fsum(item['contribution'] for item in fields)
    return {'method': 'winning-aggregate-tail-additive-v1',
            'selected_branch': ('left', 'right', 'auto')[selected],
            'branch_totals': dict(zip(('left', 'right', 'auto'), map(float, totals))),
            'score': score, 'contribution_sum': reconstruction,
            'residual': score - reconstruction, 'fields': fields,
            'top_fields': sorted(fields, key=lambda item: -item['contribution'])[:6]}


def build_engine_evidence(record, features_row, bundle, engine):
    """Bind saved output to verified engine bytes and verify all displayed values."""
    if 'cp_set' not in record:
        return {'schema_version': 1, 'method_version': ENGINE_EVIDENCE_VERSION,
                'status': 'not_used', 'reason': '本次分析未使用研判引擎，仅有 RF/LR 结果。'}
    import liushao_engine
    if engine is None:
        raise ValueError('本次分析使用过研判引擎，但当前参数无法校验；请恢复配套文件并重新分析。')
    identity = engine.get('_artifact_identity', {})
    if (not all(engine_identity(record)) or
            engine_identity(record) != tuple(identity.get(name) for name in IDENTITY_COLUMNS)):
        raise ValueError('当前引擎与这次分析的引擎身份不一致，请重新检查文件后保存。')
    if list(features_row.columns) != list(bundle['config']['features']):
        raise ValueError('Engine evidence feature order does not match frozen model')
    if len(features_row) != 1 or features_row.index[0] != record.name:
        raise ValueError('Engine evidence record and feature row are not aligned')
    from app_engine_inference import analyze_current_engine
    expected = analyze_current_engine(record.to_frame().T, features_row, bundle, engine).iloc[0]
    values = {}
    for name in liushao_engine.COLUMNS:
        value, check = record[name], expected[name]
        if name in ('cp_set', 'engine_version'):
            if str(value) != str(check):
                raise ValueError('Engine evidence result mismatch: ' + name)
            values[name] = str(value)
        elif name in ('cp_set_size', 'engine_review'):
            if value != check:
                raise ValueError('Engine evidence result mismatch: ' + name)
            values[name] = bool(value) if name == 'engine_review' else int(value)
        else:
            if not math.isclose(float(value), float(check), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Engine evidence result mismatch: ' + name)
            values[name] = float(value)
    decomposition = ecod_contributions(features_row, engine)
    if not math.isclose(decomposition['contribution_sum'], values['ecod_score'], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('ECOD contributions do not reconstruct recorded score')
    return {'schema_version': 1, 'method_version': ENGINE_EVIDENCE_VERSION, 'status': 'verified',
            'identity': dict(identity), 'results': values,
            'parameters': {'epsilon': float(engine['conformal']['epsilon']),
                           'conflict_cut': float(engine['conflict_cut']),
                           'va_method': 'exact-isotonic-two-label-refits',
                           'va_calibration_source': 'frozen-conformal-B-class-scores'},
            'source_masses': liushao_engine.source_masses(record, engine).to_dict(orient='records'),
            'ecod_decomposition': decomposition,
            'limitations': ['保存的是当时的模型输出及其计算依据，不是人工确认的真实标签。',
                            'ECOD 贡献解释原始异常分数，不是对最终攻击结论的因果解释。']}


def require_complete_engine_evidence(payload):
    if payload.get('engine', {}).get('status') == 'unavailable':
        raise ValueError(payload['engine']['reason'])
    return payload


def historical_engine_label(payload, current_identity=None):
    """Read historical identity only; never reconstruct a missing old snapshot."""
    block = payload.get('engine') if isinstance(payload, dict) else None
    if not block:
        return '旧快照未包含研判引擎证据；不能据此还原当时的引擎结论。'
    if block.get('status') == 'not_used':
        return '保存时仅使用 RF/LR，未使用研判引擎。'
    if block.get('status') != 'verified':
        return '保存时的研判引擎证据不完整，请查看原始快照说明。'
    identity = block['identity']
    message = '保存时引擎：' + identity['engine_version'] + ' · SHA256 ' + identity['engine_sha256'][:12]
    if current_identity is not None and tuple(identity.get(k) for k in IDENTITY_COLUMNS) != tuple(current_identity):
        message += '；与当前分析引擎不同，以下仅展示当时保存的证据。'
    else:
        message += '；以下仅展示当时保存的证据。'
    return message
