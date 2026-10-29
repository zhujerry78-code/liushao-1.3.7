"""Exact score accounting for the fitted, unchanged binary pipelines."""
from __future__ import annotations

import math
import numpy as np

VERSION = 'rf-path-lr-logit-v1'
TOLERANCE = 1e-10


def _pipeline(predictor):
    return getattr(predictor, 'predictor', predictor)


def _mapping(pipeline, names, count):
    imputer = pipeline.named_steps['impute']
    if not imputer.keep_empty_features:
        raise ValueError('不支持会删除空字段的填补器')
    mapping = [(name, 'value') for name in names]
    if imputer.add_indicator:
        mapping += [(names[int(i)], 'missing_indicator') for i in imputer.indicator_.features_]
    if len(mapping) != count:
        raise ValueError('预处理后的字段映射不一致')
    return mapping


def decompose_scores(frame, bundle):
    """Return unavailable on incompatibility; never manufacture zero vectors."""
    try:
        names = list(bundle['config']['features'])
        if len(frame) != 1 or list(frame.columns) != names:
            raise ValueError('评分分解需要对齐的一条原始记录')
        rf, lr = (_pipeline(bundle[key]) for key in ('rf', 'lr'))
        forest = rf.named_steps['classifier']
        classifier = lr.named_steps['classifier']
        if list(forest.classes_) != [0, 1] or list(classifier.classes_) != [0, 1]:
            raise ValueError('评分分解仅支持当前冻结模型的二分类映射')
        trees = forest.estimators_
        if not trees or forest.n_estimators != len(trees):
            raise ValueError('实际树数与模型配置不一致')
        transformed = np.asarray(rf[:-1].transform(frame), dtype=np.float32)
        mapping = _mapping(rf, names, transformed.shape[1])
        aggregate = np.zeros(transformed.shape[1], dtype=np.float64)
        base = np.float64(0)
        for tree in trees:
            if list(tree.classes_) != [0, 1]:
                raise ValueError('树类别映射与森林不兼容')
            path = set(tree.decision_path(transformed).indices.tolist())
            leaf = int(tree.apply(transformed)[0])
            values = np.asarray(tree.tree_.value[:, 0, :], dtype=np.float64)
            totals = values.sum(axis=1)
            if np.any(totals <= 0):
                raise ValueError('树节点类别质量无效')
            probabilities = values[:, 1] / totals
            base += probabilities[0]
            node = 0
            while node != leaf:
                left, right = int(tree.tree_.children_left[node]), int(tree.tree_.children_right[node])
                children = [child for child in (left, right) if child >= 0 and child in path]
                if len(children) != 1:
                    raise ValueError('实际决策路径无法唯一还原')
                child = children[0]
                aggregate[int(tree.tree_.feature[node])] += probabilities[child] - probabilities[node]
                node = child
        base = float(base / len(trees))
        aggregate /= len(trees)
        rf_rows = {name: {'field': name, 'contribution': 0., 'parts': []} for name in names}
        for (name, kind), value in zip(mapping, aggregate):
            rf_rows[name]['parts'].append({'kind': kind, 'contribution': float(value)})
        for row in rf_rows.values():
            row['contribution'] = math.fsum(part['contribution'] for part in row['parts'])
        rf_sum = math.fsum(row['contribution'] for row in rf_rows.values())
        rf_reconstructed = base + rf_sum
        rf_score = float(rf.predict_proba(frame)[0, 1])
        x = np.asarray(lr[:-1].transform(frame), dtype=np.float64)
        mapping = _mapping(lr, names, x.shape[1])
        if classifier.coef_.shape != (1, x.shape[1]):
            raise ValueError('LR 系数与预处理字段不一致')
        terms = x[0] * classifier.coef_[0]
        lr_rows = {name: {'field': name, 'term': 0., 'parts': []} for name in names}
        for (name, kind), value in zip(mapping, terms):
            lr_rows[name]['parts'].append({'kind': kind, 'term': float(value)})
        for row in lr_rows.values():
            row['term'] = math.fsum(part['term'] for part in row['parts'])
        intercept = float(classifier.intercept_[0])
        logit = intercept + math.fsum(row['term'] for row in lr_rows.values())
        actual_logit = float(classifier.decision_function(x)[0])
        # Stable sigmoid without an overflow for negative logits.
        score = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        actual_score = float(lr.predict_proba(frame)[0, 1])
        errors = {'rf_score': abs(rf_reconstructed-rf_score),
                  'lr_logit': abs(logit-actual_logit), 'lr_score': abs(score-actual_score)}
        if not np.isfinite([base, rf_sum, logit, score, *errors.values()]).all() or max(errors.values()) > TOLERANCE:
            raise ValueError('完整精度重构误差超过 1e-10')
        rows = list(rf_rows.values())
        top = sorted(rows, key=lambda row: (-abs(row['contribution']), names.index(row['field'])))[:6]
        return {'status': 'available', 'method_version': VERSION, 'tolerance': TOLERANCE,
                'errors': errors, 'rf': {'base': base, 'sum': rf_sum, 'reconstructed': rf_reconstructed,
                    'score': rf_score, 'tree_count': len(trees), 'path_dtype': 'float32',
                    'accumulation_dtype': 'float64', 'fields': rows, 'top_fields': [row['field'] for row in top]},
                'lr': {'intercept': intercept, 'logit': logit, 'actual_logit': actual_logit,
                    'score': actual_score, 'reconstructed': score, 'fields': list(lr_rows.values())}}
    except Exception as error:
        return {'status': 'unavailable', 'method_version': VERSION,
                'reason': str(error), 'label': '评分分解暂不可用'}
