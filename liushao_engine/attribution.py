"""Tree-path contribution decomposition (Saabas) for the frozen random forest.

For every tree, the attack-class fraction changes from parent to child along the
decision path; that change is credited to the parent's split feature.  Averaged
over trees: rf_score(x) = bias + sum_j contribution_j(x), exactly.
"""
import numpy as np
from scipy import sparse


def tree_paths(rf_pipeline, features):
    imputer, forest = rf_pipeline.named_steps['impute'], rf_pipeline.named_steps['classifier']
    n_original = len(features)
    # Missing-value indicator columns are credited back to the feature they flag.
    extra = [] if imputer.indicator_ is None else [int(i) for i in imputer.indicator_.features_]
    to_original = np.array(list(range(n_original)) + extra)
    rows, cols, vals, bias = [], [], [], []
    offset = 0
    for est in forest.estimators_:
        tree = est.tree_
        value = tree.value[:, 0, :]
        frac = value[:, 1] / value.sum(axis=1)
        bias.append(frac[0])
        parents = np.arange(tree.node_count)
        for children in (tree.children_left, tree.children_right):
            inner = children >= 0
            rows.append(offset + children[inner])
            cols.append(to_original[tree.feature[parents[inner]]])
            vals.append(frac[children[inner]] - frac[parents[inner]])
        offset += tree.node_count
    n_trees = len(forest.estimators_)
    weights = sparse.csr_matrix((np.concatenate(vals) / n_trees, (np.concatenate(rows), np.concatenate(cols))),
                                shape=(offset, n_original))
    return {'weights': weights, 'bias': float(np.mean(bias)), 'features': list(features)}


def contributions(paths, rf_pipeline, x):
    """Rows x features contribution matrix and the shared bias term."""
    xt = rf_pipeline.named_steps['impute'].transform(x)
    indicator, _ = rf_pipeline.named_steps['classifier'].decision_path(xt.astype(np.float32))
    return np.asarray((indicator @ paths['weights']).todense()), paths['bias']
