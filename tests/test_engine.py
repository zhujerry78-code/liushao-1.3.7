"""1.3.7 trusted triage engine: algorithm properties and frozen-artifact integration."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from liushao_engine.conformal import ATTACK, NORMAL, UNSURE, fit_conformal, predict_conformal
from liushao_engine.ecod import fit_ecod, score_ecod
from liushao_engine.evidence import bpa, combine
from liushao_engine.mmd import mmd_test
from liushao_engine.venn_abers import fit_va, predict_va

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'


def test_ecod_scores_outliers_higher():
    rng = np.random.default_rng(0)
    model = fit_ecod(rng.normal(size=(2000, 5)))
    inliers, outliers = rng.normal(size=(200, 5)), rng.normal(size=(200, 5)) + 6
    assert score_ecod(model, outliers).min() > np.median(score_ecod(model, inliers))


def test_ecod_handles_missing_and_is_fast():
    rng = np.random.default_rng(1)
    model = fit_ecod(rng.normal(size=(20000, 70)))
    x = rng.normal(size=(100000, 70))
    x[::7, 3] = np.nan
    import time
    tick = time.perf_counter()
    scores = score_ecod(model, x)
    assert np.isfinite(scores).all() and time.perf_counter() - tick < 10


def test_dempster_rule_matches_hand_computation():
    m1, m2 = np.array([[.6, .3, .1]]), np.array([[.2, .7, .1]])
    a, n, u, k = combine([m1, m2])
    conflict = .6 * .7 + .3 * .2
    assert k[0] == pytest.approx(conflict)
    assert a[0] == pytest.approx((.6 * .2 + .6 * .1 + .1 * .2) / (1 - conflict))
    assert n[0] == pytest.approx((.3 * .7 + .3 * .1 + .1 * .7) / (1 - conflict))
    assert a[0] + n[0] + u[0] == pytest.approx(1)


def test_vacuous_source_is_neutral():
    m = bpa(np.array([.9, .1]), .95)
    vacuous = bpa(np.array([.5, .5]), 0.0)
    a, n, u, k = combine([m, vacuous])
    assert np.allclose(a, m[:, 0]) and np.allclose(k, 0)


def test_conformal_marginal_coverage_per_class():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 6000)
    betp = np.clip(np.where(y == 1, rng.beta(5, 2, 6000), rng.beta(2, 5, 6000)), 0, 1)
    params = fit_conformal(betp[:3000], y[:3000], epsilon=.1)
    sets = predict_conformal(params, betp[3000:])
    yt = y[3000:]
    covered = np.where(yt == 1, sets['cp_p_attack'] > .1, sets['cp_p_normal'] > .1)
    for label in (0, 1):
        assert covered[yt == label].mean() >= .9 - .03
    assert set(np.unique(sets['cp_set'])) <= {ATTACK, NORMAL, UNSURE}


def test_venn_abers_interval_is_valid():
    rng = np.random.default_rng(3)
    s = rng.random(1000)
    y = (rng.random(1000) < s).astype(int)
    low, high = predict_va(fit_va(s, y, grid_size=64), rng.random(200))
    assert ((0 <= low) & (low <= high) & (high <= 1)).all()


def test_mmd_detects_shift_only_when_present():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 4))
    same = mmd_test(x, rng.normal(size=(300, 4)), n_perm=100)
    shifted = mmd_test(x, rng.normal(size=(300, 4)) + 1, n_perm=100)
    assert same['p_value'] > .05 and shifted['p_value'] < .05


def frozen():
    if not (ARTIFACTS / 'model.joblib').exists() or not (ARTIFACTS / 'engine.joblib').exists():
        pytest.skip('冻结模型或引擎参数缺失')
    descriptor = json.loads((ARTIFACTS / 'engine.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ARTIFACTS / 'engine.joblib').read_bytes()).hexdigest() == descriptor['engine_sha256']
    return joblib.load(ARTIFACTS / 'model.joblib'), joblib.load(ARTIFACTS / 'engine.joblib')


def test_engine_on_demo_appends_columns_and_attribution_is_exact():
    from liushao.data import read_csv_checked
    from liushao.model import infer
    import liushao_engine
    from liushao_engine.attribution import contributions, tree_paths
    bundle, engine = frozen()
    with (ARTIFACTS / 'demo.csv').open('rb') as stream:
        raw = read_csv_checked(stream)
    pred, features, _ = infer(raw, bundle, 'demo', 'demo.csv')
    out = liushao_engine.analyze(pred, features, bundle, engine)
    assert list(out.columns[:len(pred.columns)]) == list(pred.columns)
    assert out[pred.columns].equals(pred)
    assert set(liushao_engine.COLUMNS) <= set(out.columns)
    assert ((out.ds_belief >= 0) & (out.ds_belief <= out.ds_plausibility + 1e-12) & (out.ds_plausibility <= 1)).all()
    queue = liushao_engine.engine_review_ranked(out)
    assert queue.engine_review.all() and len(queue) == int(out.engine_review.sum())
    assert len(liushao_engine.engine_ranked(out)) == len(out)
    contrib, bias = contributions(tree_paths(bundle['rf'], bundle['config']['features']), bundle['rf'], features.iloc[:50])
    assert np.abs(bias + contrib.sum(axis=1) - out.rf_score.iloc[:50].to_numpy()).max() < 1e-9
    assert len(liushao_engine.source_masses(out.iloc[0], engine)) == 3
    assert len(liushao_engine.ecod_explain(features.iloc[[0]], engine)) == 6


def test_engine_falls_back_without_artifact(tmp_path):
    from app_service import with_engine
    frame = pd.DataFrame({'rf_score': [.5]})
    assert with_engine(frame, frame, {'model_version': 'x'}, tmp_path) is frame


def test_ranking_falls_back_to_base_without_engine_columns():
    from liushao.model import ranked
    from liushao_engine import engine_ranked
    frame = pd.DataFrame({'record_id': ['b', 'a'], 'rf_score': [.1, .9]})
    assert engine_ranked(frame).equals(ranked(frame))
