"""Independent direct isotonic reference, frozen worst case and cache boundaries."""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from app_engine_inference import ExactVennAbers, exact_va
from test_contest_engine_snapshot import frozen_engine_case  # noqa: F401


def direct(conformal, query):
    normal, attack = np.asarray(conformal['normal']), 1-np.asarray(conformal['attack'])
    x = np.r_[normal,attack]
    y = np.r_[np.zeros(len(normal)),np.ones(len(attack))]
    values = []
    for label in (0.,1.):
        iso = IsotonicRegression(y_min=0.,y_max=1.,out_of_bounds='clip')
        iso.fit(np.r_[x,query],np.r_[y,label])
        values.append(float(iso.predict([query])[0]))
    return sorted(values)


def test_exact_frozen_extreme_replaces_large_interpolation_error(frozen_engine_case):
    engine = frozen_engine_case[3]
    low, high = exact_va(engine,np.array([.1523090648485763]))
    assert low[0] == pytest.approx(.08163265306122448, abs=1e-12)
    assert high[0] == pytest.approx(.08843537414965986, abs=1e-12)
    from liushao_engine.venn_abers import predict_va
    old = predict_va(engine['venn_abers'],np.array([.1523090648485763]))
    assert old[1][0]-high[0] > .37


def test_exact_frozen_grid_endpoints_and_midpoints_match_direct_raw_sample(frozen_engine_case):
    engine = frozen_engine_case[3]
    grid = engine['venn_abers']['grid']
    queries = np.unique(np.r_[grid,(grid[1:]+grid[:-1])/2,np.linspace(0,1,101)])
    low,high = exact_va(engine,queries)
    expected = np.array([direct(engine['conformal'],float(q)) for q in queries])
    assert np.max(np.abs(np.column_stack([low,high])-expected)) < 1e-12


def test_tied_scores_and_one_ulp_neighbours_match_raw_reference():
    conformal = {'normal':np.array([.1,.1,.5,.8,.9]),
                 'attack':1-np.array([.1,.5,.5,.9,.9])}
    queries = np.r_[0,.1,np.nextafter(.1,0),np.nextafter(.1,1),.3,.5,.9,1]
    predictor = ExactVennAbers(conformal)
    low,high = predictor.predict(queries)
    reference = np.array([direct(conformal,float(q)) for q in queries])
    assert np.max(np.abs(np.column_stack([low,high])-reference)) < 1e-12
    before=predictor.endpoints.cache_info()
    predictor.predict(np.tile(queries,100))
    assert predictor.endpoints.cache_info().misses == before.misses


def test_exact_va_empty_and_invalid_queries():
    predictor=ExactVennAbers({'normal':[.2,.3],'attack':[.2,.3]})
    assert all(len(v)==0 for v in predictor.predict([]))
    for values in ([float('nan')],[-.1],[1.1],[[.5]]):
        with pytest.raises(ValueError): predictor.predict(values)
