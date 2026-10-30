"""AUC orientation folding must not be mislabeled as frozen calibration."""
import numpy as np
import pytest
from scripts.evaluate_engine import ecod_auc_report


def test_frozen_paic_auc_is_measured_after_calibration_not_folded_raw_auc():
    engine={'fusion':{'calibrators':{'ecod':{'x':np.array([0.,1.]),'y':np.array([.5,.5])}}}}
    result=ecod_auc_report(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),engine)
    assert result['ecod_score_raw']==1.
    assert result['ecod_raw_orientation_folded_using_test_labels']==1.
    assert result['ecod_frozen_paic_output']==.5
    assert 'ecod_score_polarity_calibrated' not in result


def test_reversed_frozen_calibrator_is_applied_without_test_label_refitting():
    engine={'fusion':{'calibrators':{'ecod':{'x':np.array([0.,1.]),'y':np.array([1.,0.])}}}}
    result=ecod_auc_report(np.array([0,0,1,1]),np.array([.9,.8,.2,.1]),engine)
    assert result['ecod_score_raw']==0.
    assert result['ecod_frozen_paic_output']==pytest.approx(1.)
