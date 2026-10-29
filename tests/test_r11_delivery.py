import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from scripts import build_windows_demo as builder
from delivery import selfcheck

def test_delivery_allowlist_contains_new_code_and_only_selected_usage_docs():
    for name in ('app_record_tools.py','app_input.py','app_session.py','docs/usage/MODEL_CARD.md',
                 'docs/usage/LOCAL_DATA.md','docs/usage/MIGRATION.md','docs/release/SOURCE_LICENSE.md',
                 'docs/engineering/license-supplement/streamlit/1.50.0-upstream/LICENSE',
                 'docs/engineering/license-supplement/playwright/playwright/driver/package/ThirdPartyNotices.txt',
                 'docs/engineering/dependencies.json', 'app_review_guidance.py',
                 'app_desktop_ready.py', 'delivery/browser-activate.cmd', 'app_navigation_scroll.py'):
        assert name in builder.FILES and builder.allowed(name)
    for name in ('docs/PROGRESS.md','docs/results/private.md','demo-state/reviews.sqlite.last-session/x.input'):
        assert not builder.allowed(name)

def test_diagnostics_exclude_actual_content_and_recovery_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(selfcheck, 'ROOT', tmp_path)
    private=tmp_path/'demo-state/reviews.sqlite.last-session';private.mkdir(parents=True)
    marker=b'PRIVATE_R11_INPUT_AND_NOTE'
    (private/'state.json').write_bytes(marker);(private/('a'*32+'.input')).write_bytes(marker)
    (tmp_path/'demo-state/reviews.sqlite').write_bytes(marker)
    archive=selfcheck.build_archive('synthetic diagnostic',tmp_path/'诊断报告-r11.txt')
    with zipfile.ZipFile(archive) as z:
        assert not any(marker in z.read(name) for name in z.namelist())
        assert not any('last-session' in name or '.sqlite' in name for name in z.namelist())

@pytest.mark.parametrize('revision', ['r11-final', 'r11-evidence-final', 'r11-ui-compact', 'r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'])
def test_r11_summary_is_required_before_assets_are_read(tmp_path, monkeypatch, revision):
    source={name:(builder.ROOT/name).read_bytes() for name in ('scripts/build_windows_demo.py','delivery/safe_files.py')}
    monkeypatch.setattr(builder,'read_source',lambda *a:('a'*40,source))
    with pytest.raises(ValueError,match='require an evaluation summary'):
        builder.build(tmp_path/'out',package_revision=revision,assets_zip=tmp_path/'absent.zip')

def test_prediction_acceptance_tolerates_roundoff_but_never_flag_changes():
    import pandas as pd
    from scripts.engineering_r11_acceptance import compare_predictions
    left=pd.DataFrame(dict(rf_score=[.2],lr_score=[.8],rf_anomaly=[True],lr_anomaly=[False],disagreement=[True],near_threshold=[False]))
    right=left.copy();right.loc[0,'rf_score']+=1e-16
    assert compare_predictions(left,right)['rf_score']<1e-12
    right.loc[0,'disagreement']=False
    with pytest.raises(AssertionError):compare_predictions(left,right)
    right=left.copy();right.loc[0,'rf_score']+=1e-5
    with pytest.raises(AssertionError):compare_predictions(left,right)
