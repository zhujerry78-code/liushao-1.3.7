import hashlib
import io
import zipfile

import pytest


def api():
    from scripts import verify_frozen_release
    return verify_frozen_release


def test_archive_hash_is_checked_before_any_member_is_accepted():
    a = api()
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('model', b'not a trusted pickle')
    content = data.getvalue()
    with pytest.raises(ValueError, match='SHA-256'):
        a.checked_zip_member(content, '0' * 64, 'model')
    assert a.checked_zip_member(content, hashlib.sha256(content).hexdigest(), 'model') == b'not a trusted pickle'


def test_metric_comparison_rejects_missing_keys_and_different_confusion_counts():
    a = api()
    expected = {'f1': .9861042183622829, 'fp': 56, 'threshold': .12}
    assert a.compare_metrics(expected, {**expected, 'f1': expected['f1'] + 1e-15})
    assert not a.compare_metrics(expected, {'f1': expected['f1'], 'fp': 56})
    assert not a.compare_metrics(expected, {**expected, 'fp': 55})
    assert not a.compare_metrics(expected, {**expected, 'threshold': .13})
