import copy
import json

import pytest

from scripts.publish_evidence import digest, encoded, public_sources, redact_paths
from tests.test_r10_evaluation import evaluation_data


@pytest.fixture
def original_evidence(evaluation_data):
    names = ['docs/phase2/audit.json', 'docs/innovation/study.json',
             'docs/release/MODEL_COMPARISON_20261016.json', 'docs/results/metrics.json',
             'docs/phase3/metrics.json', 'docs/phase2/metrics.json']
    raw = {name: encoded({'metrics': {'f1': .987}, 'url': 'https://example.org/data'}) for name in names}
    raw[names[0]] = encoded({'metrics': {'f1': .987}, 'path': r'C:\Users\private-author\Documents\data.csv'})
    raw[names[2]] = encoded({'metrics': {'f1': .986}, 'paths': [r'D:\private-work\model.joblib', r'\\server\share\data.csv']})
    raw[names[1]] = encoded({'protected_hashes': {names[0].replace('/', '\\'): digest(raw[names[0]]), 'model': 'f'*64}})
    summary = copy.deepcopy(evaluation_data)
    summary['sources'] = [{'path': name, 'sha256': digest(raw[name])} for name in names]
    return raw, encoded(summary)


def test_public_derivatives_preserve_metrics_original_bytes_and_transitive_hashes(original_evidence):
    raw, summary_bytes = original_evidence
    saved = copy.deepcopy(raw)
    public, summary_public, changes = public_sources(raw, summary_bytes)
    assert raw == saved
    summary = json.loads(summary_public)
    assert summary['publication']['original_sha256'] == digest(summary_bytes)
    assert summary['publication']['historical_evaluation_rerun'] is False
    assert len(changes) == 3
    for item in summary['sources']:
        name = item['path']
        assert digest(public[name]) == item['sha256']
        old, new = json.loads(raw[name]), json.loads(public[name])
        assert old.get('metrics') == new.get('metrics')
        if name in changes:
            assert item['original_sha256'] == digest(raw[name])
            assert new['publication']['original_sha256'] == digest(raw[name])
        else:
            assert public[name] == raw[name]
    study = json.loads(public['docs/innovation/study.json'])
    assert study['protected_hashes']['docs\\phase2\\audit.json'] == digest(public['docs/phase2/audit.json'])
    assert study['protected_hashes']['model'] == 'f'*64
    assert b'private-author' not in b''.join(public.values())
    assert b'private-work' not in b''.join(public.values())
    assert public_sources(raw, summary_bytes)[1] == summary_public
    with pytest.raises(ValueError, match='已是公开派生'):
        public_sources(public, summary_public)


def test_publication_rejects_tampered_input_and_broken_protection(original_evidence):
    raw, summary = original_evidence
    damaged = dict(raw)
    damaged['docs/phase2/audit.json'] += b' '
    with pytest.raises(ValueError, match='SHA不匹配'):
        public_sources(damaged, summary)
    damaged = dict(raw)
    study = json.loads(damaged['docs/innovation/study.json'])
    study['protected_hashes']['docs\\phase2\\audit.json'] = 'e'*64
    damaged['docs/innovation/study.json'] = encoded(study)
    data = json.loads(summary)
    next(item for item in data['sources'] if item['path'] == 'docs/innovation/study.json')['sha256'] = digest(damaged['docs/innovation/study.json'])
    with pytest.raises(ValueError, match='研究记录'):
        public_sources(damaged, encoded(data))


def test_redaction_leaves_web_sources_and_escapes_json_pointer_keys():
    result, pointers = redact_paths({'a/b~c': ['C:/private/data.csv', 'https://example.org/x', 'relative/data.csv']})
    assert pointers == ['/a~1b~0c/0']
    assert result['a/b~c'] == ['<历史本机目录>/data.csv', 'https://example.org/x', 'relative/data.csv']
