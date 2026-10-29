"""Reference and unit semantics belong to the library, without application imports."""
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

import app_evidence_rules as application_rules
from app_evidence import feature_definition as ui_feature_definition
from liushao import reference
from liushao.feature_metadata import feature_definition
from liushao.model import evidence_for


def test_library_evidence_works_when_application_modules_are_unavailable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root/'liushao', tmp_path/'liushao', ignore=shutil.ignore_patterns('__pycache__'))
    code = '''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class NoApplicationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].startswith('app_'):
            raise AssertionError('Library attempted application import: ' + fullname)
sys.meta_path.insert(0, NoApplicationImports())
import pandas as pd
from liushao.model import evidence_for
row = pd.Series({'Protocol': 0., 'Flow Duration': 2., 'Pkt Len Var': float('nan'),
                 'TotLen Fwd Pkts': 123., 'TotLen Bwd Pkts': 456., 'Tot Fwd Pkts': 3.})
bundle = {'config': {'features': list(row.index)}, 'reference': {
    'Protocol': {'q01': 6., 'median': 6., 'q99': 17.},
    'Flow Duration': {'q01': 2., 'median': 2., 'q99': 2.},
    'Pkt Len Var': {'q01': 0., 'median': 1., 'q99': 2.},
    'TotLen Fwd Pkts': {'q01': 0., 'median': 50., 'q99': 1000.},
    'TotLen Bwd Pkts': {'q01': 0., 'median': 50., 'q99': 1000.},
    'Tot Fwd Pkts': {'q01': 0., 'median': 1., 'q99': 10.}}}
table = evidence_for(row, bundle).set_index('字段')
assert table.loc['Protocol', '统计提示'] == '离散取值，需核对协议/标志/端口语义'
assert table.loc['Flow Duration', '统计提示'] == '固定参考值'
assert table.loc['Pkt Len Var', '统计提示'] == '原始缺失'
assert table.loc['Pkt Len Var', '单位'] == '字节² (B²)'
assert table.loc['TotLen Fwd Pkts', '单位'] == '字节 (B)'
assert table.loc['TotLen Bwd Pkts', '单位'] == '字节 (B)'
assert table.loc['Tot Fwd Pkts', '单位'] == '包'
assert not any(name.startswith('app_') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', code, str(tmp_path)],
                            cwd=tmp_path, capture_output=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stderr or result.stdout


def test_application_alias_shares_parameters_and_library_semantics(monkeypatch):
    assert application_rules.PARAMETERS is reference.PARAMETERS
    assert application_rules.reference_position is reference.reference_position
    ref = {'q01': 0., 'median': 50., 'q99': 100.}
    assert reference.reference_position('Flow Duration', 2., ref)['near']
    monkeypatch.setitem(application_rules.PARAMETERS, 'near_boundary', .01)
    assert not reference.reference_position('Flow Duration', 2., ref)['near']
    bundle = {'config': {'features': ['Flow Duration']}, 'reference': {'Flow Duration': ref}}
    assert evidence_for(pd.Series({'Flow Duration': 2.}), bundle).iloc[0]['统计提示'] == '参考区间内'


@pytest.mark.parametrize('field,unit', [
    ('Protocol', '协议编号'), ('Dst Port', '端口号'), ('Flow Duration', '微秒 (μs)'),
    ('Down/Up Ratio', '无量纲比值'), ('Flow Pkts/s', '包/秒'), ('Flow Byts/s', '字节/秒 (B/s)'),
    ('Pkt Len Var', '字节² (B²)'), ('Init Fwd Win Byts', '字节 (B)'),
    ('ACK Flag Cnt', '次（标记计数）'), ('Fwd Pkts/b Avg', '包/块'),
    ('Fwd Act Data Pkts', '包'), ('Fwd Header Len', '字节 (B)'),
    ('TotLen Fwd Pkts', '字节 (B)'), ('TotLen Bwd Pkts', '字节 (B)'),
    ('unknown feature', '数据源原始量纲（未确认）'),
])
def test_library_and_application_keep_the_same_field_units(field, unit):
    assert feature_definition(field)[1] == ui_feature_definition(field)[1] == unit
    bundle = {'config': {'features': [field]},
              'reference': {field: {'q01': 0., 'median': 1., 'q99': 2.}}}
    assert evidence_for(pd.Series({field: 1.}), bundle).iloc[0]['单位'] == unit
