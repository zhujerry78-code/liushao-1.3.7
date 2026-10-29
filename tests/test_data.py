import importlib
import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('liushao.data') is not None, '数据模块尚未实现'
    return importlib.import_module('liushao.data')


def test_clean_separates_labels_and_metadata_and_rejects_unusable_rows():
    d = api()
    raw = pd.DataFrame({'Timestamp': ['16/02/2018 09:00:00'] * 4,
                        'Dst Port': [80] * 4, 'Label': ['Benign', 'Bot', 'Bot', 'Bot'],
                        'a': ['1', 'Infinity', 'bad', '5'], 'b': ['2', '3', '', '6']})
    x, meta, rejected = d.clean(raw, ['a', 'b'], 'abc', 'sample.csv')
    assert list(x.columns) == ['a', 'b']
    assert meta.source_row.tolist() == [2, 3, 5]
    assert np.isnan(x.iloc[1, 0])
    assert rejected.source_row.tolist() == [4]
    assert meta.record_id.is_unique
    assert meta.label.tolist() == ['Benign', 'Bot', 'Bot']
    assert x.index.equals(meta.index)
    with pytest.raises(ValueError, match='缺少'):
        d.clean(raw.drop(columns='a'), ['a', 'b'], 'abc', 'sample.csv')


def test_block_split_is_reproducible_and_purges_near_duplicates():
    d = api()
    x = pd.DataFrame({'a': [1., 2., 3., 1.00000001, 5., 6., 7., 8., 9., 10.]})
    m = pd.DataFrame({'timestamp': pd.date_range('2018-02-16', periods=10, freq='15min'),
                      'source_file': ['a'] * 10, 'source_row': range(2, 12),
                      'record_id': [str(i) for i in range(10)], 'label': ['Benign', 'Bot'] * 5})
    first = d.split_records(x, m, seed=42)
    second = d.split_records(x, m, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, 'split'] == first.loc[3, 'split'] or 'purged' in set(first.loc[[0, 3], 'split'])
    assert set(first.split) >= {'train', 'validation', 'test'}
    assert first.groupby('block_id').split.apply(lambda s: s[s != 'purged'].nunique()).max() <= 1


def test_download_checks_integrity_and_rejects_changed_source(tmp_path):
    d = api()
    source = tmp_path / 'source.csv'
    source.write_bytes(b'a,b\n1,2\n')
    target = tmp_path / 'target.csv'
    info = d.download(source.as_uri(), target)
    assert info['sha256'] == d.sha256(source)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='哈希'):
        d.download(source.as_uri(), target, expected_sha256='0' * 64)


def test_locking_expected_hash_does_not_change_data_identity():
    d = api()
    assert hasattr(d, 'version_id'), '数据版本需要规范化下载配置'
    c = {'seed': 42, 'files': [{'name': 'x'}]}
    locked = {'seed': 42, 'files': [{'name': 'x', 'sha256': 'abc'}]}
    sources = [{'name': 'x', 'sha256': 'abc'}]
    assert d.version_id(c, sources, 'manifest') == d.version_id(locked, sources, 'manifest')


def test_csv_parser_rejects_ambiguous_headers_and_preserves_physical_rows():
    import io
    d = api()
    assert hasattr(d, 'read_csv_checked'), '需要统一的原始 CSV 校验入口'
    for content in [b'a,a,b\n1,999,2\n', b'a, a ,b\n1,999,2\n']:
        with pytest.raises(ValueError, match='重复'):
            d.read_csv_checked(io.BytesIO(content))
    raw = d.read_csv_checked(io.BytesIO(b'a,b\n1,2\n\n3,4\n'))
    x, meta, rejected = d.clean(raw, ['a', 'b'], 'hash', 'sample.csv')
    assert meta.source_row.tolist() == [2, 4]
    assert meta.record_id.tolist() == ['hash:2', 'hash:4']
    with pytest.raises(ValueError, match='多行'):
        d.read_csv_checked(io.BytesIO(b'a,b\n"1\n2",3\n'))


def test_official_chunk_parser_has_same_row_ids_and_rejects_blank_lines(tmp_path):
    d = api()
    assert hasattr(d, 'validate_csv_layout'), '官方全文件需要结构核验'
    p = tmp_path / 'input.csv'
    p.write_text('a,b\n1,2\n3,4\n', encoding='utf-8')
    assert d.validate_csv_layout(p) == ['a', 'b']
    p.write_text('a,b\n1,2\n\n3,4\n', encoding='utf-8')
    with pytest.raises(ValueError, match='空行'):
        d.validate_csv_layout(p)
