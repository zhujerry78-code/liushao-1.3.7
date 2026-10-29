import hashlib
import json
from pathlib import Path
import pytest
import app_model


def pair(tmp_path):
    original = {'model_sha256':'m','model_version':'v','config':{},'reference':{},'training_rows':2,'normal_reference_rows':1}
    raw = json.dumps(original).encode()
    root = tmp_path/'external model directory'
    root.mkdir()
    (root/'model.original.json').write_bytes(raw)
    descriptor = dict(original,training_descriptor_sha256=hashlib.sha256(raw).hexdigest(),training_descriptor_file='model.original.json')
    return root, descriptor, raw


def test_external_model_directory_is_self_contained_and_ignores_parent_decoy(tmp_path):
    root, descriptor, raw = pair(tmp_path)
    decoy = tmp_path/'resources/evaluation'
    decoy.mkdir(parents=True)
    (decoy/'model.original.json').write_text('wrong parent document')
    app_model._training_bridge(descriptor,root/'model.json')
    (root/'model.original.json').unlink()
    with pytest.raises(FileNotFoundError,match='缺少历史训练描述'):
        app_model._training_bridge(descriptor,root/'model.json')


@pytest.mark.parametrize('filename',['../model.original.json','C:/model.json','folder\\model.json','CON.json','a.txt'])
def test_training_descriptor_cannot_escape_its_directory(tmp_path, filename):
    root, descriptor, raw = pair(tmp_path)
    descriptor['training_descriptor_file'] = filename
    with pytest.raises(ValueError):
        app_model._training_bridge(descriptor,root/'model.json')


def test_original_descriptor_is_checked_again_after_hot_replacement(tmp_path):
    root, descriptor, raw = pair(tmp_path)
    app_model._training_bridge(descriptor,root/'model.json')
    (root/'model.original.json').write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='changed'):
        app_model._training_bridge(descriptor,root/'model.json')
