"""Dependency-free evaluation identity checks, including before offline install."""
import hashlib
import json


def verify_training_descriptor(descriptor, original_bytes):
    expected = descriptor.get('training_descriptor_sha256')
    if not expected:
        return
    if original_bytes is None or hashlib.sha256(original_bytes).hexdigest() != expected:
        raise ValueError('Historical training descriptor is missing or changed')
    original = json.loads(original_bytes)
    for key in ('model_sha256', 'model_version', 'config', 'reference',
                'training_rows', 'normal_reference_rows'):
        if descriptor.get(key) != original.get(key):
            raise ValueError('Release and training descriptor disagree: ' + key)


def model_info(descriptor):
    result = {key: descriptor[key] for key in ('training_rows', 'normal_reference_rows')}
    result.update({key: descriptor['config'][key] for key in
                   ('features', 'rf_threshold', 'lr_threshold', 'near_margin')})
    return result


def check_identity(summary, descriptor, descriptor_sha256, original_bytes=None):
    verify_training_descriptor(descriptor, original_bytes)
    identity = {'model_version': descriptor['model_version'], 'model_sha256': descriptor['model_sha256'],
                'descriptor_sha256': descriptor.get('training_descriptor_sha256', descriptor_sha256),
                'data_version': descriptor['config']['data_version']}
    if (summary.get('format') != 1 or summary.get('identity') != identity
            or summary.get('model_info') != model_info(descriptor)):
        raise ValueError('Evaluation identity or model metadata does not match the frozen descriptor')
    return identity
