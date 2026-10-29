"""Application-owned guards for trusted legacy model packages.

These guards do not change the data adapter, training, prediction implementation,
model artifacts, thresholds, or the agreed model handoff format.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from functools import lru_cache
import os
import hashlib
from importlib.metadata import Distribution
import io
import json
from numbers import Real
from pathlib import Path
import re
import sys
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

import liushao
from liushao.data import EXCLUDED
from liushao.model import signed_log


FORBIDDEN_FEATURES = {name.casefold() for name in EXCLUDED} | {
    'label', 'ground_truth', 'ground_truth_evaluation_only', 'attack_type_evaluation_only',
    'record_id', 'dataset_id', 'source_file', 'source_row', 'source_sha256', 'model_version',
}


class CompatibilityError(ValueError):
    """The existing model cannot be checked or used with this MVP implementation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityError(message)


def finite_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def require_fitted(estimator, attributes=None) -> None:
    try:
        check_is_fitted(estimator, attributes)
    except ValueError as error:
        raise CompatibilityError(f'预处理器或分类器尚未拟合: {error}') from error


def validate_descriptor(descriptor: dict) -> None:
    """Reject incompatible metadata before deserializing any estimator."""
    require(isinstance(descriptor, dict), '模型描述必须是对象')
    config = descriptor.get('config')
    require(isinstance(config, dict), '缺少模型 config')
    for name in ('schema_version', 'class_mapping'):
        require(name not in descriptor and name not in config,
                f'尚未约定 {name} 的兼容语义，不能自动接纳新合同声明')
    require(config.get('algorithm_version') == 'liushao-mvp-1', 'algorithm_version 不兼容')
    require(config.get('sklearn') == sklearn.__version__, 'scikit-learn 版本不兼容')
    features = config.get('features')
    require(isinstance(features, list) and bool(features), 'features 必须为非空有序列表')
    require(all(isinstance(name, str) and name and name == name.strip() for name in features),
            '特征名必须为非空且无首尾空格的字符串')
    require(len(features) == len(set(features)), '特征名重复')
    require(not FORBIDDEN_FEATURES.intersection(name.casefold() for name in features),
            '标签或来源元数据不能作为推理特征')
    require(isinstance(config.get('data_version'), str) and bool(config['data_version']),
            '缺少 data_version')
    for key in ('rf_threshold', 'lr_threshold', 'near_margin'):
        require(finite_number(config.get(key)) and 0 <= config[key] <= 1, f'{key} 无效')
    for key in ('n_estimators', 'max_depth', 'min_samples_leaf', 'seed'):
        require(type(config.get(key)) is int and config[key] >= (0 if key == 'seed' else 1),
                f'{key} 无效')
    computed = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    require(descriptor.get('model_version') == computed, 'model_version 与配置哈希不一致')
    for key in ('training_rows', 'normal_reference_rows'):
        require(type(descriptor.get(key)) is int and descriptor[key] > 0, f'{key} 无效')
    require(descriptor['normal_reference_rows'] <= descriptor['training_rows'], '正常参考条数超出训练条数')
    reference = descriptor.get('reference')
    require(isinstance(reference, dict) and set(reference) == set(features), '参考分布字段不一致')
    for name, quantiles in reference.items():
        require(isinstance(quantiles, dict) and set(quantiles) == {'q01', 'median', 'q99'},
                f'参考分位数字段不完整: {name}')
        values = [quantiles[key] for key in ('q01', 'median', 'q99')]
        require(all(value is None for value in values)
                or (all(finite_number(value) for value in values) and values == sorted(values)),
                f'参考分位数非有限或顺序错误: {name}')


@lru_cache(maxsize=2)
def _parsed_descriptor(descriptor_json, sklearn_version):
    """Cache only static work; file, source and runtime checks stay live."""
    require(sklearn_version == sklearn.__version__, 'scikit-learn 版本不兼容')
    descriptor = json.loads(descriptor_json)
    validate_descriptor(descriptor)
    return descriptor


def parse_descriptor(descriptor_json):
    return copy.deepcopy(_parsed_descriptor(descriptor_json, sklearn.__version__))


def validate_bundle(bundle: dict, descriptor: dict) -> None:
    """Check the actual fitted pipelines against the existing train.py contract."""
    validate_descriptor(descriptor)
    require(isinstance(bundle, dict), '模型包必须是对象')
    require('schema_version' not in bundle and 'class_mapping' not in bundle,
            '模型包存在未约定的 schema_version/class_mapping')
    for key in ('config', 'model_version', 'reference', 'training_rows', 'normal_reference_rows'):
        require(bundle.get(key) == descriptor[key], f'模型包与描述文件不一致: {key}')
    config = descriptor['config']
    features = config['features']
    for name in ('rf', 'lr'):
        pipeline = bundle.get(name)
        require(isinstance(pipeline, Pipeline), f'{name} 缺少 Pipeline')
        names = ['impute', 'classifier'] if name == 'rf' else ['impute', 'log', 'scale', 'classifier']
        require([step for step, _ in pipeline.steps] == names, f'{name} 预处理步骤不兼容')
        imputer = pipeline.named_steps['impute']
        require(isinstance(imputer, SimpleImputer), f'{name} 缺少训练填补器')
        require(imputer.strategy == 'median' and imputer.add_indicator
                and imputer.keep_empty_features, f'{name} 填补器配置不兼容')
        require(isinstance(imputer.missing_values, Real) and np.isnan(imputer.missing_values),
                f'{name} 填补器缺失值标记必须为 NaN')
        require_fitted(imputer, ['statistics_', 'indicator_', 'n_features_in_'])
        require(list(getattr(pipeline, 'feature_names_in_', [])) == features,
                f'{name} 内部特征顺序不一致')
        require(imputer.n_features_in_ == len(features)
                and len(imputer.statistics_) == len(features)
                and np.isfinite(imputer.statistics_).all(), f'{name} 填补器维度或统计无效')
        indicator = imputer.indicator_.features_
        require(len(set(indicator)) == len(indicator)
                and all(0 <= item < len(features) for item in indicator), f'{name} 缺失指示器无效')
        width = len(features) + len(indicator)
        classifier = pipeline.named_steps['classifier']
        require(isinstance(classifier, RandomForestClassifier if name == 'rf'
                           else LogisticRegression), f'{name} 分类器类型不兼容')
        require_fitted(classifier)
        require(classifier.n_features_in_ == width, f'{name} 分类器与预处理输出维度不一致')
        require(np.array_equal(pipeline.classes_, [0, 1]), f'{name} 类别顺序必须为 [0, 1]')
        params = classifier.get_params()
        expected = ({'n_estimators': config['n_estimators'], 'max_depth': config['max_depth'],
                     'min_samples_leaf': config['min_samples_leaf'],
                     'random_state': config['seed'], 'class_weight': 'balanced_subsample'}
                    if name == 'rf' else
                    {'max_iter': 2000, 'class_weight': 'balanced', 'random_state': config['seed']})
        require(all(params[key] == value for key, value in expected.items()),
                f'{name} 分类器参数与配置/当前实现不一致')
        if name == 'rf':
            require(len(classifier.estimators_) == config['n_estimators'], 'rf 已拟合树数量不一致')
            require(classifier.n_outputs_ == 1 and classifier.n_classes_ == 2,
                    'rf 必须为单输出二分类')
            for tree in classifier.estimators_:
                require(isinstance(tree, DecisionTreeClassifier), 'rf 树类型不兼容')
                require_fitted(tree, ['tree_', 'n_features_in_'])
                require(tree.n_features_in_ == width and tree.tree_.n_features == width
                        and tree.n_outputs_ == 1 and tree.n_classes_ == 2
                        and np.array_equal(tree.classes_, [0, 1])
                        and np.isfinite(tree.tree_.value).all(), 'rf 树结构或统计无效')
        else:
            require(classifier.coef_.shape == (1, width) and classifier.intercept_.shape == (1,)
                    and np.isfinite(classifier.coef_).all()
                    and np.isfinite(classifier.intercept_).all(), 'lr 拟合系数形状或统计无效')
            transform = pipeline.named_steps['log']
            require(isinstance(transform, FunctionTransformer) and transform.func is signed_log
                    and transform.kw_args in (None, {}) and not transform.validate,
                    'lr signed_log 预处理不兼容')
            scaler = pipeline.named_steps['scale']
            require(isinstance(scaler, StandardScaler) and scaler.with_mean and scaler.with_std,
                    'lr 缩放器不兼容')
            require_fitted(scaler, ['mean_', 'scale_', 'var_', 'n_features_in_'])
            require(scaler.n_features_in_ == width and all(
                len(getattr(scaler, key)) == width and np.isfinite(getattr(scaler, key)).all()
                for key in ('mean_', 'scale_', 'var_')), 'lr 缩放器维度或统计无效')
            require((scaler.scale_ > 0).all() and (scaler.var_ >= 0).all(), 'lr 缩放统计范围无效')


def _dependency_version(name: str, search_path: tuple[str, ...]) -> str:
    """Read current metadata without iterating Streamlit's mutable sys.path."""
    distribution = next(Distribution.discover(name=name, path=search_path), None)
    require(distribution is not None, f'无法读取模型运行依赖元数据: {name}')
    actual = distribution.metadata.get('Version')
    require(isinstance(actual, str) and bool(actual), f'无法读取模型运行依赖版本: {name}')
    return actual


def _runtime_context(descriptor: dict) -> dict:
    """Recheck the running interpreter, inference dependencies and imported source."""
    require(sys.version_info[:2] == (3, 12), '应用模型环境需要 Python 3.12')
    lock = Path(__file__).resolve().parent / 'requirements-lock.txt'
    pins = {}
    for line in lock.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, expected = line.strip().split('==', 1)
            pins[re.sub(r'[-_.]+', '-', name).lower()] = expected
    # An overlapping ScriptRunner can remove its path while this one is reading
    # metadata. A live list iterator would then skip the following search entry.
    # Snapshot paths only; every rerun still reads and validates current versions.
    search_path = tuple(sys.path)
    installed = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib'):
        actual = _dependency_version(name, search_path)
        require(pins.get(name) == actual, f'模型运行依赖与锁不兼容: {name}')
        installed[name] = actual

    sources = descriptor.get('source_code_sha256')
    require(isinstance(sources, dict), '缺少 source_code_sha256')
    normalized = {name.replace('\\', '/'): digest for name, digest in sources.items()}
    require(len(normalized) == len(sources), '重复的源码登记路径')
    required = {'liushao/__init__.py', 'liushao/model.py', 'liushao/data.py'}
    require(required <= set(normalized), '缺少模型/适配源码哈希')
    # Resolve the package that is actually imported, including in isolated test processes.
    code_root = Path(liushao.__file__).resolve().parent.parent
    evidence = {}
    for name, expected in normalized.items():
        path = (code_root / name).resolve()
        require(path.is_relative_to(code_root / 'liushao') and path.suffix == '.py',
                '源码路径超出模型模块')
        require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected) is not None,
                f'源码哈希无效: {name}')
        content = path.read_bytes()
        raw = hashlib.sha256(content).hexdigest()
        lf_content = content.replace(b'\r\n', b'\n')
        normalized_hash = hashlib.sha256(lf_content).hexdigest()
        crlf_hash = hashlib.sha256(lf_content.replace(b'\n', b'\r\n')).hexdigest()
        # Accept only newline variants of these exact bytes, including reviews.py.
        require(expected in (raw, normalized_hash, crlf_hash), f'模型推理源码不匹配: {name}')
        evidence[name] = normalized_hash
    return {'python': sys.version, 'dependencies': installed, 'source_sha256': evidence}


def _check_model_bytes(content: bytes, descriptor: dict) -> str:
    expected = descriptor.get('model_sha256')
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected) is not None,
            '模型 SHA-256 无效')
    actual = hashlib.sha256(content).hexdigest()
    require(actual == expected, '本地模型哈希不匹配，请核对已交接的模型包')
    return actual


def _training_bridge(descriptor, descriptor_path):
    if descriptor.get('training_descriptor_sha256'):
        from delivery.evaluation_resources import verify_training_descriptor
        from delivery.safe_files import regular_file, safe_name
        filename = descriptor.get('training_descriptor_file')
        if filename is not None:
            require(isinstance(filename, str) and '/' not in filename and '\\' not in filename
                    and filename.endswith('.json'), '历史训练描述必须使用同目录安全JSON文件名')
            safe_name(filename)
            original = regular_file(Path(descriptor_path).resolve().parent, filename)
        else:
            adjacent = Path(descriptor_path).resolve().parent / 'model.original.json'
            original = (regular_file(adjacent.parent, adjacent.name) if adjacent.exists()
                        else Path(__file__).resolve().parent / 'resources/evaluation/model.original.json')
        if not original.is_file():
            raise FileNotFoundError('缺少历史训练描述：' + original.name + '。请连同模型、发布描述一起恢复配套原始描述文件。')
        verify_training_descriptor(descriptor, original.read_bytes())


def prepare_model(model_path: Path, descriptor_path: Path) -> tuple[bytes, str, str]:
    """Read and validate each rerun before the app may consult its resource cache."""
    descriptor_json = Path(descriptor_path).read_bytes().decode('utf-8')
    descriptor = parse_descriptor(descriptor_json)
    _training_bridge(descriptor, descriptor_path)
    context = _runtime_context(descriptor)
    model_bytes = Path(model_path).read_bytes()
    _check_model_bytes(model_bytes, descriptor)
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    return model_bytes, descriptor_json, fingerprint


FROZEN_MODEL = '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7'
FROZEN_DESCRIPTOR = 'dc973bc0771cb19750fb99bf98cf631eb92a5c4c33f20b22e19ef913e798ed7b'


@dataclass(frozen=True)
class VerifiedModelSnapshot:
    content: bytes
    descriptor: str
    model_sha256: str
    descriptor_sha256: str
    context: str

    @property
    def identity(self):
        return '|'.join((self.model_sha256, self.descriptor_sha256, self.context))


def verified_model(model_path, descriptor_path):
    expected_model = os.environ.get('LIUSHAO_EXPECTED_MODEL_SHA256', FROZEN_MODEL)
    expected_descriptor = os.environ.get('LIUSHAO_EXPECTED_DESCRIPTOR_SHA256', FROZEN_DESCRIPTOR)
    for expected in (expected_model, expected_descriptor):
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
                '缺少可信冻结文件身份')
    descriptor_bytes = Path(descriptor_path).read_bytes()
    descriptor_hash = hashlib.sha256(descriptor_bytes).hexdigest()
    require(descriptor_hash == expected_descriptor, '描述文件不符合可信冻结身份')
    descriptor_json = descriptor_bytes.decode('utf-8')
    descriptor = parse_descriptor(descriptor_json)
    _training_bridge(descriptor, descriptor_path)
    context = _runtime_context(descriptor)
    content = Path(model_path).read_bytes()
    model_hash = _check_model_bytes(content, descriptor)
    require(model_hash == expected_model, '模型不符合可信冻结身份')
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    return VerifiedModelSnapshot(content, descriptor_json, model_hash, descriptor_hash, fingerprint)


class CheckedPredictor:
    """Validate application-boundary output before the unchanged infer computes labels."""

    def __init__(self, predictor, name: str):
        self.predictor = predictor
        self.name = name

    def predict_proba(self, values):
        probabilities = np.asarray(self.predictor.predict_proba(values))
        require(probabilities.shape == (len(values), 2), f'{self.name} 模型输出维度不兼容')
        require(np.issubdtype(probabilities.dtype, np.number) and np.isrealobj(probabilities)
                and np.isfinite(probabilities).all()
                and ((probabilities >= 0) & (probabilities <= 1)).all()
                and np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12),
                f'{self.name} 模型输出无效，无法判断')
        return probabilities


def load_model_bundle(model_bytes: bytes, descriptor_json: str) -> dict:
    """Deserialize only the verified bytes, then guard app-side predictor outputs."""
    descriptor = parse_descriptor(descriptor_json)
    _check_model_bytes(model_bytes, descriptor)
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        bundle = joblib.load(io.BytesIO(model_bytes))
    validate_bundle(bundle, descriptor)
    return {**bundle, **{name: CheckedPredictor(bundle[name], name) for name in ('rf', 'lr')}}
