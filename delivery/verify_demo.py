"""Verify immutable demo files before loading the trusted pickle-based model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .safe_files import read_json, validate_names, regular_file
    from .evaluation_resources import check_identity
except ImportError:
    from safe_files import read_json, validate_names, regular_file
    from evaluation_resources import check_identity

ROOT = Path(__file__).resolve().parents[1]
MODEL = '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7'
LEGACY_DESCRIPTOR = 'e280995a00084234c538720541db6475bb51382c4e2ffc9c0e4db8cbde933880'
ALIGNED_DESCRIPTOR = 'cf44d472c2f34b32a4eeca3ba9434fcc334a43d3eb4f5778d37055df3310d3f6'
DESCRIPTOR = 'dc973bc0771cb19750fb99bf98cf631eb92a5c4c33f20b22e19ef913e798ed7b'
REVISION_VERSIONS = {'r11-final': '1.3.0-rc.3',
                     'r11-evidence-final': '1.3.0-rc.4',
                     'r11-ui-compact': '1.3.0-rc.5',
                     'r11-launch-fix': '1.3.0-rc.6',
                     'r11-review-guidance': '1.3.0-rc.7',
                     'r11-final-readable': '1.3.0',
                     'r11-whitebox-polished': '1.3.1',
                     'r11-whitebox-final': '1.3.2',
                     'r11-whitebox-desktop': '1.3.3',
                     'r11-whitebox-simple': '1.3.4',
                     'r11-whitebox-aligned': '1.3.5',
                     'r11-whitebox-refined': '1.3.6',
                     'r12-engine': '1.3.7'}


def package_bytes(root: Path, name: str) -> bytes:
    """Read a verified package path with actionable user-facing file errors."""
    try:
        return regular_file(root, name).read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'交付文件缺失：{name}。包完整性校验未通过，安装或启动已停止；'
            '请保留旧目录和复核库，将原始运行 ZIP 重新完整解压到新目录后重试，不要用空文件补齐。') from error
    except PermissionError as error:
        raise PermissionError(
            f'无法读取交付文件：{name}。请检查本机目录权限或文件是否被占用；'
            '保留复核库，并将原始运行 ZIP 重新完整解压到可写的新目录后重试。') from error


def verify(root: Path = ROOT) -> dict:
    manifest = read_json(package_bytes(root, 'PACKAGE_MANIFEST.json'))
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('software_sha', '')):
        raise ValueError('Unexpected software baseline')
    files = manifest.get('files')
    revision = manifest.get('package_revision', '')
    if not re.fullmatch(r'(?:r9(?:-[a-z0-9-]+)?|r10-competition|r11-final|r11-evidence-final|r11-ui-compact|r11-launch-fix|r11-review-guidance|r11-final-readable|r11-whitebox-polished|r11-whitebox-final|r11-whitebox-desktop|r11-whitebox-simple|r11-whitebox-aligned|r11-whitebox-refined|r12-engine)', revision):
        raise ValueError('Unsupported package revision')
    required = {'app.py', 'app_types.py', 'app_service.py', 'SOURCE_MANIFEST.json', 'app_ui_state.py', 'app_ui_tools.py', 'app_triage.py', 'LICENSE', 'NOTICE', 'requirements-lock.txt', 'VERSION', 'artifacts/model.joblib',
                'artifacts/model.json', 'artifacts/demo.csv', 'artifacts/demo.json',
                'artifacts/demo_origins.parquet', 'liushao/model.py', 'liushao/data.py',
                'liushao/reviews.py', 'app_model.py', 'run-demo.cmd', 'install.cmd', 'delivery/desktop_start.py', '4_关闭流哨.cmd'}
    if revision in ('r10-competition', *REVISION_VERSIONS):
        required.update({'liushao_icon.svg', 'app_navigation.py', 'app_evaluation.py', 'app_score_charts.py',
                         'delivery/evaluation_resources.py', 'resources/evaluation/summary.json', 'EVALUATION_RESOURCES.json'})
    if revision in REVISION_VERSIONS:
        required.update({'app_record_tools.py', 'app_input.py', 'app_session.py', 'RELEASE_DATE',
            'delivery/migrate_reviews.py', '5_迁移旧版复核.cmd', 'docs/release/SOURCE_LICENSE.md',
            'docs/usage/MODEL_CARD.md', 'docs/usage/LOCAL_DATA.md', 'docs/usage/MIGRATION.md',
            'resources/examples/格式练习_缺少必需列.csv', 'resources/examples/格式练习_缺值与无效记录.csv'})
        required.update({'docs/engineering/dependencies.json',
            'docs/engineering/license-supplement/streamlit/1.50.0-upstream/LICENSE',
            'docs/engineering/license-supplement/playwright/playwright/driver/package/ThirdPartyNotices.txt'})
    if revision in ('r11-evidence-final', 'r11-ui-compact', 'r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.update({'app_evidence.py','app_evidence_ui.py','app_evidence_rules.py','app_decomposition.py','app_quick_search.py','docs/usage/EVIDENCE.md'})
    if revision in ('r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.update({'app_desktop_ready.py', 'delivery/browser-activate.cmd'})
    if revision in ('r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.add('app_review_guidance.py')
    if revision in ('r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.add('app_navigation_scroll.py')
    if revision in ('r11-whitebox-aligned','r11-whitebox-refined','r12-engine'):
        required.update({'app_evidence.py', 'app_evidence_ui.py', 'app_evidence_rules.py',
                         'app_decomposition.py', 'app_desktop_ready.py', 'app_review_guidance.py',
                         'app_navigation_scroll.py', 'delivery/browser-activate.cmd',
                         'delivery/preflight.py', 'delivery/service_probe.py',
                         'resources/evaluation/model.original.json',
                         'resources/evaluation/model.release.json',
                         'resources/evaluation/reevaluation.json', 'scripts/reevaluate_frozen.py'})
    if revision in ('r11-whitebox-refined', 'r12-engine'):
        required.update({'liushao/reference.py','liushao/feature_metadata.py','artifacts/model.original.json'})
    if revision == 'r12-engine':
        required.update({'liushao_engine/__init__.py', 'artifacts/engine.joblib', 'artifacts/engine.json'})
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Incomplete file manifest')
    if files['artifacts/model.joblib'].get('sha256') != MODEL:
        raise ValueError('Unexpected trusted model identity')
    expected_descriptor = (DESCRIPTOR if revision in ('r11-whitebox-refined', 'r12-engine') else
                           ALIGNED_DESCRIPTOR if revision == 'r11-whitebox-aligned' else LEGACY_DESCRIPTOR)
    if files['artifacts/model.json'].get('sha256') != expected_descriptor:
        raise ValueError('Unexpected trusted descriptor identity')
    validate_names(files)
    for name, expected in files.items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts or ':' in name or '\\' in name:
            raise ValueError(f'Unsafe manifest path: {name}')
        path = regular_file(root, name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Unsafe file: {name}')
        content = package_bytes(root, name)
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            raise ValueError(f'交付文件内容与清单不符：{name}。包完整性校验未通过，安装或启动已停止；'
                             '请保留旧目录和复核库，将原始运行 ZIP 重新完整解压到新目录后重试。')
    if revision in ('r10-competition', *REVISION_VERSIONS):
        name = 'resources/evaluation/summary.json'
        resource = read_json(package_bytes(root, 'EVALUATION_RESOURCES.json'))
        identity = check_identity(read_json(package_bytes(root, name)),
                                  read_json(package_bytes(root, 'artifacts/model.json')), expected_descriptor,
                                  package_bytes(root, 'resources/evaluation/model.original.json')
                                  if revision in ('r11-whitebox-aligned','r11-whitebox-refined','r12-engine') else None)
        if (resource.get('format') != 1 or resource.get('identity') != identity
                or not resource.get('files')
                or resource.get('files', {}).get(name) != {'bytes': files[name]['bytes'], 'sha256': files[name]['sha256']}
                or manifest.get('evaluation_summary_sha256') != files[name]['sha256']):
            raise ValueError('Inconsistent evaluation resource manifest')
        for resource_name, expected in resource['files'].items():
            if {k: files.get(resource_name, {}).get(k) for k in ('bytes', 'sha256')} != expected:
                raise ValueError('Inconsistent evaluation resource: ' + resource_name)
        if revision in ('r11-whitebox-aligned','r11-whitebox-refined','r12-engine'):
            summary = read_json(package_bytes(root, name))
            for item in summary['sources']:
                if resource['files'].get(item['path'], {}).get('sha256') != item['sha256']:
                    raise ValueError('Missing historical evidence: ' + item['path'])
            if 'resources/evaluation/reevaluation.json' not in resource['files']:
                raise ValueError('Missing frozen reevaluation')
    version = package_bytes(root, 'VERSION').decode('utf-8').strip()
    if (version != manifest.get('software_version')
            or (revision in REVISION_VERSIONS and version != REVISION_VERSIONS[revision])):
        raise ValueError('Unexpected software version')
    if revision in REVISION_VERSIONS and package_bytes(root, 'RELEASE_DATE').decode('utf-8').strip() != manifest.get('release_date'):
        raise ValueError('Inconsistent release date')
    candidates = list(root.glob('*.py')) + list(root.glob('*.cmd')) + list(root.glob('*.exe')) + list(root.glob('*.ps1'))
    for directory in ('liushao', 'liushao_engine', 'scripts', 'delivery'):
        candidates.extend(p for p in (root / directory).rglob('*')
                          if p.is_file() and p.suffix.lower() in ('.py', '.cmd', '.exe', '.ps1')
                          and '__pycache__' not in p.parts)
    candidates.extend((root / 'wheels').glob('*.whl'))
    extras = sorted(path.relative_to(root).as_posix() for path in candidates
                    if path.relative_to(root).as_posix() not in files)
    if extras:
        raise ValueError('发现未登记的可执行文件或依赖：\n' + '\n'.join(extras)
                         + '\n请将以上额外文件移出交付目录后重试；不要修改已登记文件。')
    source = read_json(package_bytes(root, 'SOURCE_MANIFEST.json'))
    if source.get('source_commit') != manifest['software_sha']:
        raise ValueError('Inconsistent source provenance')
    for name, digest in source['files'].items():
        if files.get(name, {}).get('sha256') != digest:
            raise ValueError('Inconsistent source file: ' + name)
    print(f"Verified {len(files)} files; software {manifest['software_sha']}; original frozen model c7a028b4e4cafedd.")
    return manifest


if __name__ == '__main__':
    verify()
