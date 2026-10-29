"""Current Windows delivery: byte identities, source provenance, paths and offline pip."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pytest
from delivery import setup_demo,verify_demo
from scripts import build_windows_demo


@pytest.fixture
def minimal_package(tmp_path,monkeypatch):
    names=['app.py','app_model.py','app_quality.py','app_types.py','app_service.py',
        'app_ui_state.py','app_ui_tools.py','app_triage.py','LICENSE','NOTICE','VERSION',
        'requirements-lock.txt','artifacts/model.joblib','artifacts/model.json','artifacts/demo.csv',
        'artifacts/demo.json','artifacts/demo_origins.parquet','liushao/model.py','liushao/data.py',
        'liushao/reviews.py','run-demo.cmd','install.cmd','delivery/desktop_start.py','4_关闭流哨.cmd']
    files={}
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        content=b'1.3.0-rc.2\n' if name=='VERSION' else name.encode()
        path.write_bytes(content);files[name]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    source={'source_commit':'a'*40,'files':{'app.py':files['app.py']['sha256']}}
    content=json.dumps(source).encode();(tmp_path/'SOURCE_MANIFEST.json').write_bytes(content)
    files['SOURCE_MANIFEST.json']={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(verify_demo,'MODEL',files['artifacts/model.joblib']['sha256'])
    monkeypatch.setattr(verify_demo,'DESCRIPTOR',files['artifacts/model.json']['sha256'])
    monkeypatch.setattr(verify_demo,'LEGACY_DESCRIPTOR',files['artifacts/model.json']['sha256'])
    (tmp_path/'PACKAGE_MANIFEST.json').write_text(json.dumps({'software_sha':'a'*40,'software_version':'1.3.0-rc.2','package_revision':'r9','files':files}),encoding='utf-8')
    return tmp_path


def test_changed_model_rejected_before_deserialization(minimal_package):
    (minimal_package/'artifacts/model.joblib').write_bytes(b'untrusted')
    with pytest.raises(ValueError,match='交付文件内容与清单不符'):verify_demo.verify(minimal_package)


def test_missing_lock_is_not_treated_as_success(minimal_package):
    (minimal_package/'requirements-lock.txt').unlink()
    with pytest.raises(FileNotFoundError):verify_demo.verify(minimal_package)


@pytest.mark.parametrize('name', ['app.py', 'docs/results/metrics.json', 'artifacts/model.joblib'])
@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_package_integrity_errors_name_the_file_and_chinese_recovery(minimal_package, name, damage):
    manifest_path = minimal_package/'PACKAGE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_bytes())
    path = minimal_package/name
    if name not in manifest['files']:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = b'{"historical_evidence":true}'
        path.write_bytes(content)
        manifest['files'][name] = {'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest()}
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    if damage == 'missing':
        path.unlink()
        error = FileNotFoundError
    else:
        path.write_bytes(b'changed after release')
        error = ValueError
    with pytest.raises(error) as raised:
        verify_demo.verify(minimal_package)
    message = str(raised.value)
    assert name in message and '包完整性校验未通过' in message
    assert '保留旧目录和复核库' in message and '重新完整解压' in message


def test_missing_package_manifest_has_chinese_recovery(minimal_package):
    (minimal_package/'PACKAGE_MANIFEST.json').unlink()
    with pytest.raises(FileNotFoundError, match='交付文件缺失：PACKAGE_MANIFEST.json'):
        verify_demo.verify(minimal_package)


def test_manifest_cannot_omit_model_validation(minimal_package):
    path=minimal_package/'PACKAGE_MANIFEST.json';value=json.loads(path.read_bytes());del value['files']['artifacts/model.joblib']
    path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='Incomplete file manifest'):verify_demo.verify(minimal_package)


def test_manifest_cannot_escape_package(minimal_package):
    path=minimal_package/'PACKAGE_MANIFEST.json';value=json.loads(path.read_bytes());value['files']['../outside']={'bytes':0,'sha256':'0'*64}
    path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='Unsafe manifest path'):verify_demo.verify(minimal_package)


def test_builder_refuses_different_fitted_model(tmp_path):
    assets=tmp_path/'artifacts';assets.mkdir();(assets/'model.joblib').write_bytes(b'not original')
    with pytest.raises(ValueError,match='Trusted artifact mismatch: model.joblib'):
        build_windows_demo.build(tmp_path/'output',assets)
    assert not list((tmp_path/'output').glob('*.zip'))


@pytest.fixture
def miniature_build(tmp_path,monkeypatch):
    source={name:b'# synthetic build fixture' for name in build_windows_demo.FILES}
    source['VERSION']=b'1.3.0-rc.2\n'
    source['RELEASE_DATE']=b'2026-10-21\n'
    for name in ('scripts/build_windows_demo.py','delivery/safe_files.py','app_evaluation.py','app_input.py'):
        source[name]=(build_windows_demo.ROOT/name).read_bytes()
    source['delivery/evaluation_resources.py']=(build_windows_demo.ROOT/'delivery/evaluation_resources.py').read_bytes()
    source.update({name:name.encode() for name in ['1_一键安装并启动.cmd','2_启动流哨.cmd','3_环境自检.cmd',
        '4_关闭流哨.cmd','install.cmd','run-demo.cmd','delivery/find-python.cmd','delivery/msg/banner.txt',
        'third_party/licenses/example/LICENSE']})
    monkeypatch.setattr(build_windows_demo,'read_source',lambda *a:('a'*40,dict(source)))
    asset_content={name:name.encode() for name in build_windows_demo.ARTIFACTS}
    asset_content['model.json']=b'{"config":{"features":["a","b"]}}'
    pins={name:hashlib.sha256(content).hexdigest() for name,content in asset_content.items()}
    monkeypatch.setattr(build_windows_demo,'ARTIFACTS',pins)
    assets=tmp_path/'assets.zip'
    with zipfile.ZipFile(assets,'w') as z:
        for name in pins:z.writestr('fixture/artifacts/'+name,asset_content[name])
        for name in ('LICENSE','NOTICE'):z.writestr('fixture/'+name,name.encode())
    monkeypatch.setattr(build_windows_demo,'ASSETS_SHA256',hashlib.sha256(assets.read_bytes()).hexdigest())
    return assets,source


def test_build_keeps_chinese_entrypoints_and_helper_layout(tmp_path,miniature_build):
    assets,source=miniature_build;archive=build_windows_demo.build(tmp_path/'output',assets_zip=assets)
    with zipfile.ZipFile(archive) as z:
        prefix='Liushao-1.3.0-rc.2-Windows-r9/'
        for name in ('1_一键安装并启动.cmd','4_关闭流哨.cmd','delivery/find-python.cmd','delivery/msg/banner.txt'):
            assert z.read(prefix+name)==source[name]
        assert prefix+'find-python.cmd' not in z.namelist()


def test_all_application_and_delivery_files_share_source_commit(tmp_path,miniature_build):
    assets,source=miniature_build;archive=build_windows_demo.build(tmp_path/'output',assets_zip=assets)
    with zipfile.ZipFile(archive) as z:
        prefix='Liushao-1.3.0-rc.2-Windows-r9/';manifest=json.loads(z.read(prefix+'PACKAGE_MANIFEST.json'))
        provenance=json.loads(z.read(prefix+'SOURCE_MANIFEST.json'))
        assert manifest['software_sha']==provenance['source_commit']=='a'*40
        assert manifest['software_version']=='1.3.0-rc.2' and manifest['package_revision']=='r9'
        for name,row in manifest['files'].items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==row['sha256']
        for name,content in source.items():assert provenance['files'][name]==hashlib.sha256(content).hexdigest()
        assert 'docs/PROGRESS.md' not in z.read(prefix+'README.md').decode()


def test_same_environment_builds_identical_runtime_and_source_archives(tmp_path,miniature_build):
    assets,_=miniature_build
    one=build_windows_demo.build(tmp_path/'one',assets_zip=assets)
    two=build_windows_demo.build(tmp_path/'two',assets_zip=assets)
    assert one.read_bytes()==two.read_bytes()
    assert next(p for p in one.parent.glob('*.zip') if '源码' in p.name).read_bytes()==next(p for p in two.parent.glob('*.zip') if '源码' in p.name).read_bytes()


@pytest.mark.parametrize('revision,version', [
    ('r11-final', '1.3.0-rc.3'),
    ('r11-evidence-final', '1.3.0-rc.4'),
    ('r11-ui-compact', '1.3.0-rc.5'),
    ('r11-launch-fix', '1.3.0-rc.6'),
    ('r11-review-guidance', '1.3.0-rc.7'),
    ('r11-final-readable', '1.3.0'),
    ('r11-whitebox-polished', '1.3.1'),
    ('r11-whitebox-final', '1.3.2'),
    ('r11-whitebox-desktop', '1.3.3'),
    ('r11-whitebox-simple', '1.3.4'),
])
def test_r11_builder_rejects_revision_version_mismatch(tmp_path, miniature_build, revision, version):
    assets, source = miniature_build
    source['VERSION'] = b'1.3.0-rc.2\n'
    with pytest.raises(ValueError, match='Source version mismatch'):
        build_windows_demo.build(tmp_path/'output', assets_zip=assets,
            package_revision=revision, evaluation_summary=tmp_path/'summary.json',
            evaluation_summary_sha256='a'*64)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('revision,version', [
    ('r11-ui-compact', '1.3.0-rc.5'),
    ('r11-launch-fix', '1.3.0-rc.6'),
    ('r11-review-guidance', '1.3.0-rc.7'),
    ('r11-final-readable', '1.3.0'),
    ('r11-whitebox-polished', '1.3.1'),
    ('r11-whitebox-final', '1.3.2'),
    ('r11-whitebox-desktop', '1.3.3'),
    ('r11-whitebox-simple', '1.3.4'),
])
def test_compact_delivery_keeps_navigation_instructions(tmp_path, miniature_build, monkeypatch, revision, version):
    assets, source = miniature_build
    source['VERSION'] = (version+'\n').encode()
    source['RELEASE_DATE'] = b'2026-10-23\n'
    summary = tmp_path/'summary.json'
    summary.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(build_windows_demo, 'read_summary', lambda *args: {})
    monkeypatch.setattr(build_windows_demo, 'check_identity', lambda *args: {})
    archive = build_windows_demo.build(tmp_path/'output', assets_zip=assets,
        package_revision=revision, evaluation_summary=summary,
        evaluation_summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest())
    with zipfile.ZipFile(archive) as packaged:
        prefix = f'Liushao-{version}-Windows-{revision}/'
        manifest = json.loads(packaged.read(prefix+'PACKAGE_MANIFEST.json'))
        assert manifest['software_version'] == version
        assert manifest['release_date'] == '2026-10-23'
        readme = packaged.read(prefix+'README.md').decode('utf-8')
        assert '查看完整评分依据' in readme and '返回复核' in readme
        assert '打开详情不会保存' in readme
        for name in ('app_evidence.py', 'app_evidence_ui.py', 'app_decomposition.py',
                     'app_evidence_rules.py', 'app_quick_search.py', 'docs/usage/EVIDENCE.md'):
            assert prefix+name in packaged.namelist()
        if revision in ('r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
            assert prefix+'app_desktop_ready.py' in packaged.namelist()
            assert prefix+'delivery/browser-activate.cmd' in packaged.namelist()
            assert '流哨专用浏览器窗口' in readme and '页面就绪回执' in readme
            assert '保留普通浏览器' in readme and '不要通过关闭浏览器安全沙箱处理' in readme
        if revision in ('r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
            assert prefix+'app_review_guidance.py' in packaged.namelist()
            assert '这条记录需要你核对' in readme
            assert '当前值、参考和具体需查什么' in readme
        if revision in ('r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
            assert prefix+'app_navigation_scroll.py' in packaged.namelist()
            assert 'RF路径贡献与替换后变化' in readme and '追加到备注' in readme
        if revision in ('r11-whitebox-final', 'r11-whitebox-desktop'):
            assert '1.3.2 启动修复' in readme and '有效期内重试页面就绪检查' in readme
        if revision == 'r11-whitebox-desktop':
            assert '1.3.3 桌面排版' in readme and '避免内容重叠' in readme
            assert '备注草稿分段显示' in readme and '精简重复声明' in readme
        if revision == 'r11-whitebox-simple':
            assert '1.3.4 界面精简' in readme and '合并重复提示' in readme
            assert '1.3.2 启动修复' not in readme and '1.3.3 桌面排版' not in readme


def _r11_package(minimal_package, monkeypatch, revision, version):
    """A synthetic delivery manifest isolates required-file and release-version checks."""
    path = minimal_package/'PACKAGE_MANIFEST.json'
    manifest = json.loads(path.read_bytes())
    required = {'liushao_icon.svg', 'app_navigation.py', 'app_evaluation.py', 'app_score_charts.py',
        'delivery/evaluation_resources.py', 'resources/evaluation/summary.json', 'EVALUATION_RESOURCES.json',
        'app_record_tools.py', 'app_input.py', 'app_session.py', 'RELEASE_DATE',
        'delivery/migrate_reviews.py', '5_迁移旧版复核.cmd', 'docs/release/SOURCE_LICENSE.md',
        'docs/usage/MODEL_CARD.md', 'docs/usage/LOCAL_DATA.md', 'docs/usage/MIGRATION.md',
        'resources/examples/格式练习_缺少必需列.csv', 'resources/examples/格式练习_缺值与无效记录.csv',
        'docs/engineering/dependencies.json',
        'docs/engineering/license-supplement/streamlit/1.50.0-upstream/LICENSE',
        'docs/engineering/license-supplement/playwright/playwright/driver/package/ThirdPartyNotices.txt'}
    if revision != 'r11-final':
        required.update({'app_evidence.py', 'app_evidence_ui.py', 'app_evidence_rules.py',
            'app_decomposition.py', 'app_quick_search.py', 'docs/usage/EVIDENCE.md'})
    if revision in ('r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.update({'app_desktop_ready.py', 'delivery/browser-activate.cmd'})
    if revision in ('r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.add('app_review_guidance.py')
    if revision in ('r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        required.add('app_navigation_scroll.py')
    contents = {name: name.encode('utf-8') for name in required}
    contents.update({'VERSION': (version+'\n').encode(), 'RELEASE_DATE': b'2026-10-23\n',
        'artifacts/model.json': b'{}', 'resources/evaluation/summary.json': b'{}'})
    summary_sha = hashlib.sha256(contents['resources/evaluation/summary.json']).hexdigest()
    contents['EVALUATION_RESOURCES.json'] = json.dumps({'format': 1, 'identity': {},
        'files': {'resources/evaluation/summary.json': {'bytes': 2, 'sha256': summary_sha}}}).encode()
    for name, content in contents.items():
        target = minimal_package/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest['files'][name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    manifest.update({'software_version': version, 'release_date': '2026-10-23',
        'package_revision': revision, 'evaluation_summary_sha256': summary_sha})
    path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(verify_demo, 'DESCRIPTOR', manifest['files']['artifacts/model.json']['sha256'])
    monkeypatch.setattr(verify_demo, 'LEGACY_DESCRIPTOR', manifest['files']['artifacts/model.json']['sha256'])
    monkeypatch.setattr(verify_demo, 'check_identity', lambda *args: {})
    return manifest


@pytest.mark.parametrize('revision,version', [
    ('r11-final', '1.3.0-rc.3'),
    ('r11-evidence-final', '1.3.0-rc.4'),
    ('r11-ui-compact', '1.3.0-rc.5'),
    ('r11-launch-fix', '1.3.0-rc.6'),
    ('r11-review-guidance', '1.3.0-rc.7'),
    ('r11-final-readable', '1.3.0'),
    ('r11-whitebox-polished', '1.3.1'),
    ('r11-whitebox-final', '1.3.2'),
    ('r11-whitebox-desktop', '1.3.3'),
    ('r11-whitebox-simple', '1.3.4'),
])
def test_r11_verify_requires_its_exact_release_version(minimal_package, monkeypatch, revision, version):
    manifest = _r11_package(minimal_package, monkeypatch, revision, version)
    assert verify_demo.verify(minimal_package)['software_version'] == version
    # A self-consistent rehash must not let a different release impersonate this revision.
    wrong = b'1.3.0-rc.2\n'
    (minimal_package/'VERSION').write_bytes(wrong)
    manifest['software_version'] = '1.3.0-rc.2'
    manifest['files']['VERSION'] = {'bytes': len(wrong), 'sha256': hashlib.sha256(wrong).hexdigest()}
    (minimal_package/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Unexpected software version'):
        verify_demo.verify(minimal_package)


def test_compact_verify_keeps_all_rc4_evidence_requirements(minimal_package, monkeypatch):
    manifest = _r11_package(minimal_package, monkeypatch, 'r11-ui-compact', '1.3.0-rc.5')
    del manifest['files']['app_decomposition.py']
    (minimal_package/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete file manifest'):
        verify_demo.verify(minimal_package)


@pytest.mark.parametrize('revision,version', [
    ('r11-launch-fix', '1.3.0-rc.6'),
    ('r11-review-guidance', '1.3.0-rc.7'),
    ('r11-final-readable', '1.3.0'),
    ('r11-whitebox-polished', '1.3.1'),
    ('r11-whitebox-final', '1.3.2'),
    ('r11-whitebox-desktop', '1.3.3'),
    ('r11-whitebox-simple', '1.3.4'),
])
@pytest.mark.parametrize('name', ['app_desktop_ready.py', 'delivery/browser-activate.cmd'])
def test_launch_fix_verify_requires_frontend_ready_helper(minimal_package, monkeypatch, revision, version, name):
    manifest = _r11_package(minimal_package, monkeypatch, revision, version)
    del manifest['files'][name]
    (minimal_package/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete file manifest'):
        verify_demo.verify(minimal_package)


@pytest.mark.parametrize('revision,version', [
    ('r11-review-guidance', '1.3.0-rc.7'),
    ('r11-final-readable', '1.3.0'),
    ('r11-whitebox-polished', '1.3.1'),
    ('r11-whitebox-final', '1.3.2'),
    ('r11-whitebox-desktop', '1.3.3'),
    ('r11-whitebox-simple', '1.3.4'),
])
@pytest.mark.parametrize('name', ['app_review_guidance.py', 'app_evidence.py', 'app_evidence_ui.py',
    'app_evidence_rules.py', 'app_decomposition.py', 'app_quick_search.py', 'docs/usage/EVIDENCE.md'])
def test_guidance_verify_keeps_review_and_evidence_requirements(minimal_package, monkeypatch, revision, version, name):
    manifest = _r11_package(minimal_package, monkeypatch, revision, version)
    del manifest['files'][name]
    (minimal_package/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete file manifest'):
        verify_demo.verify(minimal_package)


@pytest.mark.parametrize('revision,version', [('r11-whitebox-polished', '1.3.1'), ('r11-whitebox-final', '1.3.2'), ('r11-whitebox-desktop', '1.3.3'), ('r11-whitebox-simple', '1.3.4')])
def test_whitebox_verify_requires_navigation_helper(minimal_package, monkeypatch, revision, version):
    manifest = _r11_package(minimal_package, monkeypatch, revision, version)
    del manifest['files']['app_navigation_scroll.py']
    (minimal_package/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete file manifest'):
        verify_demo.verify(minimal_package)


def test_document_validation_checks_anchors_and_encoded_paths(tmp_path):
    (tmp_path/'manual.html').write_text('<h1 id="start">Start</h1>',encoding='utf-8')
    (tmp_path/'data file.txt').write_text('data',encoding='utf-8')
    readme=tmp_path/'README.md';readme.write_text('[manual](manual.html#start) [data](data%20file.txt)',encoding='utf-8')
    build_windows_demo.validate_documents(tmp_path)
    readme.write_text('[manual](manual.html#missing)',encoding='utf-8')
    with pytest.raises(ValueError,match='missing anchor'):build_windows_demo.validate_documents(tmp_path)


def test_document_validation_checks_same_page_anchor(tmp_path):
    (tmp_path/'manual.html').write_text('<a href="#missing">Start</a>',encoding='utf-8')
    with pytest.raises(ValueError,match='missing anchor'):build_windows_demo.validate_documents(tmp_path)


def test_document_validation_rejects_local_links_outside_package(tmp_path):
    (tmp_path/'README.md').write_text('[data](../external.txt)',encoding='utf-8')
    with pytest.raises(ValueError,match='outside package'):build_windows_demo.validate_documents(tmp_path)


def test_document_validation_rejects_missing_image_or_readme_target(tmp_path):
    (tmp_path/'README.md').write_text('[manual](absent.md)',encoding='utf-8')
    with pytest.raises(ValueError,match='missing file'):build_windows_demo.validate_documents(tmp_path)
    (tmp_path/'README.md').write_text('<img src="guide-images/missing.png">',encoding='utf-8')
    with pytest.raises(ValueError,match='missing file'):build_windows_demo.validate_documents(tmp_path)


@pytest.fixture
def offline_installer(tmp_path, monkeypatch):
    """Exercise the real installer and pip using a tiny, network-free wheel."""
    package = tmp_path / '安装包 with spaces'
    delivery = package / 'delivery'
    delivery.mkdir(parents=True)
    shutil.copyfile(setup_demo.__file__, delivery / 'setup_demo.py')
    shutil.copyfile(Path(setup_demo.__file__).with_name('local_lock.py'), delivery/'local_lock.py')
    # Package integrity has separate tests; this fixture isolates installation.
    (delivery / 'verify_demo.py').write_text('print("fixture verified")\n', encoding='utf-8')
    (package / 'requirements-lock.txt').write_text(
        'liushao-installer-fixture==1.0\n', encoding='utf-8')
    wheels = package / 'wheels'
    wheels.mkdir()
    with zipfile.ZipFile(wheels / 'liushao_installer_fixture-1.0-py3-none-any.whl', 'w') as wheel:
        wheel.writestr('liushao_installer_fixture.py', 'VALUE = "installed locally"\n')
        prefix = 'liushao_installer_fixture-1.0.dist-info/'
        wheel.writestr(prefix + 'METADATA',
                       'Metadata-Version: 2.1\nName: liushao-installer-fixture\nVersion: 1.0\n')
        wheel.writestr(prefix + 'WHEEL',
                       'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        wheel.writestr(prefix + 'RECORD', '')
    wheel=next(wheels.glob('*.whl'))
    (wheels/'SHA256SUMS.txt').write_text(hashlib.sha256(wheel.read_bytes()).hexdigest()+'  '+wheel.name+'\n',encoding='utf-8')
    monkeypatch.setattr(setup_demo, 'ROOT', package)
    monkeypatch.setattr(sys, 'argv', ['setup_demo.py', '--offline'])
    return package


@pytest.mark.skipif(os.name != 'nt', reason='Windows delivery installer')
@pytest.mark.parametrize('setting', ['PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PIP_CONFIG_FILE'])
def test_install_ignores_external_pip_destinations(offline_installer, monkeypatch, setting):
    package = offline_installer
    outside = package.parent / 'outside environment'
    if setting == 'PIP_CONFIG_FILE':
        configuration = package.parent / 'pip.ini'
        configuration.write_text(f'[install]\ntarget = {outside}\n', encoding='utf-8')
        value = str(configuration)
    else:
        value = 'true' if setting == 'PIP_USER' else str(outside)
    monkeypatch.setenv(setting, value)
    assert setup_demo.main() == 0
    python = package / '.venv/Scripts/python.exe'
    result = subprocess.run(
        [str(python), '-I', '-c',
         'import liushao_installer_fixture; print(liushao_installer_fixture.VALUE)'],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'installed locally'
    assert not outside.exists(), 'Installation escaped the package environment'
    assert os.environ[setting] == value, 'Installer changed the calling process configuration'


@pytest.mark.parametrize('fault, message', [
    ('missing', 'Locked dependency missing'),
    ('wrong-version', 'Locked dependency version mismatch'),
    ('external', 'Locked dependency is outside'),
    ('empty-lock', 'dependency lock is empty'),
    ('wrong-interpreter', 'must use this package'),
])
def test_environment_check_rejects_false_installation_success(tmp_path, monkeypatch, fault, message):
    environment = tmp_path / '.venv'
    location = tmp_path / 'outside' if fault == 'external' else environment / 'Lib/site-packages'
    location.mkdir(parents=True)
    (tmp_path / 'requirements-lock.txt').write_text(
        '' if fault == 'empty-lock' else 'liushao-check-fixture==1.0\n', encoding='utf-8')
    if fault != 'missing':
        distribution = location / 'liushao_check_fixture-1.0.dist-info'
        distribution.mkdir()
        version = '2.0' if fault == 'wrong-version' else '1.0'
        (distribution / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: liushao-check-fixture\nVersion: {version}\n',
            encoding='utf-8')
    monkeypatch.syspath_prepend(str(location))
    monkeypatch.setattr(setup_demo, 'ROOT', tmp_path)
    if fault != 'wrong-interpreter':
        monkeypatch.setattr(sys, 'prefix', str(environment))
    with pytest.raises(RuntimeError, match=message):
        setup_demo.check_environment()
