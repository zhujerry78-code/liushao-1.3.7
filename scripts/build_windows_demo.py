"""Build deterministic local delivery from one Git commit or verified source snapshot."""
from __future__ import annotations
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from app_evaluation import SUMMARY_PATH, read_summary
from delivery.evaluation_resources import check_identity
from delivery.safe_files import read_json, validate_names, regular_file
from app_input import problem_sample
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'HEAD'
PACKAGE_REVISION = 'r9'
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
ASSETS_SHA256 = 'e38828bdf0f7b37b878313958a2963c6c7581fd69138c20d7e3968ad956b6383'
ARTIFACTS = {
    'model.joblib': '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7',
    'model.json': 'e280995a00084234c538720541db6475bb51382c4e2ffc9c0e4db8cbde933880',
    'demo.csv': 'c78db00f062b0760fe4786fef4ab091c55c6a0b352266c45c03cf7ccb820de0b',
    'demo.json': 'f659acedda59cea6b4bdf8952146f7273c7e7bf4122454daa96dad703d353789',
    'demo_origins.parquet': '294a9a44ab8c00e8256361644eb302f08e48579f49050b4880657910d81bcbff',
}
FILES = {'liushao_icon.svg', 'app_navigation.py', 'app_evaluation.py', 'app_score_charts.py', 'app.py', 'app_model.py', 'app_quality.py', 'app_ui_state.py', 'app_ui_tools.py',
         'app_triage.py', 'app_types.py', 'app_service.py', 'VERSION', 'requirements.txt',
         'requirements-lock.txt', '.streamlit/config.toml', 'start.cmd',
         'scripts/build_windows_demo.py', 'delivery/safe_files.py'}
PREFIXES = ('liushao/', 'delivery/', 'scripts/', 'tests/', 'configs/', 'third_party/')
DOCUMENTS = {'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'docs/release/SOURCE_LICENSE.md',
             'docs/usage/MODEL_CARD.md', 'docs/usage/LOCAL_DATA.md', 'docs/usage/MIGRATION.md', 'docs/usage/COMPLETE_MATERIALS.md',
             'docs/engineering/license-supplement/streamlit/1.50.0-upstream/LICENSE',
             'docs/engineering/license-supplement/playwright/playwright/driver/package/ThirdPartyNotices.txt'}
FILES.update({'app_evidence.py','app_evidence_ui.py','app_evidence_rules.py','app_decomposition.py','app_quick_search.py','docs/usage/EVIDENCE.md'})
FILES.update(DOCUMENTS | {'RELEASE_DATE', 'app_record_tools.py', 'app_input.py', 'app_session.py'})
FILES.add('docs/engineering/dependencies.json')
FILES.add('delivery/materials_bootstrap.ps1')
FILES.add('app_desktop_ready.py')
FILES.add('app_navigation_scroll.py')
FILES.add('delivery/browser-activate.cmd')
FILES.add('app_review_guidance.py')



class DocumentLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self.identifiers: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get('id'):
            self.identifiers.add(attributes['id'])
        for key in ('href', 'src'):
            if attributes.get(key):
                self.links.append(attributes[key])

def validate_documents(package: Path) -> None:
    """Require complete local navigation without pulling in historical reports."""
    documents = [path for path in package.rglob('*')
                 if path.suffix in {'.md', '.html'}
                 and not path.relative_to(package).as_posix().startswith(
                     ('third_party/', 'docs/engineering/license-supplement/'))]
    failures = []
    for document in documents:
        content = document.read_text(encoding='utf-8')
        parsed = DocumentLinks()
        parsed.feed(content)
        links = parsed.links
        if document.suffix == '.md':
            links += [a or b for a, b in re.findall(
                r'\]\((?:<([^>]+)>|([^\s\)]+))\)', content)]
        for link in links:
            reference = urlsplit(link)
            if reference.scheme in {'https', 'http', 'mailto'}:
                continue
            target = ((document.parent / unquote(reference.path)).resolve()
                      if reference.path else document.resolve())
            label = f'{document.relative_to(package).as_posix()} -> {link}'
            if reference.scheme or not target.is_relative_to(package.resolve()):
                failures.append(label + ' (outside package)')
            elif not target.exists():
                failures.append(label + ' (missing file)')
            elif reference.fragment and target.suffix == '.html':
                destination = DocumentLinks()
                destination.feed(target.read_text(encoding='utf-8'))
                if unquote(reference.fragment) not in destination.identifiers:
                    failures.append(label + ' (missing anchor)')
    if failures:
        raise ValueError('Incomplete delivery documentation:\n' + '\n'.join(failures))

def sha(content):
    return hashlib.sha256(content).hexdigest()


def allowed(name):
    if name in FILES: return True
    if name.startswith('delivery/') and Path(name).suffix not in ('.py','.cmd','.txt'): return False
    return (name in FILES or name.startswith(PREFIXES) or ('/' not in name and name.endswith('.cmd'))) and not any(
        part in ('__pycache__', '.pytest_cache') for part in Path(name).parts)


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.quotepath=false', *args], cwd=ROOT)


def read_source(source_ref=None, snapshot=None, expected_digest=None):
    if snapshot is not None:
        content = regular_file(snapshot, 'SOURCE_MANIFEST.json').read_bytes()
        if not expected_digest or sha(content) != expected_digest:
            raise ValueError('Source manifest does not match independently retained digest')
        manifest = read_json(content)
        names = manifest['files']
        validate_names(names)
        files = {}
        for name, digest in names.items():
            if not allowed(name): raise ValueError('Unexpected source entry: ' + name)
            value = regular_file(snapshot, name).read_bytes()
            if sha(value) != digest: raise ValueError('Changed source: ' + name)
            files[name] = value
        for path in snapshot.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.py','.cmd','.exe','.whl'):
                name=path.relative_to(snapshot).as_posix()
                regular_file(snapshot,name)
                if name not in names: raise ValueError('Unlisted source executable: ' + name)
        commit = manifest['source_commit']
    else:
        commit = git('rev-parse', (source_ref or BASELINE)+'^{commit}').decode().strip()
        names = git('ls-tree', '-rz', '--name-only', commit).decode('utf-8').rstrip('\0').split('\0')
        files = {}
        for name in names:
            if allowed(name):
                mode = git('ls-tree', commit, '--', name).decode().split()[0]
                if mode != '100644' and mode != '100755': raise ValueError('Non-regular source: ' + name)
                files[name] = git('show', commit+':'+name)
    if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('Invalid source commit identity')
    if not FILES <= set(files): raise ValueError('Incomplete source snapshot')
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:-rc\.\d+)?', files['VERSION'].decode().strip()):
        raise ValueError('Source version mismatch')
    return commit, files


def archive(destination, members, prefix, release_date='2026-10-21'):
    validate_names(members)
    stamp = date.fromisoformat(release_date)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(prefix+'/'+name, (stamp.year,stamp.month,stamp.day,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = sha(destination.read_bytes())
    destination.with_suffix('.zip.sha256').write_text(digest+'  '+destination.name+'\n', encoding='utf-8')
    return digest


def build(output, artifacts=None, *, source_ref=None, source_snapshot=None,
          source_manifest_sha256=None, assets_zip=None, package_revision='r9',
          evaluation_summary=None, evaluation_summary_sha256=None,
          release_descriptor=None, release_descriptor_sha256=None,
          evaluation_inputs=None, reevaluation_sha256=None):
    if artifacts is not None:
        for name, expected in ARTIFACTS.items():
            if sha((artifacts/name).read_bytes()) != expected:
                raise ValueError('Trusted artifact mismatch: ' + name)
    commit, source = read_source(source_ref, source_snapshot, source_manifest_sha256)
    for name in ('scripts/build_windows_demo.py','delivery/safe_files.py','app_evaluation.py',
                 'delivery/evaluation_resources.py','app_input.py'):
        if name not in source: continue
        if source[name].replace(b'\r\n',b'\n') != (ROOT/name).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('Run the builder and helpers from the selected source: ' + name)
    if not re.fullmatch(r'(?:r9(?:-[a-z0-9-]+)?|r10-competition|r11-final|r11-evidence-final|r11-ui-compact|r11-launch-fix|r11-review-guidance|r11-final-readable|r11-whitebox-polished|r11-whitebox-final|r11-whitebox-desktop|r11-whitebox-simple|r11-whitebox-aligned|r11-whitebox-refined)', package_revision): raise ValueError('Invalid revision')
    competition = package_revision in ('r10-competition', *REVISION_VERSIONS)
    if competition and (evaluation_summary is None or not evaluation_summary_sha256):
        raise ValueError('Competition builds require an evaluation summary and independent SHA256')
    version = source['VERSION'].decode().strip()
    release_date = source['RELEASE_DATE'].decode().strip()
    date.fromisoformat(release_date)
    if package_revision in REVISION_VERSIONS and version != REVISION_VERSIONS[package_revision]:
        raise ValueError('Source version mismatch')
    if assets_zip is None: raise ValueError('A verified --assets-zip is required')
    if sha(assets_zip.read_bytes()) != ASSETS_SHA256: raise ValueError('Untrusted assets ZIP')
    members = dict(source)
    with zipfile.ZipFile(assets_zip) as assets:
        if assets.testzip() is not None: raise ValueError('Corrupt asset archive')
        names = assets.namelist()
        validate_names([n.rstrip('/') for n in names if not n.endswith('/')])
        roots = {n.split('/')[0] for n in names}
        if len(roots) != 1: raise ValueError('Unexpected archive roots')
        prefix = next(iter(roots))+'/'
        for name in names:
            relative = name[len(prefix):]
            if name.endswith('/'): continue
            if (relative.startswith(('runtime/', 'wheels/', 'artifacts/'))
                    or relative in ('LICENSE', 'NOTICE')):
                if (assets.getinfo(name).external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Asset archive links forbidden')
                if relative not in source: members[relative] = assets.read(name)
    for name, expected in ARTIFACTS.items():
        content = (artifacts / name).read_bytes() if artifacts else members['artifacts/'+name]
        if sha(content) != expected: raise ValueError('Trusted artifact mismatch: ' + name)
        members['artifacts/'+name] = content
    aligned = package_revision in ('r11-whitebox-aligned', 'r11-whitebox-refined')
    if aligned:
        if release_descriptor is None or not release_descriptor_sha256 or evaluation_inputs is None or not reevaluation_sha256:
            raise ValueError('Aligned release requires independently pinned descriptor and reevaluation inputs')
        original_bytes = members['artifacts/model.json']
        release_bytes = release_descriptor.read_bytes()
        if sha(release_bytes) != release_descriptor_sha256:
            raise ValueError('Release descriptor digest mismatch')
        from delivery.evaluation_resources import verify_training_descriptor
        descriptor = read_json(release_bytes)
        verify_training_descriptor(descriptor, original_bytes)
        for name, digest in descriptor['source_code_sha256'].items():
            if sha(members[name.replace('\\', '/')].replace(b'\r\n', b'\n')) != digest:
                raise ValueError('Release source registration mismatch: ' + name)
        members['artifacts/model.json'] = release_bytes
        if package_revision == 'r11-whitebox-refined':
            members['artifacts/model.original.json'] = original_bytes
    evaluation_resources = {}
    examples = {'resources/examples/格式练习_'+kind+'.csv': problem_sample(
        read_json(members['artifacts/model.json'])['config']['features'], kind)
        for kind in ('缺少必需列', '缺值与无效记录')}
    members.update(examples)
    if evaluation_summary is not None:
        summary_bytes = evaluation_summary.read_bytes()
        if not evaluation_summary_sha256 or sha(summary_bytes) != evaluation_summary_sha256:
            raise ValueError('Evaluation summary does not match independently retained SHA256')
        summary = read_summary(evaluation_summary, ARTIFACTS['model.joblib'], ARTIFACTS['model.json'])
        identity = check_identity(summary, read_json(members['artifacts/model.json']),
                                  release_descriptor_sha256 if aligned else ARTIFACTS['model.json'],
                                  original_bytes if aligned else None)
        evaluation_resources[SUMMARY_PATH] = summary_bytes
        if aligned:
            for item in summary['sources']:
                value = regular_file(evaluation_inputs, item['path']).read_bytes()
                if sha(value) != item['sha256']:
                    raise ValueError('Historical evidence digest mismatch: ' + item['path'])
                evaluation_resources[item['path']] = value
            evaluation_resources['resources/evaluation/model.original.json'] = original_bytes
            evaluation_resources['resources/evaluation/model.release.json'] = release_bytes
            if package_revision == 'r11-whitebox-refined':
                evaluation_resources['artifacts/model.original.json'] = original_bytes
            replay = regular_file(evaluation_inputs, 'resources/evaluation/reevaluation.json').read_bytes()
            if sha(replay) != reevaluation_sha256:
                raise ValueError('Reevaluation digest mismatch')
            replay_report = read_json(replay)
            if (replay_report['status'] != 'passed' or replay_report['training_executed'] is not False
                    or replay_report['identity']['model_sha256'] != ARTIFACTS['model.joblib']
                    or replay_report['identity']['descriptor_sha256'] != release_descriptor_sha256
                    or replay_report['identity']['data_version'] != identity['data_version']):
                raise ValueError('Reevaluation identity mismatch')
            evaluation_resources['resources/evaluation/reevaluation.json'] = replay
        evaluation_resources['EVALUATION_RESOURCES.json'] = (json.dumps(
            {'format': 1, 'identity': identity, 'files': {n: {'bytes': len(v), 'sha256': sha(v)} for n, v in evaluation_resources.items()}},
            ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        members.update(evaluation_resources)
    source_manifest = {'format':1, 'source_commit':commit,
                       'files':{n:sha(v) for n,v in sorted(source.items())}}
    members['SOURCE_MANIFEST.json'] = (json.dumps(source_manifest,ensure_ascii=False,indent=2)+'\n').encode()
    members['README.md'] = (f"# 流哨 {version} / {package_revision}\n\n"
        "解压到独立可写目录，双击 1_一键安装并启动.cmd。首次安装只用包内依赖。\n"
        "以后双击 2_启动流哨.cmd；关闭使用 4_关闭流哨.cmd；排查使用 3_环境自检.cmd。\n"
        "复核保存在 demo-state/reviews.sqlite；不要覆盖旧目录或复制虚拟环境。\n"
        "下载结果页说明三种文件用途。未保存填写须先保存或明确放弃。\n"
        "模型提示辅助人工检查；未提示异常不保证安全。试验工具限制在对应页面说明。\n"
        "源码默认数据库为 data/reviews.sqlite；LIUSHAO_REVIEW_DB 可明确指定路径。\n"
        "操作说明中的模型与评测展示同一冻结模型的历史结果；不是本次上传文件的准确率。\n"
        "操作记录 JSON 当前用于留档，没有导入恢复功能。回退旧目录不合并新版新增记录。\n"
        "活跃数据库使用 SQLite backup 备份，不单独复制可能有 WAL 的主文件。\n").encode()
    members['README.md'] += ('\n迁移顺序：解压 → 5_迁移旧版复核.cmd → 1_一键安装并启动.cmd。新库已存在则停止，不覆盖。\n'
        '上传文件默认不保留；勾选后成功分析才保留最近一次，重启点击“继续上次检查”。\n'
        '下载页另有只含已确认记录的复核结论单。\n\n'
        '[模型卡](docs/usage/MODEL_CARD.md) · [本地文件说明](docs/usage/LOCAL_DATA.md) · '
        '[迁移说明](docs/usage/MIGRATION.md) · [复核与评分](docs/usage/EVIDENCE.md) · [许可范围](docs/release/SOURCE_LICENSE.md) · '
        '[第三方与数据来源](THIRD_PARTY_NOTICES.md) · [完整材料与重建](docs/usage/COMPLETE_MATERIALS.md)\n').encode('utf-8')
    if package_revision in ('r11-ui-compact', 'r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        members['README.md'] += ('\n检查记录：上方列表选一条，下方看分数尺和简短证据，填写判断后保存。\n'
            '“查看完整评分依据”打开独立详情，含评分图解、证据明细、原始数据与方法三个标签页。\n'
            '点击“返回复核”保留当前记录、筛选、检索、页码和未保存填写；打开详情不会保存。\n').encode('utf-8')
    if package_revision in ('r11-launch-fix', 'r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        members['README.md'] += ('\n启动会自动打开流哨专用浏览器窗口，并等待页面就绪回执。服务已运行但页面未就绪时，请按启动提示处理。\n'
            '4_关闭流哨.cmd 停止本目录服务；仍显示的专用窗口可直接点右上角关闭。保留普通浏览器和其他版本。\n'
            '若仍出现灰色空白，请保留启动日志；不要通过关闭浏览器安全沙箱处理。\n').encode('utf-8')
    if package_revision in ('r11-review-guidance', 'r11-final-readable', 'r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        members['README.md'] += ('\n“我的判断”上方单列“这条记录需要你核对”，把当前值、参考和具体需查什么放在一起。\n'
            '先读这一区，再结合日志、资产用途和时间范围填写判断；这些提示不会替你确认攻击。\n').encode('utf-8')
    if package_revision in ('r11-whitebox-polished', 'r11-whitebox-final', 'r11-whitebox-desktop', 'r11-whitebox-simple'):
        members['README.md'] += ('\n白盒版：核查表分别列出当前值、训练参考、RF路径贡献与替换后变化。\n'
            '“评分依据”可展开全部字段与计算精度；整批数据分析保留77项区间统计和CSV下载。\n'
            '备注草稿先预览，再点击“追加到备注”；已有文字会保留，仍需人工核查并点击保存。\n'
            '参考偏离是统计比较，不是攻击风险等级；模型、判断阈值及演示数据延用原白盒包。\n'
            '本包不含其他人的虚拟环境、复核数据库和历史日志。请在新目录首次安装。\n').encode('utf-8')
    if package_revision in ('r11-whitebox-final', 'r11-whitebox-desktop'):
        members['README.md'] += ('\n1.3.2 启动修复：浏览器在有效期内重试页面就绪检查，等待界面渲染完成后返回回执。\n'
            '短暂渲染延迟或回执请求失败可重试；超时、异常页面或无效启动请求仍不会记为启动成功。\n').encode('utf-8')
    if package_revision == 'r11-whitebox-desktop':
        members['README.md'] += ('\n1.3.3 桌面排版：展开区调整为完整宽度，避免内容重叠；备注草稿分段显示。\n'
            '检查流程精简重复声明，保留必要的核查说明与保存提示。\n').encode('utf-8')
    if package_revision == 'r11-whitebox-simple':
        members['README.md'] += ('\n1.3.4 界面精简：缩短核查与分析说明，合并重复提示；详细方法按需展开。\n').encode('utf-8')
    if aligned:
        members['README.md'] += ('\n前两阶段整改：邻居统计按当前CSV物理行关联；缺失试算标注状态变化，参考边界按中位数分段计算。\n'
            '安装根路径建议不超过100字符；101至120字符警告后允许，超过120字符拒绝。使用本机可写、非同步目录，安装后不要移动或改名。\n'
            '服务就绪但窗口未确认时保留服务，按提示重新打开，或再次运行2_启动流哨.cmd。\n'
            '六份历史评测来源随包，新冻结复评与条件区间另列；原模型与判断阈值未改变。\n'
            '独立复评资源包包含处理数据和新预测；运行包不附完整评测数据，历史全量重训脚本不在本次执行范围。\n').encode('utf-8')
        if package_revision == 'r11-whitebox-refined':
            members['README.md'] += ('\n1.3.6收尾修正：启动完整性优先，缺少任何登记文件（含历史证据）时请保留复核库并完整重新解压原包。\n'
                '公开证据仅对本机绝对路径及对应SHA引用生成明确标注的派生副本，历史指标未改写。\n'
                '原始训练描述随模型放在artifacts/model.original.json；移动模型目录时连同两个描述JSON一起复制。\n'
                '区间与单位规则下沉至纯库模块；总发出/返回包长度按字节显示，评分与阈值未变。\n').encode('utf-8')
    manifest = {'format':1,'software_version':version,'software_sha':commit,'release_date':release_date,
                'package_revision':package_revision,'model_version':'c7a028b4e4cafedd',
                'assets_zip_sha256':ASSETS_SHA256,'manifest_excludes_itself':True,
                **({'evaluation_summary_sha256': evaluation_summary_sha256} if evaluation_resources else {}),
                'files':{n:{'bytes':len(v),'sha256':sha(v),
                        'origin':'source-snapshot' if n in source else 'verified-assets-or-generated'}
                        for n,v in sorted(members.items())}}
    members['PACKAGE_MANIFEST.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    members['SHA256SUMS.txt'] = ''.join(sha(v)+'  '+n+'\n' for n,v in sorted(members.items())).encode()
    output.mkdir(parents=True,exist_ok=False)
    package = output/f'Liushao-{version}-Windows-{package_revision}'
    package.mkdir()
    validate_names(members)
    for name, content in members.items():
        path = package/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    validate_documents(package)
    title = '比赛完善版' if competition else '工程完善版'
    destination = output/f'流哨_{title}_{package_revision}_{release_date}.zip'
    digest = archive(destination,members,package.name,release_date)
    source_zip = output/f'流哨_{title}_{package_revision}_源码_{release_date}.zip'
    source_hash = sha(members['SOURCE_MANIFEST.json'])
    evaluation_args = (' --evaluation-summary ' + SUMMARY_PATH + ' --evaluation-summary-sha256 ' + evaluation_summary_sha256) if evaluation_resources else ''
    if aligned:
        evaluation_args += (' --release-descriptor resources/evaluation/model.release.json --release-descriptor-sha256 '
                            + release_descriptor_sha256 + ' --evaluation-inputs . --reevaluation-sha256 ' + reevaluation_sha256)
    instructions = (f'流哨 {version} / {package_revision} 源码包\n'
        '模型、演示资源、Python 与 wheel 在独立运行包；源码不会自动下载或训练模型。\n'
        '本包来自本地提交 ' + commit + '。\n'
        '重建请在 Python 3.12、源码根目录执行：\n'
        'python -m scripts.build_windows_demo --source-snapshot . --source-manifest-sha256 '
        + source_hash + ' --assets-zip <已核验本地资源ZIP路径> --package-revision ' + package_revision
        + evaluation_args + ' --output <全新输出目录>\n'
        '资源ZIP SHA-256：' + ASSETS_SHA256 + '\n'
        '请同时核对独立保存的 SOURCE_MANIFEST.sha256；清单不等于数字签名。\n')
    archive(source_zip,dict(source,**{'SOURCE_MANIFEST.json':members['SOURCE_MANIFEST.json'],
        'SOURCE_README.txt':instructions.encode('utf-8'), 'LICENSE':members['LICENSE'], 'NOTICE':members['NOTICE']},
        **evaluation_resources, **examples),f'liushao-{package_revision}-source',release_date)
    (output/'SOURCE_MANIFEST.sha256').write_text(source_hash+'  SOURCE_MANIFEST.json\n',encoding='utf-8')
    (output/'SHA256SUMS.txt').write_text(''.join(sha(p.read_bytes())+'  '+p.name+'\n'
        for p in sorted(output.glob('*.zip'))),encoding='utf-8')
    print(json.dumps({'zip':str(destination),'sha256':digest,'source_commit':commit,
                      'source_manifest_sha256':sha(members['SOURCE_MANIFEST.json'])},ensure_ascii=False))
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--source-ref')
    choice.add_argument('--source-snapshot',type=Path)
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--assets-zip',required=True,type=Path)
    parser.add_argument('--package-revision',default='r9')
    parser.add_argument('--evaluation-summary',type=Path)
    parser.add_argument('--evaluation-summary-sha256')
    parser.add_argument('--release-descriptor', type=Path)
    parser.add_argument('--release-descriptor-sha256')
    parser.add_argument('--evaluation-inputs', type=Path)
    parser.add_argument('--reevaluation-sha256')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    build(args.output.resolve(),source_ref=args.source_ref,source_snapshot=args.source_snapshot,
          source_manifest_sha256=args.source_manifest_sha256,assets_zip=args.assets_zip,
          package_revision=args.package_revision,evaluation_summary=args.evaluation_summary,
          evaluation_summary_sha256=args.evaluation_summary_sha256,
          release_descriptor=args.release_descriptor, release_descriptor_sha256=args.release_descriptor_sha256,
          evaluation_inputs=args.evaluation_inputs, reevaluation_sha256=args.reevaluation_sha256)


if __name__=='__main__': main()
