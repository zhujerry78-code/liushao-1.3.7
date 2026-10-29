"""Stable application routes; labels and legacy widget keys are independent."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    id: str
    label: str
    icon: str
    group: str
    key: str


PAGES = (
    Page('overview', '查看结果', 'dashboard', 'main', 'nav_查看结果'),
    Page('review', '检查记录', 'fact_check', 'main', 'nav_检查记录'),
    Page('quality', '文件问题', 'description', 'main', 'nav_文件问题'),
    Page('download', '下载结果', 'download', 'main', 'nav_下载结果'),
    Page('auxiliary', '辅助检查', 'manage_search', 'tools', 'nav_辅助检查'),
    Page('batch', '整批数据分析', 'bar_chart', 'tools', 'nav_整批数据分析'),
    Page('help', '操作说明', 'help', 'help', 'nav_操作说明'),
    Page('about', '关于流哨', 'info', 'help', 'nav_关于流哨'),
)
BY_ID = {page.id: page for page in PAGES}
BY_ID['evidence_detail'] = Page('evidence_detail', '评分依据', 'insights', 'detail', 'nav_evidence_detail')
LEGACY = {page.label: page.id for page in PAGES}
LEGACY.update({'模型与评测':'help','evaluation':'help'})
LEGACY.update({page.key[4:]: page.id for page in PAGES if page.key != 'nav_evaluation'})


def page_id(value):
    """Accept callbacks from an older running UI without migrating other state."""
    if not isinstance(value, str):
        return 'overview'
    return value if value in BY_ID else LEGACY.get(value, 'overview')


def page_label(value):
    return BY_ID[page_id(value)].label
