"""Read aggregate JSON only; render shareable scientific figures in separate env."""
import hashlib
import importlib.metadata as md
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/innovation/figures'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    study_path = ROOT / 'docs/innovation/study.json'
    r = json.loads(study_path.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    cjk = Path('C:/Windows/Fonts/msyh.ttc')
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
        family = font_manager.FontProperties(fname=str(cjk)).get_name()
    else:
        choices = ['Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei']
        available = {font.name for font in font_manager.fontManager.ttflist}
        family = next((font for font in choices if font in available), None)
        if family is None:
            raise RuntimeError('请安装中文字体 Microsoft YaHei 或 Noto Sans CJK SC')
    plt.rcParams.update({'font.family': [family, 'DejaVu Serif'], 'font.size': 9,
                         'axes.labelsize': 10, 'axes.linewidth': .75, 'axes.unicode_minus': False,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.2,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
                         'savefig.dpi': 300, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    colors = {'combined': '#5A76A8', 'vote_uncertainty': '#C96A6A', 'margin': '#4FA38F',
              'random_all': '#999999', 'random_alert': '#333333'}
    generated = []
    def save(fig, name):
        for extension in ['svg', 'pdf', 'png']:
            path = OUT / (name + '.' + extension)
            fig.savefig(path, bbox_inches='tight', metadata={'Creator': 'Liushao frozen-prediction analysis'})
            generated.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        plt.close(fig)
    def budget_values(name):
        return [next(row['errors_found'] for row in r['budgets'] if row['strategy'] == name and row['k'] == k) for k in [20,50,100]]

    # Fig1: target-consistent strategies, all predeclared budgets, no smoothing.
    fig, ax = plt.subplots(figsize=(6.8, 4.0), layout='constrained')
    for name, label, offset in [('vote_uncertainty','RF 0.5中心度（熵排序）',2),
                                ('combined','原组合 / 仅分歧',2), ('margin','阈值距离 / 仅近阈值',-4)]:
        values = budget_values(name)
        ax.plot([20,50,100], values, 'o-', color=colors[name], lw=1.8, ms=4, label=label)
        for x,y in zip([20,50,100], values):
            dy = offset * 3
            if x == 100:
                dy = {'combined': -15, 'margin': 9, 'vote_uncertainty': -14}[name]
            ax.annotate(str(y), (x,y), xytext=(0,dy), textcoords='offset points', ha='center', color=colors[name])
    for name,label,key in [('random_alert','告警内随机（期望）','alert'),('random_all','全候选随机（期望）','all')]:
        ax.plot([20,50,100], [r['random_references'][str(k)][key]['expected_errors'] for k in [20,50,100]],
                'o-', color=colors[name], lw=1.0, ms=3, label=label)
    ax.axhline(r['total_errors'], color='#777777', lw=.8, ls=':')
    ax.text(102, r['total_errors']+.5, '总错误56', ha='right', va='bottom', color='#666666', fontsize=8)
    ax.set(xlabel='复核预算 K（条）', ylabel='发现的RF错误（条）', xticks=[20,50,100], xlim=(14,106), ylim=(-1,61))
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'fig1_error_discovery')

    # Fig2: disjoint partitions, so workloads are not double-counted.
    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout='constrained')
    keys = ['disagreement_only', 'intersection', 'near_only']
    labels = ['仅分歧独有', '两者交集', '仅近阈值独有']
    counts = [r['workload'][key]['records'] for key in keys]
    errors = [r['workload'][key]['errors'] for key in keys]
    others = np.array(counts) - errors
    x = np.arange(3)
    ax.bar(x, errors, width=.55, color='#C96A6A', label='RF错误（均为误报）')
    ax.bar(x, others, bottom=errors, width=.55, color='#D8DEE8', label='RF判断正确')
    for i,(n,e) in enumerate(zip(counts,errors)):
        ax.text(i,n+3,f'n={n}',ha='center')
        ax.text(i,e/2,str(e),ha='center',va='center',color='white',weight='bold')
    ax.set(xticks=x, xticklabels=labels, ylabel='需要复核的记录（条）', ylim=(0,max(counts)+40))
    ax.legend(loc='upper center', bbox_to_anchor=(.5,1.02), ncol=2, frameon=False)
    save(fig, 'fig2_workload_partitions')

    # Fig3: complete original hint queue, not a newly optimized order.
    fig, ax = plt.subplots(figsize=(7.1, 3.8), layout='constrained')
    length = r['workload']['combined']['records']
    ranks = np.arange(length + 1)
    error_ranks = np.array(r['ranking']['error_ranks'])
    cumulative = np.array([(error_ranks <= rank).sum() for rank in ranks])
    ax.axvspan(0,175,color='#5A76A8',alpha=.06)
    ax.axvspan(175,length,color='#4FA38F',alpha=.08)
    ax.axvspan(35,146,color='#777777',alpha=.08)
    ax.step(ranks,cumulative,where='post',color=colors['combined'],lw=1.8)
    ax.axvline(175,color='#999999',lw=.8,ls=':')
    for k in [20,50,100]:
        value = cumulative[k]
        ax.plot(k,value,'o',ms=4,color=colors['combined'])
        ax.annotate(f'K={k}: {value}',(k,value),xytext=(5,10),textcoords='offset points',fontsize=8)
    ax.annotate('第23条错误在第147名', (147,23), xytext=(155,12),
                arrowprops={'arrowstyle':'->','color':'#555555','lw':.8}, fontsize=8)
    ax.text(85,54,'分歧优先区：175条',ha='center',fontsize=8,color='#5A76A8')
    ax.text(249,54,'近阈值独有区：147条',ha='center',fontsize=8,color='#3A8270')
    ax.set(xlabel='原待核实队列名次（条）',ylabel='累计发现的RF错误（条）',xlim=(0,length),ylim=(0,56),xticks=[0,20,50,100,147,175,250,322])
    save(fig, 'fig3_original_queue_plateau')

    # Fig4: paired differences; no claim from overlapping separate intervals.
    fig, ax = plt.subplots(figsize=(7.6,4.0), layout='constrained')
    row_labels=[]
    for j,k in enumerate([20,50,100]):
        for i,(base,key,color,label) in enumerate([
            ('margin','combined_minus_margin','#4FA38F','阈值距离'),
            ('vote_uncertainty','combined_minus_vote_uncertainty','#C96A6A','0.5中心度')]):
            ci = r['bootstrap']['budgets'][str(k)][key]
            point = budget_values('combined')[j] - budget_values(base)[j]
            y=5-(j*2+i)
            ax.plot([ci['p025'],ci['p975']],[y,y],color=color,lw=1.6)
            ax.plot(point,y,'o',color=color,ms=5)
            ax.text(ci['p975']+1,y,f"[{ci['p025']:.0f}, {ci['p975']:.0f}]",va='center',fontsize=8,color=color,
                    bbox={'facecolor':'white','edgecolor':'none','pad':.2})
            row_labels.append((y,f'K={k}，相对{label}'))
    ax.axvline(0,color='#555555',lw=.8,ls=':')
    ax.set(yticks=[y for y,_ in row_labels],yticklabels=[label for _,label in row_labels],
           xlabel='原组合比基线多发现的错误（条）',ylim=(-.7,5.7))
    ax.margins(x=.22)
    save(fig, 'fig4_paired_block_ranges')

    # Record the independent drawing environment and its additional license files.
    metadata=[]
    existing = json.loads((ROOT/'docs/results/dependencies.json').read_text(encoding='utf-8'))
    known={(row['name'].lower().replace('_','-'),row['version']) for row in existing['packages']}
    for dist in sorted(md.distributions(),key=lambda d:d.metadata['Name'].lower()):
        name=dist.metadata['Name']
        row={'name':name,'version':dist.version,'license_expression':dist.metadata.get('License-Expression'),
             'license_metadata':dist.metadata.get('License'),'project_urls':dist.metadata.get_all('Project-URL',[]),
             'notices':[],'already_recorded_in_main_env':(name.lower().replace('_','-'),dist.version) in known}
        if not row['already_recorded_in_main_env']:
            for rel in dist.files or []:
                if rel.name.lower().startswith(('license','licence','copying','copyright','notice')):
                    source=Path(dist.locate_file(rel))
                    if source.is_file():
                        dest=ROOT/'docs/innovation/licenses'/name/str(rel).replace('..','_')
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(source,dest)
                        row['notices'].append({'path':dest.relative_to(ROOT).as_posix(),'sha256':digest(dest)})
        metadata.append(row)
    manifest={'source_study_sha256':digest(study_path),'matplotlib':matplotlib.__version__,
              'font':family,'png_dpi':300,'smoothing':False,'files':generated,
              'script_sha256':digest(__file__),'dependencies':metadata,
              'mapping':{'fig1':'x=K (records), y=errors found (records), groups=fixed policies',
                         'fig2':'x=disjoint D-only/intersection/T-only, y=review records, stack=error/correct',
                         'fig3':'x=original rank, y=cumulative errors; actual step counts without smoothing',
                         'fig4':'x=paired difference in error counts, rows=K/baseline, bars=conditional block 2.5%-97.5%'}}
    (ROOT/'docs/innovation/figures_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'docs/innovation/requirements-figures-lock.txt').write_text(freeze,encoding='utf-8')
    print(f'Generated {len(generated)} files with {matplotlib.__version__}, {family}')


if __name__=='__main__':
    main()
