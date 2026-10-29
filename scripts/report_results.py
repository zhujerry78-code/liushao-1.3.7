"""Build the technical result note from actual machine-readable outputs."""
import json
from pathlib import Path


def main():
    def read(name):
        return json.loads(Path(name).read_text(encoding='utf-8'))
    data = read('data/processed/summary.json')
    metrics = read('artifacts/metrics.json')
    repro = read('docs/results/reproducibility.json')
    browser = read('docs/results/browser_verification.json')
    dependencies = read('docs/results/dependencies.json')
    rows = []
    for name, key in [('随机森林', 'rf'), ('逻辑回归', 'lr')]:
        m = metrics[key]
        rows.append(f'| {name} | {m["precision"]:.6f} | {m["recall"]:.6f} | {m["f1"]:.6f} | {m["fpr"]:.6f} | {m["fp"]} | {m["fn"]} | {m["threshold"]:.2f} |')
    budget_rows = []
    for entry in metrics['review']['same_budget_error_discovery']:
        budget_rows.append(f'| {entry["common_budget"]} | {entry["original"]["errors_found"]} | {entry["score"]["errors_found"]} | {entry["review"]["errors_found"]} |')
    sources = '\n'.join(f'- `{s["name"]}`：{s["bytes"]:,} 字节，SHA-256 `{s["sha256"]}`。' for s in data['sources'])
    text = f'''# 流哨 MVP 实测记录

生成依据：本地真实数据处理、训练、复训与浏览器验收。实现完成日期为 2026-10-16（Asia/Shanghai）。本文为开发实测记录，保留材料 V1.0，不直接覆盖参赛 PDF/PPT。

## 数据与第一阶段

来源为 CSE-CIC-IDS2018 官方 AWS 两个文件，原始记录共 {sum(a['raw_rows'] for a in data['audits']):,} 条。原始标签统计中包含一条重复表头记录，固定抽样未选中该行。实际抽样 {sum(a['sampled_rows'] for a in data['audits']):,} 条；77 个模型特征，标签、时间、IP、Flow ID、端口不作为预测特征。

{sources}

文件先核验字节数和内容哈希，再全量检查唯一表头、字段数和单行结构。读取时按 50,000 行分块，按 seed42、源文件哈希及原始物理行号的 SHA-256 模10取0进行约10%抽样，独立于类别。非数值、正负无穷及超出 float32 范围的值转缺失；全部特征无效时拒绝推理。部分缺失通过训练中位数填补，并在界面明示。

按“来源文件 + 15分钟时间块”分组，固定 seed42 随机排列时间块，以约60%/20%/20%的块数分配训练/验证/测试。块内记录数量不同，因此记录比例不等于60%/20%/20%。跨集合完全重复及六位有效数字相同的特征组优先保留训练、其次验证、最后测试。此近重复定义不是任意距离相似度保证。

- 训练：{data['split_counts']['train']:,}；验证：{data['split_counts']['validation']:,}；测试：{data['split_counts']['test']:,}。
- 跨集合重复剔除：{data['split_counts']['purged']:,}；保留集合重叠为0。
- 上传 CSV 会在 pandas 解析前检查表头；空行不改变物理行号，当前拒绝多行字段。
- 两次独立处理的特征、manifest、拒绝记录文件哈希及数据版本完全相同。
- 数据版本：`{metrics['data_version']}`。

## 模型与真实测试结果

RF：160棵树、最大深度18、叶节点最少2条、balanced_subsample、seed42。LR：训练中位数填补、signed-log、训练 StandardScaler、balanced、max_iter2000、seed42。预处理和正常参考仅拟合训练集。

两模型分别在验证集上，从0.05至0.95、步长0.01的阈值网格中选择 F1 最高者；并列取更高阈值。近阈值宽度使用验证得分到主阈值距离的10%分位数，限制于0.01至0.10。测试集不参与选择。

| 模型 | Precision | Recall | F1 | FPR | 误报 | 漏报 | 阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

测试样本支持数：正常13,532、Bot1,986、SlowHTTPTest1；Hulk为0。因此上述指标主要反映这批正常与Bot样本的二分类表现。100%测试召回率不代表全攻击类别、未来网络或真实部署中的100%检测率。

模型版本 `{metrics['model_version']}`。独立复训的类别预测和评测指标一致；得分最大绝对差：RF {repro['score_max_absolute_differences']['rf_score']:.3g}，LR {repro['score_max_absolute_differences']['lr_score']:.3g}。序列化包包含运行耗时，不要求包字节完全相同。

## 技术改进点的可核验效果

双队列把“优先找疑似攻击”和“核对模型可能犯错的记录”分开，不相加两个模型的分数。高异常列表按主模型得分降序；待核实列表先分歧，再按到主阈值距离升序；最终均用稳定记录编号消除并列歧义。

在同一15,519条测试候选全集中，原始顺序 Precision@20/50/100 为45%/72%/77%，高得分排序均为100%；相应攻击召回率仅为1.01%/2.52%/5.03%，因为这只是有限检查预算。

待核实提示 {metrics['review']['flagged']} 条，覆盖 {metrics['review']['errors_total']} 条RF误报中的 {metrics['review']['errors_flagged']} 条（91.07%）。提示中的错误比例15.84%，不能把提示直接当作错误或攻击。当前RF无漏报，无法据此证明该机制能改善漏报。

| 同等检查条数 | 原始顺序发现错误 | 高得分顺序发现错误 | 待核实顺序发现错误 |
| --- | ---: | ---: | ---: |
{chr(10).join(budget_rows)}

这是错误发现的离线对照；并未改变分类器，不能表述为“复核模块提高了F1”，也不能声称真人处理节时。原始文件顺序受数据采集安排影响，后续可增加更多固定种子或场景验证，但本次不择优报告。

## 可运行界面与复核闭环

已实现演示数据/CSV导入、字段与无效行检查、双模型预测、得分和状态筛选、两类队列、详情来源、训练正常参考、人工结论和备注、追加式历史，以及预测/历史/运行清单导出。人工复核不会覆盖模型预测或数据集真值。原始标签不显示在推理界面。

真实 Edge 浏览器完成 {len(browser['checks'])} 项检查；记录见 `browser_verification.json`。单元/集成回归结果见 `pytest.xml`。浏览器验收使用隔离数据库，未向实际复核库写入测试备注。桌面界面截图保存在本地 `test-results/`，不纳入敏感结果发布。

## 性能边界与开源记录

本机逻辑CPU {metrics['machine']['logical_cpus']}，内存约 {metrics['machine']['ram_bytes']/1024**3:.1f} GiB，Python {metrics['machine']['python']}。本次RF训练 {metrics['timing']['fit_seconds']['rf']:.3f} 秒，LR训练 {metrics['timing']['fit_seconds']['lr']:.3f} 秒；测试集15,519条推理 {metrics['timing']['test_predict_seconds']:.3f} 秒，不含文件读取与模型加载。该快照不是吞吐基准或实时服务承诺。

已记录 {len(dependencies['packages'])} 个安装包，收集 {sum(len(p['bundled_notices']) for p in dependencies['packages'])} 份发行包许可证/NOTICE。实际版本与完整环境分别见 `dependencies.json`、`requirements-lock.txt`。项目自有代码最终发布许可沿用仓库当前待负责人确认状态，不能以依赖许可代替项目许可。

## 未实现范围

IP/时间线索聚合、复杂可视化、大模型问答、实时抓包、自动阻断、自动重训、CIC-UNSW与IoT2023适配、跨环境评测及真实用户试验。不能在参赛材料中把它们列为已有成果。
'''
    Path('docs/results/MVP_RESULTS.md').write_text(text, encoding='utf-8')
    print('Wrote docs/results/MVP_RESULTS.md')


if __name__ == '__main__':
    main()
