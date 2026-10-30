# 研判引擎（1.3.7 应用，1.3.7 冻结参数）

## 链路

多源异构检测（RF、LR、ECOD）→ Dempster–Shafer 证据融合 → Mondrian 共形预测 → Venn–Abers 概率端点 → 树路径与 ECOD 尾部归因。

| 模块 | 方法 | 产出 |
|---|---|---|
| 无监督检测 | ECOD（Li et al. 2022）：逐特征经验分布左右尾概率取 −log 后求和，取左尾、右尾和偏度自动方向三者中的最大值；只用训练集正常流量拟合 | `ecod_score` |
| 证据融合 | 各证据源分数经验证集 A 半区保序回归校准并学习方向，概率截断在 [10⁻⁶,1−10⁻⁶]；可靠度 r = clip(2·AUC − 1,0,0.99)，再作 Shafer 折扣：m(攻击)=r·p，m(正常)=r·(1−p)，m(Θ)=1−r，最后按 Dempster 规则组合 | `ds_belief`、`ds_plausibility`、`ds_conflict`（K）、`ds_betp` |
| 不确定性量化 | Mondrian 分割共形预测：在验证集 B 半区上分标签校准，非一致性分数取 1 − BetP 或 BetP，ε = 0.05 | `cp_set`（正常 / 攻击 / 不确定）、`cp_p_attack`、`cp_p_normal` |
| 概率端点 | 从冻结共形 B 分组恢复校准样本，对查询 BetP 分别附加标签 0/1 作精确保序双拟合，替代旧网格插值；不是固定置信水平的置信区间 | `va_low`、`va_high` |
| 可解释 | RF 树路径（Saabas）分解，与证据页原有的精确分数分解一致；ECOD 逐特征尾部贡献 | 证据页“可信研判引擎”面板 |

## 数据使用

- ECOD：训练集中的正常流量，随机抽 20,000 行。
- 校准器、可靠度和冲突阈值：验证集按标签分层随机对半切分（种子 42）后的 A 半区。
- 共形预测和 Venn–Abers：B 半区。两个半区互不重叠，test 不参与任何拟合。
- 参数文件为 `artifacts/engine.joblib`，描述文件 `artifacts/engine.json` 记录了数据 SHA、引擎源码 SHA 和底座模型 SHA。

## 界面行为

- 主队列（异常分数从高到低）按 `ds_belief` 排序，相同时再按 RF 分数排序。
- “需要再检查”收录两类记录：共形预测集给出“不确定”的，以及冲突度 K 不低于 A 半区 99% 分位的。队列内先排 BetP 最接近 0.5 的记录，再按 K 从高到低排；这条排序规则是在验证集上选定的。
- RF/LR 分数、阈值、证据页和旧复核都保留。标准启动先全量校验：任何登记文件（包括引擎参数）缺失或哈希不符会停止启动。应用分析时引擎计算失败才保留 RF/LR 的结果、排序和队列。
- ECOD 贡献使用实际获胜总尾部分支；全部字段加总还原分数。展示前六项不代表最终攻击结论的因果贡献。
- 保存同时包含完整引擎输出、参数/描述 SHA、应用推理版本、质量分配和完整 ECOD 贡献。历史读取保存值，不重算或补造缺失证据。

## 复现

```
python -m scripts.fit_engine --data <processed 数据目录>
python -m scripts.evaluate_engine --data <processed 数据目录> --cross-day <Thursday-15-02-2018 CSV> --output docs/results/contest_engine_evaluation.json
```

拟合脚本用于说明原参数来源，本次不重训。当前评测脚本默认使用本版精确 VA，指定 `--legacy-grid-va` 才比较旧网格方法。`docs/results/engine_evaluation.json` 保留 1.3.7 历史原值，其中 `ecod_score_polarity_calibrated` 实际是用测试标签计算的原始 AUC 方向折叠，不是真实 PAIC 输出；新评测分别输出 `ecod_raw_orientation_folded_using_test_labels` 与 `ecod_frozen_paic_output`。详细冻结范围及修复见[本版变更](../release/CONTEST_CHANGELOG.md)。

已有实算记录：[当前算法留出结果](../results/contest_script_holdout_evaluation.json)、[补强实验总结果](../results/contest_frozen_evaluation.json)。这些数值来自同一冻结模型和同一应用推理实现，版本更正没有重跑或改写实验；其复用依据和本次新验证见[版本映射与等价性记录](../results/contest_version_identity.json)。留出与跨日结果须分别解读，数学修复不代表跨场景性能普遍提升。
