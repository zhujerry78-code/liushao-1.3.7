# 可信研判引擎（1.3.7）

## 链路

多源异构检测（RF、LR、ECOD）→ Dempster–Shafer 证据融合 → Mondrian 共形预测 → Venn–Abers 概率区间 → 树路径与 ECOD 尾部归因。

| 模块 | 方法 | 产出 |
|---|---|---|
| 无监督检测 | ECOD（Li et al. 2022）：逐特征经验分布左右尾概率取 −log 后求和，取左尾、右尾和偏度自动方向三者中的最大值；只用训练集正常流量拟合 | `ecod_score` |
| 证据融合 | 各证据源分数先经验证集 A 半区等渗回归校准为概率 p，等渗回归会自动学习方向；再按可靠度 r = 2·AUC − 1 做 Shafer 折扣：m(攻击)=r·p，m(正常)=r·(1−p)，m(Θ)=1−r；最后按 Dempster 规则组合 | `ds_belief`、`ds_plausibility`、`ds_conflict`（K）、`ds_betp` |
| 不确定性量化 | Mondrian 分割共形预测：在验证集 B 半区上分标签校准，非一致性分数取 1 − BetP 或 BetP，ε = 0.05 | `cp_set`（正常 / 攻击 / 不确定）、`cp_p_attack`、`cp_p_normal` |
| 概率区间 | 归纳式 Venn–Abers：两次保序回归，区间预先在校准分数分位网格上计算 | `va_low`、`va_high` |
| 可解释 | RF 树路径（Saabas）分解，与证据页原有的精确分数分解一致；ECOD 逐特征尾部贡献 | 证据页“可信研判引擎”面板 |

## 数据使用

- ECOD：训练集中的正常流量，随机抽 20,000 行。
- 校准器、可靠度和冲突阈值：验证集按标签分层随机对半切分（种子 42）后的 A 半区。
- 共形预测和 Venn–Abers：B 半区。两个半区互不重叠，test 不参与任何拟合。
- 参数文件为 `artifacts/engine.joblib`，描述文件 `artifacts/engine.json` 记录了数据 SHA、引擎源码 SHA 和底座模型 SHA。

## 界面行为

- 主队列（异常分数从高到低）按 `ds_belief` 排序，相同时再按 RF 分数排序。
- “需要再检查”收录两类记录：共形预测集给出“不确定”的，以及冲突度 K 不低于 A 半区 99% 分位的。队列内先排 BetP 最接近 0.5 的记录，再按 K 从高到低排；这条排序规则是在验证集上选定的。
- 原有的 RF/LR 分数、阈值、证据页和复核记录都保留。引擎参数缺失、哈希不符或运行出错时，自动退回 1.3.6 的排序和队列。

## 复现

```
python -m scripts.fit_engine --data <processed 数据目录>
python -m scripts.evaluate_engine --data <processed 数据目录> --cross-day <Thursday-15-02-2018 CSV> --output docs/results/engine_evaluation.json
```

评估结果见 `docs/results/engine_evaluation.json`。
