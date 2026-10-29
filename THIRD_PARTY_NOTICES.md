# 第三方组件、数据与许可范围

自研源码、测试、脚本、必要配置与配套使用说明沿用 [Apache-2.0](LICENSE)，保留 [NOTICE](NOTICE) 和既有作者署名。具体范围见[源码许可与分发范围](docs/release/SOURCE_LICENSE.md)。历史 MIT 文本是未采用提案，不构成双重许可。

冻结模型 c7a028b4e4cafedd、描述和正常参考不纳入源码许可，仅限赛事评审与非商业学术研究定向使用。demo.csv 与 demo_origins.parquet 保留 CSE-CIC-IDS2018 官方引用与再分发条款。代码、模型、数据及第三方组件分别处理；本地交付不等于公开发布批准。

## 数据集

实际训练来源为 CSE-CIC-IDS2018 的 2018-02-16、2018-03-02；配套演示是原留出测试的固定抽样，不能作为新的独立评测。使用或再分发保留数据集引用与 AWS 链接。

- [CIC 官方条款](https://www.unb.ca/cic/datasets/ids-2018.html)
- [AWS 来源](https://registry.opendata.aws/cse-cic-ids2018/)
- 引用：Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, ICISSP 2018, pp.108–116。

## 第三方组件

CPython 与 50 个锁定依赖包保留各自许可证与适用声明。现有 72 份依赖声明及运行时声明随包保留，见 [third_party](third_party/)、[逐份清单](docs/engineering/dependencies.json)和 docs/engineering/license-supplement；运行包另有 runtime 与 wheels。这些声明不替代项目、模型或数据的授权范围。完整依赖版本由 requirements-lock.txt 锁定。

真实未脱敏流量、真人原始记录、日志、聊天、复核数据库、输入缓存及敏感导出不纳入发行物。文件保留与复核行为见[本地数据说明](docs/usage/LOCAL_DATA.md)。

Codex 辅助开发与自动化验证不等于真人参与者研究，也不代替成果归属或实际使用效果证据。
