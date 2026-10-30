# 1.3.7 独立源码包

源码包保留应用、算法源码、测试、脚本、配置、版本和说明；不包含 `runtime/`、`wheels/`、`artifacts/model.joblib` 或 `artifacts/engine.joblib`。它用于审查实现及准备开发环境，不能独立完成离线安装和冻结模型推理。请使用同版完整运行 ZIP 启动。

源码包的 `PACKAGE_MANIFEST.json` 标记 `package_kind=source`，只列实际包含的文件；自己的 `SHA256SUMS.txt` 同时覆盖该清单。`SOURCE_MANIFEST.json` 保留与运行包一致的源码快照。兼容字段 `source_commit` / `software_sha` 是内容快照标识，不声称是 Git 提交；另有明确的身份类型及 SHA256 内容快照。

已有 Python 3.12 时可只用标准库校验：

```text
python -m scripts.package_contest_release --verify-source .
```

开发运行另需配套受授权冻结模型、引擎及依赖。不要从不可信来源加载 pickle/joblib；缺少有意排除的二进制不代表源码损坏。

已封版且通过完整性校验的运行目录可确定性重打包：

```text
python -m scripts.package_contest_release --output <全新的输出目录>
```

命令检查全量运行文件，生成运行 ZIP 和具有独立清单的源码 ZIP，核查 CRC 并记录外层 SHA256；不下载、不训练、不补缺失资源、不覆盖已有 ZIP。它是从完整运行目录重新打包，不声称仅靠源码 ZIP 能从零生成模型和全部资源。

发行维护者完成代码、评测、说明后才运行 `scripts.seal_release` 更新清单；使用者不要以重封方式绕过损坏检查。清单证明内容一致性，不是作者身份的数字签名。旧完整材料构建说明适用于其原版交付布局，本版以本页为准。
