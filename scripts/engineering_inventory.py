"""Inventory this .venv and reuse existing notices without changing the original lock.

Run `python -m scripts.engineering_inventory` to create the new engineering inventory.
Run again with `--check` to verify its versions, metadata and notice hashes without writes.
This records package metadata and files; it does not grant redistribution permission.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/engineering"
INVENTORY = OUTPUT / "dependencies.json"
SUMMARY = OUTPUT / "DEPENDENCIES.md"
SUPPLEMENT = OUTPUT / "license-supplement"
EXISTING = ROOT / "third_party/licenses"
NOTICE_PREFIXES = (
    "license", "licence", "copying", "notice", "copyright",
    "thirdpartynotices", "thirdpartylicenses",
)
CODE_SUFFIXES = {".py", ".pyc", ".pyi", ".js", ".ts", ".c", ".h", ".cpp", ".exe", ".dll"}
UPSTREAM_NOTICES = ({
    "package": "streamlit", "version": "1.50.0", "tag": "1.50.0",
    "commit": "52736534b6488da3bcb0c9d8bba855855a963678",
    "source_url": "https://raw.githubusercontent.com/streamlit/streamlit/"
                  "52736534b6488da3bcb0c9d8bba855855a963678/LICENSE",
    "sha256": "c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c",
    "filename": "LICENSE",
},)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def pins(name: str) -> dict[str, str]:
    result = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            package, version = line.split("==", 1)
            key = normalize(package)
            if key in result:
                raise ValueError(f"Duplicate requirement: {key}")
            result[key] = version
    return result


def existing_notices() -> tuple[dict, dict]:
    exact, normalized = {}, {}
    for path in sorted(EXISTING.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            exact.setdefault(digest(content), []).append(path)
            normalized.setdefault(digest(content.replace(b"\r\n", b"\n")), []).append(path)
    return exact, normalized


def choose_existing(paths: list[Path], package: str) -> Path:
    # Prefer the package's own copy when several notices have identical contents.
    return min(paths, key=lambda path: (
        normalize(path.relative_to(EXISTING).parts[0]) != normalize(package), relative(path)
    ))


def notice_inventory(distribution, exact: dict, normalized: dict, check_only: bool) -> list:
    result = []
    for item in sorted(distribution.files or [], key=str):
        if not re.sub(r"[^a-z]", "", item.name.lower()).startswith(NOTICE_PREFIXES):
            continue
        if Path(item.name).suffix.lower() in CODE_SUFFIXES:
            continue
        source = Path(distribution.locate_file(item)).resolve()
        if not source.is_file():
            continue
        if not source.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("Notice file is outside the active virtual environment")
        content = source.read_bytes()
        installed_hash = digest(content)
        lf_hash = digest(content.replace(b"\r\n", b"\n"))
        wheel_hash = item.hash
        wheel_verified = None
        if wheel_hash and wheel_hash.mode == "sha256":
            actual_record_value = base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).decode("ascii").rstrip("=")
            wheel_verified = actual_record_value == wheel_hash.value
            if not wheel_verified:
                raise ValueError(f"Installed notice differs from RECORD: {item}")
        if installed_hash in exact:
            target = choose_existing(exact[installed_hash], distribution.metadata["Name"])
            match = "exact_bytes"
        elif lf_hash in normalized:
            target = choose_existing(normalized[lf_hash], distribution.metadata["Name"])
            match = "line_endings_only"
        else:
            item_path = PurePosixPath(str(item).replace("\\", "/"))
            if item_path.is_absolute() or ".." in item_path.parts:
                raise ValueError("Unsafe distribution notice path")
            target = SUPPLEMENT / normalize(distribution.metadata["Name"]) / item_path
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f"Existing supplement differs: {relative(target)}")
            elif check_only:
                raise ValueError(f"Missing supplement: {relative(target)}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            match = "new_supplement_original_bytes"
        repository_content = target.read_bytes()
        result.append({
            "installed_relative_path": str(item).replace("\\", "/"),
            "installed_bytes": len(content), "installed_sha256": installed_hash,
            "repository_path": relative(target),
            "repository_sha256": digest(repository_content),
            "normalized_lf_sha256": lf_hash,
            "match": match,
            "normalized_contents_equal": (
                content.replace(b"\r\n", b"\n") == repository_content.replace(b"\r\n", b"\n")
            ),
            "installed_record_sha256_verified": wheel_verified,
        })
    return result


def upstream_notice_inventory(installed: dict[str, str], check_only: bool) -> list[dict]:
    """Keep fixed upstream licenses separate from notices present in installed wheels."""
    result = []
    for spec in UPSTREAM_NOTICES:
        if installed.get(spec["package"]) != spec["version"]:
            raise ValueError(f"Upstream supplement version does not match: {spec['package']}")
        target = (SUPPLEMENT / spec["package"] / f"{spec['version']}-upstream"
                  / spec["filename"])
        if target.exists():
            content = target.read_bytes()
        elif check_only:
            raise ValueError(f"Missing upstream supplement: {relative(target)}")
        else:
            with urlopen(spec["source_url"], timeout=30) as response:
                content = response.read()
        if digest(content) != spec["sha256"]:
            raise ValueError(f"Upstream supplement hash mismatch: {relative(target)}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        result.append({
            **spec, "origin": "official_versioned_source_not_installed_wheel",
            "repository_path": relative(target), "bytes": len(content),
        })
    return result


def build_inventory(check_only: bool) -> dict:
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise ValueError("Use this repository's .venv interpreter")
    locked, direct = pins("requirements-lock.txt"), pins("requirements.txt")
    exact, normalized = existing_notices()
    protected = [ROOT / name for name in (
        "requirements.txt", "requirements-lock.txt", "docs/results/dependencies.json"
    )] + sorted(path for path in EXISTING.rglob("*") if path.is_file())
    before = {relative(path): digest(path.read_bytes()) for path in protected}
    distributions = sorted(metadata.distributions(), key=lambda dist: dist.metadata["Name"].lower())
    packages = []
    for distribution in distributions:
        meta = distribution.metadata
        name, version = meta["Name"], distribution.version
        key = normalize(name)
        scope = ("declared_direct" if key in direct else
                 "locked_indirect" if key in locked else "environment_tool" if key == "pip"
                 else "unexpected_unlocked")
        packages.append({
            "name": name, "normalized_name": key, "version": version,
            "scope": scope, "locked_version": locked.get(key),
            "lock_matches": version == locked[key] if key in locked else None,
            "homepage": meta.get("Home-page"),
            "project_urls": meta.get_all("Project-URL", []),
            "pypi_lookup_url": f"https://pypi.org/project/{quote(name)}/{quote(version)}/",
            "license_expression": meta.get("License-Expression"),
            "license_metadata": meta.get("License"),
            "license_classifiers": [value for value in meta.get_all("Classifier", [])
                                    if value.startswith("License ::")],
            "requires_dist_metadata": meta.get_all("Requires-Dist", []),
            "bundled_notices": notice_inventory(distribution, exact, normalized, check_only),
        })
    installed = {row["normalized_name"]: row["version"] for row in packages}
    upstream_notices = upstream_notice_inventory(installed, check_only)
    mismatches = [{"name": name, "expected": expected, "installed": installed.get(name)}
                  for name, expected in locked.items() if installed.get(name) != expected]
    unknown = sorted(set(installed) - set(locked) - {"pip"})
    direct_mismatch = sorted(name for name, version in direct.items() if locked.get(name) != version)
    notices = [notice for package in packages for notice in package["bundled_notices"]]
    after = {relative(path): digest(path.read_bytes()) for path in protected}
    validation = {
        "passed": (len(packages) == 51 and len(locked) == 50 and "pip" in installed
                   and len(installed) == len(packages) and not mismatches and not unknown
                   and not direct_mismatch and before == after
                   and all(notice["normalized_contents_equal"] for notice in notices)),
        "locked_version_mismatches": mismatches, "unexpected_unlocked_packages": unknown,
        "direct_lock_mismatches": direct_mismatch,
        "original_locks_and_notices_unchanged": before == after,
        "duplicate_distribution_names": len(installed) != len(packages),
    }
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"interpreter": ".venv", "python": platform.python_version(),
                        "operating_system": platform.system(), "machine": platform.machine()},
        "sources": {"direct_requirements": "requirements.txt",
                    "lock": "requirements-lock.txt",
                    "lock_sha256": digest((ROOT / "requirements-lock.txt").read_bytes()),
                    "metadata": "importlib.metadata from the active .venv",
                    "existing_notices": "third_party/licenses"},
        "counts": {
            "installed_distributions": len(packages), "locked_packages": len(locked),
            "declared_direct_packages": len(direct),
            "locked_indirect_packages": len(locked.keys() - direct.keys()),
            "environment_tools": sum(row["scope"] == "environment_tool" for row in packages),
            "bundled_notice_entries": len(notices),
            "upstream_notice_entries": len(upstream_notices),
            "total_notice_entries": len(notices) + len(upstream_notices),
            "existing_exact_matches": sum(row["match"] == "exact_bytes" for row in notices),
            "existing_line_ending_matches": sum(
                row["match"] == "line_endings_only" for row in notices
            ),
            "supplement_entries": sum(
                row["match"] == "new_supplement_original_bytes" for row in notices
            ),
            "notice_record_hashes_verified": sum(
                row["installed_record_sha256_verified"] is True for row in notices
            ),
        },
        "validation": validation,
        "limits": [
            "License fields and upstream URLs are distribution metadata, not a legal conclusion.",
            "Requires-Dist includes optional extras; it is not an assertion that every extra is used.",
            "Only legal notice bytes are checked against installed RECORD; package source is not audited.",
            "The inventory does not redistribute installed package implementations or runtime binaries.",
            "Browser, FFmpeg, dataset and model permissions are separate from Python distributions.",
            "The project's own license remains awaiting the owner's confirmation.",
            "No third-party source modifications are performed by this inventory script.",
        ],
        "packages": packages,
        "upstream_notices": upstream_notices,
    }


def license_label(package: dict) -> str:
    if package["license_expression"]:
        return package["license_expression"]
    classifiers = package["license_classifiers"]
    if classifiers:
        return "; ".join(value.rsplit(" :: ", 1)[-1] for value in classifiers)
    raw = package["license_metadata"] or ""
    return raw if raw and len(raw) < 90 and "\n" not in raw else "原始元数据/声明见 JSON"


def markdown(report: dict) -> str:
    counts = report["counts"]
    pip = next(row["version"] for row in report["packages"] if row["normalized_name"] == "pip")
    lines = [
        "# 本轮 Python 环境与依赖声明", "",
        f"采集时间（UTC）：{report['generated_at_utc']}。环境："
        f"{report['environment']['operating_system']} / Python {report['environment']['python']}。",
        "", f"实际安装 {counts['installed_distributions']} 个分发包："
        f"{counts['locked_packages']} 个严格匹配原锁定版本，另有环境安装工具 pip {pip}。"
        f"其中 {counts['declared_direct_packages']} 个列在 requirements.txt，"
        f"{counts['locked_indirect_packages']} 个是锁文件中的间接依赖。"
        "pip 不在项目锁文件中，本次仅记录其实际工具版本。", "",
        f"发现 {counts['bundled_notice_entries']} 份随包 LICENSE/COPYING/NOTICE 等声明。"
        f"其中 {counts['existing_exact_matches']} 份与原仓库文件逐字节一致，"
        f"{counts['existing_line_ending_matches']} 份仅有 LF/CRLF 换行差异。"
        f"新增补充声明 {counts['supplement_entries']} 份；已有内容不重复复制。"
        f"{counts['notice_record_hashes_verified']} 份随包声明的 SHA-256 与安装 RECORD 一致。",
        "", f"另收录 {counts['upstream_notice_entries']} 份官方固定版本来源声明，"
        f"合计 {counts['total_notice_entries']} 份。Streamlit 1.50.0 安装包中没有许可原文，"
        "其 LICENSE 从对应官方提交补充，单独列在 upstream_notices，"
        "不计作 wheel 自带文件或 RECORD 验证项。",
        "", "[机器清单](dependencies.json) 记录精确版本、上游元数据链接、"
        "原始许可证字段、依赖声明和每份文件的安装字节哈希、仓库字节哈希、"
        "换行规范化哈希及匹配方式。所有交付路径均为仓库或分发包相对路径。",
        "", "## 使用和修改范围", "",
        "应用使用 pandas/numpy 处理表格，scikit-learn 训练与推理，joblib 保存模型，"
        "Streamlit 提供界面；pytest、Playwright 与 psutil 用于测试或运行测量。"
        "其余锁定包按安装依赖随环境提供；清单不把可选 extras 自动计作已使用功能。",
        "", "本脚本读取已安装元数据与声明，并补充哈希固定的官方许可原文；"
        "未复制依赖实现代码，也未修改"
        "原 requirements.txt、requirements-lock.txt、third_party/licenses 或 docs/results。"
        "本轮核验仅覆盖许可声明字节，不是对全部第三方源码的审计。",
        "", "## 分发包清单", "",
        "下表的许可证名称来自安装元数据，不构成适用条款或兼容性判定。"
        "上游完整链接及原文保存在机器清单；PyPI 链接用于定位对应版本。", "",
        "| 分发包 | 精确版本 | 角色 | 许可证元数据摘要 | 声明数 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    roles = {"declared_direct": "直接声明", "locked_indirect": "锁定间接依赖",
             "environment_tool": "安装工具（未锁定）", "unexpected_unlocked": "异常未锁定"}
    for package in report["packages"]:
        label = license_label(package).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| [{package['name']}]({package['pypi_lookup_url']}) | {package['version']} | "
            f"{roles[package['scope']]} | {label} | {len(package['bundled_notices'])} |"
        )
    lines.extend([
        "", "## 本轮新增声明", "",
        "历史 third_party/licenses 的 70 份原文及历史快照均未修改。"
        "此前文件名筛选遗漏 Playwright 的 ThirdPartyNotices.txt；本轮补收其安装原文，"
        "记录其中列出的驱动内嵌组件及条款，且与安装 RECORD 的 SHA-256 一致。",
        "", "| 来源 | 补充文件 | 验证 |", "| --- | --- | --- |",
    ])
    for package in report["packages"]:
        for notice in package["bundled_notices"]:
            if notice["match"] == "new_supplement_original_bytes":
                path = notice["repository_path"].removeprefix("docs/engineering/")
                lines.append(f"| {package['name']} {package['version']} 安装包 | "
                             f"[{Path(path).name}]({path}) | 安装 RECORD SHA-256 |")
    for notice in report["upstream_notices"]:
        path = notice["repository_path"].removeprefix("docs/engineering/")
        lines.append(f"| [{notice['package']} {notice['tag']} 官方提交]"
                     f"({notice['source_url']}) | [{notice['filename']}]({path}) | "
                     f"固定 SHA-256：`{notice['sha256']}` |")
    lines.extend([
        "", "## 内嵌组件和分发范围", "",
        "分发包的主许可证名称不能覆盖内嵌组件。NumPy/SciPy 声明包含 OpenBLAS、"
        "LAPACK 和 GCC runtime 及其例外；Pillow、PyArrow 与 Playwright 的声明"
        "还包含图像库、压缩库或驱动组件的单独条款。保留完整原文，不能仅摘录主许可证。",
        "", "当前 Git 交付范围为本项目源码、依赖锁与许可原文，不打包 .venv、wheel、"
        "Python/SQLite 运行时、浏览器、FFmpeg、数据或模型。若以后分发这些内容，"
        "须按实际发布物另行核对；本清单不宣称二进制交付的全部内嵌组件已经验收。",
    ])
    lines.extend([
        "", "## 复核方法与边界", "",
        "在仓库根目录使用同一虚拟环境：", "", "```powershell",
        r".\.venv\Scripts\python.exe -m scripts.engineering_inventory",
        r".\.venv\Scripts\python.exe -m scripts.engineering_inventory --check",
        "```", "",
        "第一条更新本目录清单，缺少的随包声明按原字节保存到 license-supplement/，"
        "缺少的官方来源声明从固定提交下载并先校验固定哈希；"
        "第二条不联网、不写入，重算并比对计数、锁定版本、元数据和声明哈希。"
        "锁版本漂移、额外未锁定包、现存补充声明被改动或快照不一致均返回非零状态。",
        "", "本项目自研代码的正式许可证仍待负责人确认。本清单不授予数据、模型、"
        "浏览器或 FFmpeg 的再分发权限，也不对项目和依赖许可证兼容性作法律结论。"
        "Edge 与录像所需 FFmpeg 是 Python 分发包之外的独立运行工具，不计入上述 51 包。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing outputs without writing")
    args = parser.parse_args()
    report = build_inventory(check_only=args.check)
    if args.check:
        saved = json.loads(INVENTORY.read_text(encoding="utf-8"))
        report["generated_at_utc"] = saved["generated_at_utc"]
        matches = report == saved and SUMMARY.read_text(encoding="utf-8") == markdown(saved)
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        INVENTORY.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        SUMMARY.write_text(markdown(report), encoding="utf-8")
        matches = True
    passed = report["validation"]["passed"] and matches
    print(json.dumps({"passed": passed, "mode": "check" if args.check else "generate",
                      "snapshot_matches": matches, "counts": report["counts"],
                      "validation": report["validation"]}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
