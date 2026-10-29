"""Execute the Windows launcher with an isolated interpreter and Streamlit stub.

These checks cover launch arguments and failure handling, not model compatibility
or application acceptance. The model files are deliberately inert placeholders.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="start.cmd requires Windows cmd.exe")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launcher_venv(tmp_path_factory):
    template = tmp_path_factory.mktemp("launcher-venv") / ".venv"
    venv.EnvBuilder(with_pip=False).create(template)
    return template


@pytest.fixture
def launcher_workspace(tmp_path, launcher_venv):
    workspace = tmp_path / "临时仓库 空格(启动测试)"
    workspace.mkdir()
    shutil.copy2(ROOT / "start.cmd", workspace / "start.cmd")
    shutil.copytree(launcher_venv, workspace / ".venv")
    (workspace / "app.py").write_text("# Launcher fixture; no application is run.\n")
    stub = workspace / "streamlit"
    stub.mkdir()
    (stub / "__init__.py").write_text("", encoding="utf-8")
    (stub / "__main__.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['LIUSHAO_LAUNCHER_CAPTURE']).write_text(\n"
        "    json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:],\n"
        "                'artifacts_dir': os.environ.get('LIUSHAO_ARTIFACTS_DIR')},\n"
        "               ensure_ascii=False), encoding='utf-8')\n"
        "raise SystemExit(int(os.environ.get('LIUSHAO_LAUNCHER_EXIT_CODE', '0')))\n",
        encoding="utf-8",
    )
    return workspace


def model_package(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.joblib").write_bytes(b"inert launcher fixture")
    (directory / "model.json").write_text("{}", encoding="utf-8")
    return directory


def run_launcher(workspace, *, artifacts=None, arguments=(), child_exit=0,
                 delayed_expansion=False):
    caller = workspace.parent / "其他工作目录 (调用方)"
    caller.mkdir(exist_ok=True)
    capture = workspace.parent / "launcher-invocation.json"
    environment = os.environ.copy()
    for name in ("LIUSHAO_ARTIFACTS_DIR", "PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    environment.update({
        "PYTHONUTF8": "1",
        "LIUSHAO_LAUNCHER_CAPTURE": str(capture),
        "LIUSHAO_LAUNCHER_EXIT_CODE": str(child_exit),
    })
    if artifacts is not None:
        environment["LIUSHAO_ARTIFACTS_DIR"] = str(artifacts)
    extra = " " + subprocess.list2cmdline(arguments) if arguments else ""
    expansion = "on" if delayed_expansion else "off"
    command = (
        f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /v:{expansion} /s /c '
        f'""{workspace / "start.cmd"}"{extra}"'
    )
    result = subprocess.run(
        command,
        cwd=caller,
        env=environment,
        input=b"\n" * 4,
        capture_output=True,
        timeout=30,
        check=False,
    )
    invocation = json.loads(capture.read_text(encoding="utf-8")) if capture.exists() else None
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    return result.returncode, invocation, output


def assert_package_rejected(result):
    code, invocation, output = result
    assert code == 1, output
    assert invocation is None, "An incomplete selected package must not launch Streamlit"
    lowered = output.lower()
    assert "handoff" in lowered or "readme" in lowered, output
    assert "training" not in lowered and "scripts.train" not in lowered, output


def test_default_package_changes_to_repository_and_forwards_arguments(launcher_workspace):
    model_package(launcher_workspace / "artifacts")
    extra = ("--server.port", "8527", "--browser.serverAddress", "测试 主机 (localhost)")

    code, invocation, output = run_launcher(launcher_workspace, arguments=extra)

    assert code == 0, output
    assert invocation is not None, output
    assert Path(invocation["cwd"]) == launcher_workspace
    assert Path(invocation["artifacts_dir"]) == launcher_workspace / "artifacts"
    assert invocation["argv"] == ["run", "app.py", "--server.headless", "false", *extra]


def test_external_package_preserves_special_path_with_delayed_expansion(launcher_workspace):
    external = model_package(launcher_workspace.parent / "可信交接 模型包(!冻结版!)")
    assert not (launcher_workspace / "artifacts").exists()

    code, invocation, output = run_launcher(
        launcher_workspace, artifacts=external, delayed_expansion=True
    )

    assert code == 0, output
    assert invocation is not None, output
    assert Path(invocation["cwd"]) == launcher_workspace
    assert invocation["artifacts_dir"] == str(external)


def test_explicit_missing_directory_does_not_fall_back_to_default(launcher_workspace):
    model_package(launcher_workspace / "artifacts")
    missing = launcher_workspace.parent / "不存在的模型目录"

    assert_package_rejected(run_launcher(launcher_workspace, artifacts=missing))


@pytest.mark.parametrize("selected", ["default", "external"])
@pytest.mark.parametrize("missing", ["model.joblib", "model.json"])
def test_missing_selected_package_file_stops_before_launch(launcher_workspace, selected, missing):
    default = model_package(launcher_workspace / "artifacts")
    directory = default
    if selected == "external":
        directory = model_package(launcher_workspace.parent / "外部交接 包(不完整)")
    (directory / missing).unlink()

    assert_package_rejected(run_launcher(
        launcher_workspace, artifacts=directory if selected == "external" else None
    ))


def test_missing_virtual_environment_stops_with_installation_help(launcher_workspace):
    model_package(launcher_workspace / "artifacts")
    (launcher_workspace / ".venv").rename(launcher_workspace / ".venv-saved")

    code, invocation, output = run_launcher(launcher_workspace)

    assert code == 1, output
    assert invocation is None
    assert "readme" in output.lower() or "install" in output.lower(), output


def test_child_failure_exit_code_survives_pause(launcher_workspace):
    model_package(launcher_workspace / "artifacts")

    code, invocation, output = run_launcher(launcher_workspace, child_exit=23)

    assert invocation is not None, output
    assert code == 23, output
