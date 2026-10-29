"""Regression checks for omitted notices and separately sourced upstream licenses."""
import base64
import io
from importlib.metadata import PackagePath
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import engineering_inventory as inventory


@pytest.fixture(autouse=True)
def forbid_unexpected_network(monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("License checks must not access the network")

    monkeypatch.setattr(inventory, "urlopen", unexpected_request)


def distribution(tmp_path, content, name="playwright"):
    relative = PackagePath("playwright/driver/package/ThirdPartyNotices.txt")
    relative.hash = SimpleNamespace(
        mode="sha256",
        value=base64.urlsafe_b64encode(bytes.fromhex(inventory.digest(content))).decode().rstrip("="),
    )
    path = tmp_path / "venv" / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return SimpleNamespace(
        files=[relative], metadata={"Name": name},
        locate_file=lambda item: tmp_path / "venv" / item,
    )


def test_third_party_notices_are_copied_and_checked_without_changing_bytes(tmp_path, monkeypatch):
    content = b"Bundled component\r\nCopyright holder\r\n"
    package = distribution(tmp_path, content)
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(inventory.sys, "prefix", str(tmp_path / "venv"))

    notices = inventory.notice_inventory(package, {}, {}, check_only=False)

    assert len(notices) == 1
    assert notices[0]["installed_record_sha256_verified"] is True
    assert (tmp_path / notices[0]["repository_path"]).read_bytes() == content
    assert inventory.notice_inventory(package, {}, {}, check_only=True) == notices


def test_tampered_installed_notice_is_rejected(tmp_path, monkeypatch):
    package = distribution(tmp_path, b"Original notice\n")
    Path(package.locate_file(package.files[0])).write_bytes(b"Changed notice\n")
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(inventory.sys, "prefix", str(tmp_path / "venv"))

    with pytest.raises(ValueError, match="differs from RECORD"):
        inventory.notice_inventory(package, {}, {}, check_only=False)
    assert not (tmp_path / "supplement").exists()


@pytest.mark.parametrize("tampered", [False, True])
def test_check_rejects_missing_or_tampered_supplement_without_repair(
    tmp_path, monkeypatch, tampered
):
    package = distribution(tmp_path, b"Original notice\n")
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(inventory.sys, "prefix", str(tmp_path / "venv"))
    target = tmp_path / "supplement" / "playwright" / package.files[0]
    if tampered:
        target.parent.mkdir(parents=True)
        target.write_bytes(b"Altered copy\n")

    message = "Existing supplement differs" if tampered else "Missing supplement"
    with pytest.raises(ValueError, match=message):
        inventory.notice_inventory(package, {}, {}, check_only=True)
    if tampered:
        assert target.read_bytes() == b"Altered copy\n"
    else:
        assert not target.exists()


def test_upstream_check_rejects_missing_license_without_fetching(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "SUPPLEMENT", tmp_path / "supplement")
    with pytest.raises(ValueError, match="Missing upstream supplement"):
        inventory.upstream_notice_inventory({"streamlit": "1.50.0"}, check_only=True)
    assert not (tmp_path / "supplement").exists()


def test_upstream_license_is_version_and_hash_bound(tmp_path, monkeypatch):
    content = b"Official upstream license\n"
    spec = {
        "package": "streamlit", "version": "1.50.0", "tag": "1.50.0",
        "commit": "a" * 40, "source_url": "https://example.invalid/fixed/LICENSE",
        "sha256": inventory.digest(content), "filename": "LICENSE",
    }
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(inventory, "UPSTREAM_NOTICES", (spec,))
    path = tmp_path / "supplement" / "streamlit" / "1.50.0-upstream" / "LICENSE"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    entries = inventory.upstream_notice_inventory({"streamlit": "1.50.0"}, check_only=True)
    assert entries[0]["origin"] == "official_versioned_source_not_installed_wheel"
    assert entries[0]["source_url"] == spec["source_url"]
    assert entries[0]["sha256"] == inventory.digest(content)
    with pytest.raises(ValueError, match="version does not match"):
        inventory.upstream_notice_inventory({"streamlit": "1.51.0"}, check_only=True)
    path.write_bytes(b"Replaced license\n")
    with pytest.raises(ValueError, match="Upstream supplement hash mismatch"):
        inventory.upstream_notice_inventory({"streamlit": "1.50.0"}, check_only=True)


def test_upstream_download_checks_hash_before_writing(tmp_path, monkeypatch):
    content = b"Official upstream license\n"
    spec = {
        "package": "streamlit", "version": "1.50.0", "tag": "1.50.0",
        "commit": "a" * 40, "source_url": "https://example.invalid/fixed/LICENSE",
        "sha256": inventory.digest(content), "filename": "LICENSE",
    }
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b"Unexpected upstream response\n")

    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(inventory, "UPSTREAM_NOTICES", (spec,))
    monkeypatch.setattr(inventory, "urlopen", download)
    with pytest.raises(ValueError, match="Upstream supplement hash mismatch"):
        inventory.upstream_notice_inventory({"streamlit": "1.50.0"}, check_only=False)
    assert calls == [(spec["source_url"], 30)]
    assert not (tmp_path / "supplement").exists()
