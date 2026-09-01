from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from gala_sim.assets import (
    AssetCatalog, AssetFile, AssetSelection, DatasetAsset, _acquire_dataset,
    _verify_file, acquire_assets, download_file, safe_extract_zip,
)
from gala_sim.workspace import WorkspacePaths


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_loads_all_public_manifests() -> None:
    catalog = AssetCatalog.load(ROOT / "configs")
    assert set(catalog.models) == {
        "r2_gaussian", "fact_gs", "exact_gs", "gr_gaussian",
    }
    assert set(catalog.datasets) == {"chest", "walnut", "hdtomo_usb"}
    assert catalog.models["gr_gaussian"].implementation == "independent_reimplementation"
    assert catalog.datasets["hdtomo_usb"].files[0].name == "usb.zip"


def test_acquisition_dry_run_does_not_create_workspace(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "configs").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    workspace = WorkspacePaths.discover(repository=repository)
    report = acquire_assets(
        AssetCatalog.load(ROOT / "configs"), AssetSelection(all=True), workspace,
        dry_run=True,
    )
    assert len(report.items) == 7
    assert all(item.action in {"clone", "download", "local"} for item in report.items)
    assert not workspace.root.exists()


def test_safe_zip_extraction_rejects_parent_escape(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "output")


def test_safe_zip_extraction_preserves_valid_content(tmp_path: Path) -> None:
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("dataset/metadata.txt", "geometry")
    output = tmp_path / "output"
    safe_extract_zip(archive, output)
    content = output / "dataset" / "metadata.txt"
    assert content.read_text(encoding="utf-8") == "geometry"
    assert hashlib.sha256(content.read_bytes()).hexdigest()


def test_safe_zip_extraction_rejects_symbolic_links(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, "target")
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "output")


class _Response:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.content


def test_download_resumes_with_an_http_range(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.with_suffix(".bin.part").write_bytes(b"abc")
    observed = {}

    def get(url, *, headers, stream, timeout):
        observed.update(url=url, headers=headers, stream=stream, timeout=timeout)
        return _Response(b"def", 206)

    monkeypatch.setattr("requests.get", get)
    assert download_file("https://data.example/asset", destination, expected_size=6) == destination
    assert destination.read_bytes() == b"abcdef"
    assert observed["headers"] == {"Range": "bytes=3-"}


def test_download_reuses_a_complete_local_file(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"ready")
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail("network used"))
    assert download_file("https://data.example/asset", destination, expected_size=5) == destination


def test_download_checks_free_space_before_network(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "gala_sim.assets.shutil.disk_usage",
        lambda path: shutil._ntuple_diskusage(total=10, used=10, free=0),
    )
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: pytest.fail("network used"))
    with pytest.raises(OSError, match="insufficient free space"):
        download_file("https://data.example/asset", tmp_path / "asset.bin", expected_size=5)


def test_checksum_failure_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_file(path, AssetFile("asset.bin", 5, sha256="0" * 64))


def test_completed_dataset_acquisition_is_verified_without_download(
    monkeypatch, tmp_path: Path,
) -> None:
    archive = tmp_path / "cache/downloads/fixture/data.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/value.txt", "fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "datasets/fixture"
    destination.mkdir(parents=True)
    (destination / "asset-manifest.json").write_text(json.dumps({
        "id": "fixture", "identity_sha256": digest,
        "source_files": [{"name": "data.zip", "sha256": digest}],
    }), encoding="utf-8")
    asset = DatasetAsset(
        "fixture", "Fixture", "http", "https://data.example/asset", "CC0",
        "https://license.example", (AssetFile("data.zip", archive.stat().st_size),),
        None, None, tmp_path / "fixture.yaml",
    )
    monkeypatch.setattr("gala_sim.assets.download_file", lambda *args, **kwargs: pytest.fail("download used"))
    identity, action = _acquire_dataset(asset, destination, tmp_path / "cache", resume=True)
    assert identity == digest
    assert action == "verify"
