from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from gala_sim.config import ConfigError, load_config
from gala_sim.identity import sha256_tree
from gala_sim.manifest import build_freeze_record, dataset_record, verify_freeze_record


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_config_is_hashable_but_pending() -> None:
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    assert len(config.sha256) == 64
    assert config.parameter("clock.frequency")["value"] == 500000000
    assert not config.ready
    with pytest.raises(ConfigError, match="seed_fifo_entries"):
        config.require_ready()


def test_hash_tree_is_independent_of_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"a")
    (first / "b").write_bytes(b"b")
    (second / "b").write_bytes(b"b")
    (second / "a").write_bytes(b"a")
    assert sha256_tree(first) == sha256_tree(second)


def test_unavailable_dataset_is_machine_readable() -> None:
    dataset = dataset_record(None, "Chest", "https://data.example/chest", "https://license.example",
                             "source_download_not_present")
    assert dataset.status == "unavailable_data"
    assert dataset.reason == "source_download_not_present"


def test_dataset_record_contains_file_inventory(tmp_path: Path) -> None:
    (tmp_path / "meta_data.json").write_text(json.dumps({"scanner": {"DSO": 1}},), encoding="utf-8")
    (tmp_path / "vol_gt.npy").write_bytes(b"volume")
    dataset = dataset_record(tmp_path, "Chest", "https://data.example/chest", "https://license.example")
    assert dataset.status == "planned"
    assert dataset.metadata_sha256 is not None
    assert dataset.geometry == {"DSO": 1}
    assert [item["path"] for item in dataset.files or []] == ["meta_data.json", "vol_gt.npy"]


def test_freeze_record_self_hash_is_verified() -> None:
    dataset = dataset_record(None, "Chest", "https://data.example/chest", "https://license.example",
                             "fixture_not_downloaded")
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    source = {
        "name": "fixture", "url": "https://source.example", "commit": "c" * 40,
        "root": "/tmp/source", "tree_sha256": "a" * 64,
        "upstream_patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "license_path": "/tmp/source/LICENSE.md", "license_sha256": "b" * 64,
    }
    from gala_sim.manifest import SourceRecord
    record = build_freeze_record(config, SourceRecord(**source), dataset, 0, ROOT)
    verify_freeze_record(record)
    record["status"] = "running"
    with pytest.raises(ValueError, match="self-hash"):
        verify_freeze_record(record)


def test_freeze_command_records_unavailable_data(tmp_path: Path) -> None:
    source = Path("/home/madrid/Desktop/GALA-runtime/upstream/r2_gaussian")
    if not (source / ".git").exists():
        pytest.skip("external locked source checkout is not available")
    output = tmp_path / "freeze.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "tools/freeze_inputs.py"),
        "--source-root", str(source), "--dataset-reason", "fixture_not_downloaded",
        "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "unavailable_data"
    assert record["model"]["commit"] == "f2579bfddd9aac009cb797c8503bef8119bbd022"
    assert len(record["run_manifest_sha256"]) == 64
