from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gala_sim.config import ConfigError, load_config, load_source_manifest
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
    with pytest.raises(TypeError):
        config.parameters["clock"] = {}  # type: ignore[index]


def test_config_rejects_invalid_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "schema_version: gala-config-v1\nparameters:\n  x:\n"
        "    value: 3\n    unit: count\n    source: test\n    scope: test\n"
        "    status: frozen\n    allowed_range: [4, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="allowed_range"):
        load_config(path)


def test_source_manifests_are_strict() -> None:
    model = load_source_manifest(ROOT / "configs/models/r2_gaussian.yaml",
                                 schema_version="gala-model-source-v1")
    dataset = load_source_manifest(ROOT / "configs/datasets/chest.yaml",
                                   schema_version="gala-dataset-source-v1")
    assert model.require("commit") == "f2579bfddd9aac009cb797c8503bef8119bbd022"
    assert "meta_data.json" in dataset.require("required_files")


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
    np.save(tmp_path / "vol_gt.npy", np.asarray([[[0.0, 1.0]]], dtype=np.float32))
    dataset = dataset_record(tmp_path, "Chest", "https://data.example/chest", "https://license.example")
    assert dataset.status == "planned"
    assert dataset.metadata_sha256 is not None
    assert dataset.geometry == {"DSO": 1}
    assert [item["path"] for item in dataset.files or []] == ["meta_data.json", "vol_gt.npy"]
    assert dataset.reference_volume == {
        "path": "vol_gt.npy",
        "sha256": dataset.files[1]["sha256"],  # type: ignore[index]
        "shape": [1, 1, 2],
        "dtype": "<f4",
        "data_min": 0.0,
        "data_max": 1.0,
    }


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
    assert record["schema_version"] == "gala-input-freeze-v2"
    assert record["quality"]["parameters"]["quality.lpips_network"]["value"] == "alex"
    verify_freeze_record(record)
    mismatched = replace(dataset, reference_volume={
        "path": "vol_gt.npy", "sha256": "d" * 64, "shape": [256, 256, 256],
        "dtype": "<f4", "data_min": 0.0, "data_max": 2.0,
    })
    with pytest.raises(ValueError, match="data range"):
        build_freeze_record(config, SourceRecord(**source), mismatched, 0, ROOT)
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
