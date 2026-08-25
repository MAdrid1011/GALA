from __future__ import annotations

from pathlib import Path

import pytest

from gala_sim.adapters.native_reference import NativeReferenceError, run_native_reference
from gala_sim.config import load_config
from gala_sim.identity import canonical_json, sha256_bytes
from gala_sim.tools.preflight import ComputeProcess


ROOT = Path(__file__).resolve().parents[1]


def _freeze(tmp_path: Path, config_sha256: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "gala-input-freeze-v3",
        "config": {"sha256": config_sha256},
        "repository": {"commit": "a" * 40},
        "model": {"commit": "b" * 40},
        "dataset": {"manifest_sha256": "c" * 64},
        "environment": {},
        "training": {"command": {"argv": ["/usr/bin/python3", "train.py", "-s", str(tmp_path), "-m", str(tmp_path / "model")], "working_directory": str(tmp_path)}},
    }
    value["run_manifest_sha256"] = sha256_bytes(canonical_json(value))
    return value


def test_native_reference_requires_passed_matching_preflight(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    freeze = _freeze(tmp_path, config.sha256)
    with pytest.raises(NativeReferenceError, match="passed native-preflight"):
        run_native_reference(config, freeze, {
            "schema_version": "gala-native-preflight-v1",
            "status": "failed_preflight",
            "freeze_manifest_sha256": freeze["run_manifest_sha256"],
        }, tmp_path / "run", sample_fn=lambda: None)


def test_native_reference_rejects_external_gpu_before_launch(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    freeze = _freeze(tmp_path, config.sha256)
    sample = type("Sample", (), {
        "compute_processes": (ComputeProcess("GPU-fixture", 42, "external", 128),),
    })()
    with pytest.raises(NativeReferenceError, match="gpu_busy_external"):
        run_native_reference(config, freeze, {
            "schema_version": "gala-native-preflight-v1", "status": "passed",
            "freeze_manifest_sha256": freeze["run_manifest_sha256"],
        }, tmp_path / "run", sample_fn=lambda: sample)
    status = (tmp_path / "run" / "status.json").read_text(encoding="utf-8")
    assert '"status": "failed_preflight"' in status
    assert '"reason": "gpu_busy_external"' in status
    assert f'"config_sha256": "{config.sha256}"' in status
    assert f'"freeze_manifest_sha256": "{freeze["run_manifest_sha256"]}"' in status
    assert f'"repository_commit": "{"a" * 40}"' in status
    assert '"reproduction"' in status
    assert not (tmp_path / "model").exists()


def test_native_reference_records_sampling_failure(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    freeze = _freeze(tmp_path, config.sha256)
    with pytest.raises(NativeReferenceError, match="gpu_sampling_unavailable"):
        run_native_reference(config, freeze, {
            "schema_version": "gala-native-preflight-v1", "status": "passed",
            "freeze_manifest_sha256": freeze["run_manifest_sha256"],
        }, tmp_path / "run", sample_fn=lambda: (_ for _ in ()).throw(
            RuntimeError("fixture sampler failure")
        ))
    status = (tmp_path / "run" / "status.json").read_text(encoding="utf-8")
    assert '"status": "failed_preflight"' in status
