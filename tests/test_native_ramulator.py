from __future__ import annotations

import json
from pathlib import Path

import pytest

from gala_sim.timing.memory import MissingMemoryBackend, NativeRamulator2Binding
from gala_sim.timing.memory.native import _inspect_configuration


def _configuration(impl: str = "External") -> str:
    controller = """
      - impl: LPDDR5
        dram:
          impl: LPDDR5
          channel_width: 32
          timing: [6400]
"""
    return (
        f"frontend:\n  impl: {impl}\n  clock_ratio: 1\n"
        "memory_system:\n  impl: GenericDRAM\n  clock_ratio: 1\n  controllers:\n"
        + controller + controller
    )


def test_native_binding_inspects_lpddr5_configuration_identity(tmp_path: Path) -> None:
    path = tmp_path / "ramulator.yaml"
    path.write_text(_configuration(), encoding="utf-8")
    assert _inspect_configuration(path) == {
        "channels": 2,
        "channel_width_bits": 32,
        "data_rate_mtps": 6400,
    }


def test_native_binding_rejects_non_external_frontend(tmp_path: Path) -> None:
    path = tmp_path / "ramulator.yaml"
    path.write_text(_configuration("LoadStoreTrace"), encoding="utf-8")
    with pytest.raises(MissingMemoryBackend, match="External"):
        _inspect_configuration(path)


def test_native_binding_verifies_bridge_hash_before_loading(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge.so"
    bridge.write_bytes(b"not a shared library")
    manifest = tmp_path / "build.json"
    manifest.write_text(json.dumps({
        "schema_version": "gala-ramulator2-bridge-build-v1",
        "bridge_library": str(bridge),
        "bridge_library_sha256": "0" * 64,
        "ramulator_version": "v2.1.0",
        "ramulator_commit": "a" * 40,
    }), encoding="utf-8")
    config = tmp_path / "ramulator.yaml"
    config.write_text(_configuration(), encoding="utf-8")
    with pytest.raises(MissingMemoryBackend, match="SHA-256"):
        NativeRamulator2Binding.from_build_manifest(manifest, config)
