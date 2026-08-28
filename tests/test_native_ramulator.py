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


def test_native_binding_records_hashes_without_verifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / "bridge.so"
    bridge.write_bytes(b"bridge")
    ramulator = tmp_path / "libramulator.so"
    ramulator.write_bytes(b"ramulator")
    manifest = tmp_path / "build.json"
    manifest.write_text(json.dumps({
        "schema_version": "gala-ramulator2-bridge-build-v1",
        "bridge_library": str(bridge),
        "bridge_library_sha256": "0" * 64,
        "ramulator_library": str(ramulator),
        "ramulator_library_sha256": "1" * 64,
        "ramulator_version": "v2.1.0",
        "ramulator_commit": "a" * 40,
    }), encoding="utf-8")
    config = tmp_path / "ramulator.yaml"
    config.write_text(_configuration(), encoding="utf-8")
    captured: dict[str, object] = {}

    def capture_init(
        self: NativeRamulator2Binding, bridge_library: Path,
        configuration: Path, **metadata: object,
    ) -> None:
        captured.update({
            "bridge_library": bridge_library,
            "configuration": configuration,
            **metadata,
        })

    monkeypatch.setattr(NativeRamulator2Binding, "__init__", capture_init)
    NativeRamulator2Binding.from_build_manifest(manifest, config)
    assert captured["expected_bridge_sha256"] == "0" * 64
    assert captured["ramulator_library_sha256"] == "1" * 64
