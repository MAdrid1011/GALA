from __future__ import annotations

from pathlib import Path

import pytest

from gala_sim.adapters.native_reference import (
    NativeReferenceError,
    _quality_with_frozen_interpreter,
    run_native_reference,
)
from gala_sim.config import load_config
from gala_sim.identity import canonical_json, sha256_bytes
from gala_sim.metrics import QualityConfig
from gala_sim.tools.preflight import ComputeProcess
from gala_sim.tools.preflight import GpuSample


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


def test_native_reference_watchdog_terminates_inactive_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    freeze = _freeze(tmp_path, config.sha256)

    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    class Process:
        pid = 987654321

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode if self.returncode is not None else 0

    clock = Clock()
    monkeypatch.setattr("gala_sim.adapters.native_reference.subprocess.Popen", Process)
    idle = GpuSample(0.0, 0.0, 0, None, None, None, None)
    with pytest.raises(NativeReferenceError, match="watchdog_inactivity_timeout"):
        run_native_reference(
            config, freeze, {
                "schema_version": "gala-native-preflight-v1", "status": "passed",
                "freeze_manifest_sha256": freeze["run_manifest_sha256"],
            }, tmp_path / "run", sample_fn=lambda: idle,
            inactivity_timeout_seconds=2.0,
            monotonic_fn=clock.now, sleep_fn=clock.sleep,
        )
    status = (tmp_path / "run" / "status.json").read_text(encoding="utf-8")
    assert '"status": "failed_preflight"' in status
    assert '"reason": "watchdog_inactivity_timeout"' in status
    reference = (tmp_path / "run" / "gpu_reference.json").read_text(encoding="utf-8")
    assert '"status": "terminated"' in reference


def test_quality_uses_frozen_interpreter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = QualityConfig(
        data_min=0.0,
        data_max=1.0,
        ssim_window=3,
        ssim_sigma=1.5,
        ssim_boundary="reflect",
        lpips_slices=((0,), (0,), (0,)),
        lpips_network="alex",
        lpips_version="0.1",
        lpips_backbone_sha256="a" * 64,
        lpips_calibration_sha256="b" * 64,
    )
    captured: dict[str, object] = {}

    def fake_check_output(command: list[str], **kwargs: object) -> str:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return 'weight loader notice\n{"lpips": 0.03, "psnr": 31.5, "ssim": 0.92}\n'

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    result = _quality_with_frozen_interpreter(
        tmp_path / "reference.npy",
        tmp_path / "candidate.npy",
        config,
        "/frozen/python",
    )

    assert result == {"psnr": 31.5, "ssim": 0.92, "lpips": 0.03}
    assert captured["command"][0] == "/frozen/python"  # type: ignore[index]
    assert str(ROOT) in captured["environment"]["PYTHONPATH"]  # type: ignore[index]
