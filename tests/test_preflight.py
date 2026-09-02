from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from gala_sim.config import GalaConfig, load_config
from gala_sim.timing.memory import Ramulator2Backend
from gala_sim.timing.resources import ResourceUsage
from gala_sim.tools.cycle_preflight import run_cycle_preflight, write_cycle_preflight
from gala_sim.identity import canonical_json, sha256_bytes
from gala_sim.tools.preflight import (
    ComputeProcess,
    GpuSample,
    decide_long_run,
    predict_runtime,
    run_native_preflight,
    sample_gpustat,
)


def _sample(utilization: float) -> GpuSample:
    return GpuSample(1.0, utilization, 1024, None, None, None, None)


def test_preflight_prediction_requires_gpu_floor_for_long_runs() -> None:
    predicted = predict_runtime(
        measured_seconds=20.0, measured_iterations=10,
        total_iterations=200, warmup_iterations=10,
    )
    blocked = decide_long_run(
        predicted_seconds=predicted, threshold_seconds=300.0,
        samples=(_sample(20.0), _sample(40.0)), utilization_floor_percent=60.0,
    )
    assert not blocked.allowed
    assert blocked.reason == "long_run_gpu_floor_failed"
    passed = decide_long_run(
        predicted_seconds=predicted, threshold_seconds=300.0,
        samples=(_sample(80.0),), utilization_floor_percent=60.0,
    )
    assert passed.allowed


def test_preflight_rejects_invalid_prediction() -> None:
    with pytest.raises(ValueError):
        predict_runtime(measured_seconds=0, measured_iterations=1,
                        total_iterations=2, warmup_iterations=0)


def test_gpustat_sample_distinguishes_compute_processes(monkeypatch) -> None:
    gpustat = json.dumps({"gpus": [{
        "index": 0, "uuid": "GPU-fixture", "utilization.gpu": 97,
        "memory.used": 256, "memory.total": 4096,
    }]})
    compute = "GPU-fixture, 42, /opt/external-work, 128\n"

    def check_output(command, **_kwargs):
        return gpustat if command[0] == "gpustat" else compute

    monkeypatch.setattr("gala_sim.tools.preflight.subprocess.check_output", check_output)
    sample = sample_gpustat()
    assert sample.utilization_percent == 97.0
    assert sample.compute_processes == (
        ComputeProcess("GPU-fixture", 42, "/opt/external-work", 128 * 1024 * 1024),
    )
    assert sample.memory_total_bytes == 4096 * 1024 * 1024


def _freeze(config: GalaConfig, root: Path, script: Path) -> dict[str, object]:
    command = {
        "working_directory": str(root),
        "argv": [str(Path(sys.executable).resolve()), script.name, "-s", str(root / "data"),
                 "-m", str(root / "formal")],
    }
    value: dict[str, object] = {
        "schema_version": "gala-input-freeze-v3",
        "config": {"sha256": config.sha256},
        "repository": {"commit": "a" * 40},
        "training": {
            "command": command,
            "effective_arguments": {"iterations": 30000},
        },
    }
    value["run_manifest_sha256"] = sha256_bytes(canonical_json(value))
    return value


def test_native_preflight_rejects_external_compute_before_launch(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    script = tmp_path / "train.py"
    script.write_text("raise AssertionError('must not launch')\n", encoding="utf-8")
    busy = GpuSample(
        1.0, 99.0, 256, None, None, None, None, 0, "GPU-fixture",
        (ComputeProcess("GPU-fixture", 42, "external", 128),),
    )
    output = tmp_path / "preflight"
    report = run_native_preflight(
        config, _freeze(config, tmp_path, script), output,
        reproduction="fixture native-preflight", sample_fn=lambda: busy,
    )
    assert report.status == "failed_preflight"
    assert report.reason == "gpu_busy_external"
    assert not (output / "calibration_model").exists()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["reason"] == "gpu_busy_external"


def test_native_preflight_predicts_from_isolated_short_run(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    script = tmp_path / "train.py"
    script.write_text("pass\n", encoding="utf-8")
    free = GpuSample(1.0, 80.0, 1024, None, None, None, None, 0, "GPU-fixture", ())
    output = tmp_path / "preflight"
    report = run_native_preflight(
        config, _freeze(config, tmp_path, script), output,
        reproduction="fixture native-preflight", sample_fn=lambda: free,
        measurement_reader=lambda _root, _warmup, _end: 10.0,
        sleep_fn=lambda _seconds: None,
    )
    assert report.status == "passed"
    assert report.reason == "long_run_gpu_floor_passed"
    assert report.prediction is not None
    assert report.prediction["predicted_seconds"] == pytest.approx(5998.0)
    command = report.calibration["command"]
    assert command[-6:] == [
        "--iterations", "60", "--test_iterations", "60", "--save_iterations", "60",
    ]
    assert command[command.index("-m") + 1] == str(output / "calibration_model")
    assert (output / "preflight.json").is_file()


class _RamulatorBinding:
    def __init__(self) -> None:
        self.pending: list[int] = []
        self.completed: list[int] = []

    def metadata(self) -> dict[str, object]:
        return {
            "implementation": "Ramulator 2", "version": "2.1.0",
            "commit": "d" * 40,
            "config_sha256": "b" * 64,
            "channels": 8, "transaction_bytes": 64,
            "channel_width_bits": 32, "data_rate_mtps": 6400,
        }

    def try_issue(self, address: int, is_write: bool, request_id: int) -> bool:
        self.pending.append(request_id)
        return True

    def tick(self) -> None:
        self.completed.extend(self.pending)
        self.pending.clear()

    def drain_completions(self) -> tuple[int, ...]:
        result = tuple(self.completed)
        self.completed.clear()
        return result

    def clone(self) -> "_RamulatorBinding":
        return type(self)()


def _ready_config() -> GalaConfig:
    metadata = lambda value, unit: {
        "value": value, "unit": unit, "source": "fixture", "scope": "hardware",
        "status": "frozen",
    }
    parameters = {
        "top": {
            "num_pods": metadata(4, "count"),
            "shared_sram_bytes": metadata(2883584, "byte"),
        },
        "compute": {
            "clusters_per_pod": metadata(5, "cluster"),
            "fma_lanes_per_cluster": metadata(16, "lane"),
            "transcendental_lanes_per_cluster": metadata(2, "lane"),
        },
        "memory": {
            "channels": metadata(8, "channel"),
            "channel_width_bits": metadata(32, "bit"),
            "data_rate": metadata(6400, "MT/s"),
            "ramulator_version": metadata("2.1.0", "version"),
            "ramulator_commit": metadata("d" * 40, "git_commit"),
            "ramulator_config_sha256": metadata("b" * 64, "sha256"),
        },
        "cache": {"sector_bytes": metadata(64, "byte")},
    }
    return GalaConfig(Path("fixture.yaml"), parameters, "a" * 64, True)


def test_cycle_preflight_records_all_formal_blockers(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    report = run_cycle_preflight(
        config, reproduction="gala-sim cycle-preflight --config configs/architecture/gala.yaml",
    )
    assert report.status == "failed_preflight"
    assert report.pending == ()
    assert "ramulator2_binding" in report.missing_bindings
    assert "resource_envelope" in report.missing_bindings
    write_cycle_preflight(report, tmp_path / "run")
    assert (tmp_path / "run" / "preflight.json").is_file()
    assert '"status": "failed_preflight"' in (tmp_path / "run" / "status.json").read_text()


def test_cycle_preflight_passes_with_binding_and_resource_snapshot() -> None:
    usage = ResourceUsage(
        shared_sram_bytes=1, pods=4, clusters=20, fma_lanes=320,
        transcendental_lanes=40, external_channels=8, regions={"fixture": 1},
    )
    report = run_cycle_preflight(
        _ready_config(), memory_backend=Ramulator2Backend(_RamulatorBinding()),
        resource_usage=usage, reproduction="fixture",
    )
    assert report.status == "passed"
    assert report.missing_bindings == ()
