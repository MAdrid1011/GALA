from __future__ import annotations

from pathlib import Path

import pytest

from gala_sim.config import GalaConfig, load_config
from gala_sim.timing.memory import Ramulator2Backend
from gala_sim.timing.resources import ResourceUsage
from gala_sim.tools.cycle_preflight import run_cycle_preflight, write_cycle_preflight
from gala_sim.tools.preflight import GpuSample, decide_long_run, predict_runtime


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


class _RamulatorBinding:
    def __init__(self) -> None:
        self.pending: list[int] = []
        self.completed: list[int] = []

    def metadata(self) -> dict[str, object]:
        return {
            "implementation": "Ramulator 2", "version": "2.1.0",
            "config_sha256": "b" * 64,
            "channels": 8, "transaction_bytes": 64,
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
        "memory": {"channels": metadata(8, "channel")},
        "cache": {"sector_bytes": metadata(64, "byte")},
    }
    return GalaConfig(Path("fixture.yaml"), parameters, "a" * 64, True)


def test_cycle_preflight_records_all_formal_blockers(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    report = run_cycle_preflight(
        config, reproduction="gala-sim cycle-preflight --config configs/architecture/gala.yaml",
    )
    assert report.status == "failed_preflight"
    assert "relation.seed_fifo_entries" in report.pending
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
