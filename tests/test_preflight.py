from __future__ import annotations

import pytest

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
