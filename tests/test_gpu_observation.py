from __future__ import annotations

import pytest

from gala_sim.tools.gpu_observation import (
    GpuObservationError,
    ensure_gpu_isolated,
    external_compute_processes,
    sample_gpu_snapshot,
)


def test_gpu_snapshot_filters_compute_processes_to_gpu_zero() -> None:
    responses = iter((
        "GPU-0, 17, 123, 4096\n",
        "GPU-0, 99, train, 456\nGPU-1, 88, other, 789\n",
    ))

    def runner(*_args, **_kwargs):
        return next(responses)

    snapshot = sample_gpu_snapshot(runner=runner)
    assert snapshot["utilization_percent"] == 17
    assert snapshot["memory_used_mib"] == 123
    assert snapshot["memory_total_mib"] == 4096
    assert snapshot["compute_processes"] == [
        {"pid": 99, "process_name": "train", "memory_used_mib": 456},
    ]


def test_gpu_isolation_requires_consecutive_empty_compute_inventories() -> None:
    samples = iter((
        {"compute_processes": []},
        {"compute_processes": []},
    ))
    report = ensure_gpu_isolated(
        sample_count=2, sample_interval_seconds=0,
        sample_fn=lambda: next(samples), sleep_fn=lambda _seconds: None,
    )
    assert report["status"] == "isolated"
    assert report["sample_count"] == 2


def test_gpu_isolation_rejects_external_compute_process() -> None:
    with pytest.raises(GpuObservationError, match="gpu_busy_external"):
        ensure_gpu_isolated(
            sample_count=1,
            sample_fn=lambda: {"compute_processes": [{"pid": 73}]},
        )


def test_external_compute_processes_excludes_the_owned_probe_context() -> None:
    sample = {"compute_processes": [{"pid": 11}, {"pid": 12}]}
    assert external_compute_processes(sample, owner_pid=11) == [{"pid": 12}]
