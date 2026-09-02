from __future__ import annotations

from threading import Event

import pytest

from gala_sim.tools.gpu_probe_watchdog import GpuWatchdog, ensure_host_memory_reserve


def test_host_memory_gate_accepts_exact_reserve() -> None:
    assert ensure_host_memory_reserve(8, memory_fn=lambda: 8) == 8


def test_host_memory_gate_rejects_low_available_memory() -> None:
    with pytest.raises(RuntimeError, match="host_memory_reserve_exhausted"):
        ensure_host_memory_reserve(8, memory_fn=lambda: 7)


def test_host_memory_gate_allows_unavailable_observation() -> None:
    assert ensure_host_memory_reserve(8, memory_fn=lambda: None) is None


def test_runtime_watchdog_interrupts_on_host_memory_reserve() -> None:
    interrupted = Event()
    watchdog = GpuWatchdog(
        timeout_seconds=300,
        sample_interval_seconds=0.001,
        minimum_available_host_memory_bytes=8,
        minimum_free_gpu_memory_bytes=0,
        sample_fn=lambda: {
            "utilization_percent": 0,
            "memory_used_mib": 0,
            "memory_total_mib": 1,
            "compute_processes": [],
        },
        host_memory_fn=lambda: 7,
        interrupt_fn=interrupted.set,
    )
    watchdog.start()
    assert interrupted.wait(timeout=1)
    watchdog.stop()
    assert watchdog.failure == "host_memory_reserve_exhausted"


def test_runtime_watchdog_interrupts_on_gpu_memory_reserve() -> None:
    interrupted = Event()
    watchdog = GpuWatchdog(
        timeout_seconds=300,
        sample_interval_seconds=0.001,
        minimum_available_host_memory_bytes=0,
        minimum_free_gpu_memory_bytes=2 * 1024 * 1024,
        sample_fn=lambda: {
            "utilization_percent": 50,
            "memory_used_mib": 15,
            "memory_total_mib": 16,
            "compute_processes": [],
        },
        host_memory_fn=lambda: 100,
        interrupt_fn=interrupted.set,
    )
    watchdog.start()
    assert interrupted.wait(timeout=1)
    watchdog.stop()
    assert watchdog.failure == "gpu_memory_reserve_exhausted"


def test_runtime_watchdog_rejects_lost_gpu_sampling() -> None:
    interrupted = Event()
    watchdog = GpuWatchdog(
        timeout_seconds=300,
        sample_interval_seconds=0.001,
        minimum_available_host_memory_bytes=0,
        minimum_free_gpu_memory_bytes=0,
        sample_fn=lambda: None,
        host_memory_fn=lambda: 100,
        interrupt_fn=interrupted.set,
    )
    watchdog.start()
    assert interrupted.wait(timeout=1)
    watchdog.stop()
    assert watchdog.failure == "gpu_sampling_unavailable"
