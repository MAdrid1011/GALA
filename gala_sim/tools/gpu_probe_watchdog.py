"""Shared resource and progress guard for in-process CUDA probes."""

from __future__ import annotations

import _thread
from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time
from typing import Any, Callable, Sequence

from gala_sim.tools.gpu_observation import (
    GpuObservationError,
    external_compute_processes,
    sample_gpu_snapshot,
)


DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MINIMUM_FREE_GPU_MEMORY_BYTES = 1024 * 1024 * 1024


def available_host_memory_bytes() -> int | None:
    """Return Linux MemAvailable, or ``None`` when it cannot be observed."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


def process_cpu_seconds(pid: int) -> float | None:
    """Return user plus system CPU time for one Linux process."""

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(
            encoding="utf-8"
        ).rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def ensure_host_memory_reserve(
    minimum_available_bytes: int = DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES,
    *,
    memory_fn: Callable[[], int | None] = available_host_memory_bytes,
) -> int | None:
    """Reject a probe before imports and allocations consume the host reserve."""

    if minimum_available_bytes < 0:
        raise ValueError("host memory reserve cannot be negative")
    available = memory_fn()
    if available is not None and available < minimum_available_bytes:
        raise RuntimeError("host_memory_reserve_exhausted")
    return available


def _gpu_sample() -> dict[str, Any] | None:
    try:
        return sample_gpu_snapshot()
    except GpuObservationError:
        return None


@dataclass
class GpuWatchdog:
    """Interrupt an in-process probe when resources or progress become unsafe."""

    timeout_seconds: float
    sample_interval_seconds: float = 1.0
    owner_pid: int | None = None
    minimum_available_host_memory_bytes: int = (
        DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES
    )
    minimum_free_gpu_memory_bytes: int = DEFAULT_MINIMUM_FREE_GPU_MEMORY_BYTES
    sample_fn: Callable[[], dict[str, Any] | None] = _gpu_sample
    host_memory_fn: Callable[[], int | None] = available_host_memory_bytes
    process_cpu_fn: Callable[[int], float | None] = process_cpu_seconds
    monotonic_fn: Callable[[], float] = time.monotonic
    interrupt_fn: Callable[[], None] = _thread.interrupt_main

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise ValueError("watchdog intervals must be positive")
        if self.minimum_available_host_memory_bytes < 0:
            raise ValueError("host memory reserve cannot be negative")
        if self.minimum_free_gpu_memory_bytes < 0:
            raise ValueError("GPU memory reserve cannot be negative")
        self.samples: list[dict[str, Any]] = []
        self.last_progress_at = self.monotonic_fn()
        self.last_gpu_active_at = self.last_progress_at
        self.last_cpu_active_at = self.last_progress_at
        self.last_cpu_seconds = (
            self.process_cpu_fn(self.owner_pid)
            if self.owner_pid is not None else None
        )
        self.failure: str | None = None
        self.external_compute_processes: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def timed_out(self) -> bool:
        return self.failure == "watchdog_inactivity_timeout"

    @property
    def contention_detected(self) -> bool:
        return self.failure == "gpu_contention_detected"

    def start(self) -> None:
        self._thread.start()

    def progress(self) -> None:
        self.last_progress_at = self.monotonic_fn()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.sample_interval_seconds * 2)

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self.interrupt_fn()

    def _run(self) -> None:
        started = self.monotonic_fn()
        while not self._stop.wait(self.sample_interval_seconds):
            now = self.monotonic_fn()
            sample = self.sample_fn()
            host_available = self.host_memory_fn()
            if sample is None:
                self._fail("gpu_sampling_unavailable")
                return
            sample["elapsed_seconds"] = now - started
            sample["host_memory_available_bytes"] = host_available
            if self.owner_pid is not None:
                cpu_seconds = self.process_cpu_fn(self.owner_pid)
                sample["owner_cpu_seconds"] = cpu_seconds
                if (
                    cpu_seconds is not None
                    and self.last_cpu_seconds is not None
                    and cpu_seconds > self.last_cpu_seconds
                ):
                    self.last_cpu_active_at = now
                self.last_cpu_seconds = cpu_seconds
            self.samples.append(sample)
            if self.owner_pid is not None:
                try:
                    external = external_compute_processes(
                        sample, owner_pid=self.owner_pid,
                    )
                except GpuObservationError:
                    external = []
                if external:
                    self.external_compute_processes.extend(external)
                    self._fail("gpu_contention_detected")
                    return
                owned = any(
                    int(item.get("pid", -1)) == self.owner_pid
                    for item in sample.get("compute_processes", [])
                )
                if owned and int(sample.get("utilization_percent", 0)) > 0:
                    self.last_gpu_active_at = now
            total_mib = sample.get("memory_total_mib")
            used_mib = sample.get("memory_used_mib")
            if total_mib is not None and used_mib is not None:
                free_bytes = (int(total_mib) - int(used_mib)) * 1024 * 1024
                if free_bytes < self.minimum_free_gpu_memory_bytes:
                    self._fail("gpu_memory_reserve_exhausted")
                    return
            if (
                host_available is not None
                and host_available < self.minimum_available_host_memory_bytes
            ):
                self._fail("host_memory_reserve_exhausted")
                return
            if now - max(
                self.last_progress_at,
                self.last_gpu_active_at,
                self.last_cpu_active_at,
            ) >= self.timeout_seconds:
                self._fail("watchdog_inactivity_timeout")
                return


def gpu_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw samples while preserving them for later audit."""

    if not samples:
        return {"sample_count": 0, "status": "unavailable"}
    utilization = [int(item["utilization_percent"]) for item in samples]
    memory = [int(item["memory_used_mib"]) for item in samples]
    return {
        "sample_count": len(samples),
        "status": "measured",
        "mean_utilization_percent": sum(utilization) / len(utilization),
        "maximum_utilization_percent": max(utilization),
        "maximum_memory_used_mib": max(memory),
        "samples": list(samples),
    }
