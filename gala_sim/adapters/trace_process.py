"""Run one official trace process with ownership-scoped progress checks."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from gala_sim.tools.inactivity import InactivityTimeoutError, InactivityWatchdog
from gala_sim.tools.preflight import GpuSample, sample_gpustat


class TraceProcessError(RuntimeError):
    """Raised when an official trace process cannot complete safely."""


def _process_cpu_seconds(pid: int) -> float | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _available_host_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                return int(fields[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait()


def run_trace_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    trace_root: Path,
    inactivity_timeout_seconds: float = 300.0,
    sample_interval_seconds: float = 1.0,
    termination_grace_seconds: float = 10.0,
    minimum_free_memory_bytes: int = 1024 * 1024 * 1024,
    minimum_available_host_memory_bytes: int = 8 * 1024 * 1024 * 1024,
    sample_fn: Callable[[], GpuSample] = sample_gpustat,
    preflight_fn: Callable[[], None] | None = None,
    prepare_fn: Callable[[], None] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    wall_time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    host_memory_fn: Callable[[], int | None] = _available_host_memory_bytes,
) -> dict[str, Any]:
    """Run a trace job and terminate only that job after five idle minutes."""

    if inactivity_timeout_seconds <= 0 or sample_interval_seconds <= 0:
        raise ValueError("trace watchdog timing must be positive")
    if termination_grace_seconds <= 0:
        raise ValueError("trace watchdog termination grace must be positive")
    if minimum_free_memory_bytes < 0:
        raise ValueError("trace GPU memory reserve cannot be negative")
    if minimum_available_host_memory_bytes < 0:
        raise ValueError("trace host memory reserve cannot be negative")
    if preflight_fn is not None:
        preflight_fn()
    for sample_index in range(2):
        initial = sample_fn()
        if initial.compute_processes:
            raise TraceProcessError("gpu_busy_external")
        if sample_index == 0:
            sleep_fn(sample_interval_seconds)
    if prepare_fn is not None:
        prepare_fn()

    trace_root.mkdir(parents=True, exist_ok=True)
    logs = trace_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "stdout.log"
    stderr_path = logs / "stderr.log"
    watchdog = InactivityWatchdog(
        timeout_seconds=inactivity_timeout_seconds,
        monotonic_fn=monotonic_fn,
    )
    samples: list[dict[str, Any]] = []
    started = wall_time_fn()
    failure: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, \
            stderr_path.open("w", encoding="utf-8") as stderr_stream:
        process = subprocess.Popen(
            list(command), cwd=cwd, env=dict(environment),
            stdout=stdout_stream, stderr=stderr_stream, text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            try:
                sample = sample_fn()
            except RuntimeError:
                failure = "gpu_sampling_unavailable"
                _terminate_process_group(process, termination_grace_seconds)
                break
            sample_document = asdict(sample)
            host_memory_available = host_memory_fn()
            sample_document["host_memory_available_bytes"] = host_memory_available
            samples.append(sample_document)
            owned_gpu_process = any(
                item.pid == process.pid for item in sample.compute_processes
            )
            external = tuple(
                item for item in sample.compute_processes if item.pid != process.pid
            )
            if external:
                failure = "gpu_contention_detected"
                _terminate_process_group(process, termination_grace_seconds)
                break
            if (
                host_memory_available is not None
                and host_memory_available < minimum_available_host_memory_bytes
            ):
                failure = "host_memory_reserve_exhausted"
                _terminate_process_group(process, termination_grace_seconds)
                break
            if (
                owned_gpu_process
                and sample.memory_total_bytes is not None
                and sample.memory_total_bytes - sample.memory_used_bytes
                < minimum_free_memory_bytes
            ):
                failure = "gpu_memory_reserve_exhausted"
                _terminate_process_group(process, termination_grace_seconds)
                break
            try:
                watchdog.observe(
                    gpu_active=owned_gpu_process and sample.utilization_percent > 0,
                    byte_count=_tree_bytes(trace_root),
                    cpu_seconds=_process_cpu_seconds(process.pid),
                )
            except InactivityTimeoutError:
                failure = "watchdog_inactivity_timeout"
                _terminate_process_group(process, termination_grace_seconds)
                break
            sleep_fn(sample_interval_seconds)
        returncode = process.wait()
    finished = wall_time_fn()
    report = {
        "status": "passed" if returncode == 0 and failure is None else "failed",
        "command": list(command),
        "working_directory": str(cwd.resolve()),
        "started_at_unix": started,
        "finished_at_unix": finished,
        "wall_seconds": finished - started,
        "returncode": returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "gpu_samples": samples,
        "watchdog": watchdog.report(
            status="terminated" if failure is not None else "completed"
        ),
        "minimum_free_memory_bytes": minimum_free_memory_bytes,
        "minimum_available_host_memory_bytes": minimum_available_host_memory_bytes,
    }
    if failure is not None:
        report["failure"] = failure
    (trace_root / "capture_process.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failure is not None:
        raise TraceProcessError(failure)
    if returncode != 0:
        raise TraceProcessError("official_trace_command_failed")
    return report
