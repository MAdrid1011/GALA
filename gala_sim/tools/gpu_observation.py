"""GPU occupancy snapshots used to reject contended measurements."""

from __future__ import annotations

import csv
import io
import subprocess
import time
from typing import Any, Callable, Sequence


class GpuObservationError(RuntimeError):
    """GPU state could not be sampled or is unsuitable for a measurement."""


def _output_lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if line.strip()]


def _parse_compute_processes(value: str, gpu_uuid: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi`` compute rows for exactly one GPU."""

    lines = _output_lines(value)
    if not lines or all(line.lstrip().startswith("No running") for line in lines):
        return []
    processes: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(value)):
        if not row or len(row) != 4:
            raise GpuObservationError("nvidia-smi compute-process output is malformed")
        row_uuid, pid_text, process_name, memory_text = (item.strip() for item in row)
        if row_uuid != gpu_uuid:
            continue
        try:
            pid = int(pid_text)
            memory_used_mib = int(memory_text)
        except ValueError as error:
            raise GpuObservationError(
                "nvidia-smi compute-process output has a nonnumeric field"
            ) from error
        processes.append({
            "pid": pid,
            "process_name": process_name,
            "memory_used_mib": memory_used_mib,
        })
    return processes


def sample_gpu_snapshot(
    *,
    runner: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    """Return GPU 0 utilization and all compute processes using it."""

    try:
        gpu_lines = _output_lines(runner(
            [
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ))
        if len(gpu_lines) != 1:
            raise GpuObservationError("nvidia-smi did not return one GPU 0 row")
        gpu_values = [item.strip() for item in next(csv.reader([gpu_lines[0]]))]
        if len(gpu_values) != 3:
            raise GpuObservationError("nvidia-smi GPU output is malformed")
        gpu_uuid, utilization_text, memory_text = gpu_values
        utilization = int(utilization_text)
        memory_used = int(memory_text)
        process_output = runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except GpuObservationError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, csv.Error) as error:
        raise GpuObservationError("nvidia-smi GPU observation is unavailable") from error
    return {
        "gpu_uuid": gpu_uuid,
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
        "compute_processes": _parse_compute_processes(process_output, gpu_uuid),
    }


def ensure_gpu_isolated(
    *,
    sample_count: int = 2,
    sample_interval_seconds: float = 1.0,
    sample_fn: Callable[[], dict[str, Any]] = sample_gpu_snapshot,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Require consecutive snapshots with no foreign GPU compute process."""

    if sample_count <= 0 or sample_interval_seconds < 0:
        raise ValueError("GPU isolation sampling parameters are invalid")
    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        sample = sample_fn()
        processes = sample.get("compute_processes")
        if not isinstance(processes, list):
            raise GpuObservationError("GPU observation has no compute-process inventory")
        if processes:
            raise GpuObservationError("gpu_busy_external")
        samples.append(sample)
        if index + 1 < sample_count:
            sleep_fn(sample_interval_seconds)
    return {
        "status": "isolated",
        "sample_count": sample_count,
        "samples": samples,
    }


def external_compute_processes(
    sample: dict[str, Any], *, owner_pid: int,
) -> list[dict[str, Any]]:
    """Return observed CUDA contexts that do not belong to the probe process."""

    processes = sample.get("compute_processes")
    if not isinstance(processes, list):
        raise GpuObservationError("GPU observation has no compute-process inventory")
    return [item for item in processes if int(item.get("pid", -1)) != owner_pid]
