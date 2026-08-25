"""Long-run gate and GPU utilization sampling contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    utilization_percent: float
    memory_used_bytes: int
    cpu_percent: float | None
    read_bytes_per_second: float | None
    events_per_second: float | None
    async_overlap_percent: float | None


@dataclass(frozen=True)
class PreflightDecision:
    predicted_seconds: float
    threshold_seconds: float
    utilization_floor_percent: float
    allowed: bool
    reason: str


def predict_runtime(*, measured_seconds: float, measured_iterations: int,
                    total_iterations: int, warmup_iterations: int) -> float:
    if min(measured_seconds, measured_iterations, total_iterations) <= 0:
        raise ValueError("runtime prediction inputs must be positive")
    if warmup_iterations < 0 or warmup_iterations >= total_iterations:
        raise ValueError("warmup iterations must be within total iterations")
    effective = total_iterations - warmup_iterations
    return measured_seconds * effective / measured_iterations


def decide_long_run(*, predicted_seconds: float, threshold_seconds: float,
                    samples: tuple[GpuSample, ...], utilization_floor_percent: float) -> PreflightDecision:
    if predicted_seconds < 0 or threshold_seconds <= 0 or not samples:
        raise ValueError("preflight prediction and samples are invalid")
    mean_utilization = sum(sample.utilization_percent for sample in samples) / len(samples)
    allowed = predicted_seconds < threshold_seconds or mean_utilization >= utilization_floor_percent
    if predicted_seconds < threshold_seconds:
        reason = "below_long_run_threshold"
    elif allowed:
        reason = "long_run_gpu_floor_passed"
    else:
        reason = "long_run_gpu_floor_failed"
    return PreflightDecision(predicted_seconds, threshold_seconds,
                             utilization_floor_percent, allowed, reason)


def sample_gpustat() -> GpuSample:
    """Read one sample; missing gpustat is a preflight failure, not a guess."""

    try:
        output = subprocess.check_output(["gpustat", "--json"], text=True, stderr=subprocess.STDOUT)
        data: dict[str, Any] = json.loads(output)
        gpu = data["gpus"][0]
        utilization = float(gpu["utilization.gpu"])
        memory_used = int(gpu["memory.used"]) * 1024 * 1024
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, TypeError,
            ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("gpustat JSON sample is unavailable or invalid") from error
    return GpuSample(time.time(), utilization, memory_used, None, None, None, None)
