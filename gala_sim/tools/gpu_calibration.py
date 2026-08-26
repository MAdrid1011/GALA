"""Portable CUDA calibration vector used for local GPU and AGX Orin runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import time
from typing import Any, Callable, Mapping

import yaml

from gala_sim.identity import sha256_file


CALIBRATION_SCHEMA_VERSION = "gala-gpu-calibration-vector-v1"
CALIBRATION_CATEGORIES = (
    "fp32_fma", "exp", "log", "rcp", "sqrt", "memory_bandwidth",
    "atomic", "kernel_launch", "synchronization",
)


@dataclass(frozen=True)
class CalibrationConfig:
    dtype: str
    batch_sizes: tuple[int, ...]
    input_min: float
    input_max: float
    warmup_repetitions: int
    measure_repetitions: int
    launches_per_measurement: int
    atomic_contention_divisor: int

    def __post_init__(self) -> None:
        if self.dtype != "float32":
            raise ValueError("GPU calibration currently requires float32")
        if (
            not self.batch_sizes
            or any(value <= 0 for value in self.batch_sizes)
            or tuple(sorted(set(self.batch_sizes))) != self.batch_sizes
            or self.input_min <= 0
            or self.input_max <= self.input_min
            or self.warmup_repetitions <= 0
            or self.measure_repetitions <= 0
            or self.launches_per_measurement <= 0
            or self.atomic_contention_divisor <= 0
        ):
            raise ValueError("GPU calibration configuration values are invalid")

    @classmethod
    def load(cls, path: Path) -> "CalibrationConfig":
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            not isinstance(document, Mapping)
            or document.get("schema_version") != "gala-gpu-calibration-config-v1"
        ):
            raise ValueError("unsupported GPU calibration configuration")
        return cls(
            dtype=str(document["dtype"]),
            batch_sizes=tuple(int(value) for value in document["batch_sizes"]),
            input_min=float(document["input_min"]),
            input_max=float(document["input_max"]),
            warmup_repetitions=int(document["warmup_repetitions"]),
            measure_repetitions=int(document["measure_repetitions"]),
            launches_per_measurement=int(document["launches_per_measurement"]),
            atomic_contention_divisor=int(document["atomic_contention_divisor"]),
        )


def _device_measurements(
    torch: Any,
    operation: Callable[[], Any],
    warmups: int,
    repetitions: int,
) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    events = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in events]


def _summary(milliseconds: list[float], work: int, unit: str) -> dict[str, Any]:
    if not milliseconds or work <= 0:
        raise ValueError("GPU calibration measurements are empty")
    middle = float(median(milliseconds))
    return {
        "measurement_count": len(milliseconds),
        "milliseconds": milliseconds,
        "median_ms": middle,
        "work_per_invocation": work,
        "work_unit": unit,
        "median_seconds_per_work": middle / 1000.0 / work,
    }


def _device_identity(torch: Any) -> dict[str, Any]:
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[index].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unavailable"
    return {
        "device_index": index,
        "name": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "driver_version": driver,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "python_version": platform.python_version(),
    }


def run_calibration(config: CalibrationConfig) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU calibration requires CUDA")
    torch.manual_seed(0)
    vectors: dict[str, dict[str, Any]] = {name: {} for name in CALIBRATION_CATEGORIES}
    for batch_size in config.batch_sizes:
        values = torch.linspace(
            config.input_min, config.input_max, batch_size,
            dtype=torch.float32, device="cuda",
        )
        other = values.clone()
        output = torch.empty_like(values)
        operations: dict[str, tuple[Callable[[], Any], int, str]] = {
            "fp32_fma": (
                lambda: torch.addcmul(values, other, values, out=output),
                batch_size,
                "fma",
            ),
            "exp": (lambda: torch.exp(values, out=output), batch_size, "element"),
            "log": (lambda: torch.log(values, out=output), batch_size, "element"),
            "rcp": (lambda: torch.reciprocal(values, out=output), batch_size, "element"),
            "sqrt": (lambda: torch.sqrt(values, out=output), batch_size, "element"),
            "memory_bandwidth": (
                lambda: output.copy_(values), batch_size * values.element_size() * 2, "byte"
            ),
        }
        atomic_bins = max(1, batch_size // config.atomic_contention_divisor)
        atomic_output = torch.zeros(atomic_bins, dtype=torch.float32, device="cuda")
        atomic_index = torch.arange(batch_size, device="cuda", dtype=torch.int64) % atomic_bins

        def atomic_operation() -> Any:
            atomic_output.zero_()
            return atomic_output.scatter_add_(0, atomic_index, values)

        operations["atomic"] = (atomic_operation, batch_size, "atomic_add")
        for category, (operation, work, unit) in operations.items():
            measured = _device_measurements(
                torch, operation, config.warmup_repetitions,
                config.measure_repetitions,
            )
            entry = _summary(measured, work, unit)
            if category == "atomic":
                entry["target_bins"] = atomic_bins
                entry["contention_divisor"] = config.atomic_contention_divisor
            vectors[category][str(batch_size)] = entry

    launch_value = torch.zeros(1, dtype=torch.float32, device="cuda")

    def launch_batch() -> Any:
        for _ in range(config.launches_per_measurement):
            launch_value.add_(1.0)
        return launch_value

    launch_measurements = _device_measurements(
        torch, launch_batch, config.warmup_repetitions, config.measure_repetitions
    )
    vectors["kernel_launch"][str(config.launches_per_measurement)] = _summary(
        launch_measurements, config.launches_per_measurement, "launch"
    )
    for _ in range(config.warmup_repetitions):
        torch.cuda.synchronize()
    synchronization_ms = []
    for _ in range(config.measure_repetitions):
        started = time.perf_counter_ns()
        torch.cuda.synchronize()
        synchronization_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    vectors["synchronization"]["idle"] = _summary(
        synchronization_ms, 1, "synchronize_call"
    )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "passed",
        "suite": {
            "categories": list(CALIBRATION_CATEGORIES),
            "configuration": asdict(config),
            "semantics": {
                "fp32_fma": "torch.addcmul float32",
                "transcendentals": "torch float32 unary CUDA kernels",
                "memory_bandwidth": "device-to-device float32 copy, read plus write bytes",
                "atomic": "float32 scatter_add with recorded contention divisor",
                "kernel_launch": "in-place one-element float32 add launches",
                "synchronization": "idle torch.cuda.synchronize host latency",
            },
        },
        "device": _device_identity(torch),
        "vectors": vectors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_calibration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = CalibrationConfig.load(args.config)
    result = run_calibration(config)
    result["suite"]["configuration_path"] = str(args.config.resolve())
    result["suite"]["configuration_sha256"] = sha256_file(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "device": result["device"]["name"],
        "output": str(args.output.resolve()),
        "status": "passed",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
