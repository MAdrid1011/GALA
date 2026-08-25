"""Long-run gate and GPU utilization sampling contracts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from collections.abc import Callable, Mapping
import io
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from gala_sim.config import GalaConfig
from gala_sim.identity import canonical_json, sha256_bytes, sha256_file
from gala_sim.manifest import verify_freeze_record
from gala_sim.results import write_json


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    memory_used_bytes: int


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    utilization_percent: float
    memory_used_bytes: int
    cpu_percent: float | None
    read_bytes_per_second: float | None
    events_per_second: float | None
    async_overlap_percent: float | None
    gpu_index: int = 0
    gpu_uuid: str | None = None
    compute_processes: tuple[ComputeProcess, ...] = ()


@dataclass(frozen=True)
class PreflightDecision:
    predicted_seconds: float
    threshold_seconds: float
    utilization_floor_percent: float
    allowed: bool
    reason: str


@dataclass(frozen=True)
class NativePreflightReport:
    schema_version: str
    status: str
    reason: str
    freeze_manifest_sha256: str
    config_sha256: str
    repository_commit: str
    reproduction: str
    parameters: dict[str, Any]
    initial_gpu_sample: dict[str, Any]
    gpu_samples: tuple[dict[str, Any], ...]
    calibration: dict[str, Any]
    prediction: dict[str, Any] | None
    not_applicable: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["report_sha256"] = sha256_bytes(canonical_json(value))
        return value


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


def sample_gpustat(gpu_index: int = 0) -> GpuSample:
    """Read one sample; missing gpustat is a preflight failure, not a guess."""

    try:
        output = subprocess.check_output(["gpustat", "--json"], text=True, stderr=subprocess.STDOUT)
        data: dict[str, Any] = json.loads(output)
        gpu = next(item for item in data["gpus"] if int(item["index"]) == gpu_index)
        utilization = float(gpu["utilization.gpu"])
        memory_used = int(gpu["memory.used"]) * 1024 * 1024
        gpu_uuid = str(gpu["uuid"])
        compute_output = subprocess.check_output([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.STDOUT)
        compute_processes = []
        for row in csv.reader(io.StringIO(compute_output)):
            if not row or len(row) != 4 or row[0].strip() != gpu_uuid:
                continue
            compute_processes.append(ComputeProcess(
                gpu_uuid=gpu_uuid,
                pid=int(row[1].strip()),
                process_name=row[2].strip(),
                memory_used_bytes=int(row[3].strip()) * 1024 * 1024,
            ))
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, TypeError,
            ValueError, StopIteration, json.JSONDecodeError) as error:
        raise RuntimeError("gpustat JSON sample is unavailable or invalid") from error
    return GpuSample(
        time.time(), utilization, memory_used, None, None, None, None,
        gpu_index, gpu_uuid, tuple(compute_processes),
    )


def _native_parameters(config: GalaConfig) -> dict[str, Any]:
    names = (
        "preflight.long_run_threshold_seconds",
        "preflight.gpu_utilization_floor_percent",
        "preflight.gpustat_interval_seconds",
        "preflight.warmup_iterations",
        "preflight.measure_iterations",
    )
    try:
        values = {name.rsplit(".", 1)[1]: config.value(name) for name in names}
    except KeyError as error:
        raise ValueError(f"native preflight configuration is incomplete: {error.args[0]}") from error
    if (float(values["long_run_threshold_seconds"]) <= 0
            or not 0 <= float(values["gpu_utilization_floor_percent"]) <= 100
            or float(values["gpustat_interval_seconds"]) <= 0
            or int(values["warmup_iterations"]) <= 0
            or int(values["measure_iterations"]) <= 0):
        raise ValueError("native preflight configuration values are invalid")
    return values


def _validate_native_freeze(config: GalaConfig, freeze: Mapping[str, Any]) -> tuple[list[str], str, int]:
    verify_freeze_record(dict(freeze))
    if freeze.get("schema_version") != "gala-input-freeze-v3":
        raise ValueError("native preflight requires gala-input-freeze-v3")
    frozen_config = freeze.get("config")
    if not isinstance(frozen_config, Mapping) or frozen_config.get("sha256") != config.sha256:
        raise ValueError("native preflight configuration does not match input freeze")
    training = freeze.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("input freeze has no training identity")
    command = training.get("command")
    arguments = training.get("effective_arguments")
    if (not isinstance(command, Mapping) or not isinstance(command.get("argv"), list)
            or not isinstance(command.get("working_directory"), str)
            or not isinstance(arguments, Mapping)):
        raise ValueError("input freeze training identity is incomplete")
    argv = command["argv"]
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise ValueError("input freeze training command is invalid")
    interpreter = Path(argv[0])
    if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("input freeze Python interpreter is not an executable absolute path")
    working_directory = command["working_directory"]
    if not Path(working_directory).is_dir():
        raise ValueError("input freeze training working directory is unavailable")
    total_iterations = int(arguments.get("iterations", 0))
    if total_iterations <= 0:
        raise ValueError("input freeze training iteration count is invalid")
    return list(argv), working_directory, total_iterations


def _calibration_command(argv: list[str], output: Path, iterations: int) -> list[str]:
    command = list(argv)
    try:
        model_index = command.index("-m") + 1
    except (ValueError, IndexError) as error:
        raise ValueError("official command has no model output binding") from error
    command[model_index] = str(output)
    command.extend([
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--save_iterations", str(iterations),
    ])
    return command


def _process_counters(pid: int) -> tuple[float, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        cpu_seconds = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        io_values = {}
        for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                io_values[key] = int(value.strip())
        return cpu_seconds, int(io_values.get("read_bytes", 0))
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _enrich_sample(sample: GpuSample, current: tuple[float, float, int] | None,
                   previous: tuple[float, float, int] | None) -> GpuSample:
    if current is None or previous is None:
        return sample
    elapsed = current[0] - previous[0]
    if elapsed <= 0:
        return sample
    cpu_percent = max(0.0, (current[1] - previous[1]) * 100.0 / elapsed)
    read_rate = max(0.0, (current[2] - previous[2]) / elapsed)
    return replace(sample, cpu_percent=cpu_percent, read_bytes_per_second=read_rate)


def _tensorboard_measurement(model_root: Path, warmup: int, end: int,
                             python_executable: str) -> float:
    events = sorted(model_root.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError("native preflight produced an ambiguous TensorBoard event set")
    script = (
        "import json,sys; "
        "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; "
        "a=EventAccumulator(sys.argv[1], size_guidance={'scalars': 0}); a.Reload(); "
        "print(json.dumps([[v.step,v.wall_time] for v in a.Scalars('train/iter_time')]))"
    )
    try:
        raw = subprocess.check_output(
            [python_executable, "-c", script, str(events[0])],
            text=True, stderr=subprocess.STDOUT,
        )
        values = {int(step): float(wall_time) for step, wall_time in json.loads(raw)}
        measured = values[end] - values[warmup]
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        raise RuntimeError("native preflight TensorBoard measurement is unavailable") from error
    if measured <= 0:
        raise RuntimeError("native preflight TensorBoard measurement is not positive")
    return measured


def _write_native_preflight(report: NativePreflightReport, output: Path) -> None:
    value = report.as_dict()
    write_json(value, output / "preflight.json")
    write_json({
        "status": report.status,
        "reason": report.reason,
        "checks": {"native_preflight": value["report_sha256"]},
        "reproduction": report.reproduction,
        "repository_commit": report.repository_commit,
    }, output / "status.json")


def run_native_preflight(
    config: GalaConfig,
    freeze: Mapping[str, Any],
    output: Path,
    *,
    reproduction: str,
    sample_fn: Callable[[], GpuSample] = sample_gpustat,
    measurement_reader: Callable[[Path, int, int], float] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> NativePreflightReport:
    """Run the isolated short-training gate before the full native reference."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters = _native_parameters(config)
    argv, working_directory, total_iterations = _validate_native_freeze(config, freeze)
    repository = freeze.get("repository")
    repository_commit = str(repository.get("commit", "")) if isinstance(repository, Mapping) else ""
    freeze_sha = str(freeze["run_manifest_sha256"])
    not_applicable = {
        "events_per_second": "not_applicable_native_reference",
        "async_overlap_percent": "not_applicable_native_reference",
        "cycle_simulation_throughput": "not_applicable_native_reference",
    }
    try:
        initial = sample_fn()
    except RuntimeError as error:
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "gpu_sampling_unavailable",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            {"error": str(error)}, (), {"launched": False}, None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report
    if initial.compute_processes:
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "gpu_busy_external",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            asdict(initial), (), {"launched": False}, None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report

    warmup = int(parameters["warmup_iterations"])
    measured_iterations = int(parameters["measure_iterations"])
    calibration_iterations = warmup + measured_iterations
    model_output = output / "calibration_model"
    if model_output.exists():
        raise ValueError(f"native preflight output already exists: {model_output}")
    command = _calibration_command(argv, model_output, calibration_iterations)
    command_identity = {"working_directory": working_directory, "argv": command}
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    write_json({
        "status": "running", "reason": None, "checks": {},
        "reproduction": reproduction, "repository_commit": repository_commit,
    }, output / "status.json")
    started_at = time.time()
    samples: list[GpuSample] = []
    foreign: list[ComputeProcess] = []
    sampling_error: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_stream, \
                stderr_path.open("w", encoding="utf-8") as stderr_stream:
            process = subprocess.Popen(
                command, cwd=working_directory, stdout=stdout_stream, stderr=stderr_stream,
                text=True,
            )
            previous: tuple[float, float, int] | None = None
            while process.poll() is None:
                try:
                    raw_sample = sample_fn()
                except RuntimeError as error:
                    sampling_error = str(error)
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                counters = _process_counters(process.pid)
                current = ((time.monotonic(), *counters) if counters is not None else None)
                sample = _enrich_sample(raw_sample, current, previous)
                samples.append(sample)
                previous = current
                foreign = [item for item in sample.compute_processes if item.pid != process.pid]
                if foreign:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                sleep_fn(float(parameters["gpustat_interval_seconds"]))
            returncode = process.wait()
    except OSError as error:
        finished_at = time.time()
        calibration = {
            "launched": False,
            "command": command,
            "command_sha256": sha256_bytes(canonical_json(command_identity)),
            "working_directory": working_directory,
            "output_root": str(model_output),
            "started_at_unix": started_at,
            "finished_at_unix": finished_at,
            "error": str(error),
        }
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "calibration_launch_failed",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            asdict(initial), (), calibration, None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report
    finished_at = time.time()
    if not samples:
        samples.append(sample_fn())
    calibration = {
        "launched": True,
        "command": command,
        "command_sha256": sha256_bytes(canonical_json(command_identity)),
        "working_directory": working_directory,
        "output_root": str(model_output),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "wall_seconds": finished_at - started_at,
        "returncode": returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    if sampling_error is not None:
        calibration["error"] = sampling_error
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "gpu_sampling_unavailable",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            asdict(initial), tuple(asdict(item) for item in samples), calibration,
            None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report
    if foreign:
        calibration["foreign_compute_processes"] = [asdict(item) for item in foreign]
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "gpu_contention_detected",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            asdict(initial), tuple(asdict(item) for item in samples), calibration,
            None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report
    if returncode != 0:
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "calibration_command_failed",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            asdict(initial), tuple(asdict(item) for item in samples), calibration,
            None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report

    try:
        measured_seconds = (
            measurement_reader(model_output, warmup, calibration_iterations)
            if measurement_reader is not None
            else _tensorboard_measurement(model_output, warmup, calibration_iterations, argv[0])
        )
    except RuntimeError as error:
        calibration["error"] = str(error)
        report = NativePreflightReport(
            "gala-native-preflight-v1", "failed_preflight", "measurement_unavailable",
            freeze_sha, config.sha256, repository_commit, reproduction, parameters,
            asdict(initial), tuple(asdict(item) for item in samples), calibration,
            None, not_applicable,
        )
        _write_native_preflight(report, output)
        return report
    predicted_seconds = predict_runtime(
        measured_seconds=measured_seconds,
        measured_iterations=measured_iterations,
        total_iterations=total_iterations,
        warmup_iterations=warmup,
    )
    decision = decide_long_run(
        predicted_seconds=predicted_seconds,
        threshold_seconds=float(parameters["long_run_threshold_seconds"]),
        samples=tuple(samples),
        utilization_floor_percent=float(parameters["gpu_utilization_floor_percent"]),
    )
    prediction = {
        **asdict(decision),
        "measurement_basis": "tensorboard_train_iter_time_wall_interval",
        "measured_seconds": measured_seconds,
        "measured_iterations": measured_iterations,
        "total_iterations": total_iterations,
        "warmup_iterations": warmup,
        "uncertainty": "periodic_evaluation_and_final_quality_time_excluded",
    }
    report = NativePreflightReport(
        "gala-native-preflight-v1",
        "passed" if decision.allowed else "failed_preflight",
        decision.reason,
        freeze_sha, config.sha256, repository_commit, reproduction, parameters,
        asdict(initial), tuple(asdict(item) for item in samples), calibration,
        prediction, not_applicable,
    )
    _write_native_preflight(report, output)
    return report
