"""Run and record the frozen R²-Gaussian native reference path."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

import numpy as np

from gala_sim.config import GalaConfig
from gala_sim.identity import canonical_json, sha256_bytes, sha256_file
from gala_sim.manifest import verify_freeze_record
from gala_sim.metrics import QualityConfig, measure_quality
from gala_sim.results import RunManifest
from gala_sim.results.run import RunOutputWriter
from gala_sim.tools.preflight import GpuSample, sample_gpustat

from .chest import load_chest_manifest
from .r2_gaussian import _read_latest_metrics


class NativeReferenceError(RuntimeError):
    """Raised when the frozen native reference cannot produce a valid result."""


def _command_from_freeze(freeze: Mapping[str, Any]) -> tuple[list[str], str, Path, Path]:
    training = freeze.get("training")
    if not isinstance(training, Mapping):
        raise NativeReferenceError("input freeze has no training identity")
    command = training.get("command")
    if not isinstance(command, Mapping):
        raise NativeReferenceError("input freeze has no official command")
    argv = command.get("argv")
    working_directory = command.get("working_directory")
    if (not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv)
            or not isinstance(working_directory, str)):
        raise NativeReferenceError("input freeze official command is invalid")
    try:
        dataset = Path(argv[argv.index("-s") + 1]).resolve()
        model_output = Path(argv[argv.index("-m") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise NativeReferenceError("official command must bind -s and -m") from error
    return list(argv), working_directory, dataset, model_output


def _tensorboard_stages(model_output: Path, python_executable: str) -> dict[str, Any]:
    events = sorted(model_output.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise NativeReferenceError("official reference has no unique TensorBoard event file")
    script = (
        "import json,sys; "
        "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; "
        "a=EventAccumulator(sys.argv[1], size_guidance={'scalars': 0}); a.Reload(); "
        "print(json.dumps({k:[[v.step,v.wall_time,v.value] for v in a.Scalars(k)] "
        "for k in a.Tags().get('scalars',[]) if k in ('train/iter_time','train/total_points')}))"
    )
    try:
        raw = subprocess.check_output(
            [python_executable, "-c", script, str(events[0])],
            text=True, stderr=subprocess.STDOUT,
        )
        scalars = json.loads(raw)
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise NativeReferenceError("failed to read official TensorBoard stages") from error
    iter_times = scalars.get("train/iter_time", [])
    if not isinstance(iter_times, list) or not iter_times:
        raise NativeReferenceError("official TensorBoard has no train/iter_time series")
    return {
        "training_iteration_time_ms": [
            {"iteration": int(step), "wall_time_unix": float(wall), "milliseconds": float(value)}
            for step, wall, value in iter_times
        ],
        "training_iteration_time_total_ms": float(sum(float(row[2]) for row in iter_times)),
        "tensorboard_path": str(events[0]),
        "tensorboard_sha256": sha256_file(events[0]),
        "unmeasured_residual_seconds": "wall_seconds_minus_training_iteration_time",
    }


def _write_failed(output: Path, gpu_reference: dict[str, Any], reason: str) -> None:
    writer = RunOutputWriter(output)
    writer.write_gpu_reference(gpu_reference)
    writer.write_status("failed_preflight", reason=reason)


def _write_quality_failure(output: Path, gpu_reference: dict[str, Any], reason: str) -> None:
    writer = RunOutputWriter(output)
    writer.write_gpu_reference(gpu_reference)
    writer.write_status("failed_quality", reason=reason)


def run_native_reference(
    config: GalaConfig,
    freeze: Mapping[str, Any],
    preflight: Mapping[str, Any],
    output: Path,
    *,
    sample_fn: Callable[[], GpuSample] = sample_gpustat,
) -> dict[str, Any]:
    """Execute the exact frozen official command after a passing preflight."""

    verify_freeze_record(dict(freeze))
    if freeze.get("schema_version") != "gala-input-freeze-v3":
        raise NativeReferenceError("native reference requires gala-input-freeze-v3")
    frozen_config = freeze.get("config")
    if not isinstance(frozen_config, Mapping) or frozen_config.get("sha256") != config.sha256:
        raise NativeReferenceError("native reference configuration does not match freeze")
    if (preflight.get("schema_version") != "gala-native-preflight-v1"
            or preflight.get("status") != "passed"):
        raise NativeReferenceError("native reference requires a passed native-preflight")
    if preflight.get("freeze_manifest_sha256") != freeze.get("run_manifest_sha256"):
        raise NativeReferenceError("native preflight does not match input freeze")
    command, working_directory, dataset_root, model_output = _command_from_freeze(freeze)
    if not Path(working_directory).is_dir() or not dataset_root.is_dir():
        raise NativeReferenceError("native reference source or Chest directory is unavailable")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if model_output.exists():
        raise NativeReferenceError(f"official model output already exists: {model_output}")
    try:
        initial = sample_fn()
    except RuntimeError as error:
        _write_failed(output, {"status": "failed", "error": str(error)}, "gpu_sampling_unavailable")
        raise NativeReferenceError("gpu_sampling_unavailable") from error
    if initial.compute_processes:
        _write_failed(
            output,
            {
                "status": "failed",
                "external_compute_processes": [asdict(item) for item in initial.compute_processes],
            },
            "gpu_busy_external",
        )
        raise NativeReferenceError("gpu_busy_external")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "stdout.log"
    stderr_path = logs / "stderr.log"
    started = time.time()
    samples: list[dict[str, Any]] = []
    contention: list[dict[str, Any]] = []
    sampling_error: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, \
            stderr_path.open("w", encoding="utf-8") as stderr_stream:
        process = subprocess.Popen(
            command, cwd=working_directory, stdout=stdout_stream, stderr=stderr_stream,
            text=True,
        )
        while process.poll() is None:
            try:
                sample = sample_fn()
            except RuntimeError as error:
                sampling_error = str(error)
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            samples.append(asdict(sample))
            external = [item for item in sample.compute_processes if item.pid != process.pid]
            if external:
                contention.extend(asdict(item) for item in external)
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(1.0)
        returncode = process.wait()
    finished = time.time()
    gpu_reference: dict[str, Any] = {
        "status": "passed" if returncode == 0 and not contention else "failed",
        "command": command,
        "command_sha256": sha256_bytes(canonical_json({
            "working_directory": working_directory, "argv": command,
        })),
        "working_directory": working_directory,
        "model_output_root": str(model_output),
        "started_at_unix": started,
        "finished_at_unix": finished,
        "wall_seconds": finished - started,
        "returncode": returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "gpu_samples": samples,
    }
    writer = RunOutputWriter(output)
    if sampling_error is not None:
        gpu_reference["error"] = sampling_error
        _write_failed(output, gpu_reference, "gpu_sampling_unavailable")
        raise NativeReferenceError("gpu_sampling_unavailable")
    if contention:
        gpu_reference["external_compute_processes"] = contention
        _write_failed(output, gpu_reference, "gpu_contention_detected")
        raise NativeReferenceError("gpu_contention_detected")
    if returncode != 0:
        _write_failed(output, gpu_reference, "official_command_failed")
        raise NativeReferenceError("official R²-Gaussian command failed")
    volumes = sorted(model_output.glob("point_cloud/iteration_*/vol_pred.npy"))
    if not volumes:
        _write_failed(output, gpu_reference, "missing_reconstruction_volume")
        raise NativeReferenceError("official reference produced no reconstruction volume")
    try:
        dataset = load_chest_manifest(dataset_root)
        quality = measure_quality(
            np.load(dataset.volume_path, mmap_mode="r", allow_pickle=False),
            np.load(volumes[-1], mmap_mode="r", allow_pickle=False),
            QualityConfig.from_gala(config),
        )
        gpu_reference["stages"] = _tensorboard_stages(model_output, command[0])
    except (OSError, RuntimeError, ValueError) as error:
        gpu_reference["error"] = str(error)
        _write_quality_failure(output, gpu_reference, "quality_or_stage_measurement_failed")
        raise NativeReferenceError("quality_or_stage_measurement_failed") from error
    gpu_reference["stages"]["wall_seconds"] = finished - started
    writer.write_quality({
        "psnr": quality.psnr, "ssim": quality.ssim, "lpips": quality.lpips,
        **_read_latest_metrics(model_output),
    })
    writer.write_gpu_reference(gpu_reference)
    model_info = freeze.get("model")
    dataset_info = freeze.get("dataset")
    repository_info = freeze.get("repository")
    if not all(isinstance(item, Mapping) for item in (model_info, dataset_info, repository_info)):
        raise NativeReferenceError("input freeze source identity is incomplete")
    writer.write_manifest(RunManifest(
        run_id=f"r2_gaussian_chest_native_{int(started)}", status="passed",
        model={"name": "R2-Gaussian", "commit": model_info["commit"]},
        dataset={"name": "Chest", "manifest_sha256": dataset_info["manifest_sha256"]},
        config_sha256=config.sha256, ablation_bits="native-reference", random_seed=0,
        repository_commit=str(repository_info["commit"]), environment=freeze.get("environment", {}),
    ).as_dict())
    writer.write_status("passed", checks={"quality": "passed", "official_command": "passed"})
    return {
        "status": "passed",
        "quality": {"psnr": quality.psnr, "ssim": quality.ssim, "lpips": quality.lpips},
        "output": str(output), "gpu_reference": gpu_reference,
    }
