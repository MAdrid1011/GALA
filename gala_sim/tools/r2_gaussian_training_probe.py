"""Measure the complete R2-Gaussian training loop without observer stalls."""

from __future__ import annotations

import argparse
import _thread
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Iterator, Sequence


PUBLISHED_ITERATIONS = 30_000
OVERLAY_FILENAME = "train_cuda_opt_probe.py"
COMPILER_VARIANT_FLAGS = {
    "gpu_base": (False, False),
    "1000": (True, False),
    "0100": (False, True),
    "1100": (True, True),
}
OVERLAY_TRANSFORMS = (
    (
        "remove-iteration-timing-events",
        """    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
""",
        """    # Train
""",
    ),
    ("remove-iteration-start-event", "        iter_start.record()\n", ""),
    (
        "remove-post-backward-host-sync",
        """        iter_end.record()
        torch.cuda.synchronize()

""",
        "",
    ),
    (
        "remove-progress-scalar-read",
        """            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
""",
        """            # Progress bar
            if iteration % 10 == 0:
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
""",
    ),
    (
        "batch-loss-telemetry",
        """            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )
""",
        """            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l]
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                None,
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )
""",
    ),
)


class TrainingProbeError(RuntimeError):
    """The official-loop performance probe could not produce valid evidence."""


class StopAfterPrefix(RuntimeError):
    """Stop after a completed training iteration and its optimizer update."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_training_overlay(source: str) -> tuple[str, list[dict[str, str]]]:
    """Remove observation-only synchronization from the published loop."""

    transformed = source
    manifest: list[dict[str, str]] = []
    for transform_id, expected, replacement in OVERLAY_TRANSFORMS:
        count = transformed.count(expected)
        if count != 1:
            raise TrainingProbeError(
                f"overlay transform {transform_id} expected one source fragment; found {count}"
            )
        transformed = transformed.replace(expected, replacement, 1)
        manifest.append({
            "id": transform_id,
            "source_fragment_sha256": _sha256_text(expected),
        })
    if "torch.cuda.synchronize()" in transformed or ".item()" in transformed:
        raise TrainingProbeError("overlay retains a prohibited per-iteration host synchronization")
    return transformed, manifest


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_training(
    path: Path, source_root: Path, extension_root: Path | None,
) -> ModuleType:
    search_paths = [str(source_root)]
    if extension_root is not None:
        search_paths.insert(0, str(extension_root))
    sys.path[:0] = search_paths
    try:
        spec = importlib.util.spec_from_file_location("gala_r2_training_probe", path)
        if spec is None or spec.loader is None:
            raise TrainingProbeError("cannot load generated training overlay")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for search_path in search_paths:
            sys.path.remove(search_path)


@contextmanager
def _compiler_variant_environment(variant: str) -> Iterator[None]:
    query, semantic = COMPILER_VARIANT_FLAGS[variant]
    updates = {
        "GALA_QUERY_WARP_REDUCE": "1" if query else "0",
        "GALA_SEMANTIC_WARP_REDUCE": "1" if semantic else "0",
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _gpu_sample() -> dict[str, int] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip().splitlines()[0]
        utilization, memory = (int(item.strip()) for item in output.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"utilization_percent": utilization, "memory_used_mib": memory}


@dataclass
class GpuWatchdog:
    timeout_seconds: float
    sample_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise ValueError("watchdog intervals must be positive")
        self.samples: list[dict[str, Any]] = []
        self.last_progress_at = time.monotonic()
        self.last_gpu_active_at = self.last_progress_at
        self.timed_out = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def progress(self) -> None:
        self.last_progress_at = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.sample_interval_seconds * 2)

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.wait(self.sample_interval_seconds):
            now = time.monotonic()
            sample = _gpu_sample()
            if sample is not None:
                sample["elapsed_seconds"] = now - started
                self.samples.append(sample)
                if sample["utilization_percent"] > 0:
                    self.last_gpu_active_at = now
            if now - max(self.last_progress_at, self.last_gpu_active_at) >= self.timeout_seconds:
                self.timed_out = True
                _thread.interrupt_main()
                return


def _gpu_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
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


def _state_fingerprint(gaussians: Any) -> dict[str, Any]:
    tensors = {
        "xyz": gaussians._xyz,
        "scaling": gaussians._scaling,
        "density": gaussians._density,
        "max_radii2D": gaussians.max_radii2D,
        "xyz_gradient_accum": gaussians.xyz_gradient_accum,
        "denom": gaussians.denom,
    }
    digest = hashlib.sha256()
    fields: dict[str, Any] = {}
    for name, tensor in tensors.items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        fields[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sum": float(array.astype("float64").sum()),
            "absolute_sum": float(abs(array.astype("float64")).sum()),
        }
    return {"sha256": digest.hexdigest(), "fields": fields}


def run_probe(
    *,
    source_root: Path,
    extension_root: Path | None,
    dataset_root: Path,
    initial_state: Path,
    artifact_root: Path,
    requested_iterations: int,
    warmup_iterations: int,
    progress_interval: int,
    inactivity_timeout_seconds: float,
    telemetry_mode: str,
    compiler_variant: str,
) -> dict[str, Any]:
    if not 0 <= warmup_iterations < requested_iterations <= PUBLISHED_ITERATIONS:
        raise TrainingProbeError("require 0 <= warmup < requested <= 30000 iterations")
    if progress_interval <= 0:
        raise TrainingProbeError("progress interval must be positive")
    if telemetry_mode not in {"published", "cuda_opt"}:
        raise TrainingProbeError("telemetry mode must be published or cuda_opt")
    if compiler_variant not in COMPILER_VARIANT_FLAGS:
        raise TrainingProbeError(
            "compiler variant must be gpu_base, 1000, 0100, or 1100"
        )
    source_path = source_root / "train.py"
    if not source_path.is_file() or not dataset_root.is_dir() or not initial_state.is_file():
        raise TrainingProbeError("source, dataset, or initial state is unavailable")
    artifact_root.mkdir(parents=True, exist_ok=False)
    model_root = artifact_root / "model"
    model_root.mkdir()
    source = source_path.read_text(encoding="utf-8")
    if telemetry_mode == "cuda_opt":
        training_source, transformations = render_training_overlay(source)
        training_path = artifact_root / OVERLAY_FILENAME
        training_path.write_text(training_source, encoding="utf-8")
    else:
        training_source = source
        transformations = []
        training_path = source_path

    with _working_directory(source_root), _compiler_variant_environment(compiler_variant):
        training = _load_training(training_path, source_root, extension_root)
        training.safe_state(True)
        torch = training.torch
        if not torch.cuda.is_available():
            raise TrainingProbeError("CUDA is unavailable")

        dataset = type("DatasetArgs", (), {
            "source_path": str(dataset_root),
            "model_path": str(model_root),
            "data_device": "cuda",
            "eval": True,
            "ply_path": str(initial_state),
            "scale_min": 0.0005,
            "scale_max": 0.5,
        })()
        optimization = training.OptimizationParams(argparse.ArgumentParser())
        optimization.iterations = PUBLISHED_ITERATIONS
        pipeline = type("PipelineArgs", (), {
            "compute_cov3D_python": False,
            "debug": False,
        })()

        extension = sys.modules.get("xray_gaussian_rasterization_voxelization._C")
        extension_path = getattr(extension, "__file__", None)
        if extension_root is not None:
            if extension_path is None:
                raise TrainingProbeError("official training did not load the CUDA extension")
            try:
                Path(extension_path).resolve().relative_to(extension_root.resolve())
            except ValueError as error:
                raise TrainingProbeError(
                    f"CUDA extension was loaded outside --extension-root: {extension_path}"
                ) from error

        original_initialize = training.initialize_gaussian
        original_report = training.training_report
        measurement_start = torch.cuda.Event(enable_timing=True)
        measurement_end = torch.cuda.Event(enable_timing=True)
        final_metrics: dict[str, float] = {}
        final_gaussians = 0
        final_state: dict[str, Any] | None = None
        completed_iterations = 0
        cuda_elapsed_ms: float | None = None
        watchdog = GpuWatchdog(inactivity_timeout_seconds)

        def initialize_from_checked_state(gaussians: Any, _dataset: Any, _loaded: Any = None) -> None:
            gaussians.load_ply(str(initial_state))
            gaussians.max_radii2D = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
            gaussians.spatial_lr_scale = 1.0

        def report(
            _writer: Any,
            iteration: int,
            metrics: dict[str, Any],
            _elapsed: Any,
            _testing: Any,
            scene: Any,
            _render: Any,
            _query: Any,
            ) -> None:
            nonlocal completed_iterations, cuda_elapsed_ms, final_gaussians, final_state
            completed_iterations = iteration
            watchdog.progress()
            if iteration == warmup_iterations:
                measurement_start.record()
            if iteration % progress_interval == 0 or iteration == requested_iterations:
                print(json.dumps({
                    "phase": "training",
                    "completed_iterations": iteration,
                    "requested_iterations": requested_iterations,
                }, sort_keys=True), file=sys.stderr, flush=True)
            if iteration != requested_iterations:
                return
            measurement_end.record()
            measurement_end.synchronize()
            cuda_elapsed_ms = float(measurement_start.elapsed_time(measurement_end))
            final_metrics.update({
                key: float(value.detach().item()) if hasattr(value, "detach") else float(value)
                for key, value in metrics.items()
                if key.startswith("loss_")
            })
            final_gaussians = int(scene.gaussians.get_xyz.shape[0])
            final_state = _state_fingerprint(scene.gaussians)
            raise StopAfterPrefix

        if warmup_iterations == 0:
            measurement_start.record()
        training.initialize_gaussian = initialize_from_checked_state
        training.training_report = report
        wall_started = time.perf_counter()
        watchdog.start()
        try:
            try:
                training.training(dataset, optimization, pipeline, None, [], [], [], None)
            except StopAfterPrefix:
                pass
        except KeyboardInterrupt as error:
            if watchdog.timed_out:
                raise TrainingProbeError(
                    f"GPU and iteration progress were both idle for {inactivity_timeout_seconds:g} seconds"
                ) from error
            raise
        finally:
            watchdog.stop()
            training.initialize_gaussian = original_initialize
            training.training_report = original_report
        wall_seconds = time.perf_counter() - wall_started

    if completed_iterations != requested_iterations or cuda_elapsed_ms is None or final_state is None:
        raise TrainingProbeError("official training loop did not complete the requested prefix")
    measured_iterations = requested_iterations - warmup_iterations
    seconds_per_iteration = cuda_elapsed_ms / 1000.0 / measured_iterations
    return {
        "schema_version": "gala-r2-gaussian-official-training-probe-v1",
        "status": "passed",
        "result_scope": "development_prefix_projection",
        "formal_performance_eligible": False,
        "workload": {
            "model": "R2-Gaussian",
            "dataset": "Chest",
            "compiler_variant": compiler_variant,
            "comparison_baseline": "gpu_base",
            "published_iterations": PUBLISHED_ITERATIONS,
            "requested_iterations": requested_iterations,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "included_training_operations": [
                "projection_forward", "l1", "dssim", "tv", "backward",
                "adaptive_control", "densification", "adam",
            ],
            "excluded_prefix_endpoint_operations": ["evaluation", "save", "checkpoint"],
        },
        "measurement": {
            "cuda_elapsed_ms": cuda_elapsed_ms,
            "wall_seconds_including_initialization": wall_seconds,
            "seconds_per_iteration": seconds_per_iteration,
            "projected_30000_training_seconds": seconds_per_iteration * PUBLISHED_ITERATIONS,
            "final_gaussians": final_gaussians,
            "final_losses": final_metrics,
            "final_state": final_state,
        },
        "gpu": _gpu_summary(watchdog.samples),
        "watchdog": {
            "timeout_seconds": inactivity_timeout_seconds,
            "timed_out": watchdog.timed_out,
        },
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root.resolve()),
            "source_train_sha256": _sha256_text(source),
            "dataset_root": str(dataset_root.resolve()),
            "initial_state": str(initial_state.resolve()),
            "telemetry_mode": telemetry_mode,
            "compiler_variant": compiler_variant,
            "compiler_mechanism": {
                "gpu_base": "none",
                "1000": "query-backward-warp-aggregation",
                "0100": "semantic-backward-warp-aggregation",
                "1100": "query-and-semantic-backward-warp-aggregation",
            }[compiler_variant],
            "cuda_extension_path": extension_path,
            "cuda_extension_root": (
                str(extension_root.resolve()) if extension_root is not None else None
            ),
            "training_path": str(training_path.resolve()),
            "training_source_sha256": _sha256_text(training_source),
            "transformations": transformations,
            "hash_policy": "record_only_no_hash_rejection",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--extension-root", type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--initial-state", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--inactivity-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--telemetry-mode", choices=("published", "cuda_opt"), default="cuda_opt")
    parser.add_argument(
        "--compiler-variant", choices=tuple(COMPILER_VARIANT_FLAGS), default="gpu_base"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not already exist")
    try:
        record = run_probe(
            source_root=args.source_root.resolve(),
            extension_root=(
                args.extension_root.resolve() if args.extension_root is not None else None
            ),
            dataset_root=args.dataset_root.resolve(),
            initial_state=args.initial_state.resolve(),
            artifact_root=args.artifact_root.resolve(),
            requested_iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            progress_interval=args.progress_interval,
            inactivity_timeout_seconds=args.inactivity_timeout_seconds,
            telemetry_mode=args.telemetry_mode,
            compiler_variant=args.compiler_variant,
        )
    except (OSError, ValueError, TrainingProbeError) as error:
        print(f"R2-Gaussian training probe failed: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["measurement"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
