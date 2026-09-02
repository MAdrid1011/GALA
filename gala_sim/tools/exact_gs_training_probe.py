"""Measure Exact-GS compiler variants through its uninstrumented training loop."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

from gala_sim.ablation.anchors import compiler_target_speedups
from gala_sim.adapters.fact_low_memory import install_exact_low_memory_overlay
from gala_sim.gpu_measurement import (
    build_gpu_compiler_measurement,
    platform_identity_from_isolation,
)
from gala_sim.gpu_coverage import analyze_gpu_compiler_coverage
from gala_sim.tools.gpu_probe_watchdog import (
    GpuWatchdog,
    ensure_host_memory_reserve,
    gpu_summary as _gpu_summary,
)
from gala_sim.tools.fact_gs_training_probe import (
    ProbeObservation,
    _snapshot_gradients,
    _snapshot_state,
    compare_states,
)
from gala_sim.tools.gpu_observation import GpuObservationError, ensure_gpu_isolated
from gala_sim.tools.r2_gaussian_training_probe import (
    _compiler_variant_environment,
)


PUBLISHED_ITERATIONS = 30_000
COMPILER_VARIANTS = ("gpu_base", "1000", "0100", "1100")
OVERLAY_FILENAME = "train_exact_cuda_opt_probe.py"
INCLUDED_TRAINING_OPERATIONS = [
    "exact_projection_forward",
    "l2",
    "dssim",
    "backward",
    "adaptive_control",
    "densification",
    "adam",
]
OVERLAY_TRANSFORMS = (
    (
        "remove-published-iteration-events",
        """    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
""",
        "",
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
        """            if iteration % 100 == 0:
                print(iteration,"loss: ",loss['total'].item())
""",
        "",
    ),
    (
        "batch-loss-telemetry",
        """            metrics = {}
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
        """            metrics = {}
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


class ExactTrainingProbeError(RuntimeError):
    """The Exact-GS compiler probe could not produce comparable evidence."""


class _PrefixComplete(RuntimeError):
    """Stop after the measured optimizer transaction, before endpoint work."""


@dataclass
class _StageInterval:
    stage: str
    start: Any
    end: Any


class _ExactStageProfiler:
    """Optional CUDA-event characterization of a measured Exact-GS prefix."""

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self._active = False
        self._intervals: list[_StageInterval] = []
        self._patches: list[tuple[Any, str, Any]] = []

    def begin(self) -> None:
        self._active = True

    def end(self) -> None:
        self._active = False

    def measure(self, stage: str, call: Any) -> Any:
        if not self._active:
            return call()
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return call()
        finally:
            end.record()
            self._intervals.append(_StageInterval(stage, start, end))

    def install(self, training: ModuleType) -> None:
        self._patch(
            training, "render_exact",
            lambda original: self._wrap(original, "projection_forward"),
        )
        self._patch(
            training, "l2_loss",
            lambda original: self._wrap(original, "projection_loss"),
        )
        self._patch(
            training, "ssim",
            lambda original: self._wrap(original, "projection_loss"),
        )
        self._patch(
            self._torch.Tensor, "backward",
            lambda original: self._wrap(original, "backward"),
        )
        self._patch(
            training.GaussianModel, "add_densification_stats",
            lambda original: self._wrap(original, "adaptive_control"),
        )
        self._patch(
            training.GaussianModel, "densify_and_prune",
            lambda original: self._wrap(original, "densification"),
        )

    def restore(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def result(self) -> dict[str, Any]:
        self._torch.cuda.synchronize()
        records = [
            {
                "stage": item.stage,
                "milliseconds": float(item.start.elapsed_time(item.end)),
            }
            for item in self._intervals
        ]
        grouped: dict[str, list[float]] = {}
        for record in records:
            grouped.setdefault(str(record["stage"]), []).append(
                float(record["milliseconds"])
            )
        return {
            "records": records,
            "summaries": {
                stage: {
                    "call_count": len(values),
                    "total_ms": sum(values),
                    "median_ms": statistics.median(values),
                }
                for stage, values in sorted(grouped.items())
            },
        }

    def _patch(self, owner: Any, name: str, factory: Any) -> None:
        original = getattr(owner, name)
        setattr(owner, name, factory(original))
        self._patches.append((owner, name, original))

    def _wrap(self, original: Any, stage: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self.measure(stage, lambda: original(*args, **kwargs))
        return wrapped


def render_training_overlay(source: str) -> tuple[str, list[dict[str, str]]]:
    transformed = source
    manifest: list[dict[str, str]] = []
    for transform_id, expected, replacement in OVERLAY_TRANSFORMS:
        count = transformed.count(expected)
        if count != 1:
            raise ExactTrainingProbeError(
                f"overlay transform {transform_id} expected one source fragment; found {count}"
            )
        transformed = transformed.replace(expected, replacement, 1)
        manifest.append({
            "id": transform_id,
            "source_fragment_sha256": hashlib.sha256(
                expected.encode("utf-8")
            ).hexdigest(),
        })
    if "torch.cuda.synchronize()" in transformed or ".item()" in transformed:
        raise ExactTrainingProbeError(
            "Exact-GS overlay retains a per-iteration host synchronization"
        )
    return transformed, manifest


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_training(path: Path, source_root: Path, extension_root: Path) -> ModuleType:
    search_paths = [str(extension_root), str(source_root)]
    sys.path[:0] = search_paths
    try:
        spec = importlib.util.spec_from_file_location("gala_exact_training_probe", path)
        if spec is None or spec.loader is None:
            raise ExactTrainingProbeError("cannot load generated Exact-GS overlay")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for search_path in search_paths:
            sys.path.remove(search_path)


def _validate_inputs(
    source_root: Path,
    extension_root: Path,
    dataset_root: Path,
    initial_state: Path,
    artifact_root: Path,
    requested_iterations: int,
    warmup_iterations: int,
    inactivity_timeout_seconds: float,
) -> None:
    if not (source_root / "train.py").is_file():
        raise ExactTrainingProbeError("Exact-GS train.py is unavailable")
    if not (extension_root / "exact_gaussian_rasterization").is_dir():
        raise ExactTrainingProbeError("Exact-GS compiler overlay is unavailable")
    if not dataset_root.is_dir() or not initial_state.is_file():
        raise ExactTrainingProbeError("Exact-GS dataset or initialization is unavailable")
    if artifact_root.exists():
        raise ExactTrainingProbeError(f"probe artifact path already exists: {artifact_root}")
    if not 0 <= warmup_iterations < requested_iterations <= PUBLISHED_ITERATIONS:
        raise ExactTrainingProbeError(
            "require 0 <= warmup < requested <= 30000 iterations"
        )
    if inactivity_timeout_seconds <= 0:
        raise ExactTrainingProbeError("inactivity timeout must be positive")


def run_probe(
    *,
    source_root: Path,
    extension_root: Path,
    dataset_root: Path,
    initial_state: Path,
    artifact_root: Path,
    dataset_id: str,
    requested_iterations: int,
    warmup_iterations: int,
    inactivity_timeout_seconds: float,
    compiler_variant: str,
    profile_stages: bool = False,
) -> ProbeObservation:
    _validate_inputs(
        source_root, extension_root, dataset_root, initial_state, artifact_root,
        requested_iterations, warmup_iterations, inactivity_timeout_seconds,
    )
    if compiler_variant not in COMPILER_VARIANTS:
        raise ExactTrainingProbeError(f"unsupported compiler variant: {compiler_variant}")
    try:
        ensure_host_memory_reserve()
        gpu_isolation = ensure_gpu_isolated()
    except (GpuObservationError, RuntimeError) as error:
        raise ExactTrainingProbeError(str(error)) from error
    artifact_root.mkdir(parents=True)
    model_root = artifact_root / "model"
    model_root.mkdir()
    source_path = source_root / "train.py"
    source = source_path.read_text(encoding="utf-8")
    training_source, transformations = render_training_overlay(source)
    training_path = artifact_root / OVERLAY_FILENAME
    training_path.write_text(training_source, encoding="utf-8")

    final_arrays: dict[str, Any] | None = None
    final_gradients: dict[str, Any] | None = None
    stage_profile: dict[str, Any] | None = None
    cuda_elapsed_ms: float | None = None
    completed_iterations = 0
    with _working_directory(source_root), _compiler_variant_environment(compiler_variant):
        training = _load_training(training_path, source_root, extension_root)
        training.safe_state(True)
        torch = training.torch
        if not torch.cuda.is_available():
            raise ExactTrainingProbeError("CUDA is unavailable")
        extension = sys.modules.get("exact_gaussian_rasterization._C")
        extension_path = getattr(extension, "__file__", None)
        if extension_path is None:
            raise ExactTrainingProbeError("Exact-GS overlay extension was not loaded")
        try:
            Path(extension_path).resolve().relative_to(extension_root.resolve())
        except ValueError as error:
            raise ExactTrainingProbeError(
                f"Exact-GS extension was loaded outside the overlay: {extension_path}"
            ) from error

        dataset = type("DatasetArgs", (), {
            "source_path": str(dataset_root),
            "model_path": str(model_root),
            "data_device": "cuda",
            "eval": False,
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

        original_initialize = training.initialize_gaussian
        original_create_volume = training.Scene.creatVol_gt
        original_load_volume = training.Scene.loadVol_gt
        original_report = training.training_report
        original_training_setup = training.GaussianModel.training_setup
        low_memory_overlay = install_exact_low_memory_overlay()
        measurement_start = torch.cuda.Event(enable_timing=True)
        measurement_end = torch.cuda.Event(enable_timing=True)
        watchdog = GpuWatchdog(inactivity_timeout_seconds, owner_pid=os.getpid())
        profiler = _ExactStageProfiler(torch) if profile_stages else None
        gradient_tensors: dict[str, Any] | None = None
        optimizer_steps = 0

        def initialize(gaussians: Any, args: Any, load_gt: bool = True,
                       loaded_iter: int | None = None) -> Any:
            if load_gt:
                return loaded_iter
            return original_initialize(
                gaussians, args, load_gt=False, loaded_iter=loaded_iter,
            )

        def setup(gaussians: Any, options: Any) -> None:
            nonlocal gradient_tensors, optimizer_steps
            original_training_setup(gaussians, options)
            original_step = gaussians.optimizer.step

            def step(*args: Any, **kwargs: Any) -> Any:
                nonlocal gradient_tensors, optimizer_steps
                optimizer_steps += 1
                if optimizer_steps == requested_iterations:
                    names = {
                        "xyz": gaussians._xyz.grad,
                        "scaling": gaussians._scaling.grad,
                        "rotation": gaussians._rotation.grad,
                        "density": gaussians._density.grad,
                    }
                    if any(value is None for value in names.values()):
                        raise ExactTrainingProbeError(
                            "Exact-GS final optimizer step has missing gradients"
                        )
                    gradient_tensors = {
                        name: value.detach().clone() for name, value in names.items()
                    }
                if profiler is None:
                    return original_step(*args, **kwargs)
                return profiler.measure(
                    "optimizer", lambda: original_step(*args, **kwargs)
                )

            gaussians.optimizer.step = step

        def report(
            _writer: Any,
            iteration: int,
            _metrics: Mapping[str, Any],
            _elapsed: Any,
            _testing: Any,
            scene: Any,
            _render: Any,
            _query: Any,
        ) -> None:
            nonlocal completed_iterations, cuda_elapsed_ms, final_arrays, final_gradients
            completed_iterations = iteration
            watchdog.progress()
            if iteration == warmup_iterations:
                measurement_start.record()
                if profiler is not None:
                    profiler.begin()
            if iteration != requested_iterations:
                return
            measurement_end.record()
            if profiler is not None:
                profiler.end()
            measurement_end.synchronize()
            cuda_elapsed_ms = float(measurement_start.elapsed_time(measurement_end))
            final_arrays = _snapshot_state(scene.gaussians)
            if gradient_tensors is None:
                raise ExactTrainingProbeError("Exact-GS final gradients were not captured")
            final_gradients = {
                name: value.cpu().contiguous().numpy().copy()
                for name, value in gradient_tensors.items()
            }
            raise _PrefixComplete

        if warmup_iterations == 0:
            measurement_start.record()
            if profiler is not None:
                profiler.begin()
        training.initialize_gaussian = initialize
        training.Scene.creatVol_gt = lambda _scene, _query: None
        training.Scene.loadVol_gt = lambda _scene: None
        training.training_report = report
        training.GaussianModel.training_setup = setup
        if profiler is not None:
            profiler.install(training)
        wall_started = time.monotonic()
        watchdog.start()
        try:
            try:
                training.training(
                    dataset, optimization, pipeline, None, [], [], [], None, "Exact_GS",
                )
            except _PrefixComplete:
                pass
        except KeyboardInterrupt as error:
            if watchdog.failure is not None:
                raise ExactTrainingProbeError(watchdog.failure) from error
            raise
        finally:
            watchdog.stop()
            training.initialize_gaussian = original_initialize
            training.Scene.creatVol_gt = original_create_volume
            training.Scene.loadVol_gt = original_load_volume
            training.training_report = original_report
            training.GaussianModel.training_setup = original_training_setup
            if profiler is not None:
                profiler.end()
                try:
                    stage_profile = profiler.result()
                finally:
                    profiler.restore()
            low_memory_overlay.restore()
        wall_seconds = time.monotonic() - wall_started

    if completed_iterations != requested_iterations or cuda_elapsed_ms is None:
        raise ExactTrainingProbeError("Exact-GS official prefix did not complete")
    if final_arrays is None or final_gradients is None:
        raise ExactTrainingProbeError("Exact-GS prefix did not expose final state")
    if watchdog.failure is not None:
        raise ExactTrainingProbeError(watchdog.failure)
    record = {
        "schema_version": "gala-exact-gs-official-training-probe-v1",
        "status": "passed",
        "result_scope": "development_prefix_projection",
        "formal_performance_eligible": False,
        "workload": {
            "model": "Exact-GS",
            "model_id": "exact_gs",
            "dataset": dataset_id,
            "dataset_id": dataset_id,
            "compiler_variant": compiler_variant,
            "comparison_baseline": "gpu_base",
            "requested_iterations": requested_iterations,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": requested_iterations - warmup_iterations,
            "included_training_operations": INCLUDED_TRAINING_OPERATIONS,
            "excluded_prefix_endpoint_operations": ["evaluation", "save", "checkpoint"],
        },
        "measurement": {
            "cuda_elapsed_ms": cuda_elapsed_ms,
            "wall_seconds_including_initialization": wall_seconds,
            "seconds_per_iteration": (
                cuda_elapsed_ms / 1000.0 / (requested_iterations - warmup_iterations)
            ),
            "stage_profile": stage_profile,
        },
        "gpu": {
            **_gpu_summary(watchdog.samples),
            "isolation": gpu_isolation,
            "external_compute_processes": watchdog.external_compute_processes,
        },
        "watchdog": {
            "timeout_seconds": inactivity_timeout_seconds,
            "timed_out": watchdog.timed_out,
            "failure": watchdog.failure,
        },
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "compiler_variant": compiler_variant,
            "training_overlay": transformations,
            "extension": str(Path(extension_path).resolve()),
            "hash_policy": "record_only_no_hash_rejection",
        },
    }
    return ProbeObservation(record, final_arrays, final_gradients)


def summarize_matrix(
    observations: Mapping[str, Sequence[ProbeObservation]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if tuple(observations) != COMPILER_VARIANTS:
        raise ExactTrainingProbeError("matrix must use the four canonical compiler variants")
    counts = {len(samples) for samples in observations.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise ExactTrainingProbeError("every compiler variant needs equal nonzero repeats")
    baseline = observations["gpu_base"]
    reference_workload = dict(baseline[0].record["workload"])
    reference_workload.pop("compiler_variant")
    summary: dict[str, Any] = {}
    for variant, samples in observations.items():
        checks = []
        elapsed = []
        for index, sample in enumerate(samples):
            workload = dict(sample.record["workload"])
            workload.pop("compiler_variant")
            if workload != reference_workload:
                raise ExactTrainingProbeError(
                    f"{variant} changed the comparable official workload"
                )
            check = compare_states(
                baseline[index].gradient_arrays,
                sample.gradient_arrays,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if not check["passed"]:
                raise ExactTrainingProbeError(
                    f"{variant} changed gradients beyond tolerance"
                )
            checks.append(check)
            elapsed.append(float(sample.record["measurement"]["cuda_elapsed_ms"]))
        summary[variant] = {
            "bits": variant,
            "comparison_baseline": "gpu_base",
            "median_cuda_elapsed_ms": statistics.median(elapsed),
            "minimum_cuda_elapsed_ms": min(elapsed),
            "maximum_cuda_elapsed_ms": max(elapsed),
            "gradient_equivalence": checks,
            "samples": [sample.record for sample in samples],
        }
    baseline_ms = summary["gpu_base"]["median_cuda_elapsed_ms"]
    targets = compiler_target_speedups()
    for variant, item in summary.items():
        speedup = baseline_ms / item["median_cuda_elapsed_ms"]
        item["speedup_vs_gpu_base"] = speedup
        item["target_speedup_vs_gpu_base"] = targets.get(variant)
        item["target_met"] = variant == "gpu_base" or speedup >= targets[variant]
    eligible = all(
        sample.record.get("gpu", {}).get("isolation", {}).get("status") == "isolated"
        and not sample.record.get("gpu", {}).get("external_compute_processes")
        for samples in observations.values()
        for sample in samples
    )
    return {
        "schema_version": "gala-exact-gs-gpu-compiler-matrix-v1",
        "result_scope": "development_prefix_projection",
        "formal_performance_eligible": False,
        "performance_comparison_eligible": eligible,
        "comparison_baseline": "gpu_base",
        "summary": summary,
    }


def run_matrix(
    *,
    source_root: Path,
    extension_root: Path,
    dataset_root: Path,
    initial_state: Path,
    dataset_id: str,
    output: Path,
    requested_iterations: int,
    warmup_iterations: int,
    repeats: int,
    inactivity_timeout_seconds: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    profile_stages: bool = False,
) -> dict[str, Any]:
    if output.exists() or repeats <= 0:
        raise ExactTrainingProbeError("matrix output must be new and repeats must be positive")
    observations: dict[str, list[ProbeObservation]] = {
        variant: [] for variant in COMPILER_VARIANTS
    }
    try:
        for repeat in range(repeats):
            order = COMPILER_VARIANTS[repeat:] + COMPILER_VARIANTS[:repeat]
            for variant in order:
                observation = run_probe(
                    source_root=source_root,
                    extension_root=extension_root,
                    dataset_root=dataset_root,
                    initial_state=initial_state,
                    artifact_root=output / "artifacts" / variant / f"repeat_{repeat + 1}",
                    dataset_id=dataset_id,
                    requested_iterations=requested_iterations,
                    warmup_iterations=warmup_iterations,
                    inactivity_timeout_seconds=inactivity_timeout_seconds,
                    compiler_variant=variant,
                    profile_stages=False,
                )
                observations[variant].append(observation)
                sample_path = output / "samples" / variant / f"repeat_{repeat + 1}.json"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                sample_path.write_text(
                    json.dumps(observation.record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                gc.collect()
        result = summarize_matrix(
            observations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if profile_stages:
            diagnostic = run_probe(
                source_root=source_root,
                extension_root=extension_root,
                dataset_root=dataset_root,
                initial_state=initial_state,
                artifact_root=output / "artifacts" / "gpu_base_stage_profile",
                dataset_id=dataset_id,
                requested_iterations=requested_iterations,
                warmup_iterations=warmup_iterations,
                inactivity_timeout_seconds=inactivity_timeout_seconds,
                compiler_variant="gpu_base",
                profile_stages=True,
            )
            diagnostic_path = output / "gpu-base-stage-profile.json"
            diagnostic_path.write_text(
                json.dumps(diagnostic.record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stage_profile = diagnostic.record["measurement"]["stage_profile"]
            if not isinstance(stage_profile, Mapping):
                raise ExactTrainingProbeError("Exact-GS stage profile is missing")
            result["compiler_coverage_bounds"] = analyze_gpu_compiler_coverage(
                total_gpu_ms=float(
                    diagnostic.record["measurement"]["cuda_elapsed_ms"]
                ),
                stage_profile=stage_profile,
                coverable_stages={
                    "1000": ("backward",),
                    "0100": ("backward",),
                    "1100": ("backward",),
                },
            )
        (output / "matrix.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        summary = result["summary"]
        standard = build_gpu_compiler_measurement(
            model_id="exact_gs",
            dataset_id=dataset_id,
            iteration_range=(warmup_iterations + 1, requested_iterations),
            included_training_operations=INCLUDED_TRAINING_OPERATIONS,
            median_gpu_ms={
                variant: float(summary[variant]["median_cuda_elapsed_ms"])
                for variant in COMPILER_VARIANTS
            },
            sample_counts={
                variant: len(observations[variant]) for variant in COMPILER_VARIANTS
            },
            source_probe={
                "schema_version": result["schema_version"],
                "result_scope": result["result_scope"],
                "matrix": "matrix.json",
            },
            gpu_isolated=bool(result["performance_comparison_eligible"]),
            same_workload_across_variants=True,
            numerical_equivalence_passed=True,
            gpu_platform=platform_identity_from_isolation(
                observations["gpu_base"][0].record["gpu"].get("isolation")
            ),
        )
        (output / "gpu-compiler-measurement.json").write_text(
            json.dumps(standard, indent=2, sort_keys=False) + "\n", encoding="utf-8",
        )
        return result
    finally:
        observations.clear()
        gc.collect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", choices=("chest", "walnut", "hdtomo_usb"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inactivity-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--profile-stages", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            source_root=args.source_root.resolve(),
            extension_root=args.extension_root.resolve(),
            dataset_root=args.dataset_root.resolve(),
            initial_state=args.initial_state.resolve(),
            dataset_id=args.dataset_id,
            output=args.output.resolve(),
            requested_iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            repeats=args.repeats,
            inactivity_timeout_seconds=args.inactivity_timeout_seconds,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            profile_stages=args.profile_stages,
        )
    except (OSError, ValueError, ExactTrainingProbeError) as error:
        print(f"Exact-GS compiler matrix failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output": str(args.output.resolve()),
        "speedup_vs_gpu_base": {
            variant: item["speedup_vs_gpu_base"]
            for variant, item in result["summary"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
