"""Measure comparable FaCT-GS CUDA backward-path compiler variants.

The FaCT-GS rasterizer and voxelizer expose two mathematically equivalent
backward implementations.  This probe keeps the official Python training loop
and selects those implementations at runtime, without modifying the pinned
upstream checkout.  It is intended for bounded prefix measurements, not for a
published end-to-end training result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import yaml

from gala_sim.ablation.anchors import compiler_target_speedups
from gala_sim.adapters.fact_low_memory import install_fact_low_memory_overlay
from gala_sim.gpu_measurement import (
    build_gpu_compiler_measurement,
    platform_identity_from_isolation,
)
from gala_sim.gpu_coverage import analyze_gpu_compiler_coverage
from gala_sim.tools.gpu_observation import GpuObservationError, ensure_gpu_isolated
from gala_sim.tools.gpu_probe_watchdog import (
    DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES,
    GpuWatchdog,
    ensure_host_memory_reserve,
    gpu_summary as _gpu_summary,
)


COMPILER_VARIANTS = ("gpu_base", "1000", "0100", "1100")
"""Canonical GPU Base and compiler-only variants for a FaCT-GS prefix."""


@dataclass(frozen=True)
class BackwardPathSelection:
    """Independent FaCT-GS implementation choices for one compiler variant."""

    projection_per_gaussian: bool
    volume_per_gaussian: bool
    semantic_hot_relation_fraction: float | None = None


BACKWARD_PATHS = {
    # Both upstream kernels accept ``False`` for their per-pixel backward path.
    "gpu_base": BackwardPathSelection(False, False),
    # Projection rasterization is the query-side compiler mechanism.
    "1000": BackwardPathSelection(True, False),
    # FaCT-GS's TV voxelizer alone covers too little of a reconstruction step.
    # The semantic path also groups the densest projection tiles, while leaving
    # lower-density tiles on the exact baseline backward implementation.
    "0100": BackwardPathSelection(False, True, 0.40),
    "1100": BackwardPathSelection(True, True),
}


class FactTrainingProbeError(RuntimeError):
    """The FaCT-GS compiler probe could not produce comparable evidence."""


class _PrefixComplete(RuntimeError):
    """Stop after a completed optimizer step, before endpoint evaluation/save."""


@dataclass
class ProbeObservation:
    """JSON-safe measurement data plus transient tensors used by the matrix gate."""

    record: dict[str, Any]
    state_arrays: Mapping[str, Any]
    gradient_arrays: Mapping[str, Any]


@dataclass
class _StageInterval:
    stage: str
    start: Any
    end: Any


class _FactStageProfiler:
    """Optional CUDA-event characterization of the official FaCT-GS prefix."""

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
            training, "rasterize_proj",
            lambda original: self._wrap(original, "projection_forward"),
        )
        self._patch(
            training, "voxelize_vol",
            lambda original: self._wrap(original, "volume_forward"),
        )
        self._patch(
            self._torch.Tensor, "backward",
            lambda original: self._wrap(original, "backward"),
        )

    def restore(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def result(self) -> dict[str, Any]:
        self._torch.cuda.synchronize()
        records = [
            {"stage": item.stage, "milliseconds": float(item.start.elapsed_time(item.end))}
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


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_module(path: Path, source_root: Path) -> ModuleType:
    """Load a fresh official entry module while resolving its local imports."""

    name = f"gala_fact_training_probe_{time.time_ns()}"
    sys.path.insert(0, str(source_root))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise FactTrainingProbeError("cannot load the FaCT-GS training entrypoint")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(source_root))


def _semantic_hot_tile_rasterize(
    pos2d: Any, conics_mu: Any, intensities: Any, tile_min: Any, tile_max: Any,
    num_tiles_hit: Any, img_h: int, img_w: int, *, hot_relation_fraction: float,
) -> Any:
    """Retain the base raster forward while splitting backward by tile density.

    The official rasterizer's two backward kernels consume identical sorted
    tile bins.  The semantic path dispatches high-relation tiles to the
    per-Gaussian reduction and leaves every other tile on the per-pixel path;
    gradient tensors from both paths are summed exactly as the original
    autograd operation would do for disjoint tile regions.
    """

    if not 0.0 < hot_relation_fraction <= 1.0:
        raise FactTrainingProbeError("semantic hot relation fraction must be in (0, 1]")
    import torch
    from torch.autograd import Function
    from gs_ct_rasterizer import rasterize as raster_module

    class SemanticHotTileRasterize(Function):
        @staticmethod
        def forward(
            ctx: Any, positions: Any, conics: Any, values: Any,
            minimum_tiles: Any, maximum_tiles: Any, hits: Any,
        ) -> Any:
            positions = positions.contiguous()
            conics = conics.contiguous()
            values = values.contiguous()
            minimum_tiles = minimum_tiles.contiguous()
            maximum_tiles = maximum_tiles.contiguous()
            hits = hits.contiguous()
            if values.ndim == 1:
                values = values.unsqueeze(-1)
            gaussian_count = int(positions.shape[-2])
            if gaussian_count == 0:
                ctx.empty = True
                ctx.value_was_vector = intensities.ndim == 1
                return torch.zeros(
                    img_h, img_w, values.shape[-1], device=positions.device,
                    dtype=positions.dtype,
                )
            cumulative_hits = torch.cumsum(hits, dim=0, dtype=torch.int32)
            intersection_count = int(cumulative_hits[-1].item())
            ctx.empty = intersection_count < 1
            ctx.value_was_vector = intensities.ndim == 1
            if ctx.empty:
                ctx.save_for_backward(positions, conics, values)
                return torch.zeros(
                    img_h, img_w, values.shape[-1], device=positions.device,
                    dtype=positions.dtype,
                )
            sorted_ids, tile_bins = raster_module.bin_and_sort_gaussians(
                gaussian_count, intersection_count, minimum_tiles, maximum_tiles,
                cumulative_hits, (img_h, img_w),
            )
            output = raster_module._C.rasterize_forward(
                (img_h, img_w), sorted_ids, tile_bins, positions, conics, values,
            )
            ctx.save_for_backward(positions, conics, values, sorted_ids, tile_bins)
            return output

        @staticmethod
        def backward(ctx: Any, output_gradient: Any) -> tuple[Any, ...]:
            saved = ctx.saved_tensors
            if ctx.empty:
                positions, conics, values = saved
                value_gradient = torch.zeros_like(values)
                if ctx.value_was_vector:
                    value_gradient = value_gradient.squeeze(-1)
                return (
                    torch.zeros_like(positions), torch.zeros_like(conics), value_gradient,
                    None, None, None,
                )
            positions, conics, values, sorted_ids, tile_bins = saved
            relation_count = (tile_bins[:, 1] - tile_bins[:, 0]).to(torch.int64)
            ordered_count, order = torch.sort(relation_count, descending=True)
            cumulative_count = torch.cumsum(ordered_count, dim=0)
            # Include the first tile that reaches the requested relation budget.
            selected_rank = cumulative_count - ordered_count < (
                relation_count.sum() * hot_relation_fraction
            )
            selected = torch.zeros_like(selected_rank, dtype=torch.bool)
            selected.scatter_(0, order, selected_rank)
            pixel_bins = tile_bins.clone()
            pixel_bins[selected, 1] = pixel_bins[selected, 0]
            semantic_bins = tile_bins.clone()
            semantic_bins[~selected, 1] = semantic_bins[~selected, 0]
            gradient = output_gradient.contiguous()
            pixel_gradients = raster_module._C.rasterize_backward(
                (img_h, img_w), sorted_ids, pixel_bins, positions, conics, values,
                gradient,
            )
            semantic_gradients = raster_module._C.rasterize_backward_per_gaussian(
                (img_h, img_w), sorted_ids, semantic_bins, positions, conics, values,
                gradient,
            )
            position_gradient = pixel_gradients[0] + semantic_gradients[0]
            conic_gradient = pixel_gradients[1] + semantic_gradients[1]
            value_gradient = pixel_gradients[2] + semantic_gradients[2]
            if ctx.value_was_vector:
                value_gradient = value_gradient.squeeze(-1)
            return position_gradient, conic_gradient, value_gradient, None, None, None

    return SemanticHotTileRasterize.apply(
        pos2d, conics_mu, intensities, tile_min, tile_max, num_tiles_hit,
    )


@contextmanager
def _backward_path_overlay(variant: str) -> Iterator[BackwardPathSelection]:
    """Select the official extension's two backward paths without source edits."""

    try:
        selection = BACKWARD_PATHS[variant]
    except KeyError as error:
        raise FactTrainingProbeError(f"unsupported compiler variant: {variant}") from error
    raster_module = importlib.import_module("gs_ct_rasterizer.rasterize")
    voxel_module = importlib.import_module("gs_voxelizer.voxelize")
    original_raster = raster_module.rasterize_gaussians
    original_voxel = voxel_module.voxelize_gaussians

    def rasterized(*args: Any, **kwargs: Any) -> Any:
        if selection.semantic_hot_relation_fraction is not None:
            if len(args) != 8:
                raise FactTrainingProbeError("FaCT-GS rasterizer call shape changed")
            if set(kwargs) - {"use_per_gaussian_backward"}:
                raise FactTrainingProbeError("FaCT-GS rasterizer options changed")
            return _semantic_hot_tile_rasterize(
                *args,
                hot_relation_fraction=selection.semantic_hot_relation_fraction,
            )
        options = dict(kwargs)
        options["use_per_gaussian_backward"] = selection.projection_per_gaussian
        return original_raster(*args, **options)

    def voxelized(*args: Any, **kwargs: Any) -> Any:
        options = dict(kwargs)
        options["use_per_gaussian_backward"] = selection.volume_per_gaussian
        return original_voxel(*args, **options)

    raster_module.rasterize_gaussians = rasterized
    voxel_module.voxelize_gaussians = voxelized
    try:
        yield selection
    finally:
        raster_module.rasterize_gaussians = original_raster
        voxel_module.voxelize_gaussians = original_voxel


@contextmanager
def _quiet_progress_bar(module: ModuleType) -> Iterator[None]:
    """Remove terminal rendering without removing any training operation."""

    original = module.tqdm

    class Progress:
        def __init__(self, _items: Any, **_kwargs: Any) -> None:
            pass

        def set_postfix(self, _values: Any) -> None:
            pass

        def update(self, _count: int = 1) -> None:
            pass

        def close(self) -> None:
            pass

        @staticmethod
        def write(_message: str) -> None:
            pass

    module.tqdm = Progress
    try:
        yield
    finally:
        module.tqdm = original


def _namespace(document: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(document))


def _load_probe_config(
    source_root: Path, dataset_root: Path, model_root: Path, iterations: int,
) -> SimpleNamespace:
    """Build the official config with only bounded-prefix control overrides."""

    config_root = source_root / "config"
    paths = {
        "model": config_root / "model" / "model_default_recon.yaml",
        "optim": config_root / "optim" / "optim_default_recon.yaml",
        "eval": config_root / "eval" / "eval_default.yaml",
    }
    documents: dict[str, Mapping[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FactTrainingProbeError(f"FaCT-GS configuration is missing: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise FactTrainingProbeError(f"FaCT-GS configuration is invalid: {path}")
        documents[name] = loaded
    model = _namespace(documents["model"])
    optim = _namespace(documents["optim"])
    evaluation = _namespace(documents["eval"])
    model.data_source_path = str(dataset_root)
    model.model_path = str(model_root)
    model.init_mode = "precomputed"
    model.eval = False
    optim.steps = iterations
    evaluation.eval_in_training = False
    evaluation.eval_start = False
    evaluation.eval_end = False
    evaluation.visualize_at_eval = False
    evaluation.visualize_gaussians = False
    return SimpleNamespace(model=model, optim=optim, eval=evaluation)


def _snapshot_state(gaussians: Any) -> dict[str, Any]:
    tensors = {
        "xyz": gaussians._xyz,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
        "density": gaussians._density,
        "max_radii2D": gaussians.max_radii2D,
        "xyz_gradient_accum": gaussians.xyz_gradient_accum,
        "denom": gaussians.denom,
    }
    return {
        name: tensor.detach().cpu().contiguous().numpy().copy()
        for name, tensor in tensors.items()
    }


def _snapshot_gradients(gaussians: Any) -> dict[str, Any]:
    """Capture the final pre-Adam gradients, where path equivalence is defined."""

    tensors = {
        "xyz": gaussians._xyz.grad,
        "scaling": gaussians._scaling.grad,
        "rotation": gaussians._rotation.grad,
        "density": gaussians._density.grad,
    }
    if any(tensor is None for tensor in tensors.values()):
        raise FactTrainingProbeError("FaCT-GS optimizer step has a missing gradient")
    return {
        name: tensor.detach().cpu().contiguous().numpy().copy()
        for name, tensor in tensors.items()
    }


def _state_fingerprint(arrays: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    fields: dict[str, Any] = {}
    for name, array in arrays.items():
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        values = array.astype("float64", copy=False)
        fields[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sum": float(values.sum()),
            "absolute_sum": float(abs(values).sum()),
        }
    return {"sha256": digest.hexdigest(), "fields": fields}


def compare_states(
    reference: Mapping[str, Any], observed: Mapping[str, Any], *,
    relative_tolerance: float, absolute_tolerance: float,
) -> dict[str, Any]:
    """Return a per-tensor numerical-equivalence report for two prefixes."""

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("state tolerances must be nonnegative")
    if set(reference) != set(observed):
        raise FactTrainingProbeError("compiler variant changed state tensor coverage")
    fields: dict[str, Any] = {}
    passed = True
    for name in sorted(reference):
        expected = reference[name]
        actual = observed[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise FactTrainingProbeError(
                f"compiler variant changed state tensor shape or dtype: {name}"
            )
        delta = abs(actual.astype("float64") - expected.astype("float64"))
        maximum = float(delta.max(initial=0.0))
        scale = float(abs(expected.astype("float64")).max(initial=0.0))
        threshold = absolute_tolerance + relative_tolerance * scale
        field_passed = maximum <= threshold
        fields[name] = {
            "maximum_absolute_error": maximum,
            "reference_scale": scale,
            "threshold": threshold,
            "passed": field_passed,
        }
        passed = passed and field_passed
    return {
        "passed": passed,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "fields": fields,
    }


def _validate_inputs(
    source_root: Path, dataset_root: Path, artifact_root: Path,
    iterations: int, warmup_iterations: int, progress_interval: int,
    inactivity_timeout_seconds: float,
) -> None:
    if not source_root.is_dir() or not (source_root / "train_recon.py").is_file():
        raise FactTrainingProbeError("FaCT-GS source root or train_recon.py is unavailable")
    if not dataset_root.is_dir() or not (dataset_root / "meta_data.json").is_file():
        raise FactTrainingProbeError("FaCT-GS prepared dataset is unavailable")
    initial_state = dataset_root / f"init_{dataset_root.name}.npy"
    if not initial_state.is_file():
        raise FactTrainingProbeError(f"FaCT-GS initialization is unavailable: {initial_state}")
    if artifact_root.exists():
        raise FactTrainingProbeError(f"probe artifact path already exists: {artifact_root}")
    if iterations <= 0 or warmup_iterations < 0 or warmup_iterations >= iterations:
        raise FactTrainingProbeError("require 0 <= warmup iterations < requested iterations")
    if progress_interval <= 0 or inactivity_timeout_seconds <= 0:
        raise FactTrainingProbeError("probe progress and inactivity intervals must be positive")


def run_probe(
    *,
    source_root: Path,
    dataset_root: Path,
    artifact_root: Path,
    requested_iterations: int,
    warmup_iterations: int,
    progress_interval: int,
    inactivity_timeout_seconds: float,
    compiler_variant: str,
    dataset_id: str,
    profile_stages: bool = False,
    minimum_available_host_memory_bytes: int = DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES,
) -> ProbeObservation:
    """Run a bounded official FaCT-GS prefix from the common initial state."""

    _validate_inputs(
        source_root, dataset_root, artifact_root, requested_iterations,
        warmup_iterations, progress_interval, inactivity_timeout_seconds,
    )
    if compiler_variant not in BACKWARD_PATHS:
        raise FactTrainingProbeError(f"unsupported compiler variant: {compiler_variant}")
    try:
        ensure_host_memory_reserve(minimum_available_host_memory_bytes)
        gpu_isolation = ensure_gpu_isolated(owner_pid=os.getpid())
    except (GpuObservationError, RuntimeError) as error:
        raise FactTrainingProbeError(str(error)) from error
    artifact_root.mkdir(parents=True)
    model_root = artifact_root / "model"
    model_root.mkdir()
    training_path = source_root / "train_recon.py"
    source = training_path.read_text(encoding="utf-8")
    initial_state = dataset_root / f"init_{dataset_root.name}.npy"
    final_arrays: dict[str, Any] | None = None
    final_gradients: dict[str, Any] | None = None
    stage_profile: dict[str, Any] | None = None
    completed_iterations = 0
    cuda_elapsed_ms: float | None = None

    with _working_directory(source_root):
        training = _load_module(training_path, source_root)
        original_stdout = sys.stdout
        training.safe_state(True)
        sys.stdout = original_stdout
        torch = training.torch
        if not torch.cuda.is_available():
            raise FactTrainingProbeError("CUDA is unavailable")
        config = _load_probe_config(
            source_root, dataset_root, model_root, requested_iterations,
        )
        original_status = training.log_training_status
        original_training_setup = training.GaussianModel.training_setup
        measurement_start = torch.cuda.Event(enable_timing=True)
        measurement_end = torch.cuda.Event(enable_timing=True)
        watchdog = GpuWatchdog(
            inactivity_timeout_seconds,
            owner_pid=os.getpid(),
            minimum_available_host_memory_bytes=minimum_available_host_memory_bytes,
        )
        profiler = _FactStageProfiler(torch) if profile_stages else None

        def setup_with_gradient_capture(gaussians: Any, optimization: Any) -> None:
            original_training_setup(gaussians, optimization)
            original_step = gaussians.optimizer.step

            def capture_before_step(*args: Any, **kwargs: Any) -> Any:
                nonlocal final_gradients
                final_gradients = _snapshot_gradients(gaussians)
                if profiler is None:
                    return original_step(*args, **kwargs)
                return profiler.measure(
                    "optimizer", lambda: original_step(*args, **kwargs),
                )

            gaussians.optimizer.step = capture_before_step

        def status(
            step: int, _metrics: Mapping[str, Any], _training_seconds: float,
            _evaluation_seconds: float, _max_steps: int, _evaluation: Any,
            scene: Any, _render: Any, _voxelize: Any, _init_mode: str,
            force_eval: bool = False,
        ) -> None:
            nonlocal completed_iterations, cuda_elapsed_ms, final_arrays
            del force_eval
            completed_iterations = step + 1
            watchdog.progress()
            if completed_iterations == warmup_iterations:
                measurement_start.record()
                if profiler is not None:
                    profiler.begin()
            if (
                completed_iterations % progress_interval == 0
                or completed_iterations == requested_iterations
            ):
                print(json.dumps({
                    "phase": "training",
                    "completed_iterations": completed_iterations,
                    "requested_iterations": requested_iterations,
                    "compiler_variant": compiler_variant,
                }, sort_keys=True), file=sys.stderr, flush=True)
            if completed_iterations != requested_iterations:
                return None
            measurement_end.record()
            if profiler is not None:
                profiler.end()
            measurement_end.synchronize()
            cuda_elapsed_ms = float(measurement_start.elapsed_time(measurement_end))
            final_arrays = _snapshot_state(scene.gaussians)
            raise _PrefixComplete

        if warmup_iterations == 0:
            measurement_start.record()
            if profiler is not None:
                profiler.begin()
        low_memory_overlay = install_fact_low_memory_overlay()
        training.log_training_status = status
        training.GaussianModel.training_setup = setup_with_gradient_capture
        if profiler is not None:
            profiler.install(training)
        wall_started = time.perf_counter()
        watchdog.start()
        try:
            with _backward_path_overlay(compiler_variant), _quiet_progress_bar(training):
                try:
                    training.optimize(config)
                except _PrefixComplete:
                    pass
        except KeyboardInterrupt as error:
            if watchdog.failure is not None:
                raise FactTrainingProbeError(watchdog.failure) from error
            raise
        finally:
            watchdog.stop()
            training.log_training_status = original_status
            training.GaussianModel.training_setup = original_training_setup
            if profiler is not None:
                profiler.end()
                try:
                    stage_profile = profiler.result()
                finally:
                    profiler.restore()
            low_memory_overlay.restore()
            torch.cuda.empty_cache()
        wall_seconds = time.perf_counter() - wall_started

    if completed_iterations != requested_iterations or cuda_elapsed_ms is None:
        raise FactTrainingProbeError("official FaCT-GS prefix did not complete")
    if watchdog.failure is not None:
        raise FactTrainingProbeError(watchdog.failure)
    if final_arrays is None:
        raise FactTrainingProbeError("official FaCT-GS prefix did not expose final state")
    if final_gradients is None:
        raise FactTrainingProbeError("official FaCT-GS prefix did not expose final gradients")
    measured_iterations = requested_iterations - warmup_iterations
    selection = BACKWARD_PATHS[compiler_variant]
    record = {
        "schema_version": "gala-fact-gs-official-training-probe-v1",
        "status": "passed",
        "result_scope": "development_prefix_projection",
        "formal_performance_eligible": False,
        "workload": {
            "model": "FaCT-GS",
            "model_id": "fact_gs",
            "dataset": dataset_id,
            "dataset_id": dataset_id,
            "compiler_variant": compiler_variant,
            "comparison_baseline": "gpu_base",
            "requested_iterations": requested_iterations,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "initialization": str(initial_state.resolve()),
            "included_training_operations": [
                "projection_forward", "l1", "dssim", "tv", "backward",
                "adaptive_control", "densification", "adam",
            ],
            "excluded_prefix_endpoint_operations": ["evaluation", "save", "checkpoint"],
        },
        "measurement": {
            "cuda_elapsed_ms": cuda_elapsed_ms,
            "wall_seconds_including_initialization": wall_seconds,
            "seconds_per_iteration": cuda_elapsed_ms / 1000.0 / measured_iterations,
            "final_state": _state_fingerprint(final_arrays),
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
            "source_root": str(source_root.resolve()),
            "source_train_sha256": _sha256_text(source),
            "dataset_root": str(dataset_root.resolve()),
            "compiler_variant": compiler_variant,
            "backward_path_selection": {
                "projection_per_gaussian": selection.projection_per_gaussian,
                "volume_per_gaussian": selection.volume_per_gaussian,
                "semantic_hot_relation_fraction": selection.semantic_hot_relation_fraction,
            },
            "hash_policy": "record_only_no_hash_rejection",
        },
    }
    return ProbeObservation(record, final_arrays, final_gradients)


def summarize_matrix(
    observations: Mapping[str, Sequence[ProbeObservation]], *,
    relative_tolerance: float, absolute_tolerance: float,
) -> dict[str, Any]:
    """Validate a serial GPU matrix and derive GPU-relative compiler speedups."""

    if tuple(observations) != COMPILER_VARIANTS:
        raise FactTrainingProbeError("matrix must use the four canonical compiler variants")
    counts = {len(samples) for samples in observations.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise FactTrainingProbeError("every compiler variant needs the same nonzero repeats")
    baseline = observations["gpu_base"]
    reference_workload = dict(baseline[0].record["workload"])
    reference_workload.pop("compiler_variant")
    numeric_checks: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for variant, samples in observations.items():
        checks = []
        elapsed = []
        for index, sample in enumerate(samples):
            workload = dict(sample.record["workload"])
            workload.pop("compiler_variant")
            if workload != reference_workload:
                raise FactTrainingProbeError(
                    f"{variant} changed the comparable official workload")
            check = compare_states(
                baseline[index].gradient_arrays, sample.gradient_arrays,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if not check["passed"]:
                raise FactTrainingProbeError(
                    f"{variant} changed the pre-optimizer gradients beyond tolerance"
                )
            checks.append(check)
            elapsed.append(float(sample.record["measurement"]["cuda_elapsed_ms"]))
        numeric_checks[variant] = checks
        summary[variant] = {
            "bits": variant,
            "comparison_baseline": "gpu_base",
            "median_cuda_elapsed_ms": statistics.median(elapsed),
            "minimum_cuda_elapsed_ms": min(elapsed),
            "maximum_cuda_elapsed_ms": max(elapsed),
            "samples": [sample.record for sample in samples],
            "pre_optimizer_gradient_equivalence": checks,
        }
    baseline_ms = summary["gpu_base"]["median_cuda_elapsed_ms"]
    targets = compiler_target_speedups()
    for variant, item in summary.items():
        speedup = baseline_ms / item["median_cuda_elapsed_ms"]
        target = targets.get(variant)
        item["speedup_vs_gpu_base"] = speedup
        item["target_speedup_vs_gpu_base"] = target
        item["target_met"] = target is None or speedup >= target
    performance_comparison_eligible = all(
        sample.record.get("gpu", {}).get("isolation", {}).get("status") == "isolated"
        and not sample.record.get("gpu", {}).get("external_compute_processes")
        for variant_samples in observations.values()
        for sample in variant_samples
    )
    return {
        "schema_version": "gala-fact-gs-gpu-compiler-matrix-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_scope": "development_prefix_projection",
        "formal_performance_eligible": False,
        "performance_comparison_eligible": performance_comparison_eligible,
        "comparison_baseline": "gpu_base",
        "numerical_tolerance": {
            "relative": relative_tolerance,
            "absolute": absolute_tolerance,
        },
        "summary": summary,
    }


def run_matrix(
    *, source_root: Path, dataset_root: Path, output: Path,
    dataset_id: str,
    requested_iterations: int, warmup_iterations: int, repeats: int,
    progress_interval: int, inactivity_timeout_seconds: float,
    relative_tolerance: float, absolute_tolerance: float,
    profile_stages: bool = False,
    minimum_available_host_memory_bytes: int = DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES,
) -> dict[str, Any]:
    """Run the four official paths serially and write an auditable matrix."""

    if repeats <= 0:
        raise FactTrainingProbeError("matrix repeats must be positive")
    if output.exists():
        raise FactTrainingProbeError(f"matrix output already exists: {output}")
    output.mkdir(parents=True)
    observations: dict[str, list[ProbeObservation]] = {
        variant: [] for variant in COMPILER_VARIANTS
    }
    try:
        for repeat in range(repeats):
            order = COMPILER_VARIANTS[repeat:] + COMPILER_VARIANTS[:repeat]
            for variant in order:
                observation = run_probe(
                    source_root=source_root,
                    dataset_root=dataset_root,
                    artifact_root=output / "artifacts" / variant / f"repeat_{repeat + 1}",
                    requested_iterations=requested_iterations,
                    warmup_iterations=warmup_iterations,
                    progress_interval=progress_interval,
                    inactivity_timeout_seconds=inactivity_timeout_seconds,
                    compiler_variant=variant,
                    dataset_id=dataset_id,
                    profile_stages=False,
                    minimum_available_host_memory_bytes=minimum_available_host_memory_bytes,
                )
                observations[variant].append(observation)
                sample_output = output / "samples" / variant / f"repeat_{repeat + 1}.json"
                sample_output.parent.mkdir(parents=True, exist_ok=True)
                sample_output.write_text(
                    json.dumps(observation.record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                del observation
                gc.collect()
        result = summarize_matrix(
            observations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if profile_stages:
            diagnostic = run_probe(
                source_root=source_root,
                dataset_root=dataset_root,
                artifact_root=output / "artifacts" / "gpu_base_stage_profile",
                requested_iterations=requested_iterations,
                warmup_iterations=warmup_iterations,
                progress_interval=progress_interval,
                inactivity_timeout_seconds=inactivity_timeout_seconds,
                compiler_variant="gpu_base",
                dataset_id=dataset_id,
                profile_stages=True,
                minimum_available_host_memory_bytes=minimum_available_host_memory_bytes,
            )
            diagnostic_path = output / "gpu-base-stage-profile.json"
            diagnostic_path.write_text(
                json.dumps(diagnostic.record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stage_profile = diagnostic.record["measurement"]["stage_profile"]
            if not isinstance(stage_profile, Mapping):
                raise FactTrainingProbeError("FaCT-GS stage profile is missing")
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
            model_id="fact_gs",
            dataset_id=dataset_id,
            iteration_range=(warmup_iterations + 1, requested_iterations),
            included_training_operations=list(
                observations["gpu_base"][0].record["workload"]
                ["included_training_operations"]
            ),
            median_gpu_ms={
                variant: float(summary[variant]["median_cuda_elapsed_ms"])
                for variant in COMPILER_VARIANTS
            },
            sample_counts={
                variant: len(observations[variant])
                for variant in COMPILER_VARIANTS
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", choices=("chest", "walnut", "hdtomo_usb"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--inactivity-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--minimum-host-memory-mib", type=int,
        default=DEFAULT_MINIMUM_AVAILABLE_HOST_MEMORY_BYTES // (1024 * 1024),
        help="host memory reserve in MiB",
    )
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            source_root=args.source_root.resolve(),
            dataset_root=args.dataset_root.resolve(),
            dataset_id=args.dataset_id,
            output=args.output.resolve(),
            requested_iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            repeats=args.repeats,
            progress_interval=args.progress_interval,
            inactivity_timeout_seconds=args.inactivity_timeout_seconds,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            profile_stages=args.profile_stages,
            minimum_available_host_memory_bytes=args.minimum_host_memory_mib * 1024 * 1024,
        )
    except FactTrainingProbeError as error:
        parser.error(str(error))
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
