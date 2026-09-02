"""Measure orthogonal GR-Gaussian compiler paths on the CUDA reference math."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from gala_sim.ablation.anchors import compiler_target_speedups
from gala_sim.adapters.gr_gaussian import build_knn_graph
from gala_sim.gpu_measurement import build_gpu_compiler_measurement
from gala_sim.tools.gpu_observation import GpuObservationError, ensure_gpu_isolated
from gala_sim.tools.gpu_probe_watchdog import (
    GpuWatchdog,
    ensure_host_memory_reserve,
    gpu_summary as _gpu_summary,
)


COMPILER_VARIANTS = ("gpu_base", "1000", "0100", "1100")
VARIANT_PATHS = {
    "gpu_base": (False, False),
    "1000": (True, False),
    "0100": (False, True),
    "1100": (True, True),
}
INCLUDED_TRAINING_OPERATIONS = [
    "radiative_splat",
    "mean_squared_projection_loss",
    "graph_regularization",
    "density_gradient",
    "projected_sgd_update",
]


class GRTrainingProbeError(RuntimeError):
    """The GR-Gaussian CUDA compiler matrix is invalid or not comparable."""


@dataclass(frozen=True)
class ProbeObservation:
    record: Mapping[str, Any]
    densities: np.ndarray


def _weights(torch: Any, origins: Any, directions: Any, means: Any, scales: Any) -> Any:
    unit = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    relative = means.unsqueeze(0) - origins.unsqueeze(1)
    depth = torch.sum(relative * unit.unsqueeze(1), dim=2)
    closest = origins.unsqueeze(1) + torch.clamp_min(depth, 0.0).unsqueeze(2) * unit.unsqueeze(1)
    normalized = (closest - means.unsqueeze(0)) / scales.unsqueeze(0)
    values = torch.exp(-0.5 * torch.sum(normalized * normalized, dim=2))
    return torch.where(depth >= 0.0, values, torch.zeros_like(values))


def _prediction(torch: Any, weights: Any, densities: Any) -> Any:
    return -torch.expm1(-(weights @ densities))


def _graph_loss(torch: Any, densities: Any, edges: Any) -> Any:
    if edges.numel() == 0:
        return densities.sum() * 0.0
    difference = densities[edges[:, 0]] - densities[edges[:, 1]]
    return torch.mean(difference * difference)


def _closed_form_gradient(
    torch: Any,
    weights: Any,
    densities: Any,
    target: Any,
    edges: Any,
    graph_weight: float,
) -> tuple[Any, Any]:
    attenuation = weights @ densities
    prediction = -torch.expm1(-attenuation)
    residual = prediction - target
    jacobian = torch.exp(-attenuation).unsqueeze(1) * weights
    gradient = 2.0 * (jacobian.transpose(0, 1) @ residual) / max(1, residual.numel())
    if edges.numel() and graph_weight:
        source = edges[:, 0]
        destination = edges[:, 1]
        difference = densities[source] - densities[destination]
        graph_gradient = torch.zeros_like(densities)
        graph_gradient.index_add_(0, source, 2.0 * difference / edges.shape[0])
        graph_gradient.index_add_(0, destination, -2.0 * difference / edges.shape[0])
        gradient = gradient + graph_weight * graph_gradient
    loss = torch.mean(residual * residual) + graph_weight * _graph_loss(
        torch, densities, edges,
    )
    return loss, gradient


def _load_bundle(bundle: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(bundle, allow_pickle=False) as values:
            required = {
                "means", "scales", "densities", "ray_origins",
                "ray_directions", "target",
            }
            if not required.issubset(values.files):
                raise GRTrainingProbeError("GR-Gaussian ray bundle is incomplete")
            result = {name: np.asarray(values[name]).copy() for name in required}
    except (OSError, ValueError) as error:
        raise GRTrainingProbeError(f"cannot load GR-Gaussian ray bundle: {bundle}") from error
    if result["means"].shape != result["scales"].shape:
        raise GRTrainingProbeError("GR-Gaussian mean and scale shapes differ")
    if result["densities"].shape != (len(result["means"]),):
        raise GRTrainingProbeError("GR-Gaussian density shape is invalid")
    return result


def run_probe(
    *,
    bundle: Path,
    dataset_id: str,
    compiler_variant: str,
    requested_iterations: int,
    warmup_iterations: int,
    learning_rate: float,
    graph_weight: float,
    inactivity_timeout_seconds: float,
) -> ProbeObservation:
    """Run one complete bounded density-optimization path on CUDA."""

    if compiler_variant not in VARIANT_PATHS:
        raise GRTrainingProbeError(f"unsupported compiler variant: {compiler_variant}")
    if requested_iterations <= 0 or not 0 <= warmup_iterations < requested_iterations:
        raise GRTrainingProbeError("require 0 <= warmup iterations < requested iterations")
    if learning_rate <= 0 or graph_weight < 0 or inactivity_timeout_seconds <= 0:
        raise GRTrainingProbeError("GR-Gaussian probe controls are invalid")
    try:
        ensure_host_memory_reserve()
        isolation = ensure_gpu_isolated()
    except (GpuObservationError, RuntimeError) as error:
        raise GRTrainingProbeError(str(error)) from error
    arrays = _load_bundle(bundle)
    try:
        import torch
    except ImportError as error:
        raise GRTrainingProbeError("PyTorch is required for GR-Gaussian CUDA timing") from error
    if not torch.cuda.is_available():
        raise GRTrainingProbeError("CUDA is unavailable")

    device = torch.device("cuda")
    dtype = torch.float64
    means = torch.as_tensor(arrays["means"], dtype=dtype, device=device)
    scales = torch.as_tensor(arrays["scales"], dtype=dtype, device=device)
    origins = torch.as_tensor(arrays["ray_origins"], dtype=dtype, device=device)
    directions = torch.as_tensor(arrays["ray_directions"], dtype=dtype, device=device)
    target = torch.as_tensor(arrays["target"], dtype=dtype, device=device)
    densities = torch.as_tensor(arrays["densities"], dtype=dtype, device=device).clone()
    edge_array = build_knn_graph(arrays["means"], neighbors=min(8, len(means) - 1))
    edges = torch.as_tensor(edge_array, dtype=torch.long, device=device)
    hoist_query, closed_form_semantic = VARIANT_PATHS[compiler_variant]
    cached_weights = _weights(torch, origins, directions, means, scales) if hoist_query else None
    measurement_start = torch.cuda.Event(enable_timing=True)
    measurement_end = torch.cuda.Event(enable_timing=True)
    watchdog = GpuWatchdog(inactivity_timeout_seconds, owner_pid=os.getpid())
    final_loss = 0.0
    wall_started = time.monotonic()
    watchdog.start()
    try:
        if warmup_iterations == 0:
            measurement_start.record()
        for iteration in range(1, requested_iterations + 1):
            weights = cached_weights
            if weights is None:
                weights = _weights(torch, origins, directions, means, scales)
            if closed_form_semantic:
                loss, gradient = _closed_form_gradient(
                    torch, weights, densities, target, edges, graph_weight,
                )
                densities = torch.clamp_min(densities - learning_rate * gradient, 0.0)
            else:
                densities = densities.detach().requires_grad_(True)
                prediction = _prediction(torch, weights, densities)
                loss = torch.mean((prediction - target) ** 2) + graph_weight * _graph_loss(
                    torch, densities, edges,
                )
                loss.backward()
                with torch.no_grad():
                    densities = torch.clamp_min(
                        densities - learning_rate * densities.grad, 0.0,
                    )
            if iteration == warmup_iterations:
                measurement_start.record()
            watchdog.progress()
        measurement_end.record()
        measurement_end.synchronize()
        elapsed_ms = float(measurement_start.elapsed_time(measurement_end))
        final_loss = float(loss.detach().cpu())
        final_densities = densities.detach().cpu().numpy().copy()
    except KeyboardInterrupt as error:
        if watchdog.failure is not None:
            raise GRTrainingProbeError(watchdog.failure) from error
        raise
    finally:
        watchdog.stop()
    if watchdog.failure is not None:
        raise GRTrainingProbeError(watchdog.failure)
    return ProbeObservation({
        "schema_version": "gala-gr-gaussian-cuda-training-probe-v1",
        "status": "passed",
        "result_scope": "bounded_cuda_reference_training",
        "formal_performance_eligible": False,
        "workload": {
            "model": "GR-Gaussian",
            "model_id": "gr_gaussian",
            "dataset": dataset_id,
            "dataset_id": dataset_id,
            "compiler_variant": compiler_variant,
            "comparison_baseline": "gpu_base",
            "requested_iterations": requested_iterations,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": requested_iterations - warmup_iterations,
            "included_training_operations": INCLUDED_TRAINING_OPERATIONS,
        },
        "measurement": {
            "cuda_elapsed_ms": elapsed_ms,
            "wall_seconds_including_initialization": time.monotonic() - wall_started,
            "final_loss": final_loss,
            "final_density_sum": float(final_densities.sum()),
        },
        "compiler_paths": {
            "hoisted_query_weights": hoist_query,
            "closed_form_semantic_gradient": closed_form_semantic,
        },
        "gpu": {
            **_gpu_summary(watchdog.samples),
            "isolation": isolation,
            "external_compute_processes": [],
        },
        "watchdog": {
            "timeout_seconds": inactivity_timeout_seconds,
            "timed_out": watchdog.timed_out,
            "failure": watchdog.failure,
        },
    }, final_densities)


def summarize_matrix(
    observations: Mapping[str, Sequence[ProbeObservation]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if tuple(observations) != COMPILER_VARIANTS:
        raise GRTrainingProbeError("matrix must use the four canonical compiler variants")
    counts = {len(samples) for samples in observations.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise GRTrainingProbeError("every compiler variant needs equal nonzero repeats")
    baseline = observations["gpu_base"]
    reference_workload = dict(baseline[0].record["workload"])
    reference_workload.pop("compiler_variant")
    summary: dict[str, Any] = {}
    for variant, samples in observations.items():
        elapsed: list[float] = []
        maximum_errors: list[float] = []
        for index, sample in enumerate(samples):
            workload = dict(sample.record["workload"])
            workload.pop("compiler_variant")
            if workload != reference_workload:
                raise GRTrainingProbeError(f"{variant} changed the comparable workload")
            expected = baseline[index].densities
            actual = sample.densities
            if expected.shape != actual.shape:
                raise GRTrainingProbeError(f"{variant} changed the final density shape")
            maximum_error = float(np.max(np.abs(expected - actual), initial=0.0))
            scale = float(np.max(np.abs(expected), initial=0.0))
            if maximum_error > absolute_tolerance + relative_tolerance * scale:
                raise GRTrainingProbeError(
                    f"{variant} changed the final densities beyond tolerance"
                )
            elapsed_ms = float(sample.record["measurement"]["cuda_elapsed_ms"])
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise GRTrainingProbeError(f"{variant} has invalid CUDA elapsed time")
            elapsed.append(elapsed_ms)
            maximum_errors.append(maximum_error)
        summary[variant] = {
            "bits": variant,
            "comparison_baseline": "gpu_base",
            "median_cuda_elapsed_ms": statistics.median(elapsed),
            "minimum_cuda_elapsed_ms": min(elapsed),
            "maximum_cuda_elapsed_ms": max(elapsed),
            "maximum_density_errors": maximum_errors,
            "samples": [dict(item.record) for item in samples],
        }
    base_ms = summary["gpu_base"]["median_cuda_elapsed_ms"]
    targets = compiler_target_speedups()
    for variant, item in summary.items():
        speedup = base_ms / item["median_cuda_elapsed_ms"]
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
        "schema_version": "gala-gr-gaussian-gpu-compiler-matrix-v1",
        "result_scope": "bounded_cuda_reference_training",
        "formal_performance_eligible": False,
        "performance_comparison_eligible": eligible,
        "comparison_baseline": "gpu_base",
        "summary": summary,
    }


def run_matrix(
    *,
    bundle: Path,
    dataset_id: str,
    output: Path,
    requested_iterations: int,
    warmup_iterations: int,
    repeats: int,
    learning_rate: float,
    graph_weight: float,
    inactivity_timeout_seconds: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if output.exists() or repeats <= 0:
        raise GRTrainingProbeError("matrix output must be new and repeats must be positive")
    output.mkdir(parents=True)
    observations: dict[str, list[ProbeObservation]] = {
        variant: [] for variant in COMPILER_VARIANTS
    }
    for repeat in range(repeats):
        order = COMPILER_VARIANTS[repeat:] + COMPILER_VARIANTS[:repeat]
        for variant in order:
            observation = run_probe(
                bundle=bundle,
                dataset_id=dataset_id,
                compiler_variant=variant,
                requested_iterations=requested_iterations,
                warmup_iterations=warmup_iterations,
                learning_rate=learning_rate,
                graph_weight=graph_weight,
                inactivity_timeout_seconds=inactivity_timeout_seconds,
            )
            observations[variant].append(observation)
            sample_path = output / "samples" / variant / f"repeat_{repeat + 1}.json"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(
                json.dumps(observation.record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    result = summarize_matrix(
        observations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    (output / "matrix.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    summary = result["summary"]
    standard = build_gpu_compiler_measurement(
        model_id="gr_gaussian",
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
    )
    (output / "gpu-compiler-measurement.json").write_text(
        json.dumps(standard, indent=2, sort_keys=False) + "\n", encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", choices=("chest", "walnut", "hdtomo_usb"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--graph-weight", type=float, default=0.01)
    parser.add_argument("--inactivity-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-7)
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            bundle=args.bundle.resolve(),
            dataset_id=args.dataset_id,
            output=args.output.resolve(),
            requested_iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            repeats=args.repeats,
            learning_rate=args.learning_rate,
            graph_weight=args.graph_weight,
            inactivity_timeout_seconds=args.inactivity_timeout_seconds,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
    except (OSError, ValueError, GRTrainingProbeError) as error:
        print(f"GR-Gaussian compiler matrix failed: {error}", file=sys.stderr)
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
