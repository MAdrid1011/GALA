"""Run the four uninstrumented R2-Gaussian compiler timing variants."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from gala_sim.ablation.anchors import compiler_target_speedups
from gala_sim.gpu_coverage import analyze_gpu_compiler_coverage
from gala_sim.gpu_measurement import (
    build_gpu_compiler_measurement,
    platform_identity_from_isolation,
)
from gala_sim.tools.r2_gaussian_training_probe import (
    COMPILER_VARIANT_FLAGS,
    TrainingProbeError,
    run_probe,
)


COMPILER_VARIANTS = tuple(COMPILER_VARIANT_FLAGS)


def _close(expected: Any, observed: Any, *, rtol: float, atol: float) -> bool:
    try:
        left = float(expected)
        right = float(observed)
    except (TypeError, ValueError):
        return expected == observed
    return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)


def _validate_numerical_equivalence(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    reference_measurement = reference.get("measurement", {})
    observed_measurement = observed.get("measurement", {})
    reference_state = reference_measurement.get("final_state", {}).get("fields")
    observed_state = observed_measurement.get("final_state", {}).get("fields")
    if not isinstance(reference_state, Mapping) or not isinstance(observed_state, Mapping):
        raise TrainingProbeError("R2-Gaussian matrix sample has no final-state fields")
    if set(reference_state) != set(observed_state):
        raise TrainingProbeError("compiler variant changed final-state field coverage")
    checks: dict[str, Any] = {}
    passed = True
    for name in sorted(reference_state):
        expected = reference_state[name]
        actual = observed_state[name]
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise TrainingProbeError(f"invalid final-state field: {name}")
        identity_passed = (
            expected.get("shape") == actual.get("shape")
            and expected.get("dtype") == actual.get("dtype")
        )
        numeric_passed = all(
            _close(
                expected.get(metric), actual.get(metric),
                rtol=relative_tolerance, atol=absolute_tolerance,
            )
            for metric in ("sum", "absolute_sum")
        )
        field_passed = identity_passed and numeric_passed
        checks[name] = {"passed": field_passed}
        passed = passed and field_passed
    reference_losses = reference_measurement.get("final_losses")
    observed_losses = observed_measurement.get("final_losses")
    if not isinstance(reference_losses, Mapping) or not isinstance(observed_losses, Mapping):
        raise TrainingProbeError("R2-Gaussian matrix sample has no final losses")
    if set(reference_losses) != set(observed_losses):
        raise TrainingProbeError("compiler variant changed final-loss coverage")
    losses_passed = all(
        _close(
            reference_losses[name], observed_losses[name],
            rtol=relative_tolerance, atol=absolute_tolerance,
        )
        for name in reference_losses
    )
    passed = passed and losses_passed
    return {
        "passed": passed,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "fields": checks,
        "losses_passed": losses_passed,
    }


def summarize_matrix(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Check workload/numerical identity and summarize CUDA-event timings."""

    if tuple(records) != COMPILER_VARIANTS:
        raise TrainingProbeError("matrix must use the four canonical compiler variants")
    counts = {len(samples) for samples in records.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise TrainingProbeError("every compiler variant needs equal nonzero repeats")
    baseline = records["gpu_base"]
    reference_workload = dict(baseline[0].get("workload", {}))
    reference_workload.pop("compiler_variant", None)
    summary: dict[str, Any] = {}
    for variant, samples in records.items():
        elapsed: list[float] = []
        checks: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            workload = dict(sample.get("workload", {}))
            workload.pop("compiler_variant", None)
            if workload != reference_workload:
                raise TrainingProbeError(
                    f"{variant} changed the comparable official workload"
                )
            elapsed_ms = float(sample.get("measurement", {}).get("cuda_elapsed_ms", 0))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise TrainingProbeError(f"{variant} has invalid CUDA elapsed time")
            check = _validate_numerical_equivalence(
                baseline[index], sample,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if not check["passed"]:
                raise TrainingProbeError(
                    f"{variant} changed the final state beyond tolerance"
                )
            elapsed.append(elapsed_ms)
            checks.append(check)
        summary[variant] = {
            "bits": variant,
            "comparison_baseline": "gpu_base",
            "median_cuda_elapsed_ms": statistics.median(elapsed),
            "minimum_cuda_elapsed_ms": min(elapsed),
            "maximum_cuda_elapsed_ms": max(elapsed),
            "numerical_equivalence": checks,
            "samples": [dict(sample) for sample in samples],
        }
    base_ms = summary["gpu_base"]["median_cuda_elapsed_ms"]
    targets = compiler_target_speedups()
    for variant, item in summary.items():
        speedup = base_ms / item["median_cuda_elapsed_ms"]
        item["speedup_vs_gpu_base"] = speedup
        item["target_speedup_vs_gpu_base"] = targets.get(variant)
        item["target_met"] = variant == "gpu_base" or speedup >= targets[variant]
    eligible = all(
        sample.get("gpu", {}).get("isolation", {}).get("status") == "isolated"
        and not sample.get("gpu", {}).get("external_compute_processes")
        for samples in records.values()
        for sample in samples
    )
    return {
        "schema_version": "gala-r2-gaussian-gpu-compiler-matrix-v1",
        "result_scope": "development_prefix_projection",
        "formal_performance_eligible": False,
        "performance_comparison_eligible": eligible,
        "comparison_baseline": "gpu_base",
        "summary": summary,
    }


def run_matrix(
    *,
    source_root: Path,
    extension_root: Path | None,
    dataset_root: Path,
    initial_state: Path,
    dataset_id: str,
    output: Path,
    requested_iterations: int,
    warmup_iterations: int,
    repeats: int,
    progress_interval: int,
    inactivity_timeout_seconds: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    profile_stages: bool = False,
) -> dict[str, Any]:
    if repeats <= 0:
        raise TrainingProbeError("matrix repeats must be positive")
    if output.exists():
        raise TrainingProbeError(f"matrix output already exists: {output}")
    records: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in COMPILER_VARIANTS
    }
    for repeat in range(repeats):
        order = COMPILER_VARIANTS[repeat:] + COMPILER_VARIANTS[:repeat]
        for variant in order:
            record = run_probe(
                source_root=source_root,
                extension_root=extension_root,
                dataset_root=dataset_root,
                initial_state=initial_state,
                artifact_root=output / "artifacts" / variant / f"repeat_{repeat + 1}",
                requested_iterations=requested_iterations,
                warmup_iterations=warmup_iterations,
                progress_interval=progress_interval,
                inactivity_timeout_seconds=inactivity_timeout_seconds,
                telemetry_mode="cuda_opt",
                compiler_variant=variant,
                dataset_id=dataset_id,
                profile_stages=False,
            )
            records[variant].append(record)
            sample_path = output / "samples" / variant / f"repeat_{repeat + 1}.json"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            gc.collect()
    result = summarize_matrix(
        records,
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
            requested_iterations=requested_iterations,
            warmup_iterations=warmup_iterations,
            progress_interval=progress_interval,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            telemetry_mode="cuda_opt",
            compiler_variant="gpu_base",
            dataset_id=dataset_id,
            profile_stages=True,
        )
        diagnostic_path = output / "gpu-base-stage-profile.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage_profile = diagnostic["measurement"]["stage_profile"]
        if not isinstance(stage_profile, Mapping):
            raise TrainingProbeError("R2-Gaussian stage profile is missing")
        result["compiler_coverage_bounds"] = analyze_gpu_compiler_coverage(
            total_gpu_ms=float(diagnostic["measurement"]["cuda_elapsed_ms"]),
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
        model_id="r2_gaussian",
        dataset_id=dataset_id,
        iteration_range=(warmup_iterations + 1, requested_iterations),
        included_training_operations=list(
            records["gpu_base"][0]["workload"]["included_training_operations"]
        ),
        median_gpu_ms={
            variant: float(summary[variant]["median_cuda_elapsed_ms"])
            for variant in COMPILER_VARIANTS
        },
        sample_counts={variant: len(records[variant]) for variant in COMPILER_VARIANTS},
        source_probe={
            "schema_version": result["schema_version"],
            "result_scope": result["result_scope"],
            "matrix": "matrix.json",
        },
        gpu_isolated=bool(result["performance_comparison_eligible"]),
        same_workload_across_variants=True,
        numerical_equivalence_passed=True,
        gpu_platform=platform_identity_from_isolation(
            records["gpu_base"][0]["gpu"].get("isolation")
        ),
    )
    (output / "gpu-compiler-measurement.json").write_text(
        json.dumps(standard, indent=2, sort_keys=False) + "\n", encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
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
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            source_root=args.source_root.resolve(),
            extension_root=(
                args.extension_root.resolve() if args.extension_root is not None else None
            ),
            dataset_root=args.dataset_root.resolve(),
            initial_state=args.initial_state.resolve(),
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
        )
    except (OSError, ValueError, TrainingProbeError) as error:
        print(f"R2-Gaussian compiler matrix failed: {error}", file=sys.stderr)
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
