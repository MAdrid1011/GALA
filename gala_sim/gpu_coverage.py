"""Necessary Amdahl bounds for GPU compiler mechanisms.

Stage profiling is diagnostic evidence, not a performance result.  The bounds
computed here assume that every selected stage can be reduced to zero time, so
they can rule a target out but cannot prove that an implementation will reach
it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from gala_sim.ablation.anchors import compiler_target_speedups


GPU_COMPILER_BITS = ("1000", "0100", "1100")


class GpuCoverageError(ValueError):
    """GPU stage evidence is malformed or incomplete."""


def _positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GpuCoverageError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number <= 0.0:
        raise GpuCoverageError(f"{label} must be finite and positive")
    return number


def analyze_gpu_compiler_coverage(
    *,
    total_gpu_ms: float,
    stage_profile: Mapping[str, Any],
    coverable_stages: Mapping[str, Sequence[str]],
    target_speedups: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return optimistic stage-coverage bounds for the compiler variants."""

    total = _positive_number(total_gpu_ms, "total GPU time")
    if tuple(coverable_stages) != GPU_COMPILER_BITS:
        raise GpuCoverageError(
            "coverable stages must use canonical 1000, 0100, 1100 order"
        )
    summaries = stage_profile.get("summaries")
    if summaries is None:
        summaries = stage_profile.get("stage_summaries")
    if not isinstance(summaries, Mapping):
        raise GpuCoverageError("stage profile has no summaries")
    targets = dict(target_speedups or compiler_target_speedups())
    missing_targets = set(GPU_COMPILER_BITS) - set(targets)
    if missing_targets:
        raise GpuCoverageError("compiler coverage targets are incomplete")

    bounds: dict[str, Any] = {}
    for bits, selected_stages in coverable_stages.items():
        stages = tuple(str(stage) for stage in selected_stages)
        if not stages or len(stages) != len(set(stages)):
            raise GpuCoverageError(f"{bits} coverable stages are empty or duplicated")
        missing = [stage for stage in stages if stage not in summaries]
        if missing:
            raise GpuCoverageError(
                f"{bits} stage profile is missing: {', '.join(missing)}"
            )
        coverable = sum(
            _positive_number(
                summaries[stage].get("total_ms")
                if isinstance(summaries[stage], Mapping) else None,
                f"stage {stage} total",
            )
            for stage in stages
        )
        if coverable > total * (1.0 + 1.0e-6):
            raise GpuCoverageError(
                f"{bits} coverable stage time exceeds the measured GPU total"
            )
        coverable = min(coverable, total)
        uncovered = max(total - coverable, 0.0)
        maximum_speedup = math.inf if uncovered == 0.0 else total / uncovered
        target = _positive_number(targets[bits], f"{bits} target")
        required_fraction = 1.0 - 1.0 / target
        observed_fraction = coverable / total
        bounds[bits] = {
            "comparison_baseline": "gpu_base",
            "coverable_stages": list(stages),
            "total_gpu_ms": total,
            "maximum_coverable_ms": coverable,
            "minimum_uncovered_ms": uncovered,
            "coverable_fraction": observed_fraction,
            "required_coverable_fraction": required_fraction,
            "maximum_possible_speedup_vs_gpu_base": maximum_speedup,
            "target_speedup_vs_gpu_base": target,
            "status": (
                "not_ruled_out_by_necessary_stage_bound"
                if maximum_speedup >= target
                else "target_unreachable_without_engineering_coverage"
            ),
        }
    return {
        "schema_version": "gala-gpu-compiler-coverage-bound-v1",
        "bound_kind": "necessary_optimistic_zero_time_stage_bound",
        "comparison_baseline": "gpu_base",
        "diagnostic_only": True,
        "bounds": bounds,
    }
