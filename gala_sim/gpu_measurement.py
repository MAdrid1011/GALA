"""Validated GPU compiler measurements for one model/dataset campaign.

Trace capture and compiler timing are intentionally separate.  A trace-enabled
process may synchronize, serialize, or copy events and therefore cannot supply
the GPU Base used by compiler-only ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "gala-gpu-compiler-measurement-v1"
GPU_COMPILER_VARIANTS = ("gpu_base", "1000", "0100", "1100")
COMPILER_BITS = GPU_COMPILER_VARIANTS[1:]


class GpuMeasurementError(ValueError):
    """Raised when GPU compiler evidence is incomplete or not comparable."""


def platform_identity_from_isolation(
    isolation: Mapping[str, Any] | Any,
) -> dict[str, str] | None:
    """Extract one stable GPU identity from an isolation sampling report."""

    if not isinstance(isolation, Mapping):
        return None
    samples = isolation.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    identities = {
        (str(item.get("gpu_name", "")), str(item.get("gpu_uuid", "")))
        for item in samples
        if isinstance(item, Mapping)
    }
    if len(identities) != 1:
        return None
    gpu_name, gpu_uuid = next(iter(identities))
    if not gpu_name or not gpu_uuid:
        return None
    return {"gpu_name": gpu_name, "gpu_uuid": gpu_uuid}


def _platform_identity(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GpuMeasurementError("GPU platform identity is malformed")
    gpu_name = value.get("gpu_name")
    gpu_uuid = value.get("gpu_uuid")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise GpuMeasurementError("GPU platform name is malformed")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise GpuMeasurementError("GPU platform UUID is malformed")
    return {"gpu_name": gpu_name, "gpu_uuid": gpu_uuid}


def _positive_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GpuMeasurementError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number <= 0.0:
        raise GpuMeasurementError(f"{label} must be finite and positive")
    return number


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GpuMeasurementError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class GpuCompilerMeasurement:
    """An identity-bound, uninstrumented four-variant GPU timing matrix."""

    path: Path
    model_id: str
    dataset_id: str
    iteration_range: tuple[int, int]
    median_gpu_ms: Mapping[str, float]
    sample_counts: Mapping[str, int]
    gpu_platform: Mapping[str, str] | None
    source_document: Mapping[str, Any]

    @property
    def speedups_vs_gpu_base(self) -> Mapping[str, float]:
        baseline = self.median_gpu_ms["gpu_base"]
        return {
            bits: baseline / self.median_gpu_ms[bits]
            for bits in COMPILER_BITS
        }

    def as_campaign_reference(self, reference: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "reference": reference,
            "measurement_method": "cuda_events_uninstrumented_official_training",
            "trace_instrumentation_enabled": False,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "iteration_range": list(self.iteration_range),
            "median_gpu_ms": dict(self.median_gpu_ms),
            "sample_counts": dict(self.sample_counts),
            "gpu_platform": (
                dict(self.gpu_platform) if self.gpu_platform is not None else None
            ),
        }


def default_gpu_measurement_path(
    root: Path, model_id: str, dataset_id: str,
) -> Path:
    """Return the standard ignored-workspace location for one measurement."""

    return Path(root) / model_id / dataset_id / "gpu-compiler-measurement.json"


def load_gpu_compiler_measurement(
    path: Path,
    *,
    model_id: str,
    dataset_id: str,
    iteration_range: tuple[int, int],
) -> GpuCompilerMeasurement:
    """Load evidence and enforce semantic identity without source hash gating."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GpuMeasurementError(f"GPU measurement is unreadable: {path}") from error
    if not isinstance(document, Mapping):
        raise GpuMeasurementError("GPU measurement root must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise GpuMeasurementError("unsupported GPU compiler measurement schema")
    if document.get("model_id") != model_id:
        raise GpuMeasurementError("GPU measurement model identity mismatch")
    if document.get("dataset_id") != dataset_id:
        raise GpuMeasurementError("GPU measurement dataset identity mismatch")
    gpu_platform = _platform_identity(document.get("gpu_platform"))

    workload = document.get("workload")
    if not isinstance(workload, Mapping):
        raise GpuMeasurementError("GPU measurement has no workload identity")
    observed_range = workload.get("iteration_range")
    if observed_range != list(iteration_range):
        raise GpuMeasurementError("GPU measurement iteration range mismatch")
    start, end = iteration_range
    if start <= 0 or end < start:
        raise GpuMeasurementError("expected GPU measurement iteration range is invalid")
    if workload.get("measured_iterations") != end - start + 1:
        raise GpuMeasurementError("GPU measurement iteration count is inconsistent")
    operation_set = workload.get("included_training_operations")
    if not isinstance(operation_set, list) or not operation_set or not all(
        isinstance(item, str) and item for item in operation_set
    ):
        raise GpuMeasurementError("GPU measurement operation set is incomplete")

    if document.get("measurement_method") != (
        "cuda_events_uninstrumented_official_training"
    ):
        raise GpuMeasurementError("GPU measurement does not use official CUDA timing")
    if document.get("trace_instrumentation_enabled") is not False:
        raise GpuMeasurementError("trace-instrumented time cannot be used as GPU Base")
    if document.get("performance_comparison_eligible") is not True:
        raise GpuMeasurementError("GPU measurement is not performance-comparison eligible")
    comparability = document.get("comparability")
    if not isinstance(comparability, Mapping) or any(
        comparability.get(field) is not True
        for field in (
            "gpu_isolated",
            "same_workload_across_variants",
            "numerical_equivalence_passed",
        )
    ):
        raise GpuMeasurementError("GPU measurement comparability checks did not pass")

    variants = document.get("variants")
    if not isinstance(variants, Mapping) or tuple(variants) != GPU_COMPILER_VARIANTS:
        raise GpuMeasurementError(
            "GPU measurement must use gpu_base, 1000, 0100, 1100 order"
        )
    median_gpu_ms: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for bits in GPU_COMPILER_VARIANTS:
        item = variants[bits]
        if not isinstance(item, Mapping):
            raise GpuMeasurementError(f"GPU measurement variant {bits} is invalid")
        if item.get("comparison_baseline") != "gpu_base":
            raise GpuMeasurementError(f"GPU measurement variant {bits} has wrong baseline")
        median_gpu_ms[bits] = _positive_float(
            item.get("median_gpu_ms"), f"variants.{bits}.median_gpu_ms",
        )
        sample_counts[bits] = _positive_int(
            item.get("sample_count"), f"variants.{bits}.sample_count",
        )
    if len(set(sample_counts.values())) != 1:
        raise GpuMeasurementError("GPU measurement variants need equal sample counts")

    baseline = median_gpu_ms["gpu_base"]
    speedups: dict[str, float] = {}
    for bits in COMPILER_BITS:
        computed = baseline / median_gpu_ms[bits]
        speedups[bits] = computed
        claimed = variants[bits].get("speedup_vs_gpu_base")
        if claimed is not None and not math.isclose(
            _positive_float(claimed, f"variants.{bits}.speedup_vs_gpu_base"),
            computed,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise GpuMeasurementError(
                f"GPU measurement variant {bits} has inconsistent speedup"
            )
    # The combined compiler path must retain the benefit of either component.
    # This catches stale or hand-edited artifacts from experimental regressions
    # even when their isolation and numerical checks otherwise look valid.
    if speedups["1100"] + 1.0e-9 < max(speedups["1000"], speedups["0100"]):
        raise GpuMeasurementError(
            "GPU measurement combined variant 1100 regresses a component"
        )
    return GpuCompilerMeasurement(
        path=path,
        model_id=model_id,
        dataset_id=dataset_id,
        iteration_range=iteration_range,
        median_gpu_ms=median_gpu_ms,
        sample_counts=sample_counts,
        gpu_platform=gpu_platform,
        source_document=document,
    )


def build_gpu_compiler_measurement(
    *,
    model_id: str,
    dataset_id: str,
    iteration_range: tuple[int, int],
    included_training_operations: list[str],
    median_gpu_ms: Mapping[str, float],
    sample_counts: Mapping[str, int],
    source_probe: Mapping[str, Any],
    gpu_isolated: bool,
    same_workload_across_variants: bool,
    numerical_equivalence_passed: bool,
    gpu_platform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable artifact emitted by model-specific timing probes."""

    start, end = iteration_range
    if start <= 0 or end < start:
        raise GpuMeasurementError("GPU measurement iteration range is invalid")
    if not included_training_operations or not all(
        isinstance(item, str) and item for item in included_training_operations
    ):
        raise GpuMeasurementError("GPU measurement operation set is incomplete")
    if tuple(median_gpu_ms) != GPU_COMPILER_VARIANTS:
        raise GpuMeasurementError("GPU timing values must use canonical variant order")
    if tuple(sample_counts) != GPU_COMPILER_VARIANTS:
        raise GpuMeasurementError("GPU sample counts must use canonical variant order")
    variants: dict[str, Any] = {}
    baseline = _positive_float(median_gpu_ms["gpu_base"], "gpu_base median")
    for bits in GPU_COMPILER_VARIANTS:
        elapsed = _positive_float(median_gpu_ms[bits], f"{bits} median")
        count = _positive_int(sample_counts[bits], f"{bits} sample count")
        variants[bits] = {
            "comparison_baseline": "gpu_base",
            "median_gpu_ms": elapsed,
            "sample_count": count,
            "speedup_vs_gpu_base": baseline / elapsed,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "workload": {
            "iteration_range": [start, end],
            "measured_iterations": end - start + 1,
            "included_training_operations": list(included_training_operations),
        },
        "measurement_method": "cuda_events_uninstrumented_official_training",
        "trace_instrumentation_enabled": False,
        "performance_comparison_eligible": bool(
            gpu_isolated
            and same_workload_across_variants
            and numerical_equivalence_passed
        ),
        "comparability": {
            "gpu_isolated": bool(gpu_isolated),
            "same_workload_across_variants": bool(same_workload_across_variants),
            "numerical_equivalence_passed": bool(numerical_equivalence_passed),
        },
        "gpu_platform": _platform_identity(gpu_platform),
        "variants": variants,
        "source_probe": dict(source_probe),
    }
