"""Measure compiler-only CUDA variants against the same GPU Base workload."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNNER_VARIANTS = ("A0B0", "A1B0", "A0B1", "A1B1")
BITS_BY_VARIANT = {
    "A0B0": "gpu_base",
    "A1B0": "1000",
    "A0B1": "0100",
    "A1B1": "1100",
}
TARGET_SPEEDUPS = {"1000": 1.254, "0100": 1.282, "1100": 1.482}
EXACT_WORKLOAD_FIELDS = (
    "query_count",
    "active_gaussians",
    "candidate_pixels",
    "accepted_relations",
    "consumer_dequeues",
)
NUMERIC_RESULT_FIELDS = (
    "mean_l1_loss",
    "position_gradient_l1",
    "covariance_gradient_l1",
    "opacity_gradient_l1",
    "scale_latent_gradient_l1",
    "quaternion_latent_gradient_l1",
    "density_latent_gradient_l1",
)


class GpuCompilerProbeError(ValueError):
    """Raised when a CUDA compiler-ablation probe is not comparable."""


def _positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GpuCompilerProbeError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number <= 0.0:
        raise GpuCompilerProbeError(f"{label} must be finite and positive")
    return number


def _run_once(
    runner: Path,
    bundle_root: Path,
    variant: str,
    *,
    relation_capacity: int,
    row_start: int,
    row_count: int,
    diagnostic_warmup_runs: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    bundle = bundle_root / variant
    command = [
        str(runner),
        str(bundle / "bundle_manifest.json"),
        str(bundle / "bundle.bin"),
        str(bundle),
        "--relation-capacity",
        str(relation_capacity),
        "--row-start",
        str(row_start),
        "--row-count",
        str(row_count),
        "--diagnostic-warmup-runs",
        str(diagnostic_warmup_runs),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise GpuCompilerProbeError(
            f"{variant} exceeded the {timeout_seconds:g} second probe timeout"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GpuCompilerProbeError(f"{variant} runner failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GpuCompilerProbeError(f"{variant} runner output is not JSON") from error
    if not isinstance(result, dict) or result.get("bundle_variant") != variant:
        raise GpuCompilerProbeError(f"{variant} runner reported a different variant")
    if result.get("status") != "APPROXIMATE":
        raise GpuCompilerProbeError(
            f"{variant} runner must identify the development path as APPROXIMATE"
        )
    relation_ms = _positive_number(
        result.get("relation_generation_ms"), f"{variant}.relation_generation_ms"
    )
    primitive_ms = _positive_number(
        result.get("primitive_ms"), f"{variant}.primitive_ms"
    )
    sample = {
        "relation_generation_ms": relation_ms,
        "primitive_ms": primitive_ms,
        "total_ms": relation_ms + primitive_ms,
    }
    for field in EXACT_WORKLOAD_FIELDS:
        value = result.get(field)
        if not isinstance(value, int) or value < 0:
            raise GpuCompilerProbeError(f"{variant}.{field} must be a nonnegative integer")
        sample[field] = value
    for field in NUMERIC_RESULT_FIELDS:
        value = result.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise GpuCompilerProbeError(f"{variant}.{field} must be finite")
        sample[field] = float(value)
    return sample


def summarize_samples(
    samples: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if tuple(samples) != RUNNER_VARIANTS:
        raise GpuCompilerProbeError("samples must use the canonical four runner variants")
    repeat_counts = {len(items) for items in samples.values()}
    if len(repeat_counts) != 1 or not repeat_counts or next(iter(repeat_counts)) == 0:
        raise GpuCompilerProbeError("every variant must have the same nonzero sample count")
    reference = samples["A0B0"][0]
    for variant, items in samples.items():
        for sample in items:
            for field in EXACT_WORKLOAD_FIELDS:
                if sample.get(field) != reference.get(field):
                    raise GpuCompilerProbeError(
                        f"{variant} changed the comparable workload field {field}"
                    )
            for field in NUMERIC_RESULT_FIELDS:
                observed = float(sample[field])
                expected = float(reference[field])
                tolerance = max(1.0e-6, abs(expected) * 1.0e-5)
                if not math.isclose(observed, expected, rel_tol=1.0e-5, abs_tol=tolerance):
                    raise GpuCompilerProbeError(
                        f"{variant} changed the numerical result field {field}"
                    )
    summary: dict[str, Any] = {}
    for variant, items in samples.items():
        summary[variant] = {
            "bits": BITS_BY_VARIANT[variant],
            "comparison_baseline": "gpu_base",
            "median_total_ms": statistics.median(float(item["total_ms"]) for item in items),
            "median_relation_generation_ms": statistics.median(
                float(item["relation_generation_ms"]) for item in items
            ),
            "median_primitive_ms": statistics.median(
                float(item["primitive_ms"]) for item in items
            ),
            "minimum_total_ms": min(float(item["total_ms"]) for item in items),
            "maximum_total_ms": max(float(item["total_ms"]) for item in items),
            "samples": [dict(item) for item in items],
        }
    base_ms = summary["A0B0"]["median_total_ms"]
    for variant, item in summary.items():
        speedup = base_ms / item["median_total_ms"]
        item["speedup_vs_gpu_base"] = speedup
        bits = BITS_BY_VARIANT[variant]
        target = TARGET_SPEEDUPS.get(bits)
        item["target_speedup_vs_gpu_base"] = target
        item["target_met"] = target is None or speedup >= target
        item["speedup_vs_base_asic"] = None
    return summary


def run_probe(
    runner: Path,
    bundle_root: Path,
    output: Path,
    *,
    relation_capacity: int,
    row_start: int,
    row_count: int,
    warmup_runs: int,
    measured_runs: int,
    diagnostic_warmup_runs: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if relation_capacity <= 0 or row_start < 0 or row_count <= 0:
        raise GpuCompilerProbeError("relation capacity and row count must be positive")
    if (
        warmup_runs < 0
        or measured_runs <= 0
        or diagnostic_warmup_runs < 0
        or timeout_seconds <= 0
    ):
        raise GpuCompilerProbeError("run counts and timeout are invalid")
    if not runner.is_file():
        raise GpuCompilerProbeError(f"runner does not exist: {runner}")
    for variant in RUNNER_VARIANTS:
        for _ in range(warmup_runs):
            _run_once(
                runner,
                bundle_root,
                variant,
                relation_capacity=relation_capacity,
                row_start=row_start,
                row_count=row_count,
                diagnostic_warmup_runs=diagnostic_warmup_runs,
                timeout_seconds=timeout_seconds,
            )
    samples: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in RUNNER_VARIANTS
    }
    for repeat in range(measured_runs):
        offset = repeat % len(RUNNER_VARIANTS)
        order = RUNNER_VARIANTS[offset:] + RUNNER_VARIANTS[:offset]
        for variant in order:
            sample = _run_once(
                runner,
                bundle_root,
                variant,
                relation_capacity=relation_capacity,
                row_start=row_start,
                row_count=row_count,
                diagnostic_warmup_runs=diagnostic_warmup_runs,
                timeout_seconds=timeout_seconds,
            )
            samples[variant].append(sample)
            print(
                json.dumps({
                    "repeat": repeat + 1,
                    "variant": variant,
                    "total_ms": sample["total_ms"],
                }, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
    summary = summarize_samples(samples)
    result = {
        "schema_version": "gala-gpu-compiler-ablation-probe-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_scope": "development_gpu_compiler_probe",
        "formal_performance_eligible": False,
        "performance_comparison_eligible": False,
        "runner_status_required": "APPROXIMATE",
        "row_start": row_start,
        "row_count": row_count,
        "relation_capacity": relation_capacity,
        "warmup_runs_per_variant": warmup_runs,
        "diagnostic_warmup_runs_per_process": diagnostic_warmup_runs,
        "measured_runs_per_variant": measured_runs,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_compiler_ablation")
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relation-capacity", type=int, required=True)
    parser.add_argument("--row-start", type=int, required=True)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--diagnostic-warmup-runs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    result = run_probe(
        args.runner,
        args.bundle_root,
        args.output,
        relation_capacity=args.relation_capacity,
        row_start=args.row_start,
        row_count=args.row_count,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        diagnostic_warmup_runs=args.diagnostic_warmup_runs,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "speedup_vs_gpu_base": {
            item["bits"]: item["speedup_vs_gpu_base"]
            for item in result["summary"].values()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
