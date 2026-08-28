"""Record a non-comparable AGX Orin public-specification extrapolation.

The estimate combines measured local stage weights with explicit hardware
reference vectors.  It cannot calibrate Orin performance, check ASIC
plausibility, or produce a speedup.  Only a same-suite Orin calibration vector
can produce a comparable Orin time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from gala_sim.identity import sha256_file


SCHEMA_VERSION = "gala-gpu-orin-proxy-estimate-v1"
_CATEGORIES = (
    "fp32_fma", "exp", "log", "rcp", "sqrt", "memory_bandwidth",
    "atomic", "kernel_launch", "synchronization",
)
_COMPUTE_CATEGORIES = {"fp32_fma", "exp", "log", "rcp", "sqrt", "atomic"}


class OrinEstimateError(ValueError):
    """Raised when a proxy estimate input violates its schema."""


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise OrinEstimateError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number <= 0:
        raise OrinEstimateError(f"{label} must be finite and positive")
    return number


def _reference(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OrinEstimateError(f"cannot read proxy reference: {path}") from error
    if not isinstance(document, Mapping):
        raise OrinEstimateError("proxy reference must be a mapping")
    if document.get("schema_version") != "gala-gpu-proxy-reference-v1":
        raise OrinEstimateError("unsupported proxy reference schema")
    if document.get("formal_performance_eligible") is not False:
        raise OrinEstimateError("proxy reference cannot be formal")
    result: dict[str, Any] = {"method": str(document.get("method", ""))}
    for side in ("local", "target"):
        item = document.get(side)
        if not isinstance(item, Mapping):
            raise OrinEstimateError(f"proxy reference lacks {side} device")
        result[side] = {
            "name": str(item.get("name", "")),
            "fp32_cores": _positive(item.get("fp32_cores"), f"{side}.fp32_cores"),
            "sm_clock_hz": _positive(item.get("sm_clock_hz"), f"{side}.sm_clock_hz"),
            "memory_bandwidth_bytes_per_second": _positive(
                item.get("memory_bandwidth_bytes_per_second"),
                f"{side}.memory_bandwidth_bytes_per_second",
            ),
            "source": str(item.get("source", "")),
        }
    raw_uncertainty = document.get("uncertainty")
    if not isinstance(raw_uncertainty, Mapping):
        raise OrinEstimateError("proxy reference lacks uncertainty")
    uncertainty: dict[str, float] = {}
    for category in _CATEGORIES:
        value = _positive(raw_uncertainty.get(category), f"uncertainty.{category}")
        if value > 1:
            raise OrinEstimateError(f"uncertainty.{category} must be <= 1")
        uncertainty[category] = value
    result["uncertainty"] = uncertainty
    return result


def _ratios(reference: Mapping[str, Any]) -> dict[str, float]:
    local = reference["local"]
    target = reference["target"]
    compute_ratio = (
        _positive(local["fp32_cores"], "local.fp32_cores")
        * _positive(local["sm_clock_hz"], "local.sm_clock_hz")
        / (
            _positive(target["fp32_cores"], "target.fp32_cores")
            * _positive(target["sm_clock_hz"], "target.sm_clock_hz")
        )
    )
    memory_ratio = (
        _positive(local["memory_bandwidth_bytes_per_second"], "local.memory_bandwidth")
        / _positive(target["memory_bandwidth_bytes_per_second"], "target.memory_bandwidth")
    )
    return {
        **{category: compute_ratio for category in _COMPUTE_CATEGORIES},
        "memory_bandwidth": memory_ratio,
        # Host-side values have no public target spec; 1.0 is an explicit
        # neutral placeholder and its wide interval keeps the estimate provisional.
        "kernel_launch": 1.0,
        "synchronization": 1.0,
    }


def estimate_normalized(
    normalization: Mapping[str, Any], reference: Mapping[str, Any],
    *, workload_iterations: int | None = None,
) -> dict[str, Any]:
    if workload_iterations is not None and workload_iterations <= 0:
        raise OrinEstimateError("workload_iterations must be positive")
    stages = normalization.get("stages")
    if not isinstance(stages, Mapping) or not stages:
        raise OrinEstimateError("normalization has no stages")
    ratios = _ratios(reference)
    uncertainty = reference["uncertainty"]
    estimated_stages: dict[str, Any] = {}
    total_local_ms = 0.0
    total_estimated_ms = 0.0
    total_low_ms = 0.0
    total_high_ms = 0.0
    for name, raw_stage in stages.items():
        if not isinstance(raw_stage, Mapping):
            raise OrinEstimateError(f"stage {name} is malformed")
        local_ms = raw_stage.get("local_ms")
        weights = raw_stage.get("weights")
        if local_ms is None or not isinstance(weights, Mapping) or not weights:
            estimated_stages[str(name)] = {
                "status": "unavailable",
                "reason": "local stage timing or weights are missing",
                "local_ms": None,
                "estimated_orin_ms": None,
            }
            continue
        local_ms_value = _positive(local_ms, f"{name}.local_ms")
        weight_sum = sum(_positive(value, f"{name}.weights.{category}")
                         for category, value in weights.items())
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise OrinEstimateError(f"{name} weights do not sum to one")
        conversion = 0.0
        low_conversion = 0.0
        high_conversion = 0.0
        components: dict[str, Any] = {}
        for raw_category, raw_weight in weights.items():
            category = str(raw_category)
            if category not in ratios:
                raise OrinEstimateError(f"unsupported stage category: {category}")
            weight = float(raw_weight)
            ratio = ratios[category]
            spread = float(uncertainty[category])
            conversion += weight * ratio
            low_conversion += weight * ratio * (1.0 - spread)
            high_conversion += weight * ratio * (1.0 + spread)
            components[category] = {
                "weight": weight,
                "seconds_ratio": ratio,
                "uncertainty_fraction": spread,
            }
        estimated_ms = local_ms_value * conversion
        low_ms = max(0.0, local_ms_value * low_conversion)
        high_ms = local_ms_value * high_conversion
        total_local_ms += local_ms_value
        total_estimated_ms += estimated_ms
        total_low_ms += low_ms
        total_high_ms += high_ms
        estimated_stages[str(name)] = {
            "status": "proxy_estimate",
            "local_ms": local_ms_value,
            "estimated_orin_ms": estimated_ms,
            "interval_ms": {"low": low_ms, "high": high_ms},
            "conversion": conversion,
            "components": components,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "proxy_estimate",
        "result_scope": "agx_orin_noncomparable_public_spec_extrapolation",
        "formal_performance_eligible": False,
        "performance_comparison_eligible": False,
        "local_sampling_performance_eligible": normalization.get(
            "local_sampling_performance_eligible", False
        ),
        "method": reference.get("method"),
        "reason": (
            "public specifications do not predict end-to-end AGX Orin time; "
            "this extrapolation is not a calibration, plausibility check, or anchor"
        ),
        "ratios_seconds_per_work": ratios,
        "stages": estimated_stages,
        "total": {
            "local_ms": total_local_ms,
            "estimated_orin_ms": total_estimated_ms,
            "interval_ms": {"low": total_low_ms, "high": total_high_ms},
        },
        "uncertainty_status": "category_intervals_with_explicit_proxy_assumptions",
        "reference_devices": {"local": reference["local"], "target": reference["target"]},
        "reference_uncertainty": uncertainty,
        "workload_iterations": workload_iterations,
    }


def estimate_files(
    normalization_path: Path, reference_path: Path, output_path: Path,
    *, workload_iterations: int | None = None,
) -> dict[str, Any]:
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if not isinstance(normalization, Mapping):
        raise OrinEstimateError("normalization must be a JSON object")
    reference = _reference(reference_path)
    result = estimate_normalized(
        normalization, reference, workload_iterations=workload_iterations,
    )
    result["sources"] = {
        "normalization": {
            "path": str(normalization_path.resolve()),
            "sha256": sha256_file(normalization_path),
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_orin_estimate")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload-iterations", type=int, default=None,
        help="number of training iterations represented by the proxy total",
    )
    args = parser.parse_args(argv)
    result = estimate_files(
        args.normalization, args.reference, args.output,
        workload_iterations=args.workload_iterations,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "estimated_orin_ms": result["total"]["estimated_orin_ms"],
        "formal_performance_eligible": result["formal_performance_eligible"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
