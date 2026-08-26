"""Derive auditable local-to-Orin GPU stage normalization.

The normalizer consumes only measured stage times, Nsight counters, and the
portable calibration vectors.  It never substitutes a peak-specification
ratio or a neighboring calibration category.  Missing evidence leaves the
result explicitly provisional and withholds an Orin time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from gala_sim.identity import sha256_file


NORMALIZATION_SCHEMA_VERSION = "gala-gpu-stage-normalization-v1"
CALIBRATION_CATEGORIES = (
    "fp32_fma", "exp", "log", "rcp", "sqrt", "memory_bandwidth",
    "atomic", "kernel_launch", "synchronization",
)
_WORK_FIELDS = {
    "fp32_fma": "fp32_fma_equivalent",
    "exp": "exp_operations",
    "log": "log_operations",
    "rcp": "rcp_operations",
    "sqrt": "sqrt_operations",
    "memory_bandwidth": "dram_bytes",
    "atomic": "atomic_requests",
    "kernel_launch": "kernel_launch_count",
    "synchronization": "synchronization_calls",
}
_WORK_UNITS = {
    "fp32_fma": "fma",
    "exp": "element",
    "log": "element",
    "rcp": "element",
    "sqrt": "element",
    "memory_bandwidth": "byte",
    "atomic": "atomic_request",
    "kernel_launch": "launch",
    "synchronization": "synchronize_call",
}


class NormalizationError(ValueError):
    """Raised when an input violates the normalization contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NormalizationError(f"JSON object required: {path}")
    return value


def _as_positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise NormalizationError(f"{label} is not numeric") from error
    if not math.isfinite(number) or number <= 0:
        raise NormalizationError(f"{label} must be finite and positive")
    return number


def _numeric_work_key(key: str, category: str) -> float | None:
    if category == "synchronization" and key == "idle":
        return 1.0
    try:
        value = float(key)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def _calibration_points(
    calibration: Mapping[str, Any], category: str,
) -> tuple[tuple[float, float, str], ...]:
    vectors = calibration.get("vectors")
    if not isinstance(vectors, Mapping):
        raise NormalizationError("calibration vector has no vectors object")
    entries = vectors.get(category)
    if not isinstance(entries, Mapping):
        return ()
    points: list[tuple[float, float, str]] = []
    for raw_key, raw_entry in entries.items():
        if not isinstance(raw_entry, Mapping):
            raise NormalizationError(f"invalid {category} calibration entry")
        key = _numeric_work_key(str(raw_key), category)
        if key is None:
            continue
        unit = str(raw_entry.get("work_unit", ""))
        rate = _as_positive_number(
            raw_entry.get("median_seconds_per_work"),
            f"{category}[{raw_key}].median_seconds_per_work",
        )
        if not unit:
            raise NormalizationError(f"{category}[{raw_key}] has no work unit")
        points.append((key, rate, unit))
    return tuple(sorted(points))


def _rate_at(
    calibration: Mapping[str, Any], category: str, work: float,
) -> tuple[float, str, str] | None:
    """Return rate, unit, and exact/interpolated coverage mode."""

    points = _calibration_points(calibration, category)
    if not points or work <= 0 or not math.isfinite(work):
        return None
    if category in {"kernel_launch", "synchronization"}:
        if len(points) != 1:
            raise NormalizationError(f"{category} calibration requires one rate")
        point = points[0]
        return point[1], point[2], "exact"
    if work < points[0][0] or work > points[-1][0]:
        return None
    for key, rate, unit in points:
        if work == key:
            return rate, unit, "exact"
    for lower, upper in zip(points, points[1:]):
        if lower[0] < work < upper[0]:
            fraction = (work - lower[0]) / (upper[0] - lower[0])
            if lower[2] != upper[2]:
                raise NormalizationError(
                    f"{category} calibration work units are inconsistent"
                )
            rate = lower[1] + fraction * (upper[1] - lower[1])
            return rate, lower[2], "interpolated"
    return None


def _stage_work(summary: Mapping[str, Any]) -> dict[str, float]:
    work: dict[str, float] = {}
    for category, field in _WORK_FIELDS.items():
        raw = summary.get(field, 0)
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise NormalizationError(f"stage field {field} is not numeric") from error
        if not math.isfinite(value) or value < 0:
            raise NormalizationError(f"stage field {field} is invalid")
        if value > 0:
            work[category] = value
    return work


def _launch_amounts(record: Mapping[str, Any]) -> dict[str, float]:
    """Translate one NCU launch into measured work amounts.

    The calibration batch is the launch's actual thread footprint, while the
    amount is the counter value.  Keeping these separate avoids treating a
    whole stage's aggregate instruction count as one calibration batch.
    """

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    values = {
        "fp32_fma": 2.0 * float(metrics.get("fp32_ffma", 0.0))
        + (float(metrics.get("fp32_fadd", 0.0))
           + float(metrics.get("fp32_fmul", 0.0))) / 2.0,
        "memory_bandwidth": float(metrics.get("dram_read_bytes", 0.0))
        + float(metrics.get("dram_write_bytes", 0.0)),
        "atomic": float(metrics.get("atomic_requests", 0.0)),
        "kernel_launch": 1.0,
    }
    for category in ("exp", "log", "rcp", "sqrt"):
        if category in record:
            values[category] = float(record[category])
    return {category: amount for category, amount in values.items() if amount > 0}


def _launch_batch(category: str, record: Mapping[str, Any]) -> float | None:
    try:
        threads = float(record["threads_launched"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(threads) or threads <= 0:
        return None
    if category in {"kernel_launch", "synchronization"}:
        return 1.0
    return threads


def _launch_component_estimates(
    ncu_profile: Mapping[str, Any],
    local_calibration: Mapping[str, Any],
    orin_calibration: Mapping[str, Any],
    stage: str,
) -> tuple[
    dict[str, float], dict[str, float], dict[str, float], dict[str, float],
    dict[str, str], dict[str, list[str]], list[str],
] | None:
    launches = ncu_profile.get("launches")
    if not isinstance(launches, list):
        return None
    local_costs: dict[str, float] = {}
    orin_costs: dict[str, float] = {}
    work_totals: dict[str, float] = {}
    units: dict[str, str] = {}
    coverage: dict[str, list[str]] = {}
    reasons: list[str] = []
    found = False
    for launch in launches:
        if not isinstance(launch, Mapping) or launch.get("stage") != stage:
            continue
        found = True
        for category, amount in _launch_amounts(launch).items():
            batch = _launch_batch(category, launch)
            if batch is None:
                reasons.append(f"missing_launch_batch:{category}")
                continue
            local_point = _rate_at(local_calibration, category, batch)
            orin_point = _rate_at(orin_calibration, category, batch)
            if local_point is None:
                reasons.append(f"local_calibration_out_of_range:{category}")
                continue
            local_rate, local_unit, local_mode = local_point
            expected_unit = _WORK_UNITS[category]
            if local_unit != expected_unit:
                reasons.append(
                    f"work_unit_mismatch:{category}:{local_unit}!={expected_unit}"
                )
                continue
            local_costs[category] = local_costs.get(category, 0.0) + amount * local_rate
            work_totals[category] = work_totals.get(category, 0.0) + amount
            units[category] = local_unit
            coverage.setdefault(category, []).append(f"local_{local_mode}@{batch:g}")
            if orin_point is None:
                reasons.append(f"orin_calibration_missing_or_out_of_range:{category}")
                continue
            orin_rate, orin_unit, orin_mode = orin_point
            if orin_unit != local_unit:
                raise NormalizationError(f"{stage}:{category} local/Orin work units differ")
            orin_costs[category] = orin_costs.get(category, 0.0) + amount * orin_rate
            coverage.setdefault(category, []).append(
                f"orin_{orin_mode}@{batch:g}"
            )
    local_rates = {
        category: local_costs[category] / work_totals[category]
        for category in local_costs
    }
    orin_rates = {
        category: orin_costs[category] / work_totals[category]
        for category in orin_costs
    }
    return (
        local_costs, orin_costs, local_rates, orin_rates, units,
        coverage, reasons,
    ) if found else None


def _stage_times(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    summaries = profile.get("stage_summaries")
    if not isinstance(summaries, Mapping) or not summaries:
        raise NormalizationError("stage profile has no stage_summaries")
    return summaries


def normalize_stage_profiles(
    stage_profile: Mapping[str, Any],
    ncu_profile: Mapping[str, Any],
    local_calibration: Mapping[str, Any],
    orin_calibration: Mapping[str, Any],
    *,
    nsys_profile: Mapping[str, Any] | None = None,
    required_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build stage weights and Orin-equivalent times from measured evidence."""

    global_reasons: list[str] = []
    local_suite = local_calibration.get("suite", {})
    orin_suite = orin_calibration.get("suite", {})
    local_configuration = (
        local_suite.get("configuration_sha256")
        if isinstance(local_suite, Mapping) else None
    )
    orin_configuration = (
        orin_suite.get("configuration_sha256")
        if isinstance(orin_suite, Mapping) else None
    )
    if local_calibration.get("status") != "passed":
        global_reasons.append("local_calibration_not_passed")
    if orin_calibration.get("status") != "passed":
        global_reasons.append("orin_calibration_missing_or_not_passed")
    if (
        local_configuration is not None
        and orin_configuration is not None
        and local_configuration != orin_configuration
    ):
        global_reasons.append("calibration_configuration_mismatch")
    kernel_coverage = (
        nsys_profile.get("kernel_coverage")
        if isinstance(nsys_profile, Mapping) else None
    )
    if (
        not isinstance(nsys_profile, Mapping)
        or nsys_profile.get("status") != "passed"
        or not isinstance(kernel_coverage, Mapping)
        or kernel_coverage.get("status") != "complete"
    ):
        global_reasons.append("nsys_kernel_coverage_incomplete")
    profile_summaries = _stage_times(stage_profile)
    ncu_summaries = ncu_profile.get("stage_summaries")
    if not isinstance(ncu_summaries, Mapping):
        raise NormalizationError("Nsight Compute profile has no stage_summaries")
    required = tuple(required_stages) or tuple(sorted(
        stage for stage in profile_summaries if stage != "iteration_total"
    ))
    stages: dict[str, Any] = {}
    provisional_reasons: list[str] = list(global_reasons)
    for stage in required:
        reasons: list[str] = []
        reasons.extend(global_reasons)
        missing_evidence = False
        local = profile_summaries.get(stage)
        counters = ncu_summaries.get(stage)
        if not isinstance(local, Mapping):
            reasons.append("missing_stage_timing")
            missing_evidence = True
        if not isinstance(counters, Mapping):
            reasons.append("missing_ncu_counters")
            missing_evidence = True
        if missing_evidence:
            provisional_reasons.extend(f"{stage}:{reason}" for reason in reasons)
            stages[stage] = {
                "status": "provisional_normalization",
                "reasons": reasons,
                "weights": {},
                "weight_sum": 0.0,
                "local_ms": None,
                "orin_ms": None,
            }
            continue
        local_ms = _as_positive_number(local.get("total_ms"), f"{stage}.total_ms")
        work = _stage_work(counters)
        if not work:
            reasons.append("no_counted_work")
        if float(counters.get("unclassified_transcendental_operations", 0)) > 0:
            reasons.append("unclassified_transcendental_operations")
        if counters.get("weight_eligible") is False:
            reasons.append("ncu_weight_ineligible")
        launch_estimates = _launch_component_estimates(
            ncu_profile, local_calibration, orin_calibration, stage
        )
        if launch_estimates is not None:
            (
                local_components, orin_components, local_rates, orin_rates,
                units, coverage_lists, estimate_reasons,
            ) = launch_estimates
            reasons.extend(estimate_reasons)
            coverage = {
                category: ",".join(sorted(set(modes)))
                for category, modes in coverage_lists.items()
            }
        else:
            local_components = {}
            local_rates = {}
            orin_rates = {}
            units = {}
            coverage = {}
            for category, amount in work.items():
                local_point = _rate_at(local_calibration, category, amount)
                orin_point = _rate_at(orin_calibration, category, amount)
                if local_point is None:
                    reasons.append(f"local_calibration_out_of_range:{category}")
                    continue
                local_rate, local_unit, local_mode = local_point
                expected_unit = _WORK_UNITS[category]
                if local_unit != expected_unit:
                    reasons.append(
                        f"work_unit_mismatch:{category}:{local_unit}!={expected_unit}"
                    )
                    continue
                local_components[category] = amount * local_rate
                local_rates[category] = local_rate
                units[category] = local_unit
                coverage[category] = local_mode
                if orin_point is None:
                    reasons.append(f"orin_calibration_missing_or_out_of_range:{category}")
                    continue
                orin_rate, orin_unit, orin_mode = orin_point
                if orin_unit != local_unit:
                    raise NormalizationError(f"{stage}:{category} local/Orin work units differ")
                orin_rates[category] = orin_rate
                if orin_mode != local_mode:
                    coverage[category] = f"local_{local_mode}_orin_{orin_mode}"
        component_total = sum(local_components.values())
        if component_total <= 0:
            reasons.append("no_calibrated_work")
            weights: dict[str, float] = {}
        else:
            weights = {
                category: value / component_total
                for category, value in sorted(local_components.items())
            }
        weight_sum = float(sum(weights.values()))
        if weights and not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise NormalizationError(f"{stage} normalized weights do not sum to one")
        orin_ms = None
        if not reasons and set(orin_rates) == set(weights):
            if launch_estimates is not None:
                orin_ms = local_ms * sum(orin_components.values()) / component_total
            else:
                conversion = sum(weights[category] * orin_rates[category] / local_rates[category]
                                  for category in weights)
                orin_ms = local_ms * conversion
        else:
            if set(orin_rates) != set(weights):
                reasons.append("incomplete_orin_rate_vector")
        if reasons:
            provisional_reasons.extend(f"{stage}:{reason}" for reason in sorted(set(reasons)))
        stages[stage] = {
            "status": "passed" if not reasons else "provisional_normalization",
            "reasons": sorted(set(reasons)),
            "local_ms": local_ms,
            "orin_ms": orin_ms,
            "work": work,
            "weights": weights,
            "weight_sum": weight_sum,
            "local_seconds_per_work": local_rates,
            "orin_seconds_per_work": orin_rates,
            "work_units": units,
            "calibration_coverage": coverage,
        }
    status = "passed" if not provisional_reasons else "provisional_normalization"
    formal = status == "passed"
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "status": status,
        "result_scope": "gpu_stage_normalization",
        "formal_performance_eligible": formal,
        "local_calibration_status": local_calibration.get("status"),
        "orin_calibration_status": orin_calibration.get("status"),
        "local_calibration_device": local_calibration.get("device"),
        "orin_calibration_device": orin_calibration.get("device"),
        "required_stages": list(required),
        "stages": stages,
        "provisional_reasons": sorted(set(provisional_reasons)),
        "orin_total_ms": (
            float(sum(stage["orin_ms"] for stage in stages.values()
                      if stage.get("orin_ms") is not None))
            if formal else None
        ),
    }


def normalize_files(
    stage_profile_path: Path,
    nsys_profile_path: Path,
    ncu_profile_path: Path,
    local_calibration_path: Path,
    orin_calibration_path: Path | None,
    output_path: Path,
    required_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    orin_calibration = (
        _read_json(orin_calibration_path)
        if orin_calibration_path is not None
        else {"status": "unavailable", "vectors": {}, "device": None}
    )
    result = normalize_stage_profiles(
        _read_json(stage_profile_path), _read_json(ncu_profile_path),
        _read_json(local_calibration_path), orin_calibration,
        nsys_profile=_read_json(nsys_profile_path),
        required_stages=required_stages,
    )
    result["sources"] = {
        "stage_profile": {
            "path": str(stage_profile_path.resolve()),
            "sha256": sha256_file(stage_profile_path),
        },
        "nsys_profile": {
            "path": str(nsys_profile_path.resolve()),
            "sha256": sha256_file(nsys_profile_path),
        },
        "ncu_profile": {
            "path": str(ncu_profile_path.resolve()),
            "sha256": sha256_file(ncu_profile_path),
        },
        "local_calibration": {
            "path": str(local_calibration_path.resolve()),
            "sha256": sha256_file(local_calibration_path),
        },
        "orin_calibration": (
            {
                "path": str(orin_calibration_path.resolve()),
                "sha256": sha256_file(orin_calibration_path),
            }
            if orin_calibration_path is not None else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_normalization")
    parser.add_argument("--stage-profile", type=Path, required=True)
    parser.add_argument("--nsys-profile", type=Path, required=True)
    parser.add_argument("--ncu-profile", type=Path, required=True)
    parser.add_argument("--local-calibration", type=Path, required=True)
    parser.add_argument("--orin-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-stage", action="append", default=[])
    args = parser.parse_args(argv)
    result = normalize_files(
        args.stage_profile, args.nsys_profile, args.ncu_profile, args.local_calibration,
        args.orin_calibration, args.output, tuple(args.required_stage),
    )
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
