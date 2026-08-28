"""Build a bounded performance estimate from representative NCU samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "gala-ncu-sampled-performance-v1"
_IDENTITY_FIELDS = (
    "iteration",
    "stage",
    "call_index",
    "kernel_ordinal_in_call",
    "kernel_name_ordinal_in_capture",
    "kernel_name",
)
_METRIC_NAMES = (
    "atomic_requests",
    "dram_read_bytes",
    "dram_write_bytes",
    "fp32_fadd",
    "fp32_ffma",
    "fp32_fmul",
    "xu_instructions",
)
_OPERATION_NAMES = (
    "atomic_operations",
    "conversion_operations",
    "exp_operations",
    "log_operations",
    "rcp_operations",
    "sqrt_operations",
    "xu_auxiliary_operations",
)
_COUNTER_NAMES = (*_METRIC_NAMES, *_OPERATION_NAMES)


def _identity(launch: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(launch[field] for field in _IDENTITY_FIELDS)


def _threads(launch: Mapping[str, Any]) -> int:
    grid = launch.get("grid_size", launch.get("grid"))
    block = launch.get("block_size", launch.get("block"))
    if not isinstance(grid, Sequence) or not isinstance(block, Sequence):
        raise ValueError("NCU launch is missing grid or block dimensions")
    values = tuple(int(value) for value in (*grid, *block))
    if len(values) != 6 or any(value <= 0 for value in values):
        raise ValueError("NCU launch dimensions must contain six positive integers")
    result = 1
    for value in values:
        result *= value
    return result


def _metrics(launch: Mapping[str, Any]) -> dict[str, float]:
    values = launch.get("metrics")
    if not isinstance(values, Mapping):
        raise ValueError("NCU launch is missing metrics")
    return {
        **{name: float(values.get(name, 0.0)) for name in _METRIC_NAMES},
        **{name: float(launch.get(name, 0.0)) for name in _OPERATION_NAMES},
    }


def _median_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        name: float(median(_metrics(sample)[name] for sample in samples))
        for name in _COUNTER_NAMES
    }


def _scaled_median_metrics(
    samples: Sequence[Mapping[str, Any]], target_threads: int,
) -> tuple[dict[str, float], float]:
    rates = {
        name: [
            _metrics(sample)[name] / _threads(sample)
            for sample in samples
        ]
        for name in _COUNTER_NAMES
    }
    values = {
        name: float(median(items) * target_threads)
        for name, items in rates.items()
    }
    dispersion = 0.0
    for items in rates.values():
        center = float(median(items))
        if center <= 0 or len(items) < 2:
            continue
        relative = float(median(abs(value - center) for value in items) / center)
        dispersion = max(dispersion, relative)
    return values, dispersion


def _closest_samples(
    candidates: Iterable[Mapping[str, Any]], *, target_threads: int,
    target_iteration: int, limit: int = 7,
) -> list[Mapping[str, Any]]:
    def distance(sample: Mapping[str, Any]) -> tuple[float, int]:
        ratio = max(_threads(sample), target_threads) / min(_threads(sample), target_threads)
        return (abs(math.log2(ratio)), abs(int(sample.get("iteration", 0)) - target_iteration))

    return sorted(candidates, key=distance)[:limit]


def _summarize_stage(totals: Mapping[str, float], launches: int) -> dict[str, Any]:
    ffma = float(totals.get("fp32_ffma", 0.0))
    fadd = float(totals.get("fp32_fadd", 0.0))
    fmul = float(totals.get("fp32_fmul", 0.0))
    return {
        **{name: float(totals.get(name, 0.0)) for name in _COUNTER_NAMES},
        "kernel_launch_count": launches,
        "fp32_operations": 2.0 * ffma + fadd + fmul,
        "fp32_fma_equivalent": ffma + (fadd + fmul) / 2.0,
        "dram_bytes": float(totals.get("dram_read_bytes", 0.0))
        + float(totals.get("dram_write_bytes", 0.0)),
        "unclassified_transcendental_operations": 0.0,
        "weight_eligible": True,
        "eligibility_reason": "representative_sampling_estimate",
    }


def build_sampled_performance(
    plan: Mapping[str, Any],
    primary_profiles: Sequence[Mapping[str, Any]],
    supplemental_profiles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Estimate the planned counter population without enforcing hash identity."""

    expected: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    range_stages = {str(value) for value in plan.get("range_capture_stages", ())}
    for group in plan.get("capture_groups", ()):
        if not isinstance(group, Mapping) or group.get("capture_mode") == "nvtx_ranges":
            continue
        for launch in group.get("expected_launches", ()):
            if isinstance(launch, Mapping):
                expected[_identity(launch)] = launch

    selected: dict[tuple[Any, ...], tuple[int, Mapping[str, Any]]] = {}
    pools: list[Mapping[str, Any]] = []
    ignored_profiles = 0
    ignored_launches = 0
    duplicate_measurements = 0
    for priority, profiles in enumerate((primary_profiles, supplemental_profiles)):
        for profile in profiles:
            if profile.get("status") != "passed":
                ignored_profiles += 1
                continue
            launches = profile.get("launches")
            if not isinstance(launches, list):
                ignored_profiles += 1
                continue
            for launch in launches:
                if not isinstance(launch, Mapping):
                    ignored_launches += 1
                    continue
                try:
                    _metrics(launch)
                    _threads(launch)
                except (KeyError, TypeError, ValueError):
                    ignored_launches += 1
                    continue
                pools.append(launch)
                try:
                    key = _identity(launch)
                except KeyError:
                    continue
                if key not in expected:
                    continue
                previous = selected.get(key)
                if previous is None or priority < previous[0]:
                    selected[key] = (priority, launch)
                else:
                    duplicate_measurements += 1

    by_stage_name: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_stage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for launch in pools:
        stage = str(launch.get("stage", ""))
        name = str(launch.get("kernel_name", ""))
        if stage in range_stages:
            continue
        by_stage_name[(stage, name)].append(launch)
        by_name[name].append(launch)
        by_stage[stage].append(launch)

    signatures: dict[tuple[int, str], Mapping[str, Any]] = {}
    for signature in plan.get("signatures", ()):
        if not isinstance(signature, Mapping) or str(signature.get("stage")) in range_stages:
            continue
        for representative in signature.get("representative_iterations", ()):
            if isinstance(representative, Mapping):
                signatures[(int(representative["iteration"]), str(signature["signature_id"]))] = {
                    "signature": signature,
                    "representative": representative,
                }

    selected_by_signature: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for key, (_, launch) in selected.items():
        expected_launch = expected[key]
        selected_by_signature[
            (int(expected_launch["iteration"]), str(expected_launch["signature_id"]))
        ].append(launch)

    stage_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stage_launches: Counter[str] = Counter()
    stage_estimated: Counter[str] = Counter()
    stage_max_dispersion: dict[str, float] = defaultdict(float)
    aggregated: list[dict[str, Any]] = []
    directly_measured_launches = 0
    signature_covered_weight = 0
    total_weight = 0
    fallback_modes: Counter[str] = Counter()
    for signature_key, item in sorted(signatures.items()):
        signature = item["signature"]
        representative = item["representative"]
        stage = str(signature["stage"])
        name = str(signature["kernel_name"])
        multiplicity = int(representative["multiplicity"])
        target = {
            "grid": signature["grid"],
            "block": signature["block"],
        }
        target_threads = _threads(target)
        exact = selected_by_signature.get(signature_key, [])
        total_weight += multiplicity
        dispersion = 0.0
        if exact:
            signature_covered_weight += multiplicity
            directly_measured_launches += min(len(exact), multiplicity)
        if len(exact) >= multiplicity:
            per_launch = _median_metrics(exact)
            totals = {
                metric: sum(_metrics(sample)[metric] for sample in exact[:multiplicity])
                for metric in _COUNTER_NAMES
            }
            mode = "exact_observed_launches"
            sample_count = multiplicity
        elif exact:
            per_launch = _median_metrics(exact)
            totals = {metric: value * multiplicity for metric, value in per_launch.items()}
            mode = "partial_exact_extrapolation"
            sample_count = len(exact)
        else:
            candidates = by_stage_name.get((stage, name), [])
            mode = "same_stage_kernel_extrapolation"
            if not candidates:
                candidates = by_name.get(name, [])
                mode = "same_kernel_extrapolation"
            if not candidates:
                candidates = by_stage.get(stage, [])
                mode = "stage_extrapolation"
            if not candidates:
                raise ValueError(f"no measured NCU sample is available for stage {stage}")
            samples = _closest_samples(
                candidates,
                target_threads=target_threads,
                target_iteration=int(representative["iteration"]),
            )
            per_launch, dispersion = _scaled_median_metrics(samples, target_threads)
            totals = {metric: value * multiplicity for metric, value in per_launch.items()}
            sample_count = len(samples)
        fallback_modes[mode] += multiplicity
        if mode != "exact_observed_launches":
            stage_estimated[stage] += multiplicity
        stage_max_dispersion[stage] = max(stage_max_dispersion[stage], dispersion)
        for metric, value in totals.items():
            stage_totals[stage][metric] += value
        stage_launches[stage] += multiplicity
        aggregated.append({
            "iteration": int(representative["iteration"]),
            "stage": stage,
            "signature_id": str(signature["signature_id"]),
            "kernel_name": name,
            "grid_size": [int(value) for value in signature["grid"]],
            "measured_grid_sizes": sorted({
                tuple(int(value) for value in sample.get("grid_size", signature["grid"]))
                for sample in exact
            }),
            "block_size": [int(value) for value in signature["block"]],
            "threads_launched_per_launch": target_threads,
            "multiplicity": multiplicity,
            "measured_sample_count": sample_count,
            "measured_metrics_per_launch": per_launch,
            "multiplicity_expanded_metrics": totals,
            "aggregation_mode": mode,
            "relative_mad_upper_bound": dispersion,
        })

    range_launches = 0
    range_records: list[dict[str, Any]] = []
    seen_range_launches: set[tuple[Any, ...]] = set()
    for profile in (*primary_profiles, *supplemental_profiles):
        identity = profile.get("run_identity")
        if not isinstance(identity, Mapping) or identity.get("capture_mode") != "nvtx_ranges":
            continue
        for launch in profile.get("launches", ()):
            if not isinstance(launch, Mapping):
                continue
            stage = str(launch.get("stage", ""))
            if stage not in range_stages:
                continue
            range_identity = (
                launch.get("selected_launch_id"),
                launch.get("iteration"),
                stage,
                launch.get("call_index"),
                launch.get("kernel_name"),
                launch.get("launch_id"),
            )
            if range_identity in seen_range_launches:
                continue
            seen_range_launches.add(range_identity)
            values = _metrics(launch)
            for metric, value in values.items():
                stage_totals[stage][metric] += value
            stage_launches[stage] += 1
            range_launches += 1
            range_records.append({
                "stage": stage,
                "iteration": int(launch.get("iteration", 0)),
                "signature_id": str(launch.get("signature_id", "")),
                "kernel_name": str(launch.get("kernel_name", "")),
                "grid_size": [int(value) for value in launch["grid_size"]],
                "block_size": [int(value) for value in launch["block_size"]],
                "threads_launched": _threads(launch),
                "multiplicity": 1,
                "metrics": {name: values[name] for name in _METRIC_NAMES},
                **{name: values[name] for name in _OPERATION_NAMES},
                "aggregation_mode": "exact_observed_range_launch",
            })

    stage_summaries: dict[str, Any] = {}
    for stage, totals in sorted(stage_totals.items()):
        summary = _summarize_stage(totals, stage_launches[stage])
        summary.update({
            "estimated_kernel_launch_count": stage_estimated[stage],
            "estimated_launch_fraction": (
                stage_estimated[stage] / stage_launches[stage]
                if stage_launches[stage] else 0.0
            ),
            "maximum_relative_mad": stage_max_dispersion[stage],
        })
        stage_summaries[stage] = summary

    normalized_launches = []
    for item in aggregated:
        expanded = item["multiplicity_expanded_metrics"]
        normalized_launches.append({
            "stage": item["stage"],
            "iteration": item["iteration"],
            "signature_id": item["signature_id"],
            "kernel_name": item["kernel_name"],
            "grid_size": item["grid_size"],
            "block_size": item["block_size"],
            "threads_launched": item["threads_launched_per_launch"],
            "multiplicity": item["multiplicity"],
            "metrics": {
                name: expanded[name] for name in _METRIC_NAMES
            },
            **{name: expanded[name] for name in _OPERATION_NAMES},
            "aggregation_mode": item["aggregation_mode"],
        })
    normalized_launches.extend(range_records)

    expected_count = len(expected)
    exact_count = len(selected)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_sampling_estimate",
        "result_scope": "representative_gpu_performance_estimate",
        "formal_performance_eligible": False,
        "sampling_performance_eligible": True,
        "hash_validation": "disabled_by_user_request",
        "primary_profile_count": len(primary_profiles),
        "supplemental_profile_count": len(supplemental_profiles),
        "sampling_group_count": len(primary_profiles) + len(supplemental_profiles),
        "sampling_group_policy": "measured_primary_profiles_with_optional_supplemental_profiles",
        "ignored_profile_count": ignored_profiles,
        "ignored_launch_count": ignored_launches,
        "duplicate_measurement_count": duplicate_measurements,
        "expected_invocation_launch_count": expected_count,
        "exact_content_matched_launch_count": exact_count,
        "exact_content_coverage": exact_count / expected_count if expected_count else 0.0,
        "directly_measured_representative_launch_count": directly_measured_launches,
        "representative_signature_covered_launch_count": signature_covered_weight,
        "representative_weighted_launch_count": total_weight,
        "representative_weighted_signature_coverage": (
            signature_covered_weight / total_weight if total_weight else 0.0
        ),
        "observed_range_launch_count": range_launches,
        "fallback_modes": dict(sorted(fallback_modes.items())),
        "uncertainty_status": "sampled_extrapolation_reported_per_stage",
        "aggregated_signatures": aggregated,
        "launches": normalized_launches,
        "stage_summaries": stage_summaries,
    }


def _read_profiles(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_ncu_sampling")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--primary-profile", type=Path, action="append", default=[])
    parser.add_argument("--supplemental-profile", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.primary_profile:
        parser.error("at least one --primary-profile is required")
    result = build_sampled_performance(
        json.loads(args.plan.read_text(encoding="utf-8")),
        _read_profiles(args.primary_profile),
        _read_profiles(args.supplemental_profile),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "exact_content_coverage": result["exact_content_coverage"],
        "weighted_signature_coverage": result[
            "representative_weighted_signature_coverage"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
