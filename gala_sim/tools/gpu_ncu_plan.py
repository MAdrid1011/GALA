"""Build an auditable NCU sampling plan from exact NSYS launch inventories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from gala_sim.identity import canonical_json, sha256_bytes, sha256_file
from gala_sim.tools.gpu_profile_campaign import GpuProfileCampaign


CONFIG_SCHEMA_VERSION = "gala-ncu-launch-signature-config-v2"
PLAN_SCHEMA_VERSION = "gala-ncu-launch-signature-plan-v2"
_SUPPORTED_SIGNATURE_FIELDS = ("stage", "kernel_name", "grid", "block")
_SUPPORTED_OCCURRENCES = ("first", "middle", "last")
_SUPPORTED_NCU_OPTIONS = {
    "replay_mode": {"kernel"},
    "cache_control": {"all", "none"},
    "clock_control": {"base", "none"},
    "kernel_name_base": {"demangled"},
}
_SUPPORTED_NCU_SECTIONS = {"SourceCounters"}


@dataclass(frozen=True)
class NcuPlanConfig:
    path: Path
    campaign: Path
    campaign_sha256: str
    signature_fields: tuple[str, ...]
    counter_scope: str
    validation_occurrences: tuple[str, ...]
    metrics: tuple[str, ...]
    sections: tuple[str, ...]
    ncu_options: Mapping[str, str]

    @classmethod
    def load(cls, path: Path) -> "NcuPlanConfig":
        path = path.resolve()
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or document.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported NCU launch-signature configuration")
        repository = path.parents[2]
        options = document.get("ncu")
        if not isinstance(options, Mapping):
            raise ValueError("NCU configuration has no tool options")
        config = cls(
            path=path,
            campaign=(repository / str(document["campaign"])).resolve(),
            campaign_sha256=str(document["campaign_sha256"]),
            signature_fields=tuple(str(value) for value in document.get("signature_fields", ())),
            counter_scope=str(document.get("counter_scope", "")),
            validation_occurrences=tuple(
                str(value) for value in document.get("validation_occurrences", ())
            ),
            metrics=tuple(str(value) for value in document.get("metrics", ())),
            sections=tuple(str(value) for value in document.get("sections", ())),
            ncu_options={str(name): str(value) for name, value in options.items()},
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if not self.campaign.is_file() or sha256_file(self.campaign) != self.campaign_sha256:
            raise ValueError("NCU configuration campaign identity mismatch")
        if self.signature_fields != _SUPPORTED_SIGNATURE_FIELDS:
            raise ValueError("NCU signature fields must preserve stage, name, grid, and block")
        if self.counter_scope != "representative_iteration":
            raise ValueError("NCU counters must remain scoped to each representative iteration")
        if (
            not self.validation_occurrences
            or len(set(self.validation_occurrences)) != len(self.validation_occurrences)
            or any(value not in _SUPPORTED_OCCURRENCES for value in self.validation_occurrences)
        ):
            raise ValueError("NCU validation occurrence policy is invalid")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("NCU metrics must be nonempty and unique")
        if (
            not self.sections
            or len(set(self.sections)) != len(self.sections)
            or any(value not in _SUPPORTED_NCU_SECTIONS for value in self.sections)
        ):
            raise ValueError("NCU sections are incomplete or unsupported")
        if set(self.ncu_options) != set(_SUPPORTED_NCU_OPTIONS):
            raise ValueError("NCU tool options are incomplete or unsupported")
        for name, allowed in _SUPPORTED_NCU_OPTIONS.items():
            if self.ncu_options[name] not in allowed:
                raise ValueError(f"NCU tool option {name} is unsupported")


def _shape(value: Any, field: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"NSYS kernel {field} is invalid")
    return tuple(value)  # type: ignore[return-value]


def _signature(stage: str, kernel: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...], tuple[int, ...]]:
    name = kernel.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("NSYS kernel name is invalid")
    return stage, name, _shape(kernel.get("grid"), "grid"), _shape(kernel.get("block"), "block")


def _signature_document(
    key: tuple[str, str, tuple[int, ...], tuple[int, ...]],
) -> dict[str, Any]:
    stage, name, grid, block = key
    return {"stage": stage, "kernel_name": name, "grid": list(grid), "block": list(block)}


def _signature_id(key: tuple[str, str, tuple[int, ...], tuple[int, ...]]) -> str:
    return sha256_bytes(canonical_json(_signature_document(key)))


def _call_kernel_sequence(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _signature_document(_signature(str(call["stage"]), kernel))
        for kernel in call["kernels"]
    ]


def _anchor_indices(count: int, policy: Iterable[str]) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("NCU signature has no observed occurrence")
    positions = {
        "first": 0,
        "middle": (count - 1) // 2,
        "last": count - 1,
    }
    return tuple(sorted({positions[name] for name in policy}))


def _load_inventory(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("NSYS inventory is not a JSON object")
    return document, sha256_file(path)


def _validate_inventory(
    profile: Mapping[str, Any], campaign_sha256: str,
) -> tuple[int, int, list[Mapping[str, Any]]]:
    identity = profile.get("run_identity")
    coverage = profile.get("kernel_coverage")
    records = coverage.get("records") if isinstance(coverage, Mapping) else None
    calls = profile.get("stage_calls")
    if (
        profile.get("status") != "passed"
        or not isinstance(identity, Mapping)
        or identity.get("status") != "passed"
        or identity.get("profiling_campaign_sha256") != campaign_sha256
        or not isinstance(coverage, Mapping)
        or coverage.get("status") != "complete"
        or not isinstance(records, list)
        or len(records) != 1
        or not isinstance(calls, list)
    ):
        raise ValueError("NSYS inventory failed identity or exact-coverage validation")
    record = records[0]
    if not isinstance(record, Mapping):
        raise ValueError("NSYS inventory coverage record is invalid")
    iteration = int(record["iteration"])
    kernel_count = int(record["kernel_count"])
    if (
        kernel_count <= 0
        or int(record.get("assigned_kernel_count", -1)) != kernel_count
        or int(record.get("unassigned_kernel_count", -1)) != 0
        or int(record.get("multiply_assigned_kernel_count", -1)) != 0
    ):
        raise ValueError("NSYS inventory does not assign every kernel exactly once")
    observed = 0
    seen_calls: set[tuple[str, int]] = set()
    validated_calls: list[Mapping[str, Any]] = []
    for raw_call in calls:
        if not isinstance(raw_call, Mapping):
            raise ValueError("NSYS stage call is invalid")
        stage = raw_call.get("stage")
        call_iteration = raw_call.get("iteration")
        call_index = raw_call.get("call_index")
        kernels = raw_call.get("kernels")
        if (
            not isinstance(stage, str)
            or not stage
            or int(call_iteration) != iteration
            or not isinstance(call_index, int)
            or isinstance(call_index, bool)
            or call_index <= 0
            or not isinstance(kernels, list)
            or int(raw_call.get("kernel_launch_count", -1)) != len(kernels)
        ):
            raise ValueError("NSYS stage call fields are inconsistent")
        call_identity = (stage, call_index)
        if call_identity in seen_calls:
            raise ValueError("NSYS stage call identity is duplicated")
        seen_calls.add(call_identity)
        for kernel in kernels:
            if not isinstance(kernel, Mapping):
                raise ValueError("NSYS stage kernel is invalid")
            _signature(stage, kernel)
        observed += len(kernels)
        validated_calls.append(raw_call)
    if observed != kernel_count:
        raise ValueError("NSYS stage calls do not reproduce exact kernel coverage")
    return iteration, kernel_count, validated_calls


def build_ncu_plan(
    config: NcuPlanConfig, inventory_paths: Iterable[Path],
) -> dict[str, Any]:
    campaign = GpuProfileCampaign.load(config.campaign)
    expected_iterations = [item.iteration for item in campaign.representatives]
    loaded: dict[int, tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]]] = {}
    for raw_path in inventory_paths:
        path = raw_path.resolve()
        profile, profile_sha256 = _load_inventory(path)
        iteration, kernel_count, calls = _validate_inventory(
            profile, config.campaign_sha256
        )
        if iteration in loaded:
            raise ValueError(f"duplicate NSYS representative iteration: {iteration}")
        loaded[iteration] = (path, profile, profile_sha256, kernel_count, calls)
    if sorted(loaded) != expected_iterations:
        raise ValueError("NSYS inventories do not exactly match representative iterations")

    roles_by_iteration = {
        item.iteration: set(item.roles) for item in campaign.representatives
    }
    observed_stage_roles: dict[str, dict[str, list[int]]] = {}
    for stage, roles in campaign.required_stage_roles.items():
        observed_stage_roles[stage] = {}
        for role in roles:
            matched = [
                iteration for iteration in expected_iterations
                if role in roles_by_iteration[iteration]
                and any(call.get("stage") == stage for call in loaded[iteration][4])
            ]
            if not matched:
                raise ValueError(f"NSYS inventories miss required stage role {stage}:{role}")
            observed_stage_roles[stage][role] = matched

    occurrences: dict[
        tuple[int, tuple[str, str, tuple[int, ...], tuple[int, ...]]],
        list[dict[str, Any]],
    ] = defaultdict(list)
    call_records: dict[tuple[int, str, int], dict[str, Any]] = {}
    for iteration in expected_iterations:
        calls = loaded[iteration][4]
        for call_order, call in enumerate(calls, start=1):
            stage = str(call["stage"])
            call_index = int(call["call_index"])
            call_key = (iteration, stage, call_index)
            call_records[call_key] = {
                "iteration": iteration,
                "stage": stage,
                "call_index": call_index,
                "call_order": call_order,
                "kernel_launch_count": len(call["kernels"]),
                "kernel_sequence_sha256": sha256_bytes(
                    canonical_json(_call_kernel_sequence(call))
                ),
                "kernel_sequence": _call_kernel_sequence(call),
            }
            name_ordinals: dict[str, int] = defaultdict(int)
            signature_counts: Counter[str] = Counter()
            for kernel_ordinal, kernel in enumerate(call["kernels"], start=1):
                key = _signature(stage, kernel)
                signature_counts[_signature_id(key)] += 1
                name_ordinals[key[1]] += 1
                occurrence_key = (iteration, key)
                occurrences[occurrence_key].append({
                    "call_index": call_index,
                    "call_order": call_order,
                    "kernel_ordinal_in_call": kernel_ordinal,
                    "kernel_name_ordinal_in_call": name_ordinals[key[1]],
                })
            call_records[call_key]["signature_counts"] = [
                {"signature_id": signature_id, "count": count}
                for signature_id, count in sorted(signature_counts.items())
            ]

    signatures: dict[
        tuple[str, str, tuple[int, ...], tuple[int, ...]], dict[str, Any]
    ] = {}
    selected_calls: set[tuple[int, str, int]] = set()
    required_sample_count = 0
    iteration_signature_count: dict[int, int] = defaultdict(int)
    for (iteration, key), values in sorted(
        occurrences.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        sample_indices = _anchor_indices(len(values), config.validation_occurrences)
        samples = []
        for index in sample_indices:
            occurrence = values[index]
            selected_calls.add((iteration, key[0], int(occurrence["call_index"])))
            samples.append({
                "signature_occurrence_ordinal": index + 1,
                **occurrence,
            })
        required_sample_count += len(samples)
        iteration_signature_count[iteration] += 1
        record = signatures.setdefault(key, {
            "signature_id": _signature_id(key),
            **_signature_document(key),
            "total_multiplicity": 0,
            "representative_iterations": [],
        })
        record["total_multiplicity"] += len(values)
        record["representative_iterations"].append({
            "iteration": iteration,
            "multiplicity": len(values),
            "required_samples": samples,
        })

    capture_groups = []
    selected_launch_count = 0
    for iteration in expected_iterations:
        calls = sorted(
            (
                call_records[key] for key in selected_calls if key[0] == iteration
            ),
            key=lambda call: int(call["call_order"]),
        )
        selected_launches = sum(int(call["kernel_launch_count"]) for call in calls)
        selected_launch_count += selected_launches
        capture_groups.append({
            "iteration": iteration,
            "selected_stage_call_count": len(calls),
            "selected_kernel_launch_count": selected_launches,
            "iteration_signature_count": iteration_signature_count[iteration],
            "nvtx_includes": [
                {
                    **call,
                    "filter": (
                        f"gala_stage:{call['stage']}:iteration={iteration}:"
                        f"call={call['call_index']}:campaign={config.campaign_sha256}/"
                    ),
                }
                for call in calls
            ],
            "ncu_arguments": [
                "--nvtx",
                "--replay-mode", config.ncu_options["replay_mode"],
                "--cache-control", config.ncu_options["cache_control"],
                "--clock-control", config.ncu_options["clock_control"],
                "--kernel-name-base", config.ncu_options["kernel_name_base"],
                *[
                    argument
                    for section in config.sections
                    for argument in ("--section", section)
                ],
                "--metrics", ",".join(config.metrics),
                *[
                    argument
                    for call in calls
                    for argument in (
                        "--nvtx-include",
                        (
                            f"gala_stage:{call['stage']}:iteration={iteration}:"
                            f"call={call['call_index']}:campaign={config.campaign_sha256}/"
                        ),
                    )
                ],
                "--check-exit-code", "1",
            ],
        })

    inventory_records = []
    observed_kernel_count = 0
    for iteration in expected_iterations:
        path, profile, profile_sha256, kernel_count, _ = loaded[iteration]
        observed_kernel_count += kernel_count
        inventory_records.append({
            "iteration": iteration,
            "path": str(path),
            "sha256": profile_sha256,
            "source": profile.get("source"),
            "source_sha256": profile.get("source_sha256"),
            "kernel_count": kernel_count,
        })
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "planned",
        "formal_performance_eligible": False,
        "eligibility_reason": "pending_measured_ncu_counters_and_signature_reuse_validation",
        "campaign": {
            "path": str(config.campaign),
            "sha256": config.campaign_sha256,
        },
        "configuration": {
            "path": str(config.path),
            "sha256": sha256_file(config.path),
        },
        "signature_fields": list(config.signature_fields),
        "counter_scope": config.counter_scope,
        "validation_occurrences": list(config.validation_occurrences),
        "counter_reuse_gate": {
            "policy": "exact_signature_within_one_representative_iteration",
            "require_all_planned_samples": True,
            "require_counter_agreement_before_multiplicity_expansion": True,
            "aggregation": "measured_signature_counters_times_exact_nsys_multiplicity",
        },
        "ncu": {
            **dict(config.ncu_options),
            "metrics": list(config.metrics),
            "required_common_arguments": [
                "--nvtx", "--replay-mode", config.ncu_options["replay_mode"],
                "--cache-control", config.ncu_options["cache_control"],
                "--clock-control", config.ncu_options["clock_control"],
                "--kernel-name-base", config.ncu_options["kernel_name_base"],
                *[
                    argument
                    for section in config.sections
                    for argument in ("--section", section)
                ],
                "--metrics", ",".join(config.metrics),
            ],
            "nvtx_filter_trailing_slash_required": True,
        },
        "coverage": {
            "status": "complete",
            "representative_iterations": expected_iterations,
            "observed_kernel_launch_count": observed_kernel_count,
            "global_signature_count": len(signatures),
            "representative_iteration_signature_count": len(occurrences),
            "required_sample_count": required_sample_count,
            "capture_group_count": len(capture_groups),
            "selected_stage_call_count": len(selected_calls),
            "selected_kernel_launch_count": selected_launch_count,
            "stage_role_coverage": observed_stage_roles,
        },
        "nsys_inventories": inventory_records,
        "capture_groups": capture_groups,
        "signatures": [signatures[key] for key in sorted(signatures)],
    }
    return {**payload, "content_sha256": sha256_bytes(canonical_json(payload))}


def validate_ncu_measurement(
    plan: Mapping[str, Any], ncu_profiles: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate NCU launches against the plan without inventing missing counts."""

    reasons: list[str] = []
    plan_hash = plan.get("content_sha256")
    plan_payload = {
        str(key): value for key, value in plan.items() if key != "content_sha256"
    }
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or not isinstance(plan_hash, str)
        or sha256_bytes(canonical_json(plan_payload)) != plan_hash
    ):
        reasons.append("ncu_plan_identity_invalid")
    expected_campaign = (
        plan.get("campaign", {}).get("sha256")
        if isinstance(plan.get("campaign"), Mapping) else None
    )
    expected_calls: dict[tuple[int, str, int], dict[str, Any]] = {}
    expected_launch_count = 0
    for group in plan.get("capture_groups", ()):
        if not isinstance(group, Mapping):
            continue
        iteration = int(group["iteration"])
        for call in group.get("nvtx_includes", ()):
            if not isinstance(call, Mapping):
                continue
            key = (iteration, str(call["stage"]), int(call["call_index"]))
            counts = {
                str(item["signature_id"]): int(item["count"])
                for item in call.get("signature_counts", ())
                if isinstance(item, Mapping)
            }
            expected_calls[key] = {
                "signature_counts": counts,
                "kernel_sequence_sha256": str(call.get("kernel_sequence_sha256", "")),
            }
            expected_launch_count += sum(counts.values())

    observed_calls: Counter[tuple[int, str, int]] = Counter()
    observed_call_sequences: dict[
        tuple[int, str, int], list[tuple[int, dict[str, Any]]]
    ] = defaultdict(list)
    observed_signatures: Counter[tuple[int, str]] = Counter()
    observed_records: list[Mapping[str, Any]] = []
    metrics_by_signature: dict[tuple[int, str], list[tuple[tuple[str, float], ...]]] = defaultdict(list)
    ordinal_keys: set[tuple[int, str, int, int]] = set()
    profile_count = 0
    for profile in ncu_profiles:
        profile_count += 1
        identity = profile.get("run_identity")
        if (
            profile.get("status") != "passed"
            or not isinstance(identity, Mapping)
            or identity.get("status") != "passed"
            or identity.get("profiling_campaign_sha256") != expected_campaign
        ):
            reasons.append("ncu_profile_identity_or_status_invalid")
        incomplete = profile.get("incomplete_launches")
        if not isinstance(incomplete, list) or incomplete:
            reasons.append("ncu_counter_fields_incomplete")
        launches = profile.get("launches")
        if not isinstance(launches, list):
            reasons.append("ncu_launch_records_missing")
            continue
        for launch in launches:
            if not isinstance(launch, Mapping):
                reasons.append("ncu_launch_record_invalid")
                continue
            try:
                iteration = int(launch["iteration"])
                stage = str(launch["stage"])
                call_index = int(launch["call_index"])
                signature_key = (
                    stage,
                    str(launch["kernel_name"]),
                    tuple(int(value) for value in launch["grid_size"]),
                    tuple(int(value) for value in launch["block_size"]),
                )
                signature_id = sha256_bytes(canonical_json(_signature_document(signature_key)))
            except (KeyError, TypeError, ValueError):
                reasons.append("ncu_launch_signature_invalid")
                continue
            call_key = (iteration, stage, call_index)
            if call_key not in expected_calls:
                reasons.append(f"unexpected_ncu_call:{iteration}:{stage}:{call_index}")
                continue
            observed_calls[call_key] += 1
            observed_signatures[(iteration, signature_id)] += 1
            try:
                ordinal = int(launch["kernel_ordinal_in_call"])
                ordinal_key = (iteration, stage, call_index, ordinal)
                if ordinal_key in ordinal_keys:
                    reasons.append(
                        f"duplicate_ncu_kernel_ordinal:{iteration}:{stage}:{call_index}:{ordinal}"
                    )
                ordinal_keys.add(ordinal_key)
                observed_call_sequences[call_key].append(
                    (ordinal, _signature_document(signature_key))
                )
            except (KeyError, TypeError, ValueError):
                reasons.append("ncu_kernel_ordinal_missing")
            metrics = launch.get("metrics")
            if not isinstance(metrics, Mapping):
                reasons.append("ncu_counter_metrics_missing")
            else:
                metrics_by_signature[(iteration, signature_id)].append(
                    tuple(sorted((str(name), float(value)) for name, value in metrics.items()))
                )
            observed_records.append(launch)

    for call_key, expected in expected_calls.items():
        expected_counts = expected["signature_counts"]
        observed_count = observed_calls[call_key]
        expected_count = sum(expected_counts.values())
        if observed_count != expected_count:
            reasons.append(
                f"ncu_call_coverage_mismatch:{call_key[0]}:{call_key[1]}:{call_key[2]}"
            )
        observed_sequence = [
            signature for _, signature in sorted(observed_call_sequences[call_key])
        ]
        if sha256_bytes(canonical_json(observed_sequence)) != expected["kernel_sequence_sha256"]:
            reasons.append(
                f"ncu_call_sequence_mismatch:{call_key[0]}:{call_key[1]}:{call_key[2]}"
            )

    expected_signature_counts: Counter[tuple[int, str]] = Counter()
    for (iteration, _stage, _call), expected in expected_calls.items():
        counts = expected["signature_counts"]
        for signature_id, count in counts.items():
            expected_signature_counts[(iteration, signature_id)] += count
    if expected_launch_count != len(observed_records):
        reasons.append("ncu_selected_launch_count_mismatch")
    for key, expected_count in expected_signature_counts.items():
        if observed_signatures[key] != expected_count:
            reasons.append(f"ncu_signature_multiplicity_mismatch:{key[0]}:{key[1]}")
    for key in set(observed_signatures) - set(expected_signature_counts):
        reasons.append(f"unexpected_ncu_signature:{key[0]}:{key[1]}")

    agreement: dict[str, Any] = {}
    for key, values in sorted(metrics_by_signature.items()):
        unique = {value for value in values}
        identifier = f"{key[0]}:{key[1]}"
        expected_count = expected_signature_counts.get(key)
        exact_observations = expected_count is not None and len(values) == expected_count
        agreement[identifier] = {
            "iteration": key[0],
            "signature_id": key[1],
            "observed_launch_count": len(values),
            "counter_values_identical": len(unique) == 1,
            "aggregation_mode": (
                "exact_observed_launches" if exact_observations else "signature_reuse"
            ),
        }
        # A repeated signature is only reusable when counters agree.  If every
        # expected ordinal was actually measured, differing counters are valid
        # workload variation and must be summed per launch instead of reused.
        if not exact_observations and len(unique) != 1:
            reasons.append(f"counter_reuse_disagreement:{key[0]}:{key[1]}")

    aggregated_signatures = []
    stage_metrics: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stage_launches: Counter[str] = Counter()
    if not reasons:
        for signature in plan.get("signatures", ()):
            if not isinstance(signature, Mapping):
                continue
            signature_id = str(signature["signature_id"])
            stage = str(signature["stage"])
            grid = [int(value) for value in signature["grid"]]
            block = [int(value) for value in signature["block"]]
            threads = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]
            for representative in signature.get("representative_iterations", ()):
                if not isinstance(representative, Mapping):
                    continue
                iteration = int(representative["iteration"])
                multiplicity = int(representative["multiplicity"])
                samples = metrics_by_signature[(iteration, signature_id)]
                measured = dict(samples[0])
                if len(samples) == multiplicity:
                    totals: dict[str, float] = defaultdict(float)
                    for sample in samples:
                        for name, value in sample:
                            totals[name] += value
                    aggregation_mode = "exact_observed_launches"
                else:
                    totals = defaultdict(float, {
                        name: value * multiplicity for name, value in measured.items()
                    })
                    aggregation_mode = "signature_reuse"
                for name, value in totals.items():
                    stage_metrics[stage][name] += value
                stage_launches[stage] += multiplicity
                aggregated_signatures.append({
                    "iteration": iteration,
                    "stage": stage,
                    "signature_id": signature_id,
                    "kernel_name": signature["kernel_name"],
                    "grid_size": grid,
                    "block_size": block,
                    "threads_launched_per_launch": threads,
                    "multiplicity": multiplicity,
                    "measured_sample_count": len(samples),
                    "measured_metrics_per_launch": measured,
                    "multiplicity_expanded_metrics": dict(totals),
                    "aggregation_mode": aggregation_mode,
                })
    stage_summaries = {}
    for stage, totals in sorted(stage_metrics.items()):
        ffma = totals.get("fp32_ffma", 0.0)
        fadd = totals.get("fp32_fadd", 0.0)
        fmul = totals.get("fp32_fmul", 0.0)
        stage_summaries[stage] = {
            **dict(sorted(totals.items())),
            "kernel_launch_count": stage_launches[stage],
            "fp32_operations": 2.0 * ffma + fadd + fmul,
            "fp32_fma_equivalent": ffma + (fadd + fmul) / 2.0,
            "dram_bytes": totals.get("dram_read_bytes", 0.0)
            + totals.get("dram_write_bytes", 0.0),
            "unclassified_transcendental_operations": totals.get(
                "xu_instructions", 0.0
            ),
            "weight_eligible": False,
            "eligibility_reason": "pending_dynamic_sass_classification",
        }

    return {
        "schema_version": "gala-ncu-measured-plan-evidence-v1",
        "status": "passed" if not reasons and profile_count > 0 else "provisional_ncu_evidence",
        "formal_performance_eligible": False,
        "campaign_sha256": expected_campaign,
        "profile_count": profile_count,
        "expected_selected_launch_count": expected_launch_count,
        "observed_selected_launch_count": len(observed_records),
        "exact_call_coverage": not any(
            reason.startswith(("unexpected_ncu_call", "ncu_call_coverage", "ncu_selected_launch"))
            for reason in reasons
        ),
        "counter_reuse_gate": {
            "status": "passed" if not any(
                reason.startswith(("counter_reuse_disagreement", "ncu_signature_multiplicity"))
                for reason in reasons
            ) and not reasons else "provisional",
            "aggregation": "measured_counters_times_exact_nsys_multiplicity",
            "signature_iteration_checks": list(agreement.values()),
        },
        "aggregation_scope": "frozen_representative_iterations_only",
        "aggregated_signatures": aggregated_signatures,
        "stage_summaries": stage_summaries,
        "reasons": sorted(set(reasons)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_ncu_plan")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--plan", type=Path)
    parser.add_argument("--nsys-profile", type=Path, action="append", default=[])
    parser.add_argument("--ncu-profile", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.config is not None:
        if not args.nsys_profile or args.ncu_profile:
            raise ValueError("plan construction requires NSYS profiles only")
        result = build_ncu_plan(NcuPlanConfig.load(args.config), args.nsys_profile)
    else:
        if args.nsys_profile or not args.ncu_profile:
            raise ValueError("plan validation requires NCU profiles only")
        result = validate_ncu_measurement(
            json.loads(args.plan.read_text(encoding="utf-8")),
            [json.loads(path.read_text(encoding="utf-8")) for path in args.ncu_profile],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "output": str(args.output.resolve()),
        "status": result["status"],
    }
    if "content_sha256" in result:
        summary["content_sha256"] = result["content_sha256"]
    print(json.dumps(summary, sort_keys=True))
    return 0 if result["status"] in {"planned", "passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
