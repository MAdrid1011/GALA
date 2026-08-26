"""Build an auditable NCU sampling plan from exact NSYS launch inventories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from gala_sim.identity import canonical_json, sha256_bytes, sha256_file
from gala_sim.tools.gpu_profile_campaign import GpuProfileCampaign


CONFIG_SCHEMA_VERSION = "gala-ncu-launch-signature-config-v4"
PLAN_SCHEMA_VERSION = "gala-ncu-launch-signature-plan-v4"
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
    maximum_capture_job_count: int
    range_capture_stages: tuple[str, ...]
    preflight_profile_launch_count: int
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
            maximum_capture_job_count=int(document.get("maximum_capture_job_count", 0)),
            range_capture_stages=tuple(
                str(value) for value in document.get("range_capture_stages", ())
            ),
            preflight_profile_launch_count=int(
                document.get("preflight_profile_launch_count", 0)
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
        if self.maximum_capture_job_count <= 0:
            raise ValueError("NCU maximum capture job count must be positive")
        if len(set(self.range_capture_stages)) != len(self.range_capture_stages) or any(
            not value or not re.fullmatch(r"[a-z_]+", value)
            for value in self.range_capture_stages
        ):
            raise ValueError("NCU range-capture stages are invalid")
        if self.preflight_profile_launch_count <= 0:
            raise ValueError("NCU preflight profile launch count must be positive")
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


def _regex_alternation(values: Iterable[str]) -> str:
    escaped = sorted({
        re.escape(value).replace(":", r"\x3a") for value in values
    })
    if not escaped:
        raise ValueError("NCU regex group cannot be empty")
    return "^(" + "|".join(escaped) + ")$"


def _ordinal_regex(values: Iterable[int]) -> str:
    ordinals = sorted({int(value) for value in values})
    if not ordinals or any(value <= 0 for value in ordinals):
        raise ValueError("NCU invocation group cannot be empty")
    return "^(" + "|".join(str(value) for value in ordinals) + ")$"


def _kernel_id_filter(names: Iterable[str], ordinals: Iterable[int]) -> str:
    return "::regex:" + _regex_alternation(names) + ":" + _ordinal_regex(ordinals)


def _all_invocations_filter(names: Iterable[str]) -> str:
    return "::regex:" + _regex_alternation(names) + ":"


def _stage_range_filter(
    stage: str, iterations: Iterable[int], campaign_sha256: str,
) -> str:
    iteration_regex = "(" + "|".join(str(value) for value in sorted(set(iterations))) + ")"
    return (
        "regex:^gala_ncu_stage:" + stage + ":iteration=" + iteration_regex
        + ":call=[0-9]+:campaign=" + campaign_sha256 + "$"
    )


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


def _load_inventory_set(
    paths: Iterable[Path], campaign_sha256: str, expected_iterations: list[int],
) -> dict[int, tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]]]:
    loaded = {}
    for raw_path in paths:
        path = raw_path.resolve()
        profile, profile_sha256 = _load_inventory(path)
        iteration, kernel_count, calls = _validate_inventory(profile, campaign_sha256)
        if iteration in loaded:
            raise ValueError(f"duplicate NSYS representative iteration: {iteration}")
        loaded[iteration] = (path, profile, profile_sha256, kernel_count, calls)
    if sorted(loaded) != expected_iterations:
        raise ValueError("NSYS inventories do not exactly match representative iterations")
    return loaded


def _stability_validation(
    primary: Mapping[
        int, tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]]
    ],
    repeated: Mapping[
        int, tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]]
    ],
    iterations: list[int], range_stages: tuple[str, ...],
) -> dict[str, Any]:
    def invocation_sequence(
        inventories: Mapping[
            int, tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]]
        ],
        only_stage: str | None = None,
    ) -> list[dict[str, Any]]:
        sequence = []
        for iteration in iterations:
            for call in inventories[iteration][4]:
                stage = str(call["stage"])
                if stage in range_stages or (only_stage is not None and stage != only_stage):
                    continue
                for ordinal, kernel in enumerate(call["kernels"], start=1):
                    sequence.append({
                        "iteration": iteration,
                        "stage": stage,
                        "call_index": int(call["call_index"]),
                        "kernel_ordinal_in_call": ordinal,
                        **_signature_document(_signature(stage, kernel)),
                    })
        return sequence

    primary_sequence = invocation_sequence(primary)
    repeated_sequence = invocation_sequence(repeated)
    invocation_stages = sorted({
        str(call["stage"])
        for iteration in iterations
        for inventories in (primary, repeated)
        for call in inventories[iteration][4]
        if str(call["stage"]) not in range_stages
    })
    unstable_invocation_stages = [
        stage for stage in invocation_stages
        if invocation_sequence(primary, stage) != invocation_sequence(repeated, stage)
    ]
    primary_hash = sha256_bytes(canonical_json(primary_sequence))
    repeated_hash = sha256_bytes(canonical_json(repeated_sequence))
    if primary_hash != repeated_hash:
        suffix = ",".join(unstable_invocation_stages) or "cross_stage_order"
        raise ValueError(f"NSYS invocation stages are unstable: {suffix}")

    range_records = []
    for stage in range_stages:
        for iteration in iterations:
            def signature_counts(
                inventories: Mapping[
                    int,
                    tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]],
                ],
            ) -> Counter[str]:
                return Counter(
                    _signature_id(_signature(stage, kernel))
                    for call in inventories[iteration][4]
                    if str(call["stage"]) == stage
                    for kernel in call["kernels"]
                )

            primary_counts = signature_counts(primary)
            repeated_counts = signature_counts(repeated)
            if set(primary_counts) != set(repeated_counts):
                raise ValueError(
                    f"NSYS range stage signature set is unstable: {stage}:{iteration}"
                )
            if primary_counts:
                range_records.append({
                    "iteration": iteration,
                    "stage": stage,
                    "signature_count": len(primary_counts),
                    "primary_kernel_launch_count": sum(primary_counts.values()),
                    "repeated_kernel_launch_count": sum(repeated_counts.values()),
                    "multiplicity_identical": primary_counts == repeated_counts,
                })

    repeated_records = []
    for iteration in iterations:
        path, profile, digest, kernel_count, _ = repeated[iteration]
        repeated_records.append({
            "iteration": iteration,
            "path": str(path),
            "sha256": digest,
            "source": profile.get("source"),
            "source_sha256": profile.get("source_sha256"),
            "kernel_count": kernel_count,
        })
    return {
        "status": "passed",
        "policy": "exact_invocation_sequence_and_range_signature_set",
        "invocation_sequence_primary_sha256": primary_hash,
        "invocation_sequence_repeated_sha256": repeated_hash,
        "invocation_kernel_launch_count": len(primary_sequence),
        "range_stage_records": range_records,
        "repeated_nsys_inventories": repeated_records,
    }


def build_ncu_plan(
    config: NcuPlanConfig, inventory_paths: Iterable[Path],
    stability_inventory_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    campaign = GpuProfileCampaign.load(config.campaign)
    expected_iterations = [item.iteration for item in campaign.representatives]
    loaded = _load_inventory_set(
        inventory_paths, config.campaign_sha256, expected_iterations
    )
    stability_paths = tuple(stability_inventory_paths)
    stability = {"status": "not_provided"}
    if stability_paths:
        repeated = _load_inventory_set(
            stability_paths, config.campaign_sha256, expected_iterations
        )
        stability = _stability_validation(
            loaded, repeated, expected_iterations, config.range_capture_stages
        )

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
    global_name_ordinals: dict[str, int] = defaultdict(int)
    invocation_name_ordinals: dict[str, int] = defaultdict(int)
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
                global_name_ordinals[key[1]] += 1
                invocation_ordinal = None
                if stage not in config.range_capture_stages:
                    invocation_name_ordinals[key[1]] += 1
                    invocation_ordinal = invocation_name_ordinals[key[1]]
                occurrence_key = (iteration, key)
                occurrences[occurrence_key].append({
                    "call_index": call_index,
                    "call_order": call_order,
                    "kernel_ordinal_in_call": kernel_ordinal,
                    "kernel_name_ordinal_in_call": name_ordinals[key[1]],
                    "kernel_name_ordinal_in_capture": global_name_ordinals[key[1]],
                    "kernel_name_ordinal_in_invocation_capture": invocation_ordinal,
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

    # A CUDA Profiler API window covers all representative iterations in one
    # complete training process. NCU invocation numbers are global across the
    # active windows for a kernel name, so group names with the same ordinal
    # set and select them using one kernel-name regex plus one invocation regex.
    target_ordinals: dict[str, set[int]] = defaultdict(set)
    required_locations: set[tuple[int, str, int, int]] = set()
    for (iteration, key), values in occurrences.items():
        if key[0] in config.range_capture_stages:
            for index in _anchor_indices(len(values), config.validation_occurrences):
                occurrence = values[index]
                required_locations.add((
                    iteration, key[0], int(occurrence["call_index"]),
                    int(occurrence["kernel_ordinal_in_call"]),
                ))
            continue
        for index in _anchor_indices(len(values), config.validation_occurrences):
            occurrence = values[index]
            target_ordinals[key[1]].add(int(
                occurrence["kernel_name_ordinal_in_invocation_capture"]
            ))
            required_locations.add((
                iteration, key[0], int(occurrence["call_index"]),
                int(occurrence["kernel_ordinal_in_call"]),
            ))
    all_locations = []
    for (iteration, key), values in occurrences.items():
        for occurrence in values:
            location = (
                iteration, key[0], int(occurrence["call_index"]),
                int(occurrence["kernel_ordinal_in_call"]),
            )
            identity = {
                "iteration": iteration,
                "stage": key[0],
                "call_index": int(occurrence["call_index"]),
                "call_order": int(occurrence["call_order"]),
                "kernel_ordinal_in_call": int(occurrence["kernel_ordinal_in_call"]),
                "kernel_name_ordinal_in_call": int(
                    occurrence["kernel_name_ordinal_in_call"]
                ),
                "kernel_name_ordinal_in_capture": int(
                    occurrence["kernel_name_ordinal_in_capture"]
                ),
                "kernel_name_ordinal_in_invocation_capture": (
                    int(occurrence["kernel_name_ordinal_in_invocation_capture"])
                    if occurrence["kernel_name_ordinal_in_invocation_capture"] is not None
                    else None
                ),
                "kernel_name": key[1],
                "grid": list(key[2]),
                "block": list(key[3]),
                "signature_id": _signature_id(key),
                "required_sample": location in required_locations,
            }
            all_locations.append({
                **identity,
                "selected_launch_id": sha256_bytes(canonical_json(identity)),
            })

    def group_cost(names: set[str], ordinals: set[int]) -> int:
        return sum(
            sum(ordinal <= invocation_name_ordinals[name] for ordinal in ordinals)
            for name in names
        )

    grouped_names: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for name, ordinals in target_ordinals.items():
        grouped_names[tuple(sorted(ordinals))].add(name)
    groups: list[tuple[set[str], set[int]]] = [
        (set(names), set(ordinals))
        for ordinals, names in grouped_names.items()
    ]
    while len(groups) > config.maximum_capture_job_count:
        best: tuple[int, int, int, tuple[set[str], set[int]]] | None = None
        for left in range(len(groups)):
            for right in range(left):
                names = groups[left][0] | groups[right][0]
                ordinals = groups[left][1] | groups[right][1]
                candidate = (
                    group_cost(names, ordinals)
                    - group_cost(*groups[left]) - group_cost(*groups[right]),
                    group_cost(names, ordinals), left, right, (names, ordinals),
                )
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            raise ValueError("cannot construct the configured NCU capture groups")
        _, _, left, right, merged = best
        groups = [
            group for index, group in enumerate(groups)
            if index not in {left, right}
        ] + [merged]
    groups.sort(key=lambda item: (min(item[1]), sorted(item[0])))

    excluded_range_arguments = []
    for stage in config.range_capture_stages:
        stage_iterations = sorted({
            iteration for iteration, key in occurrences if key[0] == stage
        })
        if not stage_iterations:
            raise ValueError(f"NCU range-capture stage has no launches: {stage}")
        excluded_range_arguments.extend([
            "--nvtx-exclude",
            _stage_range_filter(stage, stage_iterations, config.campaign_sha256),
        ])

    capture_groups = []
    for job_index, (names, ordinals) in enumerate(groups, start=1):
        expected_launches = [
            location for location in all_locations
            if location["stage"] not in config.range_capture_stages
            and location["kernel_name"] in names
            and int(location["kernel_name_ordinal_in_invocation_capture"]) in ordinals
        ]
        expected_launches.sort(key=lambda item: (
            int(item["iteration"]), int(item["call_order"]),
            int(item["kernel_ordinal_in_call"]),
        ))
        # NCU measures every matching name at every selected invocation. Keep
        # those extra launches explicit so the validator never treats them as
        # unobserved or as multiplicity-expanded evidence.
        measured_launch_count = group_cost(names, ordinals)
        capture_groups.append({
            "job_index": job_index,
            "capture_mode": "kernel_invocations",
            "representative_iterations": expected_iterations,
            "kernel_names": sorted(names),
            "kernel_name_regex": _regex_alternation(names),
            "invocation_ordinals": sorted(ordinals),
            "invocation_regex": _ordinal_regex(ordinals),
            "required_sample_count": sum(
                bool(location["required_sample"]) for location in expected_launches
            ),
            "selected_kernel_launch_count": measured_launch_count,
            "extra_kernel_launch_count": measured_launch_count - sum(
                bool(location["required_sample"]) for location in expected_launches
            ),
            "expected_launches": expected_launches,
            "ncu_arguments": [
                "--profile-from-start", "off",
                "--nvtx",
                *excluded_range_arguments,
                "--replay-mode", config.ncu_options["replay_mode"],
                "--cache-control", config.ncu_options["cache_control"],
                "--clock-control", config.ncu_options["clock_control"],
                "--kernel-name-base", config.ncu_options["kernel_name_base"],
                "--kernel-id", _kernel_id_filter(names, ordinals),
                *[
                    argument
                    for section in config.sections
                    for argument in ("--section", section)
                ],
                "--metrics", ",".join(config.metrics),
                "--check-exit-code", "1",
            ],
        })
    range_stages = tuple(config.range_capture_stages)
    if range_stages:
        range_locations = [
            location for location in all_locations
            if location["stage"] in range_stages
        ]
        if not range_locations:
            raise ValueError("configured NCU range-capture stages have no launches")
        range_iterations = sorted({int(item["iteration"]) for item in range_locations})
        range_arguments = ["--profile-from-start", "off", "--nvtx"]
        for stage in range_stages:
            stage_iterations = sorted({
                int(item["iteration"]) for item in range_locations
                if item["stage"] == stage
            })
            if not stage_iterations:
                raise ValueError(f"NCU range-capture stage has no launches: {stage}")
            range_arguments.extend([
                "--nvtx-include",
                _stage_range_filter(stage, stage_iterations, config.campaign_sha256),
            ])
        range_arguments.extend([
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
            "--check-exit-code", "1",
        ])
        range_locations.sort(key=lambda item: (
            int(item["iteration"]), int(item["call_order"]),
            int(item["kernel_ordinal_in_call"]),
        ))
        capture_groups.append({
            "job_index": len(capture_groups) + 1,
            "capture_mode": "nvtx_ranges",
            "representative_iterations": range_iterations,
            "range_stages": list(range_stages),
            "kernel_names": sorted({str(item["kernel_name"]) for item in range_locations}),
            "required_sample_count": sum(bool(item["required_sample"]) for item in range_locations),
            "selected_kernel_launch_count": len(range_locations),
            "extra_kernel_launch_count": len(range_locations) - sum(
                bool(item["required_sample"]) for item in range_locations
            ),
            "expected_launches": range_locations,
            "ncu_arguments": range_arguments,
        })
    selected_launch_count = sum(
        int(group["selected_kernel_launch_count"]) for group in capture_groups
    )

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
        "range_capture_stages": list(config.range_capture_stages),
        "stability_validation": stability,
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
                "--profile-from-start", "off", "--nvtx",
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
            ],
            "capture_mode": "cuda_profiler_api",
            "invocation_scope": "representative_windows_after_nvtx_range_exclusion",
            "range_scope": "all_launches_in_configured_cross_thread_start_end_ranges",
            "preflight_profile_launch_count": config.preflight_profile_launch_count,
        },
        "coverage": {
            "status": "complete",
            "representative_iterations": expected_iterations,
            "observed_kernel_launch_count": observed_kernel_count,
            "global_signature_count": len(signatures),
            "representative_iteration_signature_count": len(occurrences),
            "required_sample_count": required_sample_count,
            "capture_group_count": len(capture_groups),
            "capture_job_count": len(capture_groups),
            "invocation_capture_job_count": sum(
                group["capture_mode"] == "kernel_invocations"
                for group in capture_groups
            ),
            "range_capture_job_count": sum(
                group["capture_mode"] == "nvtx_ranges"
                for group in capture_groups
            ),
            "selected_stage_call_count": len(selected_calls),
            "selected_kernel_launch_count": selected_launch_count,
            "extra_kernel_launch_count": selected_launch_count - required_sample_count,
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
    """Validate every profiler-API-selected NCU launch without inventing counts."""

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
    stability = plan.get("stability_validation")
    if not isinstance(stability, Mapping) or stability.get("status") != "passed":
        reasons.append("ncu_plan_stability_invalid")
    expected_campaign = (
        plan.get("campaign", {}).get("sha256")
        if isinstance(plan.get("campaign"), Mapping) else None
    )
    expected_jobs: dict[int, dict[str, Mapping[str, Any]]] = {}
    expected_job_modes: dict[int, str] = {}
    expected_launches: dict[str, Mapping[str, Any]] = {}
    for group in plan.get("capture_groups", ()):
        if not isinstance(group, Mapping):
            continue
        job_index = int(group["job_index"])
        capture_mode = str(group.get("capture_mode", ""))
        expected_job_modes[job_index] = capture_mode
        job_launches: dict[str, Mapping[str, Any]] = {}
        for launch in group.get("expected_launches", ()):
            if not isinstance(launch, Mapping):
                continue
            if capture_mode == "nvtx_ranges" and not bool(launch.get("required_sample")):
                continue
            selected_id = str(launch["selected_launch_id"])
            if selected_id in expected_launches:
                reasons.append(f"duplicate_planned_launch:{selected_id}")
            expected_launches[selected_id] = launch
            job_launches[selected_id] = launch
        expected_jobs[job_index] = job_launches

    valid_signature_iterations = {
        (int(representative["iteration"]), str(signature["signature_id"]))
        for signature in plan.get("signatures", ())
        if isinstance(signature, Mapping)
        for representative in signature.get("representative_iterations", ())
        if isinstance(representative, Mapping)
    }

    observed_records: list[Mapping[str, Any]] = []
    observed_ids: set[str] = set()
    observed_jobs: set[int] = set()
    metrics_by_signature: dict[
        tuple[int, str], list[tuple[tuple[str, float], ...]]
    ] = defaultdict(list)
    profile_count = 0
    identity_fields = (
        "iteration", "stage", "call_index", "kernel_ordinal_in_call",
        "kernel_name_ordinal_in_capture", "kernel_name",
    )
    for profile in ncu_profiles:
        profile_count += 1
        identity = profile.get("run_identity")
        try:
            job_index = int(identity["capture_job_index"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            job_index = -1
        if (
            profile.get("status") != "passed"
            or not isinstance(identity, Mapping)
            or identity.get("status") != "passed"
            or identity.get("profiling_campaign_sha256") != expected_campaign
            or identity.get("ncu_plan_content_sha256") != plan_hash
            or job_index not in expected_jobs
            or job_index in observed_jobs
        ):
            reasons.append("ncu_profile_identity_or_status_invalid")
        observed_jobs.add(job_index)
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
            selected_id = str(launch.get("selected_launch_id", ""))
            expected = expected_jobs.get(job_index, {}).get(selected_id)
            capture_mode = expected_job_modes.get(job_index)
            if capture_mode == "nvtx_ranges" and not bool(launch.get("required_sample")):
                try:
                    signature_key = (int(launch["iteration"]), str(launch["signature_id"]))
                except (KeyError, TypeError, ValueError):
                    reasons.append("ncu_range_signature_identity_invalid")
                    continue
                if signature_key not in valid_signature_iterations:
                    reasons.append("ncu_range_signature_identity_invalid")
                    continue
                metrics = launch.get("metrics")
                if not isinstance(metrics, Mapping):
                    reasons.append("ncu_counter_metrics_missing")
                else:
                    metrics_by_signature[signature_key].append(tuple(sorted(
                        (str(name), float(value)) for name, value in metrics.items()
                    )))
                observed_records.append(launch)
                continue
            if expected is None:
                reasons.append(f"unexpected_ncu_launch:{job_index}:{selected_id}")
                continue
            if selected_id in observed_ids:
                reasons.append(f"duplicate_ncu_launch:{selected_id}")
            observed_ids.add(selected_id)
            try:
                compared_fields = (
                    ("iteration", "stage", "kernel_name")
                    if capture_mode == "nvtx_ranges" else identity_fields
                )
                mismatch = any(launch[field] != expected[field] for field in compared_fields)
                mismatch = mismatch or [int(value) for value in launch["grid_size"]] != [
                    int(value) for value in expected["grid"]
                ]
                mismatch = mismatch or [int(value) for value in launch["block_size"]] != [
                    int(value) for value in expected["block"]
                ]
            except (KeyError, TypeError, ValueError):
                mismatch = True
            if mismatch:
                reasons.append(f"ncu_launch_identity_mismatch:{selected_id}")
                continue
            metrics = launch.get("metrics")
            if not isinstance(metrics, Mapping):
                reasons.append("ncu_counter_metrics_missing")
            else:
                metrics_by_signature[
                    (int(expected["iteration"]), str(expected["signature_id"]))
                ].append(tuple(sorted(
                    (str(name), float(value)) for name, value in metrics.items()
                )))
            observed_records.append(launch)

    missing_jobs = set(expected_jobs) - observed_jobs
    if missing_jobs:
        reasons.append("ncu_capture_jobs_missing")
    missing_launches = set(expected_launches) - observed_ids
    if missing_launches:
        reasons.append("ncu_selected_launches_missing")
    if len(expected_launches) != len(observed_ids):
        reasons.append("ncu_selected_launch_count_mismatch")

    agreement: dict[str, Any] = {}
    for key, values in sorted(metrics_by_signature.items()):
        unique = {value for value in values}
        identifier = f"{key[0]}:{key[1]}"
        exact_observations = False
        for signature in plan.get("signatures", ()):
            if not isinstance(signature, Mapping) or signature.get("signature_id") != key[1]:
                continue
            representative = next((
                item for item in signature.get("representative_iterations", ())
                if isinstance(item, Mapping) and int(item["iteration"]) == key[0]
            ), None)
            exact_observations = (
                isinstance(representative, Mapping)
                and len(values) == int(representative["multiplicity"])
            )
            break
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
        "expected_selected_launch_count": len(expected_launches),
        "observed_selected_launch_count": len(observed_ids),
        "observed_profiled_launch_count": len(observed_records),
        "exact_launch_coverage": not any(
            reason.startswith((
                "unexpected_ncu_launch", "duplicate_ncu_launch",
                "ncu_launch_identity", "ncu_selected_launch",
                "ncu_capture_jobs",
            ))
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
    parser.add_argument(
        "--stability-nsys-profile", type=Path, action="append", default=[]
    )
    parser.add_argument("--ncu-profile", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.config is not None:
        if not args.nsys_profile or args.ncu_profile:
            raise ValueError("plan construction requires NSYS profiles only")
        result = build_ncu_plan(
            NcuPlanConfig.load(args.config), args.nsys_profile,
            args.stability_nsys_profile,
        )
    else:
        if args.nsys_profile or args.stability_nsys_profile or not args.ncu_profile:
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
