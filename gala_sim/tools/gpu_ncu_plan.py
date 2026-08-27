"""Build an auditable NCU sampling plan from exact NSYS launch inventories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

import yaml

from gala_sim.identity import canonical_json, sha256_bytes, sha256_file
from gala_sim.tools.gpu_profile_campaign import GpuProfileCampaign


CONFIG_SCHEMA_VERSION = "gala-ncu-launch-signature-config-v5"
PLAN_SCHEMA_VERSION = "gala-ncu-launch-signature-plan-v5"
_SUPPORTED_SIGNATURE_FIELDS = ("stage", "kernel_name", "grid", "block")
_SUPPORTED_OCCURRENCES = ("first", "middle", "last")
_SUPPORTED_NCU_OPTIONS = {
    "replay_mode": {"kernel"},
    "cache_control": {"all", "none"},
    "clock_control": {"base", "none"},
    "kernel_name_base": {"demangled"},
}
_SUPPORTED_NCU_SECTIONS = {"SourceCounters"}
_IMPLEMENTATION_PATHS = (
    "gala_sim/adapters/stage_profile.py",
    "gala_sim/adapters/stage_runner.py",
    "gala_sim/tools/gpu_ncu_plan.py",
    "gala_sim/tools/gpu_ncu_runner.py",
    "gala_sim/tools/gpu_profile_artifacts.py",
    "gala_sim/tools/gpu_profile_campaign.py",
)


def _implementation_hashes(repository: Path) -> dict[str, str]:
    return {
        relative: sha256_file(repository / relative)
        for relative in _IMPLEMENTATION_PATHS
    }


@dataclass(frozen=True)
class NcuPlanConfig:
    path: Path
    campaign: Path
    campaign_sha256: str
    signature_fields: tuple[str, ...]
    counter_scope: str
    validation_occurrences: tuple[str, ...]
    maximum_capture_job_count: int
    maximum_single_kernel_group_launch_count: int
    isolated_invocation_ordinals: Mapping[str, tuple[int, ...]]
    range_capture_stages: tuple[str, ...]
    preflight_profile_launch_count: int
    watchdog_inactivity_seconds: float
    watchdog_termination_grace_seconds: float
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
            maximum_single_kernel_group_launch_count=int(
                document.get("maximum_single_kernel_group_launch_count", 0)
            ),
            isolated_invocation_ordinals={
                str(name): tuple(
                    int(ordinal) for ordinal in ordinals
                )
                for name, ordinals in (
                    document.get("isolated_invocation_ordinals", {}) or {}
                ).items()
            } if isinstance(document.get("isolated_invocation_ordinals", {}) or {}, Mapping)
            else {},
            range_capture_stages=tuple(
                str(value) for value in document.get("range_capture_stages", ())
            ),
            preflight_profile_launch_count=int(
                document.get("preflight_profile_launch_count", 0)
            ),
            watchdog_inactivity_seconds=float(
                document.get("watchdog_inactivity_seconds", 0.0)
            ),
            watchdog_termination_grace_seconds=float(
                document.get("watchdog_termination_grace_seconds", 0.0)
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
        if self.maximum_single_kernel_group_launch_count <= 0:
            raise ValueError(
                "NCU maximum single-kernel group launch count must be positive"
            )
        for name, ordinals in self.isolated_invocation_ordinals.items():
            if (
                not name or not ordinals or len(set(ordinals)) != len(ordinals)
                or any(
                    isinstance(ordinal, bool) or ordinal <= 0
                    for ordinal in ordinals
                )
            ):
                raise ValueError("NCU isolated invocation ordinals are invalid")
        if len(set(self.range_capture_stages)) != len(self.range_capture_stages) or any(
            not value or not re.fullmatch(r"[a-z_]+", value)
            for value in self.range_capture_stages
        ):
            raise ValueError("NCU range-capture stages are invalid")
        if self.preflight_profile_launch_count <= 0:
            raise ValueError("NCU preflight profile launch count must be positive")
        if (
            self.watchdog_inactivity_seconds <= 0
            or self.watchdog_termination_grace_seconds <= 0
        ):
            raise ValueError("NCU watchdog timing must be positive")
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
        *, include_grid: bool = False,
    ) -> list[dict[str, Any]]:
        sequence = []
        for iteration in iterations:
            for call in inventories[iteration][4]:
                stage = str(call["stage"])
                if stage in range_stages or (only_stage is not None and stage != only_stage):
                    continue
                for ordinal, kernel in enumerate(call["kernels"], start=1):
                    signature = _signature(stage, kernel)
                    selection = {
                        "iteration": iteration,
                        "stage": stage,
                        "call_index": int(call["call_index"]),
                        "kernel_ordinal_in_call": ordinal,
                        "kernel_name": signature[1],
                        "block": list(signature[3]),
                    }
                    if include_grid:
                        selection["grid"] = list(signature[2])
                    sequence.append({
                        **selection,
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

    primary_grid_sequence = invocation_sequence(primary, include_grid=True)
    repeated_grid_sequence = invocation_sequence(repeated, include_grid=True)
    grid_variation_count = sum(
        left.get("grid") != right.get("grid")
        for left, right in zip(primary_grid_sequence, repeated_grid_sequence, strict=True)
    )

    range_records = []
    for stage in range_stages:
        for iteration in iterations:
            def range_calls(
                inventories: Mapping[
                    int,
                    tuple[Path, dict[str, Any], str, int, list[Mapping[str, Any]]],
                ],
            ) -> list[Mapping[str, Any]]:
                return [
                    call
                    for call in inventories[iteration][4]
                    if str(call["stage"]) == stage
                ]

            primary_calls = range_calls(primary)
            repeated_calls = range_calls(repeated)
            primary_call_ids = [int(call["call_index"]) for call in primary_calls]
            repeated_call_ids = [int(call["call_index"]) for call in repeated_calls]
            if primary_call_ids != repeated_call_ids:
                raise ValueError(f"NSYS range stage calls are unstable: {stage}:{iteration}")
            if primary_calls:
                primary_signatures = {
                    _signature_id(_signature(stage, kernel))
                    for call in primary_calls for kernel in call["kernels"]
                }
                repeated_signatures = {
                    _signature_id(_signature(stage, kernel))
                    for call in repeated_calls for kernel in call["kernels"]
                }
                range_records.append({
                    "iteration": iteration,
                    "stage": stage,
                    "call_indices": primary_call_ids,
                    "primary_kernel_launch_count": sum(
                        len(call["kernels"]) for call in primary_calls
                    ),
                    "repeated_kernel_launch_count": sum(
                        len(call["kernels"]) for call in repeated_calls
                    ),
                    "exact_signature_sets_identical": (
                        primary_signatures == repeated_signatures
                    ),
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
        "policy": "exact_invocation_selection_sequence_and_range_call_identity",
        "invocation_selection_fields": [
            "iteration", "stage", "call_index", "kernel_ordinal_in_call",
            "kernel_name", "block",
        ],
        "invocation_selection_sequence_primary_sha256": primary_hash,
        "invocation_selection_sequence_repeated_sha256": repeated_hash,
        "invocation_kernel_launch_count": len(primary_sequence),
        "dynamic_grid_shape_variation_count": grid_variation_count,
        "exact_grid_sequence_identical": grid_variation_count == 0,
        "range_stage_records": range_records,
        "repeated_nsys_inventories": repeated_records,
    }


def build_ncu_plan(
    config: NcuPlanConfig, inventory_paths: Iterable[Path],
    stability_inventory_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    campaign = GpuProfileCampaign.load(config.campaign)
    repository = config.path.parents[2]
    repository_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
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
    # Keep large single-kernel groups split by their real invocation ordinals.
    # NCU replays every selected ordinal, so this bound controls replay pressure
    # without changing the required launch set or inventing a sampling subset.
    groups: list[tuple[set[str], set[int], bool]] = []
    applied_isolated_ordinals: set[tuple[str, int]] = set()

    def append_single_kernel_group(
        name: str, ordinals: set[int], *, force_partition: bool = False,
    ) -> None:
        ordered_ordinals = sorted(ordinals)
        should_partition = (
            force_partition
            or len(ordered_ordinals) > config.maximum_single_kernel_group_launch_count
        )
        if not should_partition:
            groups.append(({name}, set(ordinals), False))
            return
        for start in range(
            0, len(ordered_ordinals),
            config.maximum_single_kernel_group_launch_count,
        ):
            groups.append((
                {name},
                set(ordered_ordinals[
                    start:start + config.maximum_single_kernel_group_launch_count
                ]),
                True,
            ))

    for ordinals, names in grouped_names.items():
        ordinal_set = set(ordinals)
        remaining_names = set(names)
        # Isolation is allowed to split a configured kernel out of a shared
        # ordinal group; its other ordinals remain fully represented separately.
        for name in sorted(names):
            isolated = (
                set(config.isolated_invocation_ordinals.get(name, ()))
                & ordinal_set
            )
            if not isolated:
                continue
            for ordinal in sorted(isolated):
                groups.append(({name}, {ordinal}, True))
                applied_isolated_ordinals.add((name, ordinal))
            remainder = ordinal_set - isolated
            if remainder:
                append_single_kernel_group(name, remainder)
            remaining_names.discard(name)
        if not remaining_names:
            continue
        if len(remaining_names) == 1:
            append_single_kernel_group(next(iter(remaining_names)), ordinal_set)
        else:
            groups.append((remaining_names, ordinal_set, False))
    configured_isolated_ordinals = {
        (name, ordinal)
        for name, ordinals in config.isolated_invocation_ordinals.items()
        for ordinal in ordinals
    }
    if configured_isolated_ordinals - applied_isolated_ordinals:
        missing = sorted(configured_isolated_ordinals - applied_isolated_ordinals)
        raise ValueError(f"NCU isolated invocation ordinals are not selectable: {missing}")
    while len(groups) > config.maximum_capture_job_count:
        best: tuple[int, int, int, int, tuple[set[str], set[int], bool]] | None = None
        for left in range(len(groups)):
            for right in range(left):
                # A protected shard is deliberately kept independent.  If the
                # configured job budget cannot hold these shards, fail plan
                # generation instead of silently recreating the unsafe group.
                if groups[left][2] or groups[right][2]:
                    continue
                names = groups[left][0] | groups[right][0]
                ordinals = groups[left][1] | groups[right][1]
                candidate = (
                    group_cost(names, ordinals)
                    - group_cost(groups[left][0], groups[left][1])
                    - group_cost(groups[right][0], groups[right][1]),
                    group_cost(names, ordinals), left, right,
                    (names, ordinals, False),
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
    for job_index, (names, ordinals, _) in enumerate(groups, start=1):
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
            "range_calls": [
                {"iteration": iteration, "stage": stage, "call_index": call_index}
                for iteration, stage, call_index in sorted({
                    (
                        int(item["iteration"]), str(item["stage"]),
                        int(item["call_index"]),
                    )
                    for item in range_locations
                })
            ],
            "kernel_names": sorted({str(item["kernel_name"]) for item in range_locations}),
            "required_sample_count": len(range_locations),
            "selected_kernel_launch_count": len(range_locations),
            "extra_kernel_launch_count": 0,
            "binding_policy": "exact_observed_launches_with_dynamic_signatures",
            "expected_launches": range_locations,
            "ncu_arguments": range_arguments,
        })
    selected_launch_count = sum(
        int(group["selected_kernel_launch_count"]) for group in capture_groups
    )
    required_capture_count = sum(
        int(group["required_sample_count"]) for group in capture_groups
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
        "repository_commit": repository_commit,
        "implementation_sha256": _implementation_hashes(repository),
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
        "maximum_capture_job_count": config.maximum_capture_job_count,
        "maximum_single_kernel_group_launch_count": (
            config.maximum_single_kernel_group_launch_count
        ),
        "isolated_invocation_ordinals": {
            name: list(ordinals)
            for name, ordinals in sorted(config.isolated_invocation_ordinals.items())
        },
        "stability_validation": stability,
        "counter_reuse_gate": {
            "policy": (
                "invocation_semantic_anchor_reuse_and_exact_dynamic_range_observation"
            ),
            "require_all_planned_samples": True,
            "require_counter_agreement_before_multiplicity_expansion": True,
            "aggregation": (
                "invocation_counters_times_primary_nsys_multiplicity_plus_"
                "exact_observed_range_launches"
            ),
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
            "maximum_capture_job_count": config.maximum_capture_job_count,
            "maximum_single_kernel_group_launch_count": (
                config.maximum_single_kernel_group_launch_count
            ),
            "isolated_invocation_ordinals": {
                name: list(ordinals)
                for name, ordinals in sorted(config.isolated_invocation_ordinals.items())
            },
            "capture_mode": "cuda_profiler_api",
            "invocation_scope": "representative_windows_after_nvtx_range_exclusion",
            "range_scope": "all_launches_in_configured_cross_thread_start_end_ranges",
            "preflight_profile_launch_count": config.preflight_profile_launch_count,
            "watchdog_inactivity_seconds": config.watchdog_inactivity_seconds,
            "watchdog_termination_grace_seconds": (
                config.watchdog_termination_grace_seconds
            ),
        },
        "coverage": {
            "status": "complete",
            "representative_iterations": expected_iterations,
            "observed_kernel_launch_count": observed_kernel_count,
            "global_signature_count": len(signatures),
            "representative_iteration_signature_count": len(occurrences),
            "required_sample_count": required_capture_count,
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
            "extra_kernel_launch_count": selected_launch_count - required_capture_count,
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
    expected_range_scopes: dict[
        int, tuple[set[str], set[int], set[tuple[int, str, int]]]
    ] = {}
    expected_launches: dict[str, Mapping[str, Any]] = {}
    predicted_range_launch_count = 0
    for group in plan.get("capture_groups", ()):
        if not isinstance(group, Mapping):
            continue
        job_index = int(group["job_index"])
        capture_mode = str(group.get("capture_mode", ""))
        expected_job_modes[job_index] = capture_mode
        if capture_mode == "nvtx_ranges":
            expected_range_scopes[job_index] = (
                {str(value) for value in group.get("range_stages", ())},
                {int(value) for value in group.get("representative_iterations", ())},
                {
                    (
                        int(item["iteration"]), str(item["stage"]),
                        int(item["call_index"]),
                    )
                    for item in group.get("range_calls", ())
                    if isinstance(item, Mapping)
                },
            )
            predicted_range_launch_count += int(
                group.get("selected_kernel_launch_count", 0)
            )
        job_launches: dict[str, Mapping[str, Any]] = {}
        for launch in group.get("expected_launches", ()):
            if not isinstance(launch, Mapping):
                continue
            if capture_mode == "nvtx_ranges":
                continue
            selected_id = str(launch["selected_launch_id"])
            if selected_id in expected_launches:
                reasons.append(f"duplicate_planned_launch:{selected_id}")
            expected_launches[selected_id] = launch
            job_launches[selected_id] = launch
        expected_jobs[job_index] = job_launches

    observed_records: list[Mapping[str, Any]] = []
    observed_ids: set[str] = set()
    observed_planned_ids: set[str] = set()
    observed_jobs: set[int] = set()
    metrics_by_signature: dict[
        tuple[int, str], list[tuple[tuple[str, float], ...]]
    ] = defaultdict(list)
    observed_grids_by_signature: dict[
        tuple[int, str], list[tuple[int, int, int]]
    ] = defaultdict(list)
    range_metrics_by_signature: dict[
        tuple[int, str, str, str, tuple[int, int, int], tuple[int, int, int]],
        list[tuple[tuple[str, float], ...]],
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
            job_mode = expected_job_modes.get(job_index)
            if job_mode == "nvtx_ranges":
                try:
                    iteration = int(launch["iteration"])
                    stage = str(launch["stage"])
                    name = str(launch["kernel_name"])
                    grid_values = tuple(int(value) for value in launch["grid_size"])
                    block_values = tuple(int(value) for value in launch["block_size"])
                    call_index = int(launch["call_index"])
                    if (
                        len(grid_values) != 3 or len(block_values) != 3
                        or any(value <= 0 for value in (*grid_values, *block_values))
                    ):
                        raise ValueError("invalid NCU launch shape")
                    grid = (grid_values[0], grid_values[1], grid_values[2])
                    block = (block_values[0], block_values[1], block_values[2])
                    signature_document = {
                        "stage": stage, "kernel_name": name,
                        "grid": list(grid), "block": list(block),
                    }
                    signature_id = sha256_bytes(canonical_json(signature_document))
                    allowed_stages, allowed_iterations, allowed_calls = (
                        expected_range_scopes[job_index]
                    )
                except (KeyError, TypeError, ValueError):
                    reasons.append("ncu_range_signature_identity_invalid")
                    continue
                if (
                    not selected_id
                    or selected_id in observed_ids
                    or launch.get("signature_id") != signature_id
                    or stage not in allowed_stages
                    or iteration not in allowed_iterations
                    or (iteration, stage, call_index) not in allowed_calls
                ):
                    reasons.append("ncu_range_launch_identity_invalid")
                    continue
                observed_ids.add(selected_id)
                metrics = launch.get("metrics")
                if not isinstance(metrics, Mapping):
                    reasons.append("ncu_counter_metrics_missing")
                else:
                    range_metrics_by_signature[
                        (iteration, stage, signature_id, name, grid, block)
                    ].append(tuple(sorted(
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
            observed_planned_ids.add(selected_id)
            try:
                mismatch = any(launch[field] != expected[field] for field in identity_fields)
                observed_grid_values = tuple(
                    int(value) for value in launch["grid_size"]
                )
                observed_block = tuple(int(value) for value in launch["block_size"])
                mismatch = mismatch or len(observed_grid_values) != 3 or any(
                    value <= 0 for value in observed_grid_values
                )
                mismatch = mismatch or list(observed_block) != [
                    int(value) for value in expected["block"]
                ]
            except (KeyError, TypeError, ValueError):
                mismatch = True
            if mismatch:
                reasons.append(f"ncu_launch_identity_mismatch:{selected_id}")
                continue
            observed_grid = (
                observed_grid_values[0], observed_grid_values[1],
                observed_grid_values[2],
            )
            metrics = launch.get("metrics")
            if not isinstance(metrics, Mapping):
                reasons.append("ncu_counter_metrics_missing")
            else:
                signature_key = (
                    int(expected["iteration"]), str(expected["signature_id"])
                )
                metrics_by_signature[signature_key].append(tuple(sorted(
                    (str(name), float(value)) for name, value in metrics.items()
                )))
                observed_grids_by_signature[signature_key].append(observed_grid)
            observed_records.append(launch)

    missing_jobs = set(expected_jobs) - observed_jobs
    if missing_jobs:
        reasons.append("ncu_capture_jobs_missing")
    missing_launches = set(expected_launches) - observed_planned_ids
    if missing_launches:
        reasons.append("ncu_selected_launches_missing")
    if len(expected_launches) != len(observed_planned_ids):
        reasons.append("ncu_selected_launch_count_mismatch")

    agreement: dict[str, Any] = {}
    for key, values in sorted(metrics_by_signature.items()):
        unique = {value for value in values}
        measured_grids = sorted(set(observed_grids_by_signature[key]))
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
            "measured_grid_sizes": [list(value) for value in measured_grids],
            "dynamic_grid_shape_observed": bool(measured_grids),
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
    range_stages = {
        str(value) for value in plan.get("range_capture_stages", ())
    }
    if not reasons:
        for signature in plan.get("signatures", ()):
            if not isinstance(signature, Mapping):
                continue
            signature_id = str(signature["signature_id"])
            stage = str(signature["stage"])
            if stage in range_stages:
                continue
            planned_grid = [int(value) for value in signature["grid"]]
            planned_block = [int(value) for value in signature["block"]]
            threads = (
                planned_grid[0] * planned_grid[1] * planned_grid[2]
                * planned_block[0] * planned_block[1] * planned_block[2]
            )
            for representative in signature.get("representative_iterations", ()):
                if not isinstance(representative, Mapping):
                    continue
                iteration = int(representative["iteration"])
                multiplicity = int(representative["multiplicity"])
                samples = metrics_by_signature[(iteration, signature_id)]
                measured_grids = sorted(set(
                    observed_grids_by_signature[(iteration, signature_id)]
                ))
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
                    "grid_size": planned_grid,
                    "measured_grid_sizes": [list(value) for value in measured_grids],
                    "block_size": planned_block,
                    "threads_launched_per_launch": threads,
                    "multiplicity": multiplicity,
                    "measured_sample_count": len(samples),
                    "measured_metrics_per_launch": measured,
                    "multiplicity_expanded_metrics": dict(totals),
                    "aggregation_mode": aggregation_mode,
                })
        for range_key, samples in sorted(range_metrics_by_signature.items()):
            iteration, stage, signature_id, kernel_name, grid_tuple, block_tuple = (
                range_key
            )
            range_totals: dict[str, float] = defaultdict(float)
            for sample in samples:
                for name, value in sample:
                    range_totals[name] += value
            for name, value in range_totals.items():
                stage_metrics[stage][name] += value
            multiplicity = len(samples)
            stage_launches[stage] += multiplicity
            range_grid = list(grid_tuple)
            range_block = list(block_tuple)
            threads = (
                range_grid[0] * range_grid[1] * range_grid[2]
                * range_block[0] * range_block[1] * range_block[2]
            )
            aggregated_signatures.append({
                "iteration": iteration,
                "stage": stage,
                "signature_id": signature_id,
                "kernel_name": kernel_name,
                "grid_size": range_grid,
                "measured_grid_sizes": [range_grid],
                "block_size": range_block,
                "threads_launched_per_launch": threads,
                "multiplicity": multiplicity,
                "measured_sample_count": multiplicity,
                "measured_metrics_per_launch": None,
                "multiplicity_expanded_metrics": dict(range_totals),
                "aggregation_mode": "exact_observed_range_launches",
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
        "expected_invocation_launch_count": len(expected_launches),
        "predicted_range_launch_count": predicted_range_launch_count,
        "observed_selected_launch_count": len(observed_ids),
        "observed_invocation_launch_count": len(observed_planned_ids),
        "observed_range_launch_count": sum(
            len(values) for values in range_metrics_by_signature.values()
        ),
        "observed_profiled_launch_count": len(observed_records),
        "exact_launch_coverage": not any(
            reason.startswith((
                "unexpected_ncu_launch", "duplicate_ncu_launch",
                "ncu_launch_identity", "ncu_selected_launch",
                "ncu_capture_jobs", "ncu_range_",
            ))
            for reason in reasons
        ),
        "counter_reuse_gate": {
            "status": "passed" if not any(
                reason.startswith(("counter_reuse_disagreement", "ncu_signature_multiplicity"))
                for reason in reasons
            ) and not reasons else "provisional",
            "aggregation": (
                "invocation_samples_times_primary_nsys_multiplicity_plus_"
                "exact_observed_range_launches"
            ),
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
