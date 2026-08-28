"""Parse Nsight Systems and Nsight Compute artifacts into stage evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping

from gala_sim.identity import canonical_json, sha256_bytes, sha256_file


NSYS_SCHEMA_VERSION = "gala-nsys-stage-kernels-v1"
NCU_SCHEMA_VERSION = "gala-ncu-stage-counters-v1"
SASS_SCHEMA_VERSION = "gala-ncu-sass-classification-v1"
NCU_SELECTED_SCHEMA_VERSION = "gala-ncu-selected-launch-counters-v1"
_STAGE_PATTERN = re.compile(
    r"gala_stage:(?P<stage>[a-z_]+):iteration=(?P<iteration>[0-9]+):call=(?P<call>[0-9]+)"
)
_NCU_STAGE_PATTERN = re.compile(
    r"gala_ncu_stage:(?P<stage>[a-z_]+):iteration=(?P<iteration>[0-9]+):call=(?P<call>[0-9]+)"
)
_ITERATION_PATTERN = re.compile(
    r"gala_iteration:training:iteration=(?P<iteration>[0-9]+):call=(?P<call>[0-9]+)"
)
_CAMPAIGN_PATTERN = re.compile(r":campaign=(?P<sha256>[0-9a-f]{64})(?:$|:)")
_NCU_METRICS = {
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "fp32_ffma",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": "fp32_fadd",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": "fp32_fmul",
    "smsp__inst_executed_pipe_xu.sum": "xu_instructions",
    "lts__t_requests_op_atom.sum": "atomic_requests",
}


def _stage_identity(value: str) -> tuple[str, int, int] | None:
    match = _STAGE_PATTERN.search(value) or _NCU_STAGE_PATTERN.search(value)
    if match is None:
        return None
    return (
        match.group("stage"), int(match.group("iteration")), int(match.group("call"))
    )


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def parse_nsys_sqlite(path: Path) -> dict[str, Any]:
    path = path.resolve()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        ranges = []
        for row in connection.execute(
            "SELECT start, end, text FROM NVTX_EVENTS "
            "WHERE text LIKE 'gala_stage:%:iteration=%' AND end IS NOT NULL "
            "ORDER BY start"
        ):
            identity = _stage_identity(str(row["text"]))
            if identity is not None:
                ranges.append((identity, int(row["start"]), int(row["end"])))
        iteration_ranges = []
        campaign_hashes: set[str] = set()
        for row in connection.execute(
            "SELECT start, end, text FROM NVTX_EVENTS "
            "WHERE text LIKE 'gala_iteration:training:iteration=%' "
            "AND end IS NOT NULL ORDER BY start"
        ):
            match = _ITERATION_PATTERN.search(str(row["text"]))
            if match is not None:
                campaign_match = _CAMPAIGN_PATTERN.search(str(row["text"]))
                if campaign_match is not None:
                    campaign_hashes.add(campaign_match.group("sha256"))
                iteration_ranges.append((
                    int(match.group("iteration")), int(match.group("call")),
                    int(row["start"]), int(row["end"]),
                ))
        calls = []
        stage_kernels: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
        stage_kernel_time: dict[str, float] = defaultdict(float)
        stage_launches: dict[str, int] = defaultdict(int)
        stage_memcpy_bytes: dict[str, int] = defaultdict(int)
        stage_memcpy_count: dict[str, int] = defaultdict(int)
        for (stage, iteration, call_index), start, end in ranges:
            kernel_rows = connection.execute(
                "SELECT k.start, k.end, k.gridX, k.gridY, k.gridZ, "
                "k.blockX, k.blockY, k.blockZ, s.value AS kernel_name "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
                "ON r.correlationId = k.correlationId "
                "JOIN StringIds s ON s.id = k.demangledName "
                "WHERE r.start >= ? AND r.end <= ? ORDER BY k.start",
                (start, end),
            ).fetchall()
            kernels = []
            for row in kernel_rows:
                name = str(row["kernel_name"])
                duration_ms = (int(row["end"]) - int(row["start"])) / 1_000_000.0
                kernels.append({
                    "name": name,
                    "duration_ms": duration_ms,
                    "grid": [int(row["gridX"]), int(row["gridY"]), int(row["gridZ"])],
                    "block": [int(row["blockX"]), int(row["blockY"]), int(row["blockZ"])],
                })
                aggregate = stage_kernels[stage].setdefault(
                    name, {"launch_count": 0, "gpu_time_ms": 0.0}
                )
                aggregate["launch_count"] = int(aggregate["launch_count"]) + 1
                aggregate["gpu_time_ms"] = float(aggregate["gpu_time_ms"]) + duration_ms
                stage_kernel_time[stage] += duration_ms
                stage_launches[stage] += 1
            memcpy_bytes = 0
            memcpy_count = 0
            if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
                rows = connection.execute(
                    "SELECT m.bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
                    "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "ON r.correlationId = m.correlationId "
                    "WHERE r.start >= ? AND r.end <= ?", (start, end)
                ).fetchall()
                memcpy_bytes = sum(int(row["bytes"]) for row in rows)
                memcpy_count = len(rows)
                stage_memcpy_bytes[stage] += memcpy_bytes
                stage_memcpy_count[stage] += memcpy_count
            calls.append({
                "stage": stage,
                "iteration": iteration,
                "call_index": call_index,
                "range_duration_ms": (end - start) / 1_000_000.0,
                "kernel_launch_count": len(kernels),
                "memcpy_count": memcpy_count,
                "memcpy_bytes": memcpy_bytes,
                "kernels": kernels,
            })
        summaries = {}
        for stage in sorted(stage_kernels):
            summaries[stage] = {
                "kernel_launch_count": stage_launches[stage],
                "gpu_kernel_time_ms": stage_kernel_time[stage],
                "memcpy_count": stage_memcpy_count[stage],
                "memcpy_bytes": stage_memcpy_bytes[stage],
                "kernels": [
                    {"name": name, **values}
                    for name, values in sorted(stage_kernels[stage].items())
                ],
            }
        coverage_records = []
        unassigned_kernels = []
        multiply_assigned_kernels = []
        for iteration, iteration_call, outer_start, outer_end in iteration_ranges:
            stage_ranges = [
                (identity, start, end)
                for identity, start, end in ranges if identity[1] == iteration
            ]
            kernel_rows = connection.execute(
                "SELECT k.start, k.end, k.correlationId, "
                "s.value AS kernel_name, r.start AS runtime_start, "
                "r.end AS runtime_end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
                "ON r.correlationId = k.correlationId "
                "JOIN StringIds s ON s.id = k.demangledName "
                "WHERE r.start >= ? AND r.end <= ? ORDER BY k.start",
                (outer_start, outer_end),
            ).fetchall()
            assigned_count = 0
            iteration_unassigned = 0
            iteration_multiple = 0
            for kernel in kernel_rows:
                assignments = [
                    identity for identity, start, end in stage_ranges
                    if int(kernel["runtime_start"]) >= start
                    and int(kernel["runtime_end"]) <= end
                ]
                record = {
                    "iteration": iteration,
                    "iteration_call_index": iteration_call,
                    "kernel_name": str(kernel["kernel_name"]),
                    "correlation_id": int(kernel["correlationId"]),
                    "gpu_time_ms": (
                        int(kernel["end"]) - int(kernel["start"])
                    ) / 1_000_000.0,
                    "runtime_start": int(kernel["runtime_start"]),
                    "runtime_end": int(kernel["runtime_end"]),
                }
                if not assignments:
                    iteration_unassigned += 1
                    unassigned_kernels.append(record)
                elif len(assignments) > 1:
                    iteration_multiple += 1
                    multiply_assigned_kernels.append({
                        **record,
                        "assignments": [
                            {"stage": stage, "call_index": call_index}
                            for stage, _, call_index in assignments
                        ],
                    })
                else:
                    assigned_count += 1
            coverage_records.append({
                "iteration": iteration,
                "iteration_call_index": iteration_call,
                "kernel_count": len(kernel_rows),
                "assigned_kernel_count": assigned_count,
                "unassigned_kernel_count": iteration_unassigned,
                "multiply_assigned_kernel_count": iteration_multiple,
            })
        coverage_status = (
            "complete"
            if coverage_records and not unassigned_kernels and not multiply_assigned_kernels
            else "incomplete"
        )
        kernel_coverage = {
            "status": coverage_status,
            "iteration_count": len(coverage_records),
            "kernel_count": sum(record["kernel_count"] for record in coverage_records),
            "assigned_kernel_count": sum(
                record["assigned_kernel_count"] for record in coverage_records
            ),
            "unassigned_kernel_count": len(unassigned_kernels),
            "multiply_assigned_kernel_count": len(multiply_assigned_kernels),
            "records": coverage_records,
            "unassigned_kernels": unassigned_kernels,
            "multiply_assigned_kernels": multiply_assigned_kernels,
        }
        identity_status = "passed"
        return {
            "schema_version": NSYS_SCHEMA_VERSION,
            "status": "passed" if calls and identity_status == "passed" else "failed_preflight",
            "source": str(path),
            "source_sha256": sha256_file(path),
            "run_identity": {
                "status": identity_status,
                "profiling_campaign_sha256": (
                    next(iter(campaign_hashes)) if len(campaign_hashes) == 1 else None
                ),
            },
            "stage_summaries": summaries,
            "stage_calls": calls,
            "kernel_coverage": kernel_coverage,
        }
    finally:
        connection.close()


def _ncu_rows(path: Path) -> Iterable[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next((index for index, line in enumerate(lines) if line.startswith('"ID"')), None)
    if header is None:
        raise ValueError("Nsight Compute CSV has no header")
    return csv.DictReader(lines[header:])


def _metric_value(value: str) -> float:
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError as error:
        raise ValueError(f"Nsight Compute metric value is not numeric: {value}") from error


def _launch_shape(value: str) -> tuple[int, int, int]:
    fields = value.strip().removeprefix("(").removesuffix(")").split(",")
    if len(fields) != 3:
        raise ValueError(f"Nsight Compute launch shape is invalid: {value}")
    try:
        shape = tuple(int(field.strip()) for field in fields)
    except ValueError as error:
        raise ValueError(f"Nsight Compute launch shape is invalid: {value}") from error
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Nsight Compute launch shape is invalid: {value}")
    return shape  # type: ignore[return-value]


def _shape_product(shape: tuple[int, int, int]) -> int:
    return shape[0] * shape[1] * shape[2]


def parse_ncu_csv(path: Path) -> dict[str, Any]:
    path = path.resolve()
    launches: dict[tuple[str, int, int, str, str], dict[str, Any]] = defaultdict(dict)
    campaign_hashes: set[str] = set()
    process_ids: set[int] = set()
    for row in _ncu_rows(path):
        process_id = row.get("Process ID", "").strip()
        if process_id:
            try:
                process_ids.add(int(process_id))
            except ValueError as error:
                raise ValueError("Nsight Compute process ID is invalid") from error
        range_value = " ".join(
            row.get(key, "")
            for key in row
            if "Push/Pop_Range" in key or "Start/Stop_Range" in key
        )
        identity = _stage_identity(range_value)
        metric = row.get("Metric Name", "")
        if identity is None or metric not in _NCU_METRICS:
            continue
        campaign_match = _CAMPAIGN_PATTERN.search(range_value)
        if campaign_match is not None:
            campaign_hashes.add(campaign_match.group("sha256"))
        stage, iteration, call_index = identity
        kernel_name = row.get("Kernel Name", "")
        launch_id = row.get("ID", "")
        key = (stage, iteration, call_index, launch_id, kernel_name)
        launch = launches[key]
        launch[_NCU_METRICS[metric]] = _metric_value(row.get("Metric Value", ""))
        block = _launch_shape(row.get("Block Size", ""))
        grid = _launch_shape(row.get("Grid Size", ""))
        previous_shape = launch.setdefault("_shape", (block, grid))
        if previous_shape != (block, grid):
            raise ValueError("Nsight Compute launch shape changed between metric rows")
    stage_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stage_launch_count: dict[str, int] = defaultdict(int)
    unclassified: dict[str, float] = defaultdict(float)
    stage_threads: dict[str, list[int]] = defaultdict(list)
    incomplete_launches: list[dict[str, Any]] = []
    launch_records = []
    call_kernel_ordinals: dict[tuple[str, int, int], int] = defaultdict(int)
    call_name_ordinals: dict[tuple[str, int, int, str], int] = defaultdict(int)
    required_fields = set(_NCU_METRICS.values())
    for (stage, iteration, call_index, launch_id, kernel_name), values in launches.items():
        block, grid = values.pop("_shape")
        metrics = {name: float(value) for name, value in values.items()}
        missing = sorted(required_fields - metrics.keys())
        if missing:
            incomplete_launches.append({
                "stage": stage,
                "iteration": iteration,
                "call_index": call_index,
                "launch_id": launch_id,
                "kernel_name": kernel_name,
                "missing_counter_fields": missing,
            })
        stage_launch_count[stage] += 1
        for name, value in metrics.items():
            stage_totals[stage][name] += value
        xu = metrics.get("xu_instructions", 0.0)
        unclassified[stage] += xu
        threads_launched = _shape_product(block) * _shape_product(grid)
        stage_threads[stage].append(threads_launched)
        call_key = (stage, iteration, call_index)
        name_key = (*call_key, kernel_name)
        call_kernel_ordinals[call_key] += 1
        call_name_ordinals[name_key] += 1
        launch_records.append({
            "stage": stage,
            "iteration": iteration,
            "call_index": call_index,
            "launch_id": launch_id,
            "kernel_name": kernel_name,
            "kernel_ordinal_in_call": call_kernel_ordinals[call_key],
            "kernel_name_ordinal_in_call": call_name_ordinals[name_key],
            "block_size": list(block),
            "grid_size": list(grid),
            "threads_launched": threads_launched,
            "metrics": dict(sorted(metrics.items())),
            "transcendental_evidence": "pending_sass_classification" if xu > 0 else "not_present",
            "missing_counter_fields": missing,
        })
    summaries = {}
    for stage, totals in sorted(stage_totals.items()):
        ffma = totals.get("fp32_ffma", 0.0)
        fadd = totals.get("fp32_fadd", 0.0)
        fmul = totals.get("fp32_fmul", 0.0)
        summaries[stage] = {
            **dict(sorted(totals.items())),
            "kernel_launch_count": stage_launch_count[stage],
            "fp32_operations": 2.0 * ffma + fadd + fmul,
            "fp32_fma_equivalent": ffma + (fadd + fmul) / 2.0,
            "dram_bytes": totals.get("dram_read_bytes", 0.0)
            + totals.get("dram_write_bytes", 0.0),
            "unclassified_transcendental_operations": unclassified[stage],
            "unclassified_xu_instructions": unclassified[stage],
            "minimum_threads_launched": min(stage_threads[stage]),
            "maximum_threads_launched": max(stage_threads[stage]),
            "weight_eligible": (
                unclassified[stage] == 0
                and not any(item["stage"] == stage for item in incomplete_launches)
            ),
        }
    identity_status = "passed"
    status = (
        "passed"
        if launch_records and not incomplete_launches and identity_status == "passed"
        else "failed_preflight"
    )
    return {
        "schema_version": NCU_SCHEMA_VERSION,
        "status": status,
        "source": str(path),
        "source_sha256": sha256_file(path),
        "run_identity": {
            "status": identity_status,
            "process_ids": sorted(process_ids),
            "profiling_campaign_sha256": (
                next(iter(campaign_hashes)) if len(campaign_hashes) == 1 else None
            ),
        },
        "required_metric_names": list(_NCU_METRICS),
        "stage_summaries": summaries,
        "launches": launch_records,
        "incomplete_launches": incomplete_launches,
    }


def bind_ncu_profile_to_plan(
    ncu_profile: Mapping[str, Any], plan: Mapping[str, Any], job_index: int,
    stage_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind filtered NCU launch order to exact NSYS identities from one plan job."""

    reasons: list[str] = []
    plan_hash = plan.get("content_sha256")
    if (
        plan.get("schema_version") != "gala-ncu-launch-signature-plan-v5"
        or not isinstance(plan_hash, str)
    ):
        reasons.append("ncu_plan_identity_invalid")
    stability = plan.get("stability_validation")
    if not isinstance(stability, Mapping) or stability.get("status") != "passed":
        reasons.append("ncu_plan_stability_invalid")
    jobs = [
        item for item in plan.get("capture_groups", ())
        if isinstance(item, Mapping) and int(item.get("job_index", -1)) == job_index
    ]
    if len(jobs) != 1:
        reasons.append("ncu_capture_job_invalid")
        expected_launches: list[Mapping[str, Any]] = []
        capture_mode = "invalid"
    else:
        capture_mode = str(jobs[0].get("capture_mode", ""))
        expected_launches = [
            item for item in jobs[0].get("expected_launches", ())
            if isinstance(item, Mapping)
        ]
        if capture_mode not in {"kernel_invocations", "nvtx_ranges"}:
            reasons.append("ncu_capture_mode_invalid")
    expected_campaign = (
        plan.get("campaign", {}).get("sha256")
        if isinstance(plan.get("campaign"), Mapping) else None
    )
    ncu_identity = ncu_profile.get("run_identity")
    stage_identity = stage_profile.get("run_identity")
    stage_campaign = (
        stage_identity.get("profiling_campaign")
        if isinstance(stage_identity, Mapping) else None
    )
    expected_ranges = [
        {"start": int(iteration), "end": int(iteration)}
        for iteration in plan.get("coverage", {}).get("representative_iterations", ())
    ] if isinstance(plan.get("coverage"), Mapping) else []
    stage_process_id = (
        stage_identity.get("process_id") if isinstance(stage_identity, Mapping) else None
    )
    if (
        ncu_profile.get("status") != "passed"
        or not isinstance(ncu_identity, Mapping)
        or ncu_identity.get("status") != "passed"
        or ncu_identity.get("process_ids") != [stage_process_id]
        or stage_profile.get("status") != "passed"
        or stage_profile.get("iteration_ranges") != expected_ranges
        or not isinstance(stage_identity, Mapping)
        or stage_identity.get("cuda_profiler_api_control") is not True
        or not isinstance(stage_campaign, Mapping)
        or stage_campaign.get("profile_mode") != "representative"
        or stage_campaign.get("profile_tool") != "ncu"
    ):
        reasons.append("ncu_job_run_identity_invalid")

    def launch_key(launch: Mapping[str, Any], *, expected: bool) -> tuple[Any, ...]:
        block_name = "block" if expected else "block_size"
        return (
            int(launch["iteration"]), str(launch["stage"]), int(launch["call_index"]),
            str(launch["kernel_name"]),
            tuple(int(value) for value in launch[block_name]),
        )

    def observed_signature(launch: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        grid = [int(value) for value in launch["grid_size"]]
        block = [int(value) for value in launch["block_size"]]
        if (
            len(grid) != 3 or len(block) != 3
            or any(value <= 0 for value in (*grid, *block))
        ):
            raise ValueError("NCU launch shape is invalid")
        document = {
            "stage": str(launch["stage"]),
            "kernel_name": str(launch["kernel_name"]),
            "grid": grid,
            "block": block,
        }
        return sha256_bytes(canonical_json(document)), document

    raw_launches = [
        launch for launch in ncu_profile.get("launches", ())
        if isinstance(launch, Mapping)
    ]
    bound_launches = []
    if capture_mode == "kernel_invocations":
        expected_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        actual_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        try:
            for launch in expected_launches:
                expected_groups[launch_key(launch, expected=True)].append(launch)
            for launch in raw_launches:
                actual_groups[launch_key(launch, expected=False)].append(launch)
        except (KeyError, TypeError, ValueError):
            reasons.append("ncu_job_launch_identity_invalid")
        expected_queues = {}
        for key in set(expected_groups) | set(actual_groups):
            expected_group = sorted(expected_groups[key], key=lambda item: int(
                item["kernel_ordinal_in_call"]
            ))
            if len(expected_group) != len(actual_groups[key]):
                reasons.append("ncu_job_launch_multiplicity_mismatch")
            expected_queues[key] = deque(expected_group)
        signature_ids = {
            (
                int(signature_iteration["iteration"]), str(signature["stage"]),
                str(signature["kernel_name"]),
                tuple(int(value) for value in signature["grid"]),
                tuple(int(value) for value in signature["block"]),
            ): str(signature["signature_id"])
            for signature in plan.get("signatures", ())
            if isinstance(signature, Mapping)
            for signature_iteration in signature.get("representative_iterations", ())
            if isinstance(signature_iteration, Mapping)
        }
        matched_count = 0
        for actual in raw_launches:
            try:
                key = launch_key(actual, expected=False)
            except (KeyError, TypeError, ValueError):
                continue
            queue = expected_queues.get(key)
            if not queue:
                observed_identity = {
                    "capture_job_index": job_index,
                    "process_id": stage_process_id,
                    "launch_id": actual.get("launch_id"),
                    "iteration": actual.get("iteration"),
                    "stage": actual.get("stage"),
                    "call_index": actual.get("call_index"),
                    "kernel_name": actual.get("kernel_name"),
                    "grid_size": actual.get("grid_size"),
                    "block_size": actual.get("block_size"),
                }
                signature_identity = (
                    int(actual["iteration"]), str(actual["stage"]),
                    str(actual["kernel_name"]),
                    tuple(int(value) for value in actual["grid_size"]),
                    tuple(int(value) for value in actual["block_size"]),
                )
                bound_launches.append({
                    **dict(actual),
                    "selected_launch_id": sha256_bytes(canonical_json(observed_identity)),
                    "signature_id": signature_ids.get(signature_identity),
                    "required_sample": False,
                    "planned_anchor": None,
                })
                continue
            expected = queue.popleft()
            matched_count += 1
            bound_launches.append({
                **dict(actual),
                "selected_launch_id": str(expected["selected_launch_id"]),
                "signature_id": str(expected["signature_id"]),
                "observed_signature_id": observed_signature(actual)[0],
                "iteration": int(expected["iteration"]),
                "stage": str(expected["stage"]),
                "call_index": int(expected["call_index"]),
                "kernel_ordinal_in_call": int(expected["kernel_ordinal_in_call"]),
                "kernel_name_ordinal_in_call": int(
                    expected["kernel_name_ordinal_in_call"]
                ),
                "kernel_name_ordinal_in_capture": int(
                    expected["kernel_name_ordinal_in_capture"]
                ),
                "kernel_name_ordinal_in_invocation_capture": int(
                    expected["kernel_name_ordinal_in_invocation_capture"]
                ),
                "planned_grid_size": [int(value) for value in expected["grid"]],
                "grid_shape_matches_primary_nsys": (
                    [int(value) for value in actual["grid_size"]]
                    == [int(value) for value in expected["grid"]]
                ),
                "required_sample": bool(expected["required_sample"]),
            })
        if matched_count != len(expected_launches):
            reasons.append("ncu_job_selected_launch_count_mismatch")
    elif capture_mode == "nvtx_ranges":
        range_stages = {
            str(value) for value in jobs[0].get("range_stages", ())
        } if jobs else set()
        range_iterations = {
            int(value) for value in jobs[0].get("representative_iterations", ())
        } if jobs else set()
        range_calls = {
            (int(item["iteration"]), str(item["stage"]), int(item["call_index"]))
            for item in jobs[0].get("range_calls", ())
            if isinstance(item, Mapping)
        } if jobs else set()
        if not raw_launches:
            reasons.append("ncu_range_launches_missing")
        for actual in raw_launches:
            try:
                signature_id, signature = observed_signature(actual)
                iteration = int(actual["iteration"])
                stage = str(actual["stage"])
                call_index = int(actual["call_index"])
            except (KeyError, TypeError, ValueError):
                reasons.append("ncu_job_launch_identity_invalid")
                continue
            if (
                stage not in range_stages
                or iteration not in range_iterations
                or call_index <= 0
                or (iteration, stage, call_index) not in range_calls
            ):
                reasons.append("ncu_range_scope_mismatch")
                continue
            observed_identity = {
                "capture_job_index": job_index,
                "process_id": stage_process_id,
                "launch_id": actual.get("launch_id"),
                "iteration": actual.get("iteration"),
                "stage": actual.get("stage"),
                "call_index": actual.get("call_index"),
                "kernel_name": actual.get("kernel_name"),
                "grid_size": actual.get("grid_size"),
                "block_size": actual.get("block_size"),
            }
            bound_launches.append({
                **dict(actual),
                "selected_launch_id": sha256_bytes(canonical_json(observed_identity)),
                "signature_id": signature_id,
                "observed_signature": signature,
                "required_sample": True,
                "planned_anchor": None,
            })
    return {
        **dict(ncu_profile),
        "schema_version": NCU_SELECTED_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed_preflight",
        "formal_performance_eligible": False,
        "run_identity": {
            "status": "passed" if not reasons else "failed_preflight",
            "profiling_campaign_sha256": expected_campaign,
            "ncu_plan_content_sha256": plan_hash,
            "capture_job_index": job_index,
            "capture_mode": capture_mode,
            "process_id": stage_process_id,
            "repository_commit": stage_identity.get("repository_commit")
            if isinstance(stage_identity, Mapping) else None,
            "stage_profile_source": stage_profile.get("source"),
            "stage_profile_source_sha256": stage_profile.get("source_sha256"),
        },
        "launches": bound_launches,
        "capture_mode": capture_mode,
        "binding_reasons": sorted(set(reasons)),
    }


def _source_sections(path: Path) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    index = 0
    while index < len(rows):
        row = rows[index]
        if len(row) < 2 or row[0] != "Kernel Name":
            index += 1
            continue
        kernel_name = row[1]
        index += 1
        if index >= len(rows) or "Source" not in rows[index]:
            raise ValueError("Nsight Compute source CSV has no Source column")
        header = rows[index]
        source_index = header.index("Source")
        count_name = "Predicated-On Thread Instructions Executed"
        if count_name not in header:
            raise ValueError("Nsight Compute source CSV has no predicated thread count")
        count_index = header.index(count_name)
        warp_name = "Instructions Executed"
        if warp_name not in header:
            raise ValueError("Nsight Compute source CSV has no warp instruction count")
        warp_index = header.index(warp_name)
        index += 1
        instructions = []
        while index < len(rows) and (not rows[index] or rows[index][0] != "Kernel Name"):
            current = rows[index]
            index += 1
            if len(current) <= max(source_index, count_index, warp_index):
                continue
            source = current[source_index].strip()
            if not source:
                continue
            count = _metric_value(current[count_index] or "0")
            warp_count = _metric_value(current[warp_index] or "0")
            instructions.append({
                "source": source,
                "predicated_thread_count": count,
                "warp_instruction_count": warp_count,
            })
        sections.append({"kernel_name": kernel_name, "instructions": instructions})
    return sections


def _sass_category(
    source: str, kernel_name: str,
) -> tuple[str | None, str | None, bool]:
    instruction = source.lstrip("@!P0123456789 ").split(maxsplit=1)[0]
    if instruction == "MUFU.EX2":
        return "exp", None, True
    if instruction == "MUFU.LG2":
        return "log", None, True
    if instruction in {"MUFU.RCP", "MUFU.RCP64H"}:
        return "rcp", None, True
    if instruction in {"MUFU.SQRT", "MUFU.RSQ", "MUFU.RSQ64H"}:
        return "sqrt", None, True
    if instruction.startswith("MUFU."):
        return None, instruction, True
    # Keep non-transcendental XU work separate from arithmetic counts while
    # including it in the per-launch XU reconciliation.  These opcodes are
    # established from the collected Ampere source counters, not inferred
    # from kernel names.
    if instruction.startswith("F2I"):
        return "conversion", None, True
    if instruction.startswith(("FCHK", "FRND", "FLO", "BREV")) or instruction == "OPC":
        return "xu_auxiliary", None, True
    if instruction.startswith(("ATOM", "RED.")):
        return "atomic", None, False
    return None, None, False


def classify_sass_csv(
    source_path: Path,
    ncu_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach dynamic SASS opcode counts to the matching NCU launches."""

    launches = ncu_profile.get("launches")
    if not isinstance(launches, list):
        raise ValueError("NCU counter profile has no launches")
    raw_sections = _source_sections(source_path.resolve())
    raw_remaining = Counter(str(section["kernel_name"]) for section in raw_sections)
    launch_remaining = Counter(str(launch.get("kernel_name")) for launch in launches)
    source_index = 0
    classified_launches = [dict(launch) for launch in launches]
    unmatched_sections = []
    unmatched_launch_ids = []
    for match, launch in enumerate(launches):
        kernel_name = str(launch.get("kernel_name"))
        while (
            source_index < len(raw_sections)
            and launch_remaining[str(raw_sections[source_index]["kernel_name"])] == 0
        ):
            unmatched_sections.append(raw_sections[source_index]["kernel_name"])
            raw_remaining[str(raw_sections[source_index]["kernel_name"])] -= 1
            source_index += 1
        if (
            source_index >= len(raw_sections)
            or raw_sections[source_index]["kernel_name"] != kernel_name
        ):
            unmatched_launch_ids.append(launch.get("launch_id"))
            launch_remaining[kernel_name] -= 1
            continue
        section = raw_sections[source_index]
        raw_remaining[kernel_name] -= 1
        launch_remaining[kernel_name] -= 1
        source_index += 1
        # The NCU source page commonly emits the same SASS table twice for a
        # launch.  Consume the duplicate only when the remaining raw count is
        # larger than the remaining launch count, preserving consecutive
        # launches whose source tables happen to be identical.
        if (
            source_index < len(raw_sections)
            and raw_sections[source_index] == section
            and raw_remaining[kernel_name] > launch_remaining[kernel_name]
        ):
            raw_remaining[kernel_name] -= 1
            source_index += 1
        counts: dict[str, float] = defaultdict(float)
        unsupported: dict[str, float] = defaultdict(float)
        classified_xu_warp_instructions = 0.0
        for instruction in section["instructions"]:
            count = float(instruction["predicated_thread_count"])
            if count <= 0:
                continue
            category, unsupported_opcode, is_xu = _sass_category(
                str(instruction["source"]), str(section["kernel_name"])
            )
            if is_xu:
                classified_xu_warp_instructions += float(
                    instruction["warp_instruction_count"]
                )
            if category is not None:
                counts[category] += count
            if unsupported_opcode is not None:
                unsupported[unsupported_opcode] += count
        launch = classified_launches[match]
        for category, count in counts.items():
            launch[f"{category}_operations"] = count
        measured_xu = float(launch.get("metrics", {}).get("xu_instructions", 0.0))
        xu_difference = measured_xu - classified_xu_warp_instructions
        has_xu_metric = "xu_instructions" in launch.get("metrics", {})
        launch["sass_classification"] = {
            "status": (
                "passed" if has_xu_metric and not unsupported
                else "unsupported_or_incomplete"
            ),
            "unsupported_opcodes": dict(sorted(unsupported.items())),
            "measured_xu_warp_instructions": measured_xu,
            "classified_xu_warp_instructions": classified_xu_warp_instructions,
            "unaccounted_xu_warp_instructions": xu_difference,
            "xu_reconciliation_status": (
                "exact" if abs(xu_difference) <= 0.5 else "replay_variation"
            ),
        }
    unmatched_sections.extend(
        section["kernel_name"] for section in raw_sections[source_index:]
    )
    stage_summaries: dict[str, dict[str, Any]] = {
        str(stage): dict(summary)
        for stage, summary in ncu_profile.get("stage_summaries", {}).items()
    }
    for stage, summary in stage_summaries.items():
        stage_launches = [launch for launch in classified_launches if launch.get("stage") == stage]
        unsupported: dict[str, float] = defaultdict(float)
        unaccounted_xu = 0.0
        incomplete_classification = []
        for launch in stage_launches:
            for category in (
                "exp", "log", "rcp", "sqrt", "conversion", "xu_auxiliary", "atomic",
            ):
                summary[f"{category}_operations"] = float(summary.get(f"{category}_operations", 0.0)) + float(
                    launch.get(f"{category}_operations", 0.0)
                )
            evidence = launch.get("sass_classification", {})
            for opcode, count in evidence.get("unsupported_opcodes", {}).items():
                unsupported[str(opcode)] += float(count)
            unaccounted_xu += abs(float(evidence.get("unaccounted_xu_warp_instructions", 0.0)))
            if evidence.get("status") != "passed":
                incomplete_classification.append(launch.get("launch_id"))
        missing = [launch.get("launch_id") for launch in stage_launches if "sass_classification" not in launch]
        classification_status = (
            "passed"
            if not unsupported and not missing and not incomplete_classification
            else "incomplete"
        )
        summary["sass_classification"] = {
            "status": classification_status,
            "unsupported_opcodes": dict(sorted(unsupported.items())),
            "missing_launch_ids": missing,
            "incomplete_launch_ids": incomplete_classification,
            "unaccounted_xu_warp_instructions": unaccounted_xu,
            "xu_reconciliation_status": (
                "exact" if unaccounted_xu <= 0.5 else "replay_variation"
            ),
            "xu_coverage_complete": classification_status == "passed",
            "transcendental_opcode_coverage_complete": (
                classification_status == "passed" and not unsupported
            ),
        }
        summary["weight_eligible"] = classification_status == "passed"
        if classification_status == "passed":
            summary["unclassified_transcendental_operations"] = 0.0
        else:
            summary["unclassified_transcendental_operations"] = max(
                float(summary.get("unclassified_xu_instructions", 0.0)),
                sum(unsupported.values()),
            )
    return {
        **dict(ncu_profile),
        "schema_version": SASS_SCHEMA_VERSION,
        "source_csv": str(source_path.resolve()),
        "source_csv_sha256": sha256_file(source_path),
        "stage_summaries": stage_summaries,
        "launches": classified_launches,
        "unmatched_source_sections": unmatched_sections,
        "unmatched_launch_ids": unmatched_launch_ids,
        "status": (
            "passed"
            if not unmatched_sections and not unmatched_launch_ids
            else "failed_preflight"
        ),
    }


def _write(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_profile_artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("nsys", "ncu"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    ncu_job = commands.add_parser("ncu-job")
    ncu_job.add_argument("--input", type=Path, required=True)
    ncu_job.add_argument("--plan", type=Path, required=True)
    ncu_job.add_argument("--capture-job-index", type=int, required=True)
    ncu_job.add_argument("--stage-profile", type=Path, required=True)
    ncu_job.add_argument("--output", type=Path, required=True)
    sass = commands.add_parser("sass")
    sass.add_argument("--input", type=Path, required=True)
    sass.add_argument("--ncu-input", type=Path, required=True)
    sass.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "nsys":
        result = parse_nsys_sqlite(args.input)
    elif args.command == "ncu":
        result = parse_ncu_csv(args.input)
    elif args.command == "ncu-job":
        stage_profile = json.loads(args.stage_profile.read_text(encoding="utf-8"))
        stage_profile["source"] = str(args.stage_profile.resolve())
        stage_profile["source_sha256"] = sha256_file(args.stage_profile)
        result = bind_ncu_profile_to_plan(
            parse_ncu_csv(args.input),
            json.loads(args.plan.read_text(encoding="utf-8")),
            args.capture_job_index,
            stage_profile,
        )
    else:
        result = classify_sass_csv(
            args.input, json.loads(args.ncu_input.read_text(encoding="utf-8")),
        )
    _write(result, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()), "status": result["status"]
    }, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
