"""Plan phase-stratified compact packet groups without expanding trace events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gala_sim.clamp.events import (
    PrimitiveKind, ResourceClass, TraceEvent, UpdateBeginKind,
    dependency_dtype, event_dtype,
)
from gala_sim.mechanisms import CANONICAL_VARIANT_POLICIES
from gala_sim.timing.kernel import PackedTileStatistics, packed_tile_statistics
from gala_sim.trace import (
    Trace, VirtualPacketArchiveReader, VirtualQueryEventExpander,
    VirtualTracePacket,
)
from gala_sim.trace.virtual import (
    RASTER_BLOCK, RASTER_TEMPLATE_ID, VOXEL_BLOCK, VOXEL_TEMPLATE_ID,
)


def plan_representative_packet_groups(
    archive_root: Path,
    profiling_campaign: Path | None,
    *,
    expected_group_count: int,
    live_prefix: bool = False,
    single_iteration: int | None = None,
) -> dict[str, Any]:
    """Choose median-cost tiles from phase windows or one closed iteration.

    A one-iteration capture cannot supply an adjacent training window, but it
    still contains the complete forward, loss, backward, and optimizer
    lifecycle of that iteration.  ``single_iteration`` therefore provides a
    bounded, real-trace input for rapid end-to-end cycle validation without
    pretending that it is a multi-phase performance result.
    """
    if expected_group_count <= 0:
        raise ValueError("representative packet group count must be positive")
    if single_iteration is not None and single_iteration <= 0:
        raise ValueError("single representative iteration must be positive")
    reader = VirtualPacketArchiveReader(Path(archive_root))
    source_identity = _archive_source_identity(reader.manifest)
    if single_iteration is not None:
        descriptors = {
            (descriptor.iteration_id, descriptor.template_id): descriptor
            for descriptor in reader.packet_descriptors(iterations={single_iteration})
        }
        if not descriptors:
            raise ValueError(
                f"representative single iteration {single_iteration} has no packets"
            )
        groups = _single_iteration_groups(descriptors, reader, single_iteration)
        if len(groups) != expected_group_count:
            raise ValueError(
                f"representative single iteration produced {len(groups)} groups, "
                f"expected {expected_group_count}"
            )
        return {
            "schema_version": "gala-representative-packet-plan-v1",
            "result_scope": "representative_speedup_validation",
            "formal_performance_eligible": False,
            "campaign_complete": False,
            "live_prefix": bool(live_prefix),
            "single_iteration": True,
            "archive": str(Path(archive_root).resolve()),
            "profiling_campaign": None,
            "source_identity": source_identity,
            "group_count": len(groups),
            "iteration_count": 1,
            "iterations": [single_iteration],
            "groups": groups,
        }
    if profiling_campaign is None:
        raise ValueError("phase-window planning requires a profiling campaign")
    campaign = yaml.safe_load(Path(profiling_campaign).read_text(encoding="utf-8"))
    if not isinstance(campaign, dict):
        raise ValueError("profiling campaign root is not a mapping")
    raw_iterations = campaign.get("representative_iterations")
    if not isinstance(raw_iterations, list):
        raise ValueError("profiling campaign has no representative iterations")
    windows: list[tuple[int, int, tuple[str, ...]]] = []
    for item in raw_iterations:
        if not isinstance(item, dict):
            raise ValueError("representative iteration entry is not a mapping")
        iteration = int(item["iteration"])
        if iteration <= 1:
            continue
        roles = tuple(str(value) for value in item.get("roles", ()))
        windows.append((iteration - 1, iteration, roles))
    if not windows:
        raise ValueError("profiling campaign has no adjacent representative windows")
    campaign_window_count = len(windows)
    if live_prefix:
        prefix = reader.manifest.get("metadata", {}).get("live_prefix", {})
        if not isinstance(prefix, dict) or "last_closed_iteration" not in prefix:
            raise ValueError("live-prefix planning requires a live archive snapshot")
        last_closed_iteration = int(prefix["last_closed_iteration"])
        windows = [window for window in windows if window[1] <= last_closed_iteration]
        if not windows:
            raise ValueError("live archive prefix covers no representative window")
    target_iterations = {
        iteration for begin, end, _roles in windows for iteration in (begin, end)
    }
    descriptors = {
        (descriptor.iteration_id, descriptor.template_id): descriptor
        for descriptor in reader.packet_descriptors(iterations=target_iterations)
    }
    packets: dict[tuple[int, int], Any] = {}
    statistics: dict[tuple[int, int], PackedTileStatistics] = {}
    groups: list[dict[str, Any]] = []
    for begin, end, roles in windows:
        templates = sorted(
            {template for iteration, template in descriptors if iteration == begin}
            & {template for iteration, template in descriptors if iteration == end}
        )
        if not templates:
            raise ValueError(
                f"representative window {begin}:{end} has no common packet template"
            )
        for template_id in templates:
            keys = ((begin, template_id), (end, template_id))
            for key in keys:
                if key not in packets:
                    packets[key] = reader.packet(descriptors[key])
                    statistics[key] = packed_tile_statistics(packets[key])
            first = statistics[keys[0]]
            second = statistics[keys[1]]
            common = (
                (first.physical_packet_counts > 0)
                & (second.physical_packet_counts > 0)
            )
            common_tiles = first.tile_ids[common]
            if common_tiles.size == 0:
                raise ValueError(
                    f"representative window {begin}:{end} template {template_id} "
                    "has no common nonempty tile"
                )
            combined_packets = (
                first.physical_packet_counts[common]
                + second.physical_packet_counts[common]
            )
            median_packets = float(np.median(combined_packets))
            tile_order = np.lexsort((
                common_tiles,
                np.abs(combined_packets.astype(np.float64) - median_packets),
            ))
            tile_id = int(common_tiles[tile_order[0]])
            groups.append({
                "group_index": len(groups),
                "iterations": [begin, end],
                "roles": list(roles),
                "template_id": template_id,
                "tile_id": tile_id,
                "selection": "common_nonempty_tile_nearest_pair_median_physical_packets",
                "pair_median_physical_packets": median_packets,
                "packets": [
                    _tile_record(iteration, statistics[(iteration, template_id)], tile_id)
                    for iteration in (begin, end)
                ],
            })
    if len(groups) != expected_group_count:
        raise ValueError(
            f"representative campaign produced {len(groups)} groups, expected "
            f"{expected_group_count}"
        )
    return {
        "schema_version": "gala-representative-packet-plan-v1",
        "result_scope": "representative_speedup_validation",
        "formal_performance_eligible": False,
        "campaign_complete": len(windows) == campaign_window_count,
        "live_prefix": bool(live_prefix),
        "single_iteration": False,
        "archive": str(Path(archive_root).resolve()),
        "profiling_campaign": str(Path(profiling_campaign).resolve()),
        "source_identity": source_identity,
        "group_count": len(groups),
        "iteration_count": len(target_iterations),
        "iterations": sorted(target_iterations),
        "groups": groups,
    }


def _archive_source_identity(manifest: dict[str, Any]) -> dict[str, str] | None:
    """Return a capture identity only when its stable identifiers are present."""

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    model_id = metadata.get("model_id")
    dataset_id = metadata.get("dataset_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None
    model = metadata.get("model")
    dataset = metadata.get("dataset")
    return {
        "model_id": model_id,
        "model": model if isinstance(model, str) and model.strip() else model_id,
        "dataset_id": dataset_id,
        "dataset": (
            dataset if isinstance(dataset, str) and dataset.strip() else dataset_id
        ),
    }


def _single_iteration_groups(
    descriptors: dict[tuple[int, int], Any],
    reader: VirtualPacketArchiveReader,
    iteration: int,
) -> list[dict[str, Any]]:
    """Select one nonempty median packet tile per template in an iteration."""

    groups: list[dict[str, Any]] = []
    template_ids = sorted(
        template_id for packet_iteration, template_id in descriptors
        if packet_iteration == iteration
        and descriptors[(packet_iteration, template_id)].candidate_count > 0
    )
    for template_id in template_ids:
        statistics = packed_tile_statistics(
            reader.packet(descriptors[(iteration, template_id)])
        )
        nonempty = statistics.physical_packet_counts > 0
        tiles = statistics.tile_ids[nonempty]
        if tiles.size == 0:
            raise ValueError(
                f"representative single iteration {iteration} template "
                f"{template_id} has no nonempty tile"
            )
        physical_packets = statistics.physical_packet_counts[nonempty]
        median_packets = float(np.median(physical_packets))
        tile_order = np.lexsort((
            tiles,
            np.abs(physical_packets.astype(np.float64) - median_packets),
        ))
        tile_id = int(tiles[tile_order[0]])
        groups.append({
            "group_index": len(groups),
            "iterations": [iteration],
            "roles": ["single_iteration"],
            "template_id": template_id,
            "tile_id": tile_id,
            "selection": "nonempty_tile_nearest_median_physical_packets",
            "median_physical_packets": median_packets,
            "packets": [_tile_record(iteration, statistics, tile_id)],
        })
    return groups


def _tile_record(
    iteration: int, statistics: PackedTileStatistics, tile_id: int,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "candidate_count": int(statistics.candidate_counts[tile_id]),
        "logical_relation_count": int(statistics.logical_relation_counts[tile_id]),
        "physical_packet_count": int(statistics.physical_packet_counts[tile_id]),
        "lane_histogram": [
            int(value) for value in statistics.lane_histograms[tile_id]
        ],
    }


def build_representative_packet_trace(
    archive_root: Path,
    plan_path: Path,
    *,
    window_index: int,
    max_events: int,
    query_lanes: int,
    model_id: str | None = None,
    dataset_id: str | None = None,
) -> Trace:
    """Expand one planned representative window through the cycle replay.

    Single-iteration groups receive a dependency-closed no-op optimizer
    transaction after their backward frontier.  This preserves the complete
    execution shape without inventing a second iteration or a state change.
    """

    if window_index < 0 or max_events <= 0 or not 0 < query_lanes <= 8:
        raise ValueError("representative packet trace limits are invalid")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != "gala-representative-packet-plan-v1":
        raise ValueError("representative packet plan schema is unsupported")
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("representative packet plan has no groups")
    identity = _resolve_source_identity(
        plan.get("source_identity"), model_id=model_id, dataset_id=dataset_id,
    )
    windows: list[tuple[int, ...]] = []
    for group in groups:
        iterations = tuple(int(value) for value in group.get("iterations", ()))
        if len(iterations) not in {1, 2}:
            raise ValueError("representative packet group has an invalid window")
        if iterations not in windows:
            windows.append(iterations)
    if window_index >= len(windows):
        raise ValueError("representative packet window index is out of range")
    window = windows[window_index]
    selected_groups = [
        group for group in groups
        if tuple(int(value) for value in group["iterations"]) == window
    ]
    reader = VirtualPacketArchiveReader(Path(archive_root))
    descriptors = {
        (descriptor.iteration_id, descriptor.template_id): descriptor
        for descriptor in reader.packet_descriptors(iterations=set(window))
    }
    expander = VirtualQueryEventExpander(
        max_events=max_events,
        relation_query_lanes=query_lanes,
    )
    row_parts: list[np.ndarray] = []
    dependency_parts: list[np.ndarray] = []
    dependency_offset = 0
    query_base = 0
    packet_reports: list[dict[str, Any]] = []
    max_gaussian_id = -1
    prior_state_barrier: tuple[int, ...] = ()
    for iteration_index, iteration_id in enumerate(window):
        iteration_gradient_frontier: list[int] = []
        iteration_zero_relation_consumers: list[int] = []
        for group in sorted(selected_groups, key=lambda item: int(item["template_id"])):
            template_id = int(group["template_id"])
            descriptor = descriptors.get((iteration_id, template_id))
            if descriptor is None:
                raise ValueError(
                    f"representative packet {iteration_id}/{template_id} is absent"
                )
            source = reader.packet(descriptor)
            packet = _select_tile_packet(
                source, tile_id=int(group["tile_id"]), query_base=query_base,
            )
            query_base += packet.query_count
            zero_relation_queries = _zero_relation_query_ids(packet)
            max_gaussian_id = max(
                max_gaussian_id, int(packet.point_ids.max()),
            )
            for event_packet in expander.expand(
                packet, external_dependencies=prior_state_barrier,
            ):
                rows = event_packet.events.copy()
                iteration_gradient_frontier.extend(
                    int(event_id) for event_id in rows["event_id"][
                        rows["primitive_kind"]
                        == int(PrimitiveKind.GRADIENT_REDUCTION)
                    ]
                )
                consumer_rows = rows[
                    rows["primitive_kind"] == int(PrimitiveKind.CONSUMER)
                ]
                iteration_zero_relation_consumers.extend(
                    int(event_id) for event_id, query_id in zip(
                        consumer_rows["event_id"], consumer_rows["query_id"],
                        strict=True,
                    )
                    if int(query_id) in zero_relation_queries
                )
                rows["dependency_begin"] += dependency_offset
                row_parts.append(rows)
                dependency_parts.append(event_packet.dependencies)
                dependency_offset += event_packet.dependencies.size
            packet_reports.append({
                "iteration_id": iteration_id,
                "template_id": template_id,
                "tile_id": int(group["tile_id"]),
                "candidate_count": packet.candidate_count,
                "relation_count": packet.logical_relation_count,
                "physical_packet_count": next(
                    int(item["physical_packet_count"])
                    for item in group["packets"]
                    if int(item["iteration"]) == iteration_id
                ),
                "query_base": packet.query_base,
                "query_count": packet.query_count,
                "query_shape": list(packet.query_shape),
            })
        if len(window) == 1 or iteration_index + 1 < len(window):
            barrier_dependencies = (
                *iteration_gradient_frontier,
                *iteration_zero_relation_consumers,
            )
            barrier_rows, barrier_dependency_column = _noop_update_barrier(
                event_start=expander.next_event_id,
                iteration_id=iteration_id,
                dependencies=barrier_dependencies,
            )
            expander.next_event_id += barrier_rows.size
            barrier_rows["dependency_begin"] += dependency_offset
            row_parts.append(barrier_rows)
            dependency_parts.append(barrier_dependency_column)
            dependency_offset += barrier_dependency_column.size
            prior_state_barrier = (int(barrier_rows[-1]["event_id"]),)
    events = np.concatenate(row_parts).astype(event_dtype(), copy=False)
    dependencies = np.concatenate(dependency_parts).astype(
        dependency_dtype(), copy=False,
    )
    return Trace(
        events,
        dependencies,
        np.empty(0, dtype=np.dtype("<f4")),
        {
            "schema_version": "gala-clamp-events-v2",
            "model": identity["model"],
            "model_id": identity["model_id"],
            "dataset": identity["dataset"],
            "dataset_id": identity["dataset_id"],
            "initial_gaussian_count": max_gaussian_id + 1,
            "state_record_bytes": 128,
            "trace_sample": {
                "schema_version": "gala-representative-packet-trace-v1",
                "result_scope": "representative_window_simulation",
                "formal_performance_eligible": False,
                "quality_eligible": False,
                "selection": (
                    "single_iteration_median_physical_tile"
                    if len(window) == 1
                    else "phase_window_common_median_physical_tile"
                ),
                "cross_iteration_barrier": "validated_noop_optimizer_transaction",
                "source_plan": str(Path(plan_path).resolve()),
                "window_index": window_index,
                "iterations": list(window),
                "query_lanes": query_lanes,
                "eligible_policies": [
                    "base", "query", "residency", "full",
                    "query_oracle", "residency_oracle",
                    *CANONICAL_VARIANT_POLICIES,
                ],
                "packets": packet_reports,
                "sample_event_count": int(events.size),
                "sample_dependency_count": int(dependencies.size),
            },
        },
    )


def _resolve_source_identity(
    plan_identity: Any, *, model_id: str | None, dataset_id: str | None,
) -> dict[str, str]:
    """Use immutable capture provenance, or require explicit legacy identity."""

    identity = (
        dict(plan_identity) if isinstance(plan_identity, dict) else {}
    )
    archived_model_id = identity.get("model_id")
    archived_dataset_id = identity.get("dataset_id")
    if archived_model_id is not None and model_id not in {None, archived_model_id}:
        raise ValueError("explicit model_id conflicts with archive provenance")
    if archived_dataset_id is not None and dataset_id not in {None, archived_dataset_id}:
        raise ValueError("explicit dataset_id conflicts with archive provenance")
    resolved_model_id = archived_model_id or model_id
    resolved_dataset_id = archived_dataset_id or dataset_id
    if not isinstance(resolved_model_id, str) or not resolved_model_id.strip():
        raise ValueError("representative trace requires a model_id provenance")
    if not isinstance(resolved_dataset_id, str) or not resolved_dataset_id.strip():
        raise ValueError("representative trace requires a dataset_id provenance")
    model = identity.get("model")
    dataset = identity.get("dataset")
    return {
        "model_id": resolved_model_id,
        "model": model if isinstance(model, str) and model.strip() else resolved_model_id,
        "dataset_id": resolved_dataset_id,
        "dataset": (
            dataset if isinstance(dataset, str) and dataset.strip()
            else resolved_dataset_id
        ),
    }


def _select_tile_packet(
    source: VirtualTracePacket, *, tile_id: int, query_base: int,
) -> VirtualTracePacket:
    shape = {
        RASTER_TEMPLATE_ID: RASTER_BLOCK,
        VOXEL_TEMPLATE_ID: VOXEL_BLOCK,
    }.get(source.template_id)
    if shape is None:
        raise ValueError(f"unsupported representative template {source.template_id}")
    tiles = np.right_shift(
        np.asarray(source.point_keys, dtype=np.uint64), np.uint64(32),
    )
    selected = np.flatnonzero(tiles == tile_id)
    if selected.size == 0:
        raise ValueError(f"representative packet contains no tile {tile_id}")
    low_key_mask = np.uint64((1 << 32) - 1)
    return VirtualTracePacket(
        iteration_id=source.iteration_id,
        template_id=source.template_id,
        query_base=query_base,
        query_shape=shape,
        point_ids=source.point_ids[selected].copy(),
        point_keys=np.bitwise_and(source.point_keys[selected], low_key_mask),
        masks=source.masks[selected].copy(),
        state_version=0,
        field_mask=source.field_mask,
        loss_flags=source.loss_flags,
        ssim_radius=source.ssim_radius,
        backward_confirmed=source.backward_confirmed,
    )


def _zero_relation_query_ids(packet: VirtualTracePacket) -> set[int]:
    counts = np.zeros(packet.query_count, dtype=np.uint32)
    for _candidates, query_ids, _gaussians, _keys in packet.iter_relation_arrays(
        max(packet.logical_relation_count, 1),
    ):
        np.add.at(counts, query_ids - packet.query_base, 1)
    return set(
        (packet.query_base + np.flatnonzero(counts == 0)).astype(int).tolist()
    )


def _noop_update_barrier(
    *, event_start: int, iteration_id: int, dependencies: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not dependencies:
        raise ValueError("representative update barrier has no backward frontier")
    rows = np.empty(2, dtype=event_dtype())
    rows[:] = TraceEvent().as_tuple()
    begin_id = event_start
    end_id = event_start + 1
    rows["event_id"] = (begin_id, end_id)
    rows["iteration_id"] = iteration_id
    rows["state_version"] = 0
    rows["resource_class"] = int(ResourceClass.UPDATE)
    rows["flags"] = int(UpdateBeginKind.OPTIMIZER)
    rows[0]["primitive_kind"] = int(PrimitiveKind.UPDATE_BEGIN)
    rows[0]["dependency_begin"] = 0
    rows[0]["dependency_count"] = len(dependencies)
    rows[1]["primitive_kind"] = int(PrimitiveKind.UPDATE_END)
    rows[1]["reduction_key"] = begin_id
    rows[1]["dependency_begin"] = len(dependencies)
    rows[1]["dependency_count"] = 1
    dependency_column = np.asarray(
        (*dependencies, begin_id), dtype=dependency_dtype(),
    )
    return rows, dependency_column
