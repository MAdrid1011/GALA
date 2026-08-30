"""Plan phase-stratified compact packet groups without expanding trace events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gala_sim.timing.kernel import PackedTileStatistics, packed_tile_statistics
from gala_sim.trace import VirtualPacketArchiveReader


def plan_representative_packet_groups(
    archive_root: Path,
    profiling_campaign: Path,
    *,
    expected_group_count: int,
) -> dict[str, Any]:
    if expected_group_count <= 0:
        raise ValueError("representative packet group count must be positive")
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
    target_iterations = {
        iteration for begin, end, _roles in windows for iteration in (begin, end)
    }
    reader = VirtualPacketArchiveReader(Path(archive_root))
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
        "archive": str(Path(archive_root).resolve()),
        "profiling_campaign": str(Path(profiling_campaign).resolve()),
        "group_count": len(groups),
        "iteration_count": len(target_iterations),
        "iterations": sorted(target_iterations),
        "groups": groups,
    }


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
