"""Bounded relation-store feasibility checks for compact packet archives."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from gala_sim.config import GalaConfig
from gala_sim.trace import VirtualPacketArchiveReader


def run_relation_capacity_preflight(
    archive_root: Path,
    config: GalaConfig,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure exact source-order relation occupancy without event expansion."""

    config.require_ready()
    query_lanes = int(config.value("compute.relations_per_microcontext"))
    relation_capacity = int(config.value("query.relation_store_records"))
    candidate_ordinal_bits = int(
        config.value("query.relation_candidate_ordinal_bits")
    )
    candidate_ordinal_limit = 1 << candidate_ordinal_bits
    reader = VirtualPacketArchiveReader(archive_root)
    packets: list[dict[str, Any]] = []
    for packet_index, (kind, value) in enumerate(
        (item for item in reader.records() if item[0] == "packet")
    ):
        del kind
        wavefront = value.relation_store_wavefront(
            query_lanes=query_lanes,
            relation_capacity=relation_capacity,
        )
        record = {
            "packet_index": packet_index,
            "iteration_id": value.iteration_id,
            "template_id": value.template_id,
            "query_count": value.query_count,
            "candidate_count": value.candidate_count,
            "logical_relation_count": value.logical_relation_count,
            "max_candidates_per_tile": value.max_candidates_per_tile,
            "candidate_ordinal_bits": candidate_ordinal_bits,
            "candidate_ordinal_limit": candidate_ordinal_limit,
            "candidate_ordinal_feasible": (
                value.max_candidates_per_tile <= candidate_ordinal_limit
            ),
            **asdict(wavefront),
            "capacity_deficit_records": max(
                wavefront.peak_live_records - relation_capacity, 0,
            ),
        }
        packets.append(record)
        if progress is not None:
            progress(record)
    if not packets:
        raise ValueError("virtual packet archive contains no query packets")
    failed = [
        packet for packet in packets
        if not packet["feasible"] or not packet["candidate_ordinal_feasible"]
    ]
    capacity_failed = [packet for packet in packets if not packet["feasible"]]
    ordinal_failed = [
        packet for packet in packets if not packet["candidate_ordinal_feasible"]
    ]
    return {
        "schema_version": "gala-relation-capacity-preflight-v1",
        "result_scope": "capacity_preflight",
        "status": "passed" if not failed else "failed_preflight",
        "reason": (
            None if not failed else
            "relation_candidate_ordinal_overflow" if ordinal_failed else
            "relation_store_capacity_infeasible"
        ),
        "formal_performance_eligible": False,
        "topology_scope": "exact_for_declared_topology_not_global_lower_bound",
        "archive_formal_performance_eligible": reader.formal_performance_eligible,
        "configuration_sha256_recorded_only": config.sha256,
        "query_lanes": query_lanes,
        "relation_capacity_records": relation_capacity,
        "candidate_ordinal_bits": candidate_ordinal_bits,
        "candidate_ordinal_limit": candidate_ordinal_limit,
        "packet_count": len(packets),
        "failed_packet_count": len(failed),
        "capacity_failed_packet_count": len(capacity_failed),
        "ordinal_failed_packet_count": len(ordinal_failed),
        "maximum_peak_live_records": max(
            int(packet["peak_live_records"]) for packet in packets
        ),
        "packets": packets,
    }
