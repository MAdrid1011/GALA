"""Derive physical RelationPacket metadata for legacy quick traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from gala_sim.clamp.events import (
    PrimitiveKind,
    RELATION_PACKET_LANE_MASK_SHIFT,
    RELATION_PACKET_METADATA_VALID,
    event_dtype,
    has_relation_packet_metadata,
)

from .model import Trace


PACKET_DERIVATION_SCHEMA_VERSION = "gala-relation-packet-derivation-v1"
_RELATION_CHAIN_KINDS = (
    PrimitiveKind.CACHE_REQUEST,
    PrimitiveKind.CACHE_RETURN,
    PrimitiveKind.FORWARD,
    PrimitiveKind.ADJOINT,
    PrimitiveKind.GRADIENT_REDUCTION,
)


@dataclass(frozen=True)
class QueryDomain:
    """One row-major query domain whose final axis maps to packet lanes."""

    template_id: int
    query_base: int
    query_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.template_id <= 0 or self.query_base < 0:
            raise ValueError("query domain IDs must be positive/non-negative")
        if not self.query_shape or any(extent <= 0 for extent in self.query_shape):
            raise ValueError("query domain shape extents must be positive")

    @property
    def query_count(self) -> int:
        return int(np.prod(self.query_shape, dtype=np.int64))

    @property
    def query_end(self) -> int:
        return self.query_base + self.query_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "query_base": self.query_base,
            "query_shape": list(self.query_shape),
        }


def derive_quick_relation_packets(
    trace: Trace,
    domains: Iterable[QueryDomain],
    *,
    query_lanes: int = 8,
) -> Trace:
    """Annotate a legacy quick trace without changing its event graph.

    Relation masks contain exactly the logical rows retained by the dependency
    closed sample.  A backward stage may therefore contain a strict subset of
    the corresponding forward packet.  This exception is recorded explicitly
    and is accepted only for an ineligible quick-validation derivation.
    """

    _require_quick_scope(trace)
    if not 0 < query_lanes <= 8:
        raise ValueError("relation packet width must be in [1, 8]")
    domain_map = _domain_map(domains)
    if trace.metadata.get("relation_packet_derivation") is not None:
        raise ValueError("trace already has relation packet derivation metadata")

    events = np.array(trace.events, copy=True)
    kinds = events["primitive_kind"]
    packetized_rows = np.flatnonzero(np.isin(
        kinds,
        np.asarray(
            [int(PrimitiveKind.RELATION), int(PrimitiveKind.QUERY_CLOSE)]
            + [int(kind) for kind in _RELATION_CHAIN_KINDS],
            dtype=kinds.dtype,
        ),
    ))
    legacy_flags = events["flags"][packetized_rows]
    if np.any(legacy_flags != 0):
        raise ValueError("legacy packetized events must have zero flags")

    relation_rows = np.flatnonzero(kinds == int(PrimitiveKind.RELATION))
    if relation_rows.size == 0:
        raise ValueError("trace has no RELATION events to packetize")
    relation_events = events[relation_rows]
    relation_lanes, relation_bases = _coordinates(
        relation_events, domain_map, query_lanes,
    )
    candidate_ids = _unique_candidate_dependencies(trace, relation_rows)
    relation_masks = _group_masks(
        candidate_ids, relation_bases, relation_lanes, label="RELATION",
    )
    relation_flags = _encode_flags(relation_lanes, relation_masks)
    events["flags"][relation_rows] = relation_flags

    relation_ids = np.asarray(relation_events["relation_id"], dtype=np.int64)
    if np.any(relation_ids < 0) or np.unique(relation_ids).size != relation_ids.size:
        raise ValueError("RELATION IDs must be unique and non-negative")
    relation_order = np.argsort(relation_ids)
    sorted_relation_ids = relation_ids[relation_order]
    backward_subset_packets = 0
    for kind in _RELATION_CHAIN_KINDS:
        rows = np.flatnonzero(kinds == int(kind))
        if rows.size == 0:
            continue
        stage_relation_ids = np.asarray(events["relation_id"][rows], dtype=np.int64)
        positions = np.searchsorted(sorted_relation_ids, stage_relation_ids)
        valid = positions < sorted_relation_ids.size
        if np.any(valid):
            valid &= sorted_relation_ids[np.minimum(
                positions, sorted_relation_ids.size - 1,
            )] == stage_relation_ids
        if not np.all(valid):
            raise ValueError(f"{kind.name} refers to an unknown relation ID")
        relation_positions = relation_order[positions]
        stage_lanes = relation_lanes[relation_positions]
        stage_bases = relation_bases[relation_positions]
        stage_candidates = candidate_ids[relation_positions]
        stage_masks = _group_masks(
            stage_candidates, stage_bases, stage_lanes, label=kind.name,
        )
        parent_masks = relation_masks[relation_positions]
        if np.any((stage_masks & parent_masks) != stage_masks):
            raise ValueError(f"{kind.name} contains a lane outside its relation packet")
        if kind in {PrimitiveKind.ADJOINT, PrimitiveKind.GRADIENT_REDUCTION}:
            backward_subset_packets += _different_group_count(
                stage_candidates, stage_bases, stage_masks, parent_masks,
            )
        elif not np.array_equal(stage_masks, parent_masks):
            raise ValueError(f"{kind.name} is incomplete relative to RELATION")
        events["flags"][rows] = _encode_flags(stage_lanes, stage_masks)

    close_rows = np.flatnonzero(kinds == int(PrimitiveKind.QUERY_CLOSE))
    close_lanes, close_bases = _coordinates(
        events[close_rows], domain_map, query_lanes,
    )
    close_group = _close_group_keys(events[close_rows])
    close_masks = _group_masks(
        close_group, close_bases, close_lanes, label="QUERY_CLOSE",
    )
    events["flags"][close_rows] = _encode_flags(close_lanes, close_masks)

    metadata = dict(trace.metadata)
    metadata["relation_packet_derivation"] = {
        "schema_version": PACKET_DERIVATION_SCHEMA_VERSION,
        "result_scope": "quick_cycle_validation",
        "formal_performance_eligible": False,
        "source_trace_unchanged_except_flags": True,
        "query_lanes": query_lanes,
        "query_domains": [domain.as_dict() for domain in domain_map.values()],
        "logical_relation_events": int(relation_rows.size),
        "physical_relation_packets": int(np.unique(
            np.rec.fromarrays([candidate_ids, relation_bases])
        ).size),
        "backward_subset_stage_packets": backward_subset_packets,
        "allow_partial_backward_stage_masks": backward_subset_packets > 0,
    }
    return Trace(
        np.asarray(events, dtype=event_dtype()),
        trace.dependencies,
        trace.payload,
        metadata,
    )


def validate_packet_derivation(source: Trace, derived: Trace) -> None:
    """Prove that packet annotation changed only packetized event flags."""

    if source.events.shape != derived.events.shape:
        raise ValueError("derived trace changed the event count")
    for name in source.events.dtype.names or ():
        if name != "flags" and not np.array_equal(
            source.events[name], derived.events[name], equal_nan=True,
        ):
            raise ValueError(f"derived trace changed event field {name}")
    if not np.array_equal(source.dependencies, derived.dependencies):
        raise ValueError("derived trace changed dependencies")
    if not np.array_equal(source.payload, derived.payload, equal_nan=True):
        raise ValueError("derived trace changed payload")
    packetized = np.isin(
        source.events["primitive_kind"],
        [int(PrimitiveKind.RELATION), int(PrimitiveKind.QUERY_CLOSE)]
        + [int(kind) for kind in _RELATION_CHAIN_KINDS],
    )
    if not np.array_equal(
        source.events["flags"][~packetized], derived.events["flags"][~packetized],
    ):
        raise ValueError("derived trace changed non-packet event flags")
    if not np.all([
        has_relation_packet_metadata(int(flags))
        for flags in derived.events["flags"][packetized]
    ]):
        raise ValueError("derived trace lacks packet metadata")


def _require_quick_scope(trace: Trace) -> None:
    sample = trace.metadata.get("trace_sample")
    if not isinstance(sample, dict):
        raise ValueError("packet derivation is restricted to sampled quick traces")
    if (
        sample.get("result_scope") != "quick_cycle_validation"
        or sample.get("formal_performance_eligible") is not False
        or sample.get("quality_eligible") is not False
    ):
        raise ValueError("sample is not an ineligible quick-validation trace")


def _domain_map(domains: Iterable[QueryDomain]) -> dict[int, QueryDomain]:
    result: dict[int, QueryDomain] = {}
    for domain in domains:
        if domain.template_id in result:
            raise ValueError("query domain template IDs must be unique")
        result[domain.template_id] = domain
    if not result:
        raise ValueError("at least one query domain is required")
    return result


def _coordinates(
    rows: np.ndarray,
    domains: dict[int, QueryDomain],
    query_lanes: int,
) -> tuple[np.ndarray, np.ndarray]:
    lanes = np.empty(rows.size, dtype=np.int64)
    bases = np.empty(rows.size, dtype=np.int64)
    covered = np.zeros(rows.size, dtype=bool)
    for template_id, domain in domains.items():
        selected = rows["template_id"] == template_id
        if not np.any(selected):
            continue
        queries = np.asarray(rows["query_id"][selected], dtype=np.int64)
        offsets = queries - domain.query_base
        if np.any(offsets < 0) or np.any(offsets >= domain.query_count):
            raise ValueError(f"template {template_id} query lies outside its domain")
        axis = offsets % domain.query_shape[-1]
        selected_lanes = axis % query_lanes
        lanes[selected] = selected_lanes
        bases[selected] = queries - selected_lanes
        covered[selected] = True
    if not np.all(covered):
        missing = np.unique(rows["template_id"][~covered]).tolist()
        raise ValueError(f"packetized events lack query domains for templates {missing}")
    return lanes, bases


def _unique_candidate_dependencies(trace: Trace, relation_rows: np.ndarray) -> np.ndarray:
    rows = trace.events[relation_rows]
    if np.all(rows["dependency_count"] == 1):
        candidate_ids = np.asarray(
            trace.dependencies[rows["dependency_begin"]], dtype=np.int64,
        )
        if np.all(
            trace.events["primitive_kind"][candidate_ids]
            == int(PrimitiveKind.RELATION_CANDIDATE)
        ):
            return candidate_ids
    result = np.empty(relation_rows.size, dtype=np.int64)
    for index, row in enumerate(rows):
        begin = int(row["dependency_begin"])
        dependencies = trace.dependencies[begin:begin + int(row["dependency_count"])]
        candidates = dependencies[
            trace.events["primitive_kind"][dependencies]
            == int(PrimitiveKind.RELATION_CANDIDATE)
        ]
        if candidates.size != 1:
            raise ValueError("each RELATION needs exactly one candidate dependency")
        result[index] = int(candidates[0])
    return result


def _group_masks(
    first_key: np.ndarray,
    packet_bases: np.ndarray,
    lanes: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    if lanes.size == 0:
        return np.empty(0, dtype=np.uint8)
    order = np.lexsort((packet_bases, first_key))
    sorted_first = first_key[order]
    sorted_bases = packet_bases[order]
    starts = np.concatenate((np.asarray([0]), 1 + np.flatnonzero(
        (sorted_first[1:] != sorted_first[:-1])
        | (sorted_bases[1:] != sorted_bases[:-1])
    )))
    ends = np.concatenate((starts[1:], np.asarray([order.size])))
    lane_bits = np.left_shift(np.uint8(1), lanes[order].astype(np.uint8))
    group_masks = np.bitwise_or.reduceat(lane_bits, starts)
    if np.any(np.asarray([mask.bit_count() for mask in group_masks]) != ends - starts):
        raise ValueError(f"{label} packet repeats a lane")
    sorted_masks = np.repeat(group_masks, ends - starts)
    masks = np.empty_like(sorted_masks)
    masks[order] = sorted_masks
    return masks


def _different_group_count(
    first_key: np.ndarray,
    packet_bases: np.ndarray,
    actual_masks: np.ndarray,
    parent_masks: np.ndarray,
) -> int:
    order = np.lexsort((packet_bases, first_key))
    first = first_key[order]
    bases = packet_bases[order]
    starts = np.concatenate((np.asarray([0]), 1 + np.flatnonzero(
        (first[1:] != first[:-1]) | (bases[1:] != bases[:-1])
    )))
    return int(np.count_nonzero(
        actual_masks[order][starts] != parent_masks[order][starts]
    ))


def _close_group_keys(rows: np.ndarray) -> np.ndarray:
    return np.rec.fromarrays([
        rows["iteration_id"], rows["template_id"], rows["state_version"],
    ])


def _encode_flags(lanes: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return (
        np.uint32(RELATION_PACKET_METADATA_VALID)
        | lanes.astype(np.uint32)
        | (masks.astype(np.uint32) << np.uint32(RELATION_PACKET_LANE_MASK_SHIFT))
    )
