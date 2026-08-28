"""Rebuild complete quick packet traces from captured CUDA relation records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gala_sim.clamp.events import dependency_dtype, event_dtype

from .model import Trace
from .virtual import (
    MASK_WORD_BITS,
    RASTER_BLOCK,
    RASTER_TEMPLATE_ID,
    VOXEL_BLOCK,
    VOXEL_TEMPLATE_ID,
    VirtualQueryEventExpander,
    VirtualTracePacket,
)


CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION = "gala-captured-packet-sample-v1"


@dataclass(frozen=True)
class CapturedPacketSpec:
    """Location and query semantics for one captured physical tile/brick."""

    candidate_records: Path
    relation_records: Path
    iteration_id: int
    template_id: int
    query_base: int
    query_shape: tuple[int, ...]
    tile_id: int
    loss_flags: int
    ssim_radius: int = 0
    state_version: int = 0
    field_mask: int = 0

    def __post_init__(self) -> None:
        if min(
            self.iteration_id,
            self.template_id,
            self.query_base,
            self.tile_id,
            self.loss_flags,
            self.ssim_radius,
            self.state_version,
            self.field_mask,
        ) < 0:
            raise ValueError("captured packet identifiers and flags must be non-negative")
        if self.template_id == 0 or self.loss_flags == 0:
            raise ValueError("captured packet needs a template and loss consumer")
        if not self.query_shape or any(extent <= 0 for extent in self.query_shape):
            raise ValueError("captured packet query shape must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CapturedPacketSpec":
        required = {
            "candidate_records", "relation_records", "iteration_id", "template_id",
            "query_base", "query_shape", "tile_id", "loss_flags",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(
                "captured packet manifest lacks " + ", ".join(sorted(missing))
            )
        shape = value["query_shape"]
        if not isinstance(shape, list):
            raise ValueError("captured packet query_shape must be a JSON list")
        return cls(
            candidate_records=Path(value["candidate_records"]),
            relation_records=Path(value["relation_records"]),
            iteration_id=int(value["iteration_id"]),
            template_id=int(value["template_id"]),
            query_base=int(value["query_base"]),
            query_shape=tuple(int(extent) for extent in shape),
            tile_id=int(value["tile_id"]),
            loss_flags=int(value["loss_flags"]),
            ssim_radius=int(value.get("ssim_radius", 0)),
            state_version=int(value.get("state_version", 0)),
            field_mask=int(value.get("field_mask", 0)),
        )


def captured_virtual_packet(spec: CapturedPacketSpec) -> VirtualTracePacket:
    """Extract one exact tile/brick without scanning unrelated relation rows."""

    expected_shape = {
        RASTER_TEMPLATE_ID: RASTER_BLOCK,
        VOXEL_TEMPLATE_ID: VOXEL_BLOCK,
    }.get(spec.template_id)
    if expected_shape is None:
        raise ValueError(f"unsupported captured packet template {spec.template_id}")
    if spec.query_shape != expected_shape or spec.tile_id != 0:
        raise ValueError(
            "captured packet quick samples require one canonical tile/brick rebased "
            "to tile zero"
        )
    candidates = _record_array(spec.candidate_records)
    relations = _record_array(spec.relation_records)
    if candidates.size == 0 or relations.size == 0:
        raise ValueError("captured packet source records are empty")
    candidate_count = candidates.shape[0]
    if (
        np.any(candidates[:, 0] != 0)
        or not np.array_equal(candidates[:, 1], np.arange(candidate_count))
        or np.any(relations[:, 0] != 1)
    ):
        raise ValueError("captured relation records use an unsupported layout")
    relation_candidates = relations[:, 1]
    if np.any(relation_candidates[1:] < relation_candidates[:-1]):
        raise ValueError("captured relations are not candidate-major")

    point_keys = candidates[:, 3].view(np.uint64)
    tile_ids = np.right_shift(point_keys, np.uint64(32))
    selected = np.flatnonzero(tile_ids == spec.tile_id)
    if selected.size == 0:
        raise ValueError(f"captured records contain no tile {spec.tile_id}")
    if selected[-1] - selected[0] + 1 != selected.size:
        raise ValueError("captured tile candidates are not contiguous")
    first_candidate = int(selected[0])
    last_candidate = int(selected[-1])
    relation_begin = int(np.searchsorted(
        relation_candidates, first_candidate, side="left",
    ))
    relation_end = int(np.searchsorted(
        relation_candidates, last_candidate, side="right",
    ))
    packet_relations = relations[relation_begin:relation_end]
    if packet_relations.size and (
        np.any(packet_relations[:, 1] < first_candidate)
        or np.any(packet_relations[:, 1] > last_candidate)
    ):
        raise ValueError("captured tile relation bounds are inconsistent")

    query_count = int(np.prod(expected_shape, dtype=np.int64))
    local_queries = np.asarray(packet_relations[:, 2], dtype=np.int64)
    if local_queries.size and (
        int(local_queries.min()) < 0 or int(local_queries.max()) >= query_count
    ):
        raise ValueError("captured relation query lies outside the packet shape")
    local_candidates = np.asarray(
        packet_relations[:, 1] - first_candidate, dtype=np.int64,
    )
    masks = np.zeros(
        (selected.size, (query_count + MASK_WORD_BITS - 1) // MASK_WORD_BITS),
        dtype=np.dtype("<u4"),
    )
    if packet_relations.size:
        words = local_queries // MASK_WORD_BITS
        bits = np.left_shift(
            np.uint32(1), (local_queries % MASK_WORD_BITS).astype(np.uint32),
        )
        np.bitwise_or.at(masks, (local_candidates, words), bits)

    packet_candidates = candidates[selected]
    return VirtualTracePacket(
        iteration_id=spec.iteration_id,
        template_id=spec.template_id,
        query_base=spec.query_base,
        query_shape=spec.query_shape,
        point_ids=np.asarray(packet_candidates[:, 2], dtype=np.int64),
        point_keys=np.asarray(packet_candidates[:, 3].view(np.uint64), dtype=np.uint64),
        masks=masks,
        state_version=spec.state_version,
        field_mask=spec.field_mask,
        loss_flags=spec.loss_flags,
        ssim_radius=spec.ssim_radius,
        backward_confirmed=True,
    )


def complete_captured_packet_sample(
    specs: Iterable[CapturedPacketSpec],
    *,
    max_events: int,
    query_lanes: int = 8,
    initial_gaussian_count: int,
) -> Trace:
    """Expand captured packets through complete forward and backward chains."""

    specs = tuple(specs)
    if not specs:
        raise ValueError("captured packet sample needs at least one packet")
    if max_events <= 0 or initial_gaussian_count <= 0:
        raise ValueError("captured packet sample limits must be positive")
    expander = VirtualQueryEventExpander(
        max_events=max_events,
        relation_query_lanes=query_lanes,
    )
    row_parts: list[np.ndarray] = []
    dependency_parts: list[np.ndarray] = []
    dependency_offset = 0
    packet_reports: list[dict[str, Any]] = []
    for spec in specs:
        packet = captured_virtual_packet(spec)
        for event_packet in expander.expand(packet):
            rows = event_packet.events.copy()
            rows["dependency_begin"] += dependency_offset
            row_parts.append(rows)
            dependency_parts.append(event_packet.dependencies)
            dependency_offset += event_packet.dependencies.size
        packet_reports.append({
            "template_id": packet.template_id,
            "query_base": packet.query_base,
            "query_shape": list(packet.query_shape),
            "tile_id": spec.tile_id,
            "candidate_count": packet.candidate_count,
            "relation_count": packet.logical_relation_count,
            "query_count": packet.query_count,
            "loss_flags": packet.loss_flags,
            "ssim_radius": packet.ssim_radius,
            "candidate_records": str(spec.candidate_records),
            "relation_records": str(spec.relation_records),
        })
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
            "model": "R2-Gaussian",
            "dataset": "Chest",
            "initial_gaussian_count": initial_gaussian_count,
            "state_record_bytes": 128,
            "trace_sample": {
                "schema_version": CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION,
                "result_scope": "quick_cycle_validation",
                "formal_performance_eligible": False,
                "quality_eligible": False,
                "selection": "complete_captured_physical_packets",
                "query_lanes": query_lanes,
                "packets": packet_reports,
                "sample_event_count": int(events.size),
                "sample_dependency_count": int(dependencies.size),
            },
        },
    )


def _record_array(path: Path) -> np.ndarray:
    path = Path(path)
    record_bytes = 4 * np.dtype("<i8").itemsize
    size = path.stat().st_size
    if size % record_bytes:
        raise ValueError(f"captured record file has a partial row: {path}")
    if size == 0:
        return np.empty((0, 4), dtype=np.dtype("<i8"))
    return np.memmap(path, dtype=np.dtype("<i8"), mode="r").reshape(-1, 4)
