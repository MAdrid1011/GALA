"""Physical RelationPacket plan shared by offline and online cycle replay."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gala_sim.clamp.events import (
    PrimitiveKind,
    decode_relation_packet_flags,
    has_relation_packet_metadata,
)
from gala_sim.trace.model import Trace
from gala_sim.clamp.events import EVENT_SCHEMA_VERSION, dependency_dtype
from gala_sim.trace.virtual import VirtualEventPacket


RELATION_PACKET_STAGE_KINDS = frozenset({
    PrimitiveKind.RELATION,
    PrimitiveKind.CACHE_REQUEST,
    PrimitiveKind.CACHE_RETURN,
    PrimitiveKind.FORWARD,
    PrimitiveKind.ADJOINT,
    PrimitiveKind.GRADIENT_REDUCTION,
})
PACKETIZED_KINDS = RELATION_PACKET_STAGE_KINDS | {PrimitiveKind.QUERY_CLOSE}


class RelationPacketPlanError(ValueError):
    """The trace cannot be mapped to legal physical RelationPackets."""


@dataclass(frozen=True)
class PhysicalPacketStage:
    """Logical lane events that jointly consume one physical stage issue."""

    stage_id: int
    kind: PrimitiveKind
    event_ids: tuple[int, ...]
    lanes: tuple[int, ...]
    lane_mask: int
    query_base: int
    relation_packet_id: int | None

    def __post_init__(self) -> None:
        if self.stage_id < 0 or not self.event_ids:
            raise ValueError("physical packet stage identifiers must be non-negative")
        if len(self.event_ids) != len(self.lanes):
            raise ValueError("physical packet stage event and lane counts differ")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("physical packet stage contains duplicate events")
        if len(set(self.lanes)) != len(self.lanes):
            raise ValueError("physical packet stage contains duplicate lanes")
        if any(not self.lane_mask & (1 << lane) for lane in self.lanes):
            raise ValueError("physical packet stage contains an inactive lane")

    @property
    def head_event_id(self) -> int:
        return min(self.event_ids)

    @property
    def active_lanes(self) -> int:
        return self.lane_mask.bit_count()


@dataclass(frozen=True)
class RelationPacketPlan:
    """Frozen policy-independent physical packet plan for one exact trace."""

    query_lanes: int
    stages: tuple[PhysicalPacketStage, ...]
    event_to_stage: np.ndarray
    relation_packet_count: int
    _relation_packet_by_relation: Mapping[int, int]
    event_id_base: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.query_lanes <= 8:
            raise ValueError("relation packet width must be in [1, 8]")
        if self.event_to_stage.dtype != np.dtype("<i8"):
            raise ValueError("event-to-stage map must use little-endian int64")
        if self.relation_packet_count < 0:
            raise ValueError("relation packet count must be non-negative")
        self.event_to_stage.setflags(write=False)

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        query_lanes: int,
        require_metadata: bool | None = None,
    ) -> "RelationPacketPlan":
        if not 0 < query_lanes <= 8:
            raise ValueError("relation packet width must be in [1, 8]")
        strict = query_lanes > 1 if require_metadata is None else require_metadata
        allow_partial_backward = _allow_partial_backward_stage_masks(trace)
        events = trace.events
        kinds = events["primitive_kind"]

        relation_rows = np.flatnonzero(kinds == int(PrimitiveKind.RELATION))
        packet_members: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        relation_packet_key: dict[int, tuple[int, int]] = {}
        packet_contract: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
        for event_id_value in relation_rows:
            event_id = int(event_id_value)
            row = events[event_id]
            lane, lane_mask = _metadata_for_row(
                row, query_lanes=query_lanes, strict=strict,
            )
            candidate_ids = [
                int(dependency)
                for dependency in trace.dependency_ids(row)
                if PrimitiveKind(int(events[int(dependency)]["primitive_kind"]))
                is PrimitiveKind.RELATION_CANDIDATE
            ]
            if strict and len(candidate_ids) != 1:
                raise RelationPacketPlanError(
                    f"RELATION {event_id} needs one candidate dependency"
                )
            candidate_id = candidate_ids[0] if candidate_ids else event_id
            query_base = int(row["query_id"]) - lane
            key = (candidate_id, query_base)
            contract = (
                int(row["iteration_id"]), int(row["template_id"]),
                int(row["gaussian_id"]), int(row["state_version"]), lane_mask,
            )
            previous = packet_contract.setdefault(key, contract)
            if previous != contract:
                raise RelationPacketPlanError(
                    f"RELATION {event_id} disagrees with its physical packet"
                )
            relation_id = int(row["relation_id"])
            if relation_id < 0 or relation_id in relation_packet_key:
                if strict:
                    raise RelationPacketPlanError(
                        f"RELATION {event_id} has a non-unique relation ID"
                    )
                relation_id = -(event_id + 2)
            relation_packet_key[relation_id] = key
            packet_members.setdefault(key, []).append((lane, event_id, relation_id))

        ordered_packet_keys = sorted(
            packet_members, key=lambda key: min(item[1] for item in packet_members[key])
        )
        packet_id_by_key = {
            key: packet_id for packet_id, key in enumerate(ordered_packet_keys)
        }
        relation_packet_by_relation = {
            relation_id: packet_id_by_key[key]
            for relation_id, key in relation_packet_key.items()
        }
        for key, members in packet_members.items():
            lanes = [item[0] for item in members]
            lane_mask = packet_contract[key][-1]
            if len(set(lanes)) != len(lanes):
                raise RelationPacketPlanError(
                    f"physical relation packet {packet_id_by_key[key]} repeats a lane"
                )
            if strict and set(lanes) != _active_lanes(lane_mask):
                raise RelationPacketPlanError(
                    f"physical relation packet {packet_id_by_key[key]} is incomplete"
                )

        mutable_stages: list[tuple[
            PrimitiveKind, tuple[int, ...], tuple[int, ...], int, int, int | None
        ]] = []
        for key in ordered_packet_keys:
            members = sorted(packet_members[key])
            mutable_stages.append((
                PrimitiveKind.RELATION,
                tuple(item[1] for item in members),
                tuple(item[0] for item in members),
                packet_contract[key][-1], key[1], packet_id_by_key[key],
            ))

        for kind in sorted(
            RELATION_PACKET_STAGE_KINDS - {PrimitiveKind.RELATION},
            key=int,
        ):
            stage_members: dict[int, list[tuple[int, int, int]]] = {}
            scalar_members: list[tuple[int, int, int]] = []
            for event_id_value in np.flatnonzero(kinds == int(kind)):
                event_id = int(event_id_value)
                row = events[event_id]
                relation_id = int(row["relation_id"])
                packet_id = relation_packet_by_relation.get(relation_id)
                if packet_id is None:
                    if strict:
                        raise RelationPacketPlanError(
                            f"{kind.name} {event_id} has no physical relation packet"
                        )
                    lane, lane_mask = _metadata_for_row(
                        row, query_lanes=query_lanes, strict=False,
                    )
                    scalar_members.append((lane, lane_mask, event_id))
                    continue
                lane, lane_mask = _metadata_for_row(
                    row, query_lanes=query_lanes, strict=strict,
                )
                key = ordered_packet_keys[packet_id]
                expected_mask = packet_contract[key][-1]
                expected_lane = next(
                    (member_lane for member_lane, _member_event, member_relation
                     in packet_members[key] if member_relation == relation_id),
                    None,
                )
                if lane != expected_lane:
                    raise RelationPacketPlanError(
                        f"{kind.name} {event_id} changed physical packet metadata"
                    )
                if lane_mask != expected_mask and not (
                    allow_partial_backward
                    and kind in {
                        PrimitiveKind.ADJOINT,
                        PrimitiveKind.GRADIENT_REDUCTION,
                    }
                    and lane_mask & expected_mask == lane_mask
                ):
                    raise RelationPacketPlanError(
                        f"{kind.name} {event_id} changed physical packet metadata"
                    )
                stage_members.setdefault(packet_id, []).append(
                    (lane, lane_mask, event_id)
                )
            for packet_id in sorted(
                stage_members,
                key=lambda value: min(item[2] for item in stage_members[value]),
            ):
                members = sorted(stage_members[packet_id])
                if len({lane for lane, _mask, _event_id in members}) != len(members):
                    raise RelationPacketPlanError(
                        f"{kind.name} packet {packet_id} repeats a lane"
                    )
                member_masks = {mask for _lane, mask, _event_id in members}
                if len(member_masks) != 1:
                    raise RelationPacketPlanError(
                        f"{kind.name} packet {packet_id} disagrees on its lane mask"
                    )
                stage_mask = next(iter(member_masks))
                key = ordered_packet_keys[packet_id]
                if strict and {
                    lane for lane, _mask, _event_id in members
                } != _active_lanes(stage_mask):
                    raise RelationPacketPlanError(
                        f"{kind.name} packet {packet_id} is incomplete"
                    )
                mutable_stages.append((
                    kind,
                    tuple(event_id for _lane, _mask, event_id in members),
                    tuple(lane for lane, _mask, _event_id in members),
                    stage_mask, key[1], packet_id,
                ))
            for lane, lane_mask, event_id in scalar_members:
                mutable_stages.append((
                    kind, (event_id,), (lane,), lane_mask,
                    int(events[event_id]["query_id"]) - lane, None,
                ))

        close_members: dict[
            tuple[int, int, int, int, int], list[tuple[int, int]]
        ] = {}
        for event_id_value in np.flatnonzero(kinds == int(PrimitiveKind.QUERY_CLOSE)):
            event_id = int(event_id_value)
            row = events[event_id]
            lane, lane_mask = _metadata_for_row(
                row, query_lanes=query_lanes, strict=strict,
            )
            query_base = int(row["query_id"]) - lane
            key = (
                int(row["iteration_id"]), int(row["template_id"]),
                int(row["state_version"]), query_base, lane_mask,
            )
            close_members.setdefault(key, []).append((lane, event_id))
        for key in sorted(
            close_members, key=lambda value: min(item[1] for item in close_members[value])
        ):
            members = sorted(close_members[key])
            lanes = {lane for lane, _event_id in members}
            if len(lanes) != len(members):
                raise RelationPacketPlanError("QUERY_CLOSE packet repeats a lane")
            if strict and lanes != _active_lanes(key[-1]):
                raise RelationPacketPlanError("QUERY_CLOSE packet is incomplete")
            mutable_stages.append((
                PrimitiveKind.QUERY_CLOSE,
                tuple(event_id for _lane, event_id in members),
                tuple(lane for lane, _event_id in members),
                key[-1], key[-2], None,
            ))

        mutable_stages.sort(key=lambda item: min(item[1]))
        stages = tuple(
            PhysicalPacketStage(
                stage_id=index, kind=item[0], event_ids=item[1], lanes=item[2],
                lane_mask=item[3], query_base=item[4], relation_packet_id=item[5],
            )
            for index, item in enumerate(mutable_stages)
        )
        event_to_stage = np.full(trace.event_count, -1, dtype=np.dtype("<i8"))
        for stage in stages:
            for event_id in stage.event_ids:
                if event_to_stage[event_id] >= 0:
                    raise RelationPacketPlanError(
                        f"event {event_id} belongs to multiple physical stages"
                    )
                event_to_stage[event_id] = stage.stage_id
        return cls(
            query_lanes=query_lanes,
            stages=stages,
            event_to_stage=event_to_stage,
            relation_packet_count=len(ordered_packet_keys),
            _relation_packet_by_relation=MappingProxyType(
                relation_packet_by_relation
            ),
        )

    @classmethod
    def from_event_packets(
        cls,
        packets: tuple[VirtualEventPacket, ...],
        *,
        query_lanes: int,
    ) -> "RelationPacketPlan":
        """Build one atomic-source plan from bounded expanded sub-packets."""

        if not packets or not any(packet.event_count for packet in packets):
            raise RelationPacketPlanError("atomic virtual source has no events")
        event_base = packets[0].global_event_start
        rows: list[np.ndarray] = []
        dependencies: list[np.ndarray] = []
        dependency_offset = 0
        expected_event = event_base
        for packet in packets:
            if packet.global_event_start != expected_event:
                raise RelationPacketPlanError(
                    "atomic virtual source event packets are not contiguous"
                )
            part = packet.events.copy()
            part["event_id"] -= event_base
            part["dependency_begin"] += dependency_offset
            raw_dependencies = np.asarray(
                packet.dependencies, dtype=dependency_dtype(),
            )
            # Only RELATION rows inspect dependency kinds while building the
            # plan, and their candidate dependency is local to this source.
            # External lifecycle barriers are retained as a benign local zero
            # placeholder in this temporary planning view.
            rebased = np.where(
                raw_dependencies >= event_base,
                raw_dependencies - event_base,
                np.uint64(0),
            ).astype(dependency_dtype(), copy=False)
            rows.append(part)
            dependencies.append(rebased)
            dependency_offset += rebased.size
            expected_event += packet.event_count
        local = Trace(
            np.concatenate(rows),
            np.concatenate(dependencies).astype(dependency_dtype(), copy=False),
            np.empty(0, dtype=np.dtype("<f4")),
            {"schema_version": EVENT_SCHEMA_VERSION},
        )
        plan = cls.from_trace(local, query_lanes=query_lanes)
        stages = tuple(
            PhysicalPacketStage(
                stage_id=stage.stage_id,
                kind=stage.kind,
                event_ids=tuple(event_id + event_base for event_id in stage.event_ids),
                lanes=stage.lanes,
                lane_mask=stage.lane_mask,
                query_base=stage.query_base,
                relation_packet_id=stage.relation_packet_id,
            )
            for stage in plan.stages
        )
        return cls(
            query_lanes=plan.query_lanes,
            stages=stages,
            event_to_stage=plan.event_to_stage.copy(),
            relation_packet_count=plan.relation_packet_count,
            _relation_packet_by_relation=plan._relation_packet_by_relation,
            event_id_base=event_base,
        )

    def stage_for_event(self, event_id: int) -> PhysicalPacketStage | None:
        local_event_id = event_id - self.event_id_base
        if not 0 <= local_event_id < self.event_to_stage.size:
            raise IndexError("event ID is outside the packet plan")
        stage_id = int(self.event_to_stage[local_event_id])
        return None if stage_id < 0 else self.stages[stage_id]

    def is_stage_head(self, event_id: int) -> bool:
        stage = self.stage_for_event(event_id)
        return stage is None or stage.head_event_id == event_id

    def relation_packet_for(self, relation_id: int) -> int:
        try:
            return self._relation_packet_by_relation[relation_id]
        except KeyError as error:
            raise KeyError(f"unknown relation ID {relation_id}") from error

    def physical_stage_count(self, kind: PrimitiveKind) -> int:
        return sum(stage.kind is kind for stage in self.stages)


def _metadata_for_row(
    row: np.void, *, query_lanes: int, strict: bool,
) -> tuple[int, int]:
    flags = int(row["flags"])
    if has_relation_packet_metadata(flags):
        lane, lane_mask = decode_relation_packet_flags(flags)
        if lane >= query_lanes or lane_mask >= 1 << query_lanes:
            raise RelationPacketPlanError("packet metadata exceeds configured width")
        return lane, lane_mask
    if strict:
        kind = PrimitiveKind(int(row["primitive_kind"]))
        raise RelationPacketPlanError(f"{kind.name} event lacks packet metadata")
    return 0, 1


def _active_lanes(lane_mask: int) -> set[int]:
    return {lane for lane in range(8) if lane_mask & (1 << lane)}


def _allow_partial_backward_stage_masks(trace: Trace) -> bool:
    sample = trace.metadata.get("trace_sample")
    derivation = trace.metadata.get("relation_packet_derivation")
    return bool(
        isinstance(sample, dict)
        and sample.get("result_scope") == "quick_cycle_validation"
        and sample.get("formal_performance_eligible") is False
        and isinstance(derivation, dict)
        and derivation.get("result_scope") == "quick_cycle_validation"
        and derivation.get("formal_performance_eligible") is False
        and derivation.get("allow_partial_backward_stage_masks") is True
    )
