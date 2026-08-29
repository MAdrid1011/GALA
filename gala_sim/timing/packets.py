"""Physical RelationPacket plan shared by offline and online cycle replay."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Any

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
class RelationWindowDescriptor:
    """One hardware relation-window scope and its exact reference counts."""

    window_id: int
    relation_stage_heads: frozenset[int]
    producer_event_ids: frozenset[int]
    forward_stage_heads: frozenset[int]
    consumer_event_ids: frozenset[int]
    adjoint_event_ids: frozenset[int]
    # A relation record is reusable once every logical lane in its paired
    # adjoint stage has retired.  Empty for hand-authored descriptors that
    # use the conservative whole-window lifetime.
    relation_record_release_events: Mapping[int, frozenset[int]] = MappingProxyType({})
    sealed: bool = True

    @property
    def event_ids(self) -> tuple[int, ...]:
        return tuple(sorted(
            self.producer_event_ids
            | self.forward_stage_heads
            | self.consumer_event_ids
            | self.adjoint_event_ids
        ))

    @property
    def producer_references(self) -> int:
        return len(self.producer_event_ids)

    @property
    def forward_references(self) -> int:
        return len(self.forward_stage_heads)

    @property
    def consumer_references(self) -> int:
        return len(self.consumer_event_ids)

    @property
    def adjoint_references(self) -> int:
        return len(self.adjoint_event_ids)

    @property
    def relation_records(self) -> int:
        return len(self.relation_stage_heads)


@dataclass(frozen=True)
class RelationWindowPlan:
    """Policy-independent event-to-window identity from capture metadata."""

    descriptors: tuple[RelationWindowDescriptor, ...]
    event_to_window: np.ndarray

    def __post_init__(self) -> None:
        if self.event_to_window.dtype != np.dtype("<i8"):
            raise ValueError("event-to-window map must use little-endian int64")
        self.event_to_window.setflags(write=False)

    @classmethod
    def from_trace(
        cls, trace: Trace, packet_plan: "RelationPacketPlan",
    ) -> "RelationWindowPlan | None":
        """Build quick-sample windows or require an explicit formal sidecar.

        Formal traces may not infer hardware windows from event ordering.  The
        current captured-packet quick format explicitly declares complete CUDA
        source packets, each of which is small enough to represent one window.
        """

        explicit = trace.metadata.get("relation_windows")
        if explicit is not None:
            return cls._from_explicit(trace, packet_plan, explicit)
        sample = trace.metadata.get("trace_sample")
        if isinstance(sample, Mapping) and sample.get("result_scope") == "quick_cycle_validation":
            packets = sample.get("packets")
            if isinstance(packets, list) and packets:
                return cls._from_quick_packets(trace, packet_plan, packets)
        if trace.metadata.get("formal_performance_eligible") is True:
            raise RelationPacketPlanError(
                "formal cycle trace lacks an explicit relation-window sidecar"
            )
        return None

    @classmethod
    def from_event_packets(
        cls,
        packets: tuple[VirtualEventPacket, ...],
        packet_plan: "RelationPacketPlan",
        *,
        window_id: int,
    ) -> "RelationWindowPlan":
        event_ids = tuple(
            int(event_id) for packet in packets for event_id in packet.events["event_id"]
        )
        if not event_ids:
            raise RelationPacketPlanError("relation window has no events")
        event_base = min(event_ids)
        event_end = max(event_ids) + 1
        rows = np.concatenate([packet.events for packet in packets])
        descriptor = cls._descriptor(
            window_id, rows, packet_plan,
            event_ids=event_ids,
        )
        event_to_window = np.full(event_end - event_base, -1, dtype=np.dtype("<i8"))
        event_to_window[np.asarray(event_ids, dtype=np.int64) - event_base] = window_id
        # Online callers retain the absolute descriptor IDs; the local lookup
        # array is not used outside construction.
        return cls((descriptor,), event_to_window)

    @classmethod
    def _from_quick_packets(
        cls,
        trace: Trace,
        packet_plan: "RelationPacketPlan",
        raw_packets: list[Any],
    ) -> "RelationWindowPlan":
        specs: list[tuple[int, int, int]] = []
        for index, raw in enumerate(raw_packets):
            if not isinstance(raw, Mapping):
                raise RelationPacketPlanError("quick packet metadata is malformed")
            try:
                begin = int(raw["query_base"])
                count = int(raw["query_count"])
                candidates = int(raw["candidate_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise RelationPacketPlanError(
                    "quick packet lacks query and candidate bounds"
                ) from error
            if min(begin, count, candidates) < 0 or count == 0:
                raise RelationPacketPlanError("quick packet bounds are invalid")
            specs.append((begin, begin + count, candidates))
        for left, right, _count in specs:
            if sum(other_left < right and left < other_right
                   for other_left, other_right, _ in specs) != 1:
                raise RelationPacketPlanError("quick packet query ranges overlap")

        mapping = np.full(trace.event_count, -1, dtype=np.dtype("<i8"))
        query_ids = trace.events["query_id"]
        for window_id, (begin, end, _candidates) in enumerate(specs):
            matched = (query_ids >= begin) & (query_ids < end)
            if np.any(mapping[matched] >= 0):
                raise RelationPacketPlanError("event belongs to multiple quick windows")
            mapping[matched] = window_id

        candidate_rows = np.flatnonzero(
            trace.events["primitive_kind"] == int(PrimitiveKind.RELATION_CANDIDATE)
        )
        cursor = 0
        for window_id, (_begin, _end, count) in enumerate(specs):
            rows = candidate_rows[cursor:cursor + count]
            if rows.size != count:
                raise RelationPacketPlanError("quick packet candidate counts exceed trace")
            mapping[rows] = window_id
            cursor += count
        if cursor != candidate_rows.size:
            raise RelationPacketPlanError("quick packet candidate counts do not cover trace")

        window_kinds = {
            PrimitiveKind.RELATION_CANDIDATE, PrimitiveKind.RELATION,
            PrimitiveKind.QUERY_CLOSE, PrimitiveKind.FORWARD,
            PrimitiveKind.QUERY_REDUCTION, PrimitiveKind.CONSUMER,
            PrimitiveKind.ADJOINT, PrimitiveKind.GRADIENT_REDUCTION,
            PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN,
        }
        for event_id, row in enumerate(trace.events):
            kind = PrimitiveKind(int(row["primitive_kind"]))
            if kind in window_kinds and mapping[event_id] < 0:
                raise RelationPacketPlanError(
                    f"quick relation-window metadata does not cover event {event_id}"
                )
            if kind is PrimitiveKind.RELATION:
                window_id = int(mapping[event_id])
                for dependency in trace.dependency_ids(row):
                    dependency_id = int(dependency)
                    if PrimitiveKind(int(trace.events[dependency_id]["primitive_kind"])) is PrimitiveKind.RELATION_CANDIDATE and int(mapping[dependency_id]) != window_id:
                        raise RelationPacketPlanError(
                            "relation and candidate cross quick-window boundaries"
                        )

        descriptors = tuple(
            cls._descriptor(
                window_id,
                trace.events[mapping == window_id],
                packet_plan,
                event_ids=tuple(np.flatnonzero(mapping == window_id).tolist()),
            )
            for window_id in range(len(specs))
        )
        return cls(descriptors, mapping)

    @classmethod
    def _from_explicit(
        cls, trace: Trace, packet_plan: "RelationPacketPlan", raw: Any,
    ) -> "RelationWindowPlan":
        if not isinstance(raw, list) or not raw:
            raise RelationPacketPlanError("relation-window sidecar must be a nonempty list")
        mapping = np.full(trace.event_count, -1, dtype=np.dtype("<i8"))
        descriptors: list[RelationWindowDescriptor] = []
        for expected_id, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise RelationPacketPlanError("relation-window sidecar entry is malformed")
            try:
                window_id = int(item["window_id"])
                begin = int(item["event_begin"])
                end = int(item["event_end"])
            except (KeyError, TypeError, ValueError) as error:
                raise RelationPacketPlanError("relation-window sidecar lacks event bounds") from error
            if window_id != expected_id or not 0 <= begin < end <= trace.event_count:
                raise RelationPacketPlanError("relation-window sidecar IDs or bounds are invalid")
            if np.any(mapping[begin:end] >= 0):
                raise RelationPacketPlanError("relation-window sidecar event bounds overlap")
            mapping[begin:end] = window_id
            event_ids = tuple(range(begin, end))
            descriptors.append(cls._descriptor(
                window_id, trace.events[begin:end], packet_plan,
                event_ids=event_ids,
            ))
        return cls(tuple(descriptors), mapping)

    @staticmethod
    def _descriptor(
        window_id: int,
        rows: np.ndarray,
        packet_plan: "RelationPacketPlan",
        *,
        event_ids: tuple[int, ...],
    ) -> RelationWindowDescriptor:
        kinds = {
            event_id: PrimitiveKind(int(row["primitive_kind"]))
            for event_id, row in zip(event_ids, rows, strict=True)
        }
        relation_heads = frozenset(
            event_id for event_id, kind in kinds.items()
            if kind is PrimitiveKind.RELATION
            and packet_plan.is_stage_head(event_id)
        )
        adjoint_by_packet = {
            stage.relation_packet_id: stage
            for stage in packet_plan.stages
            if stage.kind is PrimitiveKind.ADJOINT
            and stage.relation_packet_id is not None
        }
        relation_record_release_events = {
            relation_stage.head_event_id: frozenset(
                adjoint_by_packet[relation_stage.relation_packet_id].event_ids
            )
            for relation_stage in packet_plan.stages
            if relation_stage.kind is PrimitiveKind.RELATION
            and relation_stage.head_event_id in relation_heads
            and relation_stage.relation_packet_id in adjoint_by_packet
        }
        return RelationWindowDescriptor(
            window_id=window_id,
            relation_stage_heads=relation_heads,
            producer_event_ids=frozenset(
                event_id for event_id, kind in kinds.items()
                if kind in {
                    PrimitiveKind.RELATION_CANDIDATE,
                    PrimitiveKind.RELATION,
                    PrimitiveKind.QUERY_CLOSE,
                }
            ),
            forward_stage_heads=frozenset(
                event_id for event_id, kind in kinds.items()
                if kind is PrimitiveKind.FORWARD
                and (
                    (stage := packet_plan.stage_for_event(event_id)) is None
                    or event_id == max(stage.event_ids)
                )
            ),
            consumer_event_ids=frozenset(
                event_id for event_id, kind in kinds.items()
                if kind is PrimitiveKind.CONSUMER
            ),
            adjoint_event_ids=frozenset(
                event_id for event_id, kind in kinds.items()
                if kind is PrimitiveKind.ADJOINT
            ),
            relation_record_release_events=MappingProxyType(
                relation_record_release_events
            ),
        )

    def window_for_event(self, event_id: int) -> int | None:
        if not 0 <= event_id < self.event_to_window.size:
            return None
        value = int(self.event_to_window[event_id])
        return None if value < 0 else value


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
