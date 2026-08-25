"""Structural and lifecycle checks for real CLAMP traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from gala_sim.clamp.events import PrimitiveKind

from .model import Trace


class TraceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TraceValidationReport:
    event_count: int
    dependency_count: int
    counts: dict[str, int]
    query_count: int
    gaussian_count: int
    update_count: int


def validate_trace(trace: Trace) -> TraceValidationReport:
    events = trace.events
    ids = [int(value) for value in events["event_id"]]
    if ids != list(range(len(ids))):
        raise TraceValidationError("event_id must be a dense capture-order sequence")
    if len(set(ids)) != len(ids):
        raise TraceValidationError("event_id is not unique")
    known = set(ids)
    counts: dict[str, int] = defaultdict(int)
    query_last_relation: dict[int, int] = {}
    query_closed: set[int] = set()
    gaussian_version: dict[int, int] = {}
    relation_seen: set[int] = set()
    relation_last_event: dict[int, int] = {}
    consumers: set[tuple[int, int]] = set()
    reads_by_key: dict[tuple[int, int], int] = defaultdict(int)
    for row in events:
        event_id = int(row["event_id"])
        try:
            kind = PrimitiveKind(int(row["primitive_kind"]))
        except ValueError as error:
            raise TraceValidationError(f"unknown primitive kind at event {event_id}") from error
        counts[kind.name] += 1
        begin = int(row["dependency_begin"])
        end = begin + int(row["dependency_count"])
        payload_begin = int(row["payload_offset"])
        payload_end = payload_begin + int(row["payload_length"])
        if begin < 0 or end > trace.dependencies.size:
            raise TraceValidationError(f"dependency range is invalid at event {event_id}")
        if payload_begin < 0 or payload_end > trace.payload.size:
            raise TraceValidationError(f"payload range is invalid at event {event_id}")
        dependencies = trace.dependencies[begin:end]
        if any(int(dep) not in known or int(dep) >= event_id for dep in dependencies):
            raise TraceValidationError(f"dependency is not a prior event at event {event_id}")
        query_id = int(row["query_id"])
        relation_id = int(row["relation_id"])
        gaussian_id = int(row["gaussian_id"])
        state_version = int(row["state_version"])
        if gaussian_id >= 0:
            previous = gaussian_version.get(gaussian_id, state_version)
            if state_version < previous:
                raise TraceValidationError(f"state version regressed for Gaussian {gaussian_id}")
            gaussian_version[gaussian_id] = state_version
        if kind is PrimitiveKind.RELATION:
            if relation_id < 0 or query_id < 0:
                raise TraceValidationError(f"relation lacks query/relation id at event {event_id}")
            relation_seen.add(relation_id)
            relation_last_event[relation_id] = event_id
            query_last_relation[query_id] = event_id
        elif kind is PrimitiveKind.QUERY_CLOSE:
            if query_id < 0 or query_id in query_closed:
                raise TraceValidationError(f"invalid query close at event {event_id}")
            if query_id in query_last_relation and query_last_relation[query_id] >= event_id:
                raise TraceValidationError(f"query closes before its last relation at event {event_id}")
            query_closed.add(query_id)
        elif kind is PrimitiveKind.CONSUMER:
            if query_id < 0 or relation_id < 0:
                raise TraceValidationError(f"consumer lacks query/relation id at event {event_id}")
            consumers.add((query_id, relation_id))
        elif kind is PrimitiveKind.ADJOINT:
            if query_id < 0 or relation_id < 0:
                raise TraceValidationError(f"adjoint lacks query/relation id at event {event_id}")
            if (query_id, relation_id) not in consumers:
                raise TraceValidationError(f"adjoint precedes consumer for relation {relation_id}")
        elif kind is PrimitiveKind.CACHE_REQUEST:
            if gaussian_id < 0:
                raise TraceValidationError(f"cache request lacks Gaussian id at event {event_id}")
            reads_by_key[(gaussian_id, state_version)] += 1
        elif kind is PrimitiveKind.CACHE_RETURN:
            key = (gaussian_id, state_version)
            if gaussian_id < 0 or reads_by_key[key] <= 0:
                raise TraceValidationError(f"cache return has no active request at event {event_id}")
            reads_by_key[key] -= 1
        elif kind is PrimitiveKind.SET_MODIFICATION and gaussian_id >= 0:
            if reads_by_key[(gaussian_id, state_version)] != 0:
                raise TraceValidationError(f"state release precedes final read at event {event_id}")
    return TraceValidationReport(
        event_count=len(events),
        dependency_count=int(trace.dependencies.size),
        counts=dict(counts),
        query_count=len({int(value) for value in events["query_id"] if int(value) >= 0}),
        gaussian_count=len({int(value) for value in events["gaussian_id"] if int(value) >= 0}),
        update_count=counts.get(PrimitiveKind.UPDATE_COMMIT.name, 0),
    )
