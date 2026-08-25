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
    kinds: dict[int, PrimitiveKind] = {}
    relation_event_by_id: dict[int, int] = {}
    relation_events_by_query: dict[int, set[int]] = defaultdict(set)
    forward_events_by_query: dict[int, set[int]] = defaultdict(set)
    gradient_events_by_key: dict[tuple[int, int], set[int]] = defaultdict(set)

    # Decode the frozen event kind table first so dependency checks can be
    # expressed in terms of the actual preceding primitive, not just IDs.
    for row in events:
        event_id = int(row["event_id"])
        try:
            kind = PrimitiveKind(int(row["primitive_kind"]))
        except ValueError as error:
            raise TraceValidationError(f"unknown primitive kind at event {event_id}") from error
        kinds[event_id] = kind
        counts[kind.name] += 1

    def dependencies_for(row: object) -> list[int]:
        begin = int(row["dependency_begin"])
        end = begin + int(row["dependency_count"])
        return [int(dep) for dep in trace.dependencies[begin:end]]

    def dependency_kinds_for(row: object) -> list[PrimitiveKind]:
        return [kinds[dependency] for dependency in dependencies_for(row)]

    def require_dependency_kind(row: object, expected: PrimitiveKind) -> None:
        if expected not in dependency_kinds_for(row):
            raise TraceValidationError(
                f"{expected.name} dependency is missing at event {int(row['event_id'])}"
            )

    gaussian_version: dict[int, int] = {}
    consumers: set[tuple[int, int]] = set()
    reads_by_key: dict[tuple[int, int], int] = defaultdict(int)
    query_closed: set[int] = set()
    for row in events:
        event_id = int(row["event_id"])
        kind = kinds[event_id]
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
            if relation_id in relation_event_by_id:
                raise TraceValidationError(f"relation_id is duplicated at event {event_id}")
            relation_event_by_id[relation_id] = event_id
            relation_events_by_query[query_id].add(event_id)
            if counts[PrimitiveKind.RELATION_CANDIDATE.name] > 0:
                if dependencies.size == 0 or any(
                    kinds[dependency] is not PrimitiveKind.RELATION_CANDIDATE
                    for dependency in dependencies
                ):
                    raise TraceValidationError(
                        f"relation does not depend only on candidate seeds at event {event_id}"
                    )
        elif kind is PrimitiveKind.QUERY_CLOSE:
            if query_id < 0 or query_id in query_closed:
                raise TraceValidationError(f"invalid query close at event {event_id}")
            if counts[PrimitiveKind.RELATION_CANDIDATE.name] > 0:
                expected_relations = relation_events_by_query[query_id]
                actual_relations = {
                    dependency for dependency in dependencies
                    if kinds[dependency] is PrimitiveKind.RELATION
                }
                if actual_relations != expected_relations:
                    raise TraceValidationError(
                        f"query close does not depend on every relation at event {event_id}"
                    )
            query_closed.add(query_id)
        elif kind is PrimitiveKind.FORWARD:
            if counts[PrimitiveKind.CACHE_REQUEST.name] > 0:
                require_dependency_kind(row, PrimitiveKind.RELATION)
                require_dependency_kind(row, PrimitiveKind.CACHE_RETURN)
            forward_events_by_query[query_id].add(event_id)
        elif kind is PrimitiveKind.QUERY_REDUCTION:
            expected_forwards = forward_events_by_query[query_id]
            if expected_forwards and set(dependencies) != expected_forwards:
                raise TraceValidationError(
                    f"query reduction does not depend on every forward at event {event_id}"
                )
        elif kind is PrimitiveKind.CONSUMER:
            if query_id < 0 or relation_id < 0:
                raise TraceValidationError(f"consumer lacks query/relation id at event {event_id}")
            consumers.add((query_id, relation_id))
            if counts[PrimitiveKind.QUERY_REDUCTION.name] > 0:
                require_dependency_kind(row, PrimitiveKind.QUERY_REDUCTION)
        elif kind is PrimitiveKind.ADJOINT:
            if query_id < 0 or relation_id < 0:
                raise TraceValidationError(f"adjoint lacks query/relation id at event {event_id}")
            if (query_id, relation_id) not in consumers:
                raise TraceValidationError(f"adjoint precedes consumer for relation {relation_id}")
            if counts[PrimitiveKind.CONSUMER.name] > 0:
                require_dependency_kind(row, PrimitiveKind.CONSUMER)
        elif kind is PrimitiveKind.GRADIENT_REDUCTION:
            if counts[PrimitiveKind.ADJOINT.name] > 0:
                require_dependency_kind(row, PrimitiveKind.ADJOINT)
            gradient_events_by_key[(gaussian_id, state_version)].add(event_id)
        elif kind is PrimitiveKind.CACHE_REQUEST:
            if gaussian_id < 0:
                raise TraceValidationError(f"cache request lacks Gaussian id at event {event_id}")
            if counts[PrimitiveKind.RELATION.name] > 0:
                require_dependency_kind(row, PrimitiveKind.RELATION)
            reads_by_key[(gaussian_id, state_version)] += 1
        elif kind is PrimitiveKind.CACHE_RETURN:
            key = (gaussian_id, state_version)
            if gaussian_id < 0 or reads_by_key[key] <= 0:
                raise TraceValidationError(f"cache return has no active request at event {event_id}")
            require_dependency_kind(row, PrimitiveKind.CACHE_REQUEST)
            reads_by_key[key] -= 1
        elif kind is PrimitiveKind.UPDATE_COMMIT:
            if gaussian_id < 0:
                raise TraceValidationError(f"update commit lacks Gaussian id at event {event_id}")
            expected_gradients = gradient_events_by_key[(gaussian_id, state_version)]
            if expected_gradients and set(dependencies) != expected_gradients:
                raise TraceValidationError(
                    f"update commit does not wait for every Gaussian gradient at event {event_id}"
                )
        elif kind is PrimitiveKind.SET_MODIFICATION and gaussian_id >= 0:
            if reads_by_key[(gaussian_id, state_version)] != 0:
                raise TraceValidationError(f"state release precedes final read at event {event_id}")
    if counts[PrimitiveKind.CACHE_RETURN.name] > 0 and any(
        value != 0 for value in reads_by_key.values()
    ):
        raise TraceValidationError("trace ends with an outstanding cache request")
    _validate_capture_audit(trace, counts)
    return TraceValidationReport(
        event_count=len(events),
        dependency_count=int(trace.dependencies.size),
        counts=dict(counts),
        query_count=len({int(value) for value in events["query_id"] if int(value) >= 0}),
        gaussian_count=len({int(value) for value in events["gaussian_id"] if int(value) >= 0}),
        update_count=counts.get(PrimitiveKind.UPDATE_COMMIT.name, 0),
    )


def _validate_capture_audit(trace: Trace, counts: dict[str, int]) -> None:
    audit = trace.metadata.get("capture_audit")
    if audit is None:
        return
    if not isinstance(audit, dict) or any(
        not isinstance(key, str) or not isinstance(value, int) or value < 0
        for key, value in audit.items()
    ):
        raise TraceValidationError("capture audit metadata is malformed")

    event_matches = {
        "cuda_relation_candidates": PrimitiveKind.RELATION_CANDIDATE,
        "cuda_valid_relations": PrimitiveKind.RELATION,
        "captured_query_calls": PrimitiveKind.QUERY_CLOSE,
        "captured_backward_relations": PrimitiveKind.ADJOINT,
        "optimizer_updated_gaussians": PrimitiveKind.UPDATE_COMMIT,
    }
    for audit_name, primitive_kind in event_matches.items():
        expected = int(audit.get(audit_name, 0))
        actual = int(counts.get(primitive_kind.name, 0))
        if expected != actual:
            raise TraceValidationError(
                f"capture audit {audit_name}={expected} does not match "
                f"{primitive_kind.name} events={actual}"
            )

    relation_count = int(counts.get(PrimitiveKind.RELATION.name, 0))
    for primitive_kind in (
        PrimitiveKind.CACHE_REQUEST,
        PrimitiveKind.CACHE_RETURN,
        PrimitiveKind.FORWARD,
        PrimitiveKind.CONSUMER,
        PrimitiveKind.ADJOINT,
        PrimitiveKind.GRADIENT_REDUCTION,
    ):
        actual = int(counts.get(primitive_kind.name, 0))
        if actual != relation_count:
            raise TraceValidationError(
                f"captured {primitive_kind.name} events={actual} do not match "
                f"valid relations={relation_count}"
            )

    captured_queries = int(audit.get("captured_query_calls", 0))
    captured_by_kind = sum(
        int(audit.get(name, 0))
        for name in ("captured_raster_query_calls", "captured_voxel_query_calls")
    )
    if captured_by_kind != captured_queries:
        raise TraceValidationError("captured query audit totals are inconsistent")
    if int(counts.get(PrimitiveKind.QUERY_REDUCTION.name, 0)) != captured_queries:
        raise TraceValidationError("query reduction count does not match captured queries")

    for query_kind in ("raster", "voxel"):
        official = int(audit.get(f"official_{query_kind}_query_calls", 0))
        captured = int(audit.get(f"captured_{query_kind}_query_calls", 0))
        excluded = int(audit.get(f"excluded_no_grad_{query_kind}_query_calls", 0))
        if official != captured + excluded:
            raise TraceValidationError(f"official {query_kind} query audit totals are inconsistent")
