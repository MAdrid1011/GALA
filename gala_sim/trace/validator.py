"""Structural and lifecycle checks for real CLAMP traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from gala_sim.clamp.events import ModificationKind, PrimitiveKind, UpdateBeginKind

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
    counts: dict[str, int] = defaultdict(int)
    relation_event_by_id: dict[int, int] = {}
    relation_events_by_query: dict[int, set[int]] = defaultdict(set)
    forward_events_by_query: dict[int, set[int]] = defaultdict(set)
    gradient_events_by_key: dict[tuple[int, int], set[int]] = defaultdict(set)
    initial_gaussian_count = trace.metadata.get("initial_gaussian_count")
    if initial_gaussian_count is not None and (
        not isinstance(initial_gaussian_count, int) or initial_gaussian_count < 0
    ):
        raise TraceValidationError("initial Gaussian count metadata is invalid")
    active_gaussians = (
        set(range(initial_gaussian_count)) if initial_gaussian_count is not None else None
    )
    known_gaussians = set(active_gaussians or ())

    # Decode the frozen event kind table first so dependency checks can be
    # expressed in terms of the actual preceding primitive, not just IDs.
    for expected_event_id, row in enumerate(events):
        event_id = int(row["event_id"])
        if event_id != expected_event_id:
            raise TraceValidationError("event_id must be a dense capture-order sequence")
        try:
            kind = PrimitiveKind(int(row["primitive_kind"]))
        except ValueError as error:
            raise TraceValidationError(f"unknown primitive kind at event {event_id}") from error
        counts[kind.name] += 1

    def kind_for_event(event_id: int) -> PrimitiveKind:
        return PrimitiveKind(int(events[event_id]["primitive_kind"]))

    def dependencies_for(row: object) -> list[int]:
        begin = int(row["dependency_begin"])
        end = begin + int(row["dependency_count"])
        return [int(dep) for dep in trace.dependencies[begin:end]]

    def dependency_kinds_for(row: object) -> list[PrimitiveKind]:
        return [kind_for_event(dependency) for dependency in dependencies_for(row)]

    def require_dependency_kind(row: object, expected: PrimitiveKind) -> None:
        if expected not in dependency_kinds_for(row):
            raise TraceValidationError(
                f"{expected.name} dependency is missing at event {int(row['event_id'])}"
            )

    gaussian_version: dict[int, int] = {}
    consumers: dict[int, int] = {}
    reads_by_key: dict[tuple[int, int], int] = defaultdict(int)
    query_closed: set[int] = set()
    current_update_begin: tuple[int, int, UpdateBeginKind] | None = None
    current_update_members: set[int] = set()
    latest_state_ready: tuple[int, int] | None = None
    closed_state_versions: set[int] = set()
    clone_parent_events: dict[int, int] = {}
    split_child_events: dict[int, set[int]] = defaultdict(set)
    for row in events:
        event_id = int(row["event_id"])
        kind = kind_for_event(event_id)
        begin = int(row["dependency_begin"])
        end = begin + int(row["dependency_count"])
        payload_begin = int(row["payload_offset"])
        payload_end = payload_begin + int(row["payload_length"])
        if begin < 0 or end > trace.dependencies.size:
            raise TraceValidationError(f"dependency range is invalid at event {event_id}")
        if payload_begin < 0 or payload_end > trace.payload.size:
            raise TraceValidationError(f"payload range is invalid at event {event_id}")
        dependencies = trace.dependencies[begin:end]
        if dependencies.size and bool((dependencies >= event_id).any()):
            raise TraceValidationError(f"dependency is not a prior event at event {event_id}")
        query_id = int(row["query_id"])
        relation_id = int(row["relation_id"])
        gaussian_id = int(row["gaussian_id"])
        state_version = int(row["state_version"])
        if state_version in closed_state_versions:
            raise TraceValidationError(
                f"closed state version {state_version} is used at event {event_id}"
            )
        if (
            active_gaussians is not None
            and gaussian_id >= 0
            and kind is not PrimitiveKind.SET_MODIFICATION
            and gaussian_id not in active_gaussians
        ):
            raise TraceValidationError(
                f"inactive Gaussian {gaussian_id} is used at event {event_id}"
            )
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
                    kind_for_event(int(dependency)) is not PrimitiveKind.RELATION_CANDIDATE
                    for dependency in dependencies
                ):
                    raise TraceValidationError(
                        f"relation does not depend only on candidate seeds at event {event_id}"
                    )
        elif kind is PrimitiveKind.RELATION_CANDIDATE:
            if (
                latest_state_ready is not None
                and state_version == latest_state_ready[1]
                and latest_state_ready[0] not in dependencies
            ):
                raise TraceValidationError(
                    f"relation candidate lacks prior update end at event {event_id}"
                )
        elif kind is PrimitiveKind.QUERY_CLOSE:
            if query_id < 0 or query_id in query_closed:
                raise TraceValidationError(f"invalid query close at event {event_id}")
            if counts[PrimitiveKind.RELATION_CANDIDATE.name] > 0:
                expected_relations = relation_events_by_query[query_id]
                actual_relations = {
                    dependency for dependency in dependencies
                    if kind_for_event(int(dependency)) is PrimitiveKind.RELATION
                }
                if actual_relations != expected_relations:
                    raise TraceValidationError(
                        f"query close does not depend on every relation at event {event_id}"
                    )
            if (
                latest_state_ready is not None
                and state_version == latest_state_ready[1]
                and latest_state_ready[0] not in dependencies
            ):
                raise TraceValidationError(
                    f"query close lacks prior update end at event {event_id}"
                )
            query_closed.add(query_id)
        elif kind is PrimitiveKind.FORWARD:
            if counts[PrimitiveKind.CACHE_REQUEST.name] > 0:
                require_dependency_kind(row, PrimitiveKind.RELATION)
                require_dependency_kind(row, PrimitiveKind.CACHE_RETURN)
            forward_events_by_query[query_id].add(event_id)
        elif kind is PrimitiveKind.QUERY_REDUCTION:
            expected_forwards = forward_events_by_query[query_id]
            actual_forwards = {
                int(dependency) for dependency in dependencies
                if kind_for_event(int(dependency)) is PrimitiveKind.FORWARD
            }
            if actual_forwards != expected_forwards:
                raise TraceValidationError(
                    f"query reduction does not depend on every forward at event {event_id}"
                )
            require_dependency_kind(row, PrimitiveKind.QUERY_CLOSE)
        elif kind is PrimitiveKind.CONSUMER:
            if query_id < 0 or int(row["consumer_id"]) < 0:
                raise TraceValidationError(f"consumer lacks query/consumer id at event {event_id}")
            if query_id in consumers:
                raise TraceValidationError(f"query has multiple consumers at event {event_id}")
            consumers[query_id] = event_id
            if counts[PrimitiveKind.QUERY_REDUCTION.name] > 0:
                require_dependency_kind(row, PrimitiveKind.QUERY_REDUCTION)
        elif kind is PrimitiveKind.ADJOINT:
            if query_id < 0 or relation_id < 0:
                raise TraceValidationError(f"adjoint lacks query/relation id at event {event_id}")
            consumer = consumers.get(query_id)
            if consumer is None:
                raise TraceValidationError(f"adjoint precedes consumer for relation {relation_id}")
            if counts[PrimitiveKind.CONSUMER.name] > 0:
                if consumer not in dependencies:
                    raise TraceValidationError(
                        f"adjoint does not depend on its query consumer for relation {relation_id}"
                    )
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
        elif kind is PrimitiveKind.UPDATE_BEGIN:
            try:
                begin_kind = UpdateBeginKind(int(row["flags"]))
            except ValueError as error:
                raise TraceValidationError(
                    f"update begin has invalid kind at event {event_id}"
                ) from error
            if current_update_begin is not None:
                raise TraceValidationError(f"nested update transaction at event {event_id}")
            if begin_kind is UpdateBeginKind.COLLECTION:
                clone_parent_events.clear()
                split_child_events.clear()
            current_update_begin = (event_id, state_version, begin_kind)
            current_update_members = set()
        elif kind is PrimitiveKind.UPDATE_COMMIT:
            if gaussian_id < 0:
                raise TraceValidationError(f"update commit lacks Gaussian id at event {event_id}")
            if (
                current_update_begin is None
                or current_update_begin[1] != state_version
                or current_update_begin[2] is not UpdateBeginKind.OPTIMIZER
                or current_update_begin[0] not in dependencies
            ):
                raise TraceValidationError(
                    f"update commit lacks its optimizer begin at event {event_id}"
                )
            expected_gradients = gradient_events_by_key[(gaussian_id, state_version)]
            actual_gradients = {
                int(dependency) for dependency in dependencies
                if kind_for_event(int(dependency)) is PrimitiveKind.GRADIENT_REDUCTION
            }
            if actual_gradients != expected_gradients:
                raise TraceValidationError(
                    f"update commit does not wait for every Gaussian gradient at event {event_id}"
                )
            current_update_members.add(event_id)
        elif kind is PrimitiveKind.SET_MODIFICATION:
            if gaussian_id < 0:
                raise TraceValidationError(
                    f"set modification lacks Gaussian id at event {event_id}"
                )
            if (
                current_update_begin is None
                or current_update_begin[1] != state_version
                or current_update_begin[2] is not UpdateBeginKind.COLLECTION
                or current_update_begin[0] not in dependencies
            ):
                raise TraceValidationError(
                    f"set modification lacks its collection begin at event {event_id}"
                )
            if reads_by_key[(gaussian_id, state_version)] != 0:
                raise TraceValidationError(f"state release precedes final read at event {event_id}")
            try:
                modification = ModificationKind(int(row["flags"]))
            except ValueError as error:
                raise TraceValidationError(
                    f"set modification has invalid kind at event {event_id}"
                ) from error
            parent_id = int(row["reduction_key"])
            if active_gaussians is not None:
                if modification in {
                    ModificationKind.PRUNE,
                    ModificationKind.CLONE_PARENT,
                    ModificationKind.SPLIT_PARENT,
                } and gaussian_id not in active_gaussians:
                    raise TraceValidationError(
                        f"modification uses inactive Gaussian at event {event_id}"
                    )
                if modification in {
                    ModificationKind.CLONE_CHILD,
                    ModificationKind.SPLIT_CHILD,
                }:
                    if gaussian_id in known_gaussians or parent_id not in active_gaussians:
                        raise TraceValidationError(
                            f"modification has invalid child lineage at event {event_id}"
                        )
                    active_gaussians.add(gaussian_id)
                    known_gaussians.add(gaussian_id)
                elif modification in {
                    ModificationKind.PRUNE,
                    ModificationKind.SPLIT_PARENT,
                }:
                    active_gaussians.remove(gaussian_id)
            if modification is ModificationKind.CLONE_PARENT:
                if parent_id != gaussian_id:
                    raise TraceValidationError(
                        f"clone parent lineage is invalid at event {event_id}"
                    )
                clone_parent_events[gaussian_id] = event_id
            elif modification is ModificationKind.CLONE_CHILD:
                parent_event = clone_parent_events.get(parent_id)
                if parent_event is None or parent_event not in dependencies:
                    raise TraceValidationError(
                        f"clone child lacks its parent mutation at event {event_id}"
                    )
            elif modification is ModificationKind.SPLIT_CHILD:
                split_child_events[parent_id].add(event_id)
            elif modification is ModificationKind.SPLIT_PARENT:
                if parent_id != gaussian_id or not split_child_events[parent_id]:
                    raise TraceValidationError(
                        f"split parent has no child lineage at event {event_id}"
                    )
                if not split_child_events[parent_id].issubset(set(dependencies)):
                    raise TraceValidationError(
                        f"split parent does not wait for every child at event {event_id}"
                    )
            elif modification is ModificationKind.PRUNE and parent_id != gaussian_id:
                raise TraceValidationError(
                    f"prune lineage is invalid at event {event_id}"
                )
            current_update_members.add(event_id)
        elif kind is PrimitiveKind.UPDATE_END:
            try:
                end_kind = UpdateBeginKind(int(row["flags"]))
            except ValueError as error:
                raise TraceValidationError(
                    f"update end has invalid kind at event {event_id}"
                ) from error
            if (
                current_update_begin is None
                or current_update_begin[1] != state_version
                or current_update_begin[2] is not end_kind
                or int(row["reduction_key"]) != current_update_begin[0]
            ):
                raise TraceValidationError(
                    f"update end does not match its begin at event {event_id}"
                )
            expected_dependencies = (
                current_update_members or {current_update_begin[0]}
            )
            if set(int(value) for value in dependencies) != expected_dependencies:
                raise TraceValidationError(
                    f"update end does not wait for its transaction at event {event_id}"
                )
            end_field_mask = int(row["field_mask"])
            advances_version = end_field_mask != 0
            if (
                (end_kind is UpdateBeginKind.COLLECTION or bool(current_update_members))
                != advances_version
            ):
                raise TraceValidationError(
                    f"update end field mask does not match transaction writes at event {event_id}"
                )
            latest_state_ready = (event_id, state_version + int(advances_version))
            if advances_version:
                closed_state_versions.add(state_version)
            current_update_begin = None
            current_update_members = set()
    if counts[PrimitiveKind.CACHE_RETURN.name] > 0 and any(
        value != 0 for value in reads_by_key.values()
    ):
        raise TraceValidationError("trace ends with an outstanding cache request")
    if current_update_begin is not None:
        raise TraceValidationError("trace ends with an open update transaction")
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
    if trace.metadata.get("capture_audit_schema_version") != "gala-r2-capture-audit-v3":
        raise TraceValidationError("captured trace uses an obsolete capture audit schema")

    event_matches = {
        "cuda_relation_candidates": PrimitiveKind.RELATION_CANDIDATE,
        "cuda_valid_relations": PrimitiveKind.RELATION,
        "captured_logical_queries": PrimitiveKind.QUERY_CLOSE,
        "captured_consumers": PrimitiveKind.CONSUMER,
        "captured_backward_relations": PrimitiveKind.ADJOINT,
        "optimizer_updated_gaussians": PrimitiveKind.UPDATE_COMMIT,
        "update_begin_events": PrimitiveKind.UPDATE_BEGIN,
        "update_end_events": PrimitiveKind.UPDATE_END,
        "collection_modification_events": PrimitiveKind.SET_MODIFICATION,
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
        PrimitiveKind.ADJOINT,
        PrimitiveKind.GRADIENT_REDUCTION,
    ):
        actual = int(counts.get(primitive_kind.name, 0))
        if actual != relation_count:
            raise TraceValidationError(
                f"captured {primitive_kind.name} events={actual} do not match "
                f"valid relations={relation_count}"
            )

    captured_kernels = int(audit.get("captured_query_kernel_calls", 0))
    captured_kernels_by_kind = sum(
        int(audit.get(name, 0))
        for name in ("captured_raster_kernel_calls", "captured_voxel_kernel_calls")
    )
    if captured_kernels_by_kind != captured_kernels:
        raise TraceValidationError("captured kernel audit totals are inconsistent")
    captured_queries = int(audit.get("captured_logical_queries", 0))
    for primitive_kind in (PrimitiveKind.QUERY_REDUCTION, PrimitiveKind.CONSUMER):
        if int(counts.get(primitive_kind.name, 0)) != captured_queries:
            raise TraceValidationError(
                f"{primitive_kind.name} count does not match captured logical queries"
            )
    if int(audit.get("captured_backward_calls", 0)) != captured_kernels:
        raise TraceValidationError("captured forward/backward kernel totals are inconsistent")

    begin_counts: dict[UpdateBeginKind, int] = defaultdict(int)
    end_counts: dict[UpdateBeginKind, int] = defaultdict(int)
    for row in trace.events:
        primitive = PrimitiveKind(int(row["primitive_kind"]))
        if primitive is PrimitiveKind.UPDATE_BEGIN:
            begin_counts[UpdateBeginKind(int(row["flags"]))] += 1
        elif primitive is PrimitiveKind.UPDATE_END:
            end_counts[UpdateBeginKind(int(row["flags"]))] += 1
    if begin_counts != end_counts:
        raise TraceValidationError("update begin/end audit totals are inconsistent")
    optimizer_steps = int(audit.get("optimizer_steps", 0))
    optimizer_noops = int(audit.get("optimizer_noop_steps", 0))
    if optimizer_steps != begin_counts[UpdateBeginKind.OPTIMIZER]:
        raise TraceValidationError("optimizer step audit totals are inconsistent")
    if optimizer_noops > optimizer_steps:
        raise TraceValidationError("optimizer no-op audit total is inconsistent")
    if int(audit.get("collection_modification_transactions", 0)) != begin_counts[
        UpdateBeginKind.COLLECTION
    ]:
        raise TraceValidationError("collection transaction audit totals are inconsistent")
    modification_counts: dict[ModificationKind, int] = defaultdict(int)
    for row in trace.events:
        if PrimitiveKind(int(row["primitive_kind"])) is PrimitiveKind.SET_MODIFICATION:
            modification_counts[ModificationKind(int(row["flags"]))] += 1
    for modification in ModificationKind:
        expected = int(audit.get(
            f"collection_{modification.name.lower()}_events", 0
        ))
        if expected != modification_counts[modification]:
            raise TraceValidationError(
                f"collection {modification.name.lower()} audit total is inconsistent"
            )

    for query_kind in ("raster", "voxel"):
        official = int(audit.get(f"official_{query_kind}_kernel_calls", 0))
        captured = int(audit.get(f"captured_{query_kind}_kernel_calls", 0))
        excluded = int(audit.get(f"excluded_no_grad_{query_kind}_kernel_calls", 0))
        if official != captured + excluded:
            raise TraceValidationError(f"official {query_kind} kernel audit totals are inconsistent")
