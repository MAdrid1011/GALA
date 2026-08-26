"""Structural and lifecycle checks for real CLAMP traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import tempfile
import mmap

import numpy as np

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
    # Large raw-column captures are mmap-backed.  Keeping a Python set for
    # every relation or forward event defeats the bounded trace writer, so
    # validate them with chunked NumPy passes that retain the same dependency
    # and audit invariants without materializing event IDs as Python objects.
    if (
        trace.metadata.get("trace_storage_format") == "raw_columns"
        and trace.metadata.get("capture_audit_schema_version") == "gala-r2-capture-audit-v4"
    ):
        return _validate_large_capture_trace_streaming(trace)
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


def _validate_large_capture_trace_streaming(trace: Trace) -> TraceValidationReport:
    """Validate a real raw capture with bounded chunks and a disk-backed relation index."""

    source_events = trace.events
    source_dependencies = trace.dependencies
    if isinstance(source_events, np.memmap) and isinstance(source_dependencies, np.memmap):
        events = np.memmap(
            source_events.filename, dtype=source_events.dtype, mode="c",
            shape=source_events.shape,
        )
        dependencies = np.memmap(
            source_dependencies.filename, dtype=source_dependencies.dtype, mode="c",
            shape=source_dependencies.shape,
        )
    else:
        events = source_events
        dependencies = source_dependencies
    payload = trace.payload
    audit = trace.metadata.get("capture_audit")
    if not isinstance(audit, dict):
        raise TraceValidationError("capture audit metadata is malformed")
    relation_count = int(audit.get("cuda_valid_relations", -1))
    query_count = int(audit.get("captured_logical_queries", -1))
    initial_gaussian_count = trace.metadata.get("initial_gaussian_count")
    if relation_count < 0 or query_count < 0:
        raise TraceValidationError("capture audit relation/query totals are missing")
    if not isinstance(initial_gaussian_count, int) or initial_gaussian_count < 0:
        raise TraceValidationError("initial Gaussian count metadata is invalid")
    if any(int(audit.get(name, 0)) for name in (
        "optimizer_steps", "collection_modification_transactions",
        "update_begin_events", "update_end_events", "collection_modification_events",
    )):
        raise TraceValidationError(
            "large mmap validator requires a state-transition pass for update events"
        )

    event_count = int(events.size)
    dependency_count = int(dependencies.size)
    payload_count = int(payload.size)
    chunk_size = max(1, int(trace.metadata.get("trace_chunk_events", 1)))
    dependency_chunk_size = max(chunk_size, 1_048_576)
    counts_array = np.zeros(len(PrimitiveKind) + 1, dtype=np.int64)
    relation_query_counts = np.zeros(query_count, dtype=np.uint64)
    query_close_seen = np.zeros(query_count, dtype=bool)
    consumer_seen = np.zeros(query_count, dtype=bool)
    backward_pending = np.zeros(relation_count, dtype=bool)
    next_relation = {
        kind: 0 for kind in (
            PrimitiveKind.RELATION, PrimitiveKind.CACHE_REQUEST,
            PrimitiveKind.CACHE_RETURN, PrimitiveKind.FORWARD,
        )
    }
    relation_index_directory = None
    filename = getattr(events, "filename", None)
    if filename is not None:
        relation_index_directory = str(__import__("pathlib").Path(filename).parent)
    with tempfile.TemporaryFile(dir=relation_index_directory) as relation_index_file:
        relation_event_ids = np.memmap(
            relation_index_file, dtype=np.uint64, mode="w+", shape=(relation_count,)
        )
        relation_event_ids[:] = np.iinfo(np.uint64).max
        start = 0
        while start < event_count:
            end = min(start + chunk_size, event_count)
            rows = events[start:end]
            begins = np.asarray(rows["dependency_begin"], dtype=np.int64)
            dep_counts = np.asarray(rows["dependency_count"], dtype=np.int64)
            dep_ends = begins + dep_counts
            bounded_count = int(np.searchsorted(
                dep_ends, int(begins[0]) + dependency_chunk_size, side="right"
            ))
            if bounded_count == 0:
                bounded_count = 1
            if bounded_count < rows.size:
                end = start + bounded_count
                rows = rows[:bounded_count]
                begins = begins[:bounded_count]
                dep_counts = dep_counts[:bounded_count]
                dep_ends = dep_ends[:bounded_count]
            event_ids = np.arange(start, end, dtype=np.uint64)
            if not np.array_equal(rows["event_id"], event_ids):
                mismatch = int(start + np.flatnonzero(rows["event_id"] != event_ids)[0])
                raise TraceValidationError(
                    f"event_id must be a dense capture-order sequence at event {mismatch}"
                )
            kinds = np.asarray(rows["primitive_kind"], dtype=np.int64)
            if kinds.size and (int(kinds.min()) < 1 or int(kinds.max()) > len(PrimitiveKind)):
                raise TraceValidationError(f"unknown primitive kind at event {start}")
            counts_array += np.bincount(kinds, minlength=len(PrimitiveKind) + 1)
            expected_begin = 0 if start == 0 else int(
                events[start - 1]["dependency_begin"]
                + events[start - 1]["dependency_count"]
            )
            if (
                int(begins[0]) != expected_begin
                or bool((begins < 0).any())
                or bool((dep_ends > dependency_count).any())
                or (begins.size > 1 and bool((begins[1:] != dep_ends[:-1]).any()))
            ):
                raise TraceValidationError(f"dependency range is invalid at event {start}")
            flat_dependencies = np.asarray(
                dependencies[int(begins[0]):int(dep_ends[-1])], dtype=np.uint64
            )
            if flat_dependencies.size:
                owners = np.repeat(event_ids, dep_counts)
                if bool((flat_dependencies >= owners).any()):
                    raise TraceValidationError(
                        f"dependency is not a prior event at event {start}"
                    )
            payload_begins = np.asarray(rows["payload_offset"], dtype=np.int64)
            payload_counts = np.asarray(rows["payload_length"], dtype=np.int64)
            payload_ends = payload_begins + payload_counts
            expected_payload = 0 if start == 0 else int(
                events[start - 1]["payload_offset"]
                + events[start - 1]["payload_length"]
            )
            if (
                int(payload_begins[0]) != expected_payload
                or bool((payload_begins < 0).any())
                or bool((payload_ends > payload_count).any())
                or (
                    payload_begins.size > 1
                    and bool((payload_begins[1:] != payload_ends[:-1]).any())
                )
            ):
                raise TraceValidationError(f"payload range is invalid at event {start}")

            def positions(kind: PrimitiveKind) -> np.ndarray:
                return np.flatnonzero(kinds == int(kind))

            def one_dependency(pos: np.ndarray, owner: PrimitiveKind,
                               expected: PrimitiveKind) -> np.ndarray:
                if pos.size == 0:
                    return np.empty(0, dtype=np.uint64)
                if bool((dep_counts[pos] != 1).any()):
                    raise TraceValidationError(
                        f"{owner.name} must have exactly one dependency"
                    )
                deps = np.asarray(dependencies[begins[pos]], dtype=np.uint64)
                if bool((events["primitive_kind"][deps] != int(expected)).any()):
                    raise TraceValidationError(
                        f"{owner.name} has an invalid dependency kind"
                    )
                return deps

            candidate_pos = positions(PrimitiveKind.RELATION_CANDIDATE)
            if candidate_pos.size:
                gaussian_ids = np.asarray(rows["gaussian_id"][candidate_pos], dtype=np.int64)
                if bool((gaussian_ids < 0).any()) or bool((gaussian_ids >= initial_gaussian_count).any()):
                    raise TraceValidationError("relation candidate uses an inactive Gaussian")

            relation_pos = positions(PrimitiveKind.RELATION)
            relation_deps = one_dependency(
                relation_pos, PrimitiveKind.RELATION, PrimitiveKind.RELATION_CANDIDATE
            )
            if relation_pos.size:
                ids = np.asarray(rows["relation_id"][relation_pos], dtype=np.int64)
                expected = np.arange(
                    next_relation[PrimitiveKind.RELATION],
                    next_relation[PrimitiveKind.RELATION] + ids.size,
                    dtype=np.int64,
                )
                if not np.array_equal(ids, expected):
                    raise TraceValidationError("relation_id is not dense in capture order")
                queries = np.asarray(rows["query_id"][relation_pos], dtype=np.int64)
                gaussians = np.asarray(rows["gaussian_id"][relation_pos], dtype=np.int64)
                if (
                    bool((queries < 0).any()) or bool((queries >= query_count).any())
                    or bool((gaussians < 0).any())
                    or bool((gaussians >= initial_gaussian_count).any())
                ):
                    raise TraceValidationError("relation IDs are outside the active domains")
                if bool((events["gaussian_id"][relation_deps] != gaussians).any()):
                    raise TraceValidationError("relation Gaussian does not match its candidate")
                relation_event_ids[ids] = event_ids[relation_pos]
                np.add.at(relation_query_counts, queries, 1)
                next_relation[PrimitiveKind.RELATION] += int(ids.size)

            close_pos = positions(PrimitiveKind.QUERY_CLOSE)
            if close_pos.size:
                queries = np.asarray(rows["query_id"][close_pos], dtype=np.int64)
                if bool((queries < 0).any()) or bool((queries >= query_count).any()):
                    raise TraceValidationError("query close ID is outside the captured domain")
                if bool(query_close_seen[queries].any()):
                    raise TraceValidationError("query has multiple close events")
                if bool((dep_counts[close_pos] != relation_query_counts[queries]).any()):
                    raise TraceValidationError("query close does not depend on every relation")
                for first, last in _contiguous_position_runs(close_pos):
                    run_queries = np.asarray(rows["query_id"][first:last], dtype=np.int64)
                    run_counts = dep_counts[first:last]
                    run_deps = dependencies[int(begins[first]):int(dep_ends[last - 1])]
                    if bool((events["primitive_kind"][run_deps] != int(PrimitiveKind.RELATION)).any()):
                        raise TraceValidationError("query close has a non-relation dependency")
                    if bool((events["query_id"][run_deps] != np.repeat(run_queries, run_counts)).any()):
                        raise TraceValidationError("query close relation belongs to another query")
                query_close_seen[queries] = True

            for owner, expected_kind in (
                (PrimitiveKind.CACHE_REQUEST, PrimitiveKind.RELATION),
                (PrimitiveKind.CACHE_RETURN, PrimitiveKind.CACHE_REQUEST),
            ):
                pos = positions(owner)
                deps = one_dependency(pos, owner, expected_kind)
                if pos.size:
                    ids = np.asarray(rows["relation_id"][pos], dtype=np.int64)
                    expected = np.arange(
                        next_relation[owner], next_relation[owner] + ids.size,
                        dtype=np.int64,
                    )
                    if not np.array_equal(ids, expected):
                        raise TraceValidationError(f"{owner.name} relation IDs are not dense")
                    if bool((events["relation_id"][deps] != ids).any()):
                        raise TraceValidationError(f"{owner.name} relation dependency is mismatched")
                    if bool((events["gaussian_id"][deps] != rows["gaussian_id"][pos]).any()):
                        raise TraceValidationError(f"{owner.name} Gaussian dependency is mismatched")
                    next_relation[owner] += int(ids.size)

            forward_pos = positions(PrimitiveKind.FORWARD)
            if forward_pos.size:
                if bool((dep_counts[forward_pos] != 2).any()):
                    raise TraceValidationError("forward must have two dependencies")
                ids = np.asarray(rows["relation_id"][forward_pos], dtype=np.int64)
                expected = np.arange(
                    next_relation[PrimitiveKind.FORWARD],
                    next_relation[PrimitiveKind.FORWARD] + ids.size,
                    dtype=np.int64,
                )
                if not np.array_equal(ids, expected):
                    raise TraceValidationError("forward relation IDs are not dense")
                deps = np.column_stack((
                    np.asarray(dependencies[begins[forward_pos]], dtype=np.uint64),
                    np.asarray(dependencies[begins[forward_pos] + 1], dtype=np.uint64),
                ))
                if (
                    bool((events["primitive_kind"][deps[:, 0]] != int(PrimitiveKind.RELATION)).any())
                    or bool((events["primitive_kind"][deps[:, 1]] != int(PrimitiveKind.CACHE_RETURN)).any())
                    or bool((events["relation_id"][deps] != ids[:, None]).any())
                ):
                    raise TraceValidationError("forward dependencies are mismatched")
                next_relation[PrimitiveKind.FORWARD] += int(ids.size)

            reduction_pos = positions(PrimitiveKind.QUERY_REDUCTION)
            if reduction_pos.size:
                queries = np.asarray(rows["query_id"][reduction_pos], dtype=np.int64)
                if bool((dep_counts[reduction_pos] != relation_query_counts[queries] + 1).any()):
                    raise TraceValidationError("query reduction does not wait for every forward")
                for first, last in _contiguous_position_runs(reduction_pos):
                    run_queries = np.asarray(rows["query_id"][first:last], dtype=np.int64)
                    run_counts = dep_counts[first:last]
                    run_deps = dependencies[int(begins[first]):int(dep_ends[last - 1])]
                    run_kinds = events["primitive_kind"][run_deps]
                    if not bool(np.isin(
                        run_kinds,
                        [int(PrimitiveKind.QUERY_CLOSE), int(PrimitiveKind.FORWARD)],
                    ).all()):
                        raise TraceValidationError("query reduction dependency kind is invalid")
                    if bool((events["query_id"][run_deps] != np.repeat(run_queries, run_counts)).any()):
                        raise TraceValidationError("query reduction dependency belongs to another query")

            consumer_pos = positions(PrimitiveKind.CONSUMER)
            if consumer_pos.size:
                queries = np.asarray(rows["query_id"][consumer_pos], dtype=np.int64)
                if bool((queries < 0).any()) or bool((queries >= query_count).any()):
                    raise TraceValidationError("consumer query ID is outside the captured domain")
                if bool(consumer_seen[queries].any()):
                    raise TraceValidationError("query has multiple consumers")
                for first, last in _contiguous_position_runs(consumer_pos):
                    run_deps = dependencies[int(begins[first]):int(dep_ends[last - 1])]
                    if bool((events["primitive_kind"][run_deps] != int(PrimitiveKind.QUERY_REDUCTION)).any()):
                        raise TraceValidationError("consumer has a non-reduction dependency")
                consumer_seen[queries] = True

            adjoint_pos = positions(PrimitiveKind.ADJOINT)
            adjoint_deps = one_dependency(
                adjoint_pos, PrimitiveKind.ADJOINT, PrimitiveKind.CONSUMER
            )
            if adjoint_pos.size:
                ids = np.asarray(rows["relation_id"][adjoint_pos], dtype=np.int64)
                if bool((ids < 0).any()) or bool((ids >= relation_count).any()):
                    raise TraceValidationError("adjoint relation ID is outside the captured domain")
                if bool(backward_pending[ids].any()):
                    raise TraceValidationError("relation has multiple pending adjoints")
                relation_events = np.asarray(relation_event_ids[ids], dtype=np.uint64)
                if bool((relation_events == np.iinfo(np.uint64).max).any()):
                    raise TraceValidationError("adjoint precedes its relation")
                if (
                    bool((events["query_id"][adjoint_deps] != rows["query_id"][adjoint_pos]).any())
                    or bool((events["query_id"][relation_events] != rows["query_id"][adjoint_pos]).any())
                    or bool((events["gaussian_id"][relation_events] != rows["gaussian_id"][adjoint_pos]).any())
                ):
                    raise TraceValidationError("adjoint relation or consumer identity is mismatched")
                backward_pending[ids] = True

            gradient_pos = positions(PrimitiveKind.GRADIENT_REDUCTION)
            gradient_deps = one_dependency(
                gradient_pos, PrimitiveKind.GRADIENT_REDUCTION, PrimitiveKind.ADJOINT
            )
            if gradient_pos.size:
                ids = np.asarray(rows["relation_id"][gradient_pos], dtype=np.int64)
                if bool((ids < 0).any()) or bool((ids >= relation_count).any()):
                    raise TraceValidationError("gradient relation ID is outside the captured domain")
                if not bool(backward_pending[ids].all()):
                    raise TraceValidationError("gradient has no unique pending adjoint")
                if (
                    bool((events["relation_id"][gradient_deps] != ids).any())
                    or bool((events["gaussian_id"][gradient_deps] != rows["gaussian_id"][gradient_pos]).any())
                    or bool((events["query_id"][gradient_deps] != rows["query_id"][gradient_pos]).any())
                ):
                    raise TraceValidationError("gradient and adjoint identities are mismatched")
                backward_pending[ids] = False

            _release_mmap_pages(events)
            _release_mmap_pages(dependencies)
            # The original read-only mappings remain live on the Trace object;
            # release their sequentially scanned pages as well as the COW views.
            _release_mmap_pages(source_events)
            _release_mmap_pages(source_dependencies)
            start = end

        relation_event_ids.flush()
        if any(value != relation_count for value in next_relation.values()):
            raise TraceValidationError("relation pipeline counts are inconsistent")
        if bool((relation_event_ids == np.iinfo(np.uint64).max).any()):
            raise TraceValidationError("relation index is incomplete")
    if not bool(query_close_seen.all()):
        raise TraceValidationError("captured query is missing its close event")
    if not bool(consumer_seen.all()):
        raise TraceValidationError("captured query is missing its consumer")
    if bool(backward_pending.any()):
        raise TraceValidationError("captured relation is missing its gradient")
    counts = {
        PrimitiveKind(kind).name: int(counts_array[kind])
        for kind in range(1, len(PrimitiveKind) + 1)
        if counts_array[kind]
    }
    _validate_capture_audit(trace, counts)
    return TraceValidationReport(
        event_count=event_count,
        dependency_count=dependency_count,
        counts=counts,
        query_count=query_count,
        gaussian_count=initial_gaussian_count,
        update_count=0,
    )


def _contiguous_position_runs(positions: np.ndarray):
    if positions.size == 0:
        return
    splits = np.flatnonzero(np.diff(positions) != 1) + 1
    for run in np.split(positions, splits):
        yield int(run[0]), int(run[-1]) + 1


def _release_mmap_pages(array: object) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


def _validate_large_capture_trace(trace: Trace) -> TraceValidationReport:
    """Validate a raw-column capture without relation-sized Python objects.

    The real R²-Gaussian capture emits the query pipeline in contiguous
    primitive sections.  This validator checks each section's exact IDs,
    dependency kinds, query/Gaussian identity and audit totals in bounded
    chunks.  It deliberately rejects update-containing large traces until
    their state-transition pass is available instead of silently weakening
    lifecycle checks.
    """

    events = trace.events
    dependencies = trace.dependencies
    payload = trace.payload
    chunk_size = max(1, int(trace.metadata.get("trace_chunk_events", 1)))
    event_count = int(events.size)
    dependency_count = int(dependencies.size)
    payload_count = int(payload.size)
    kind_values = np.asarray(events["primitive_kind"], dtype=np.uint16)
    if kind_values.size:
        known_kinds = np.isin(kind_values, np.arange(1, len(PrimitiveKind) + 1))
        if not bool(known_kinds.all()):
            bad = int(np.flatnonzero(~known_kinds)[0])
            raise TraceValidationError(f"unknown primitive kind at event {bad}")
    counts_array = np.bincount(kind_values.astype(np.int64), minlength=len(PrimitiveKind) + 1)
    counts = {
        PrimitiveKind(kind).name: int(counts_array[kind])
        for kind in range(1, len(PrimitiveKind) + 1)
        if counts_array[kind]
    }

    for start in range(0, event_count, chunk_size):
        end = min(start + chunk_size, event_count)
        rows = events[start:end]
        expected_ids = np.arange(start, end, dtype=np.uint64)
        if not np.array_equal(rows["event_id"], expected_ids):
            mismatch = int(start + np.flatnonzero(rows["event_id"] != expected_ids)[0])
            raise TraceValidationError(f"event_id must be a dense capture-order sequence at event {mismatch}")
        begins = np.asarray(rows["dependency_begin"], dtype=np.int64)
        dep_counts = np.asarray(rows["dependency_count"], dtype=np.int64)
        ends = begins + dep_counts
        if bool((begins < 0).any()) or bool((dep_counts < 0).any()) or bool((ends > dependency_count).any()):
            raise TraceValidationError(f"dependency range is invalid at event {start}")
        if start == 0:
            expected_begin = 0
        else:
            expected_begin = int(events[start - 1]["dependency_begin"] + events[start - 1]["dependency_count"])
        if int(begins[0]) != expected_begin or (
            begins.size > 1 and bool(begins[1:] != ends[:-1]).any()
        ):
            raise TraceValidationError(f"dependency columns are not contiguous at event {start}")
        if ends.size:
            flat_dependencies = dependencies[int(begins[0]):int(ends[-1])]
            flat_event_ids = np.repeat(expected_ids, dep_counts)
            if flat_dependencies.size and bool((flat_dependencies >= flat_event_ids).any()):
                raise TraceValidationError(f"dependency is not a prior event at event {start}")
        payload_begins = np.asarray(rows["payload_offset"], dtype=np.int64)
        payload_counts = np.asarray(rows["payload_length"], dtype=np.int64)
        payload_ends = payload_begins + payload_counts
        if bool((payload_begins < 0).any()) or bool((payload_counts < 0).any()) or bool((payload_ends > payload_count).any()):
            raise TraceValidationError(f"payload range is invalid at event {start}")
        if start == 0:
            expected_payload_begin = 0
        else:
            expected_payload_begin = int(events[start - 1]["payload_offset"] + events[start - 1]["payload_length"])
        if int(payload_begins[0]) != expected_payload_begin or (
            payload_begins.size > 1 and bool(payload_begins[1:] != payload_ends[:-1]).any()
        ):
            raise TraceValidationError(f"payload columns are not contiguous at event {start}")

    if any(counts.get(PrimitiveKind(kind).name, 0) for kind in (
        PrimitiveKind.UPDATE_BEGIN, PrimitiveKind.UPDATE_COMMIT,
        PrimitiveKind.UPDATE_END, PrimitiveKind.SET_MODIFICATION,
    )):
        raise TraceValidationError(
            "large mmap validator requires a state-transition pass for update events"
        )

    def rows_for(kind: PrimitiveKind) -> np.ndarray:
        return np.flatnonzero(kind_values == int(kind)).astype(np.int64, copy=False)

    def dependency_slice(event_ids: np.ndarray) -> np.ndarray:
        if event_ids.size == 0:
            return np.empty(0, dtype=np.uint64)
        begins = np.asarray(events["dependency_begin"][event_ids], dtype=np.int64)
        counts_local = np.asarray(events["dependency_count"][event_ids], dtype=np.int64)
        if event_ids.size > 1 and bool(
            (begins[1:] != begins[:-1] + counts_local[:-1]).any()
        ):
            raise TraceValidationError("selected event dependencies are not contiguous")
        return np.asarray(dependencies[int(begins[0]):int(begins[-1] + counts_local[-1])], dtype=np.uint64)

    candidate_ids = rows_for(PrimitiveKind.RELATION_CANDIDATE)
    relation_ids = rows_for(PrimitiveKind.RELATION)
    close_ids = rows_for(PrimitiveKind.QUERY_CLOSE)
    request_ids = rows_for(PrimitiveKind.CACHE_REQUEST)
    return_ids = rows_for(PrimitiveKind.CACHE_RETURN)
    forward_ids = rows_for(PrimitiveKind.FORWARD)
    reduction_ids = rows_for(PrimitiveKind.QUERY_REDUCTION)
    consumer_ids = rows_for(PrimitiveKind.CONSUMER)
    adjoint_ids = rows_for(PrimitiveKind.ADJOINT)
    gradient_ids = rows_for(PrimitiveKind.GRADIENT_REDUCTION)

    initial_gaussian_count = trace.metadata.get("initial_gaussian_count")
    if initial_gaussian_count is not None and (
        not isinstance(initial_gaussian_count, int) or initial_gaussian_count < 0
    ):
        raise TraceValidationError("initial Gaussian count metadata is invalid")
    if initial_gaussian_count is not None:
        gaussian_values = np.asarray(events["gaussian_id"], dtype=np.int64)
        used = gaussian_values[gaussian_values >= 0]
        if used.size and int(used.max()) >= initial_gaussian_count:
            raise TraceValidationError("large trace uses a Gaussian outside the initial active set")

    def require_ids_dense(ids: np.ndarray, field: str) -> None:
        values = np.asarray(events[field][ids], dtype=np.int64)
        if values.size and not np.array_equal(values, np.arange(values.size, dtype=np.int64)):
            raise TraceValidationError(f"{field} is not dense in capture order")

    require_ids_dense(relation_ids, "relation_id")
    require_ids_dense(request_ids, "relation_id")
    require_ids_dense(return_ids, "relation_id")
    require_ids_dense(forward_ids, "relation_id")
    require_ids_dense(adjoint_ids, "relation_id")
    require_ids_dense(gradient_ids, "relation_id")
    if relation_ids.size != request_ids.size or relation_ids.size != return_ids.size or relation_ids.size != forward_ids.size:
        raise TraceValidationError("relation pipeline counts are inconsistent")
    if relation_ids.size != adjoint_ids.size or relation_ids.size != gradient_ids.size:
        raise TraceValidationError("backward relation counts are inconsistent")

    def check_dependencies(kind: PrimitiveKind, ids: np.ndarray, expected: PrimitiveKind) -> np.ndarray:
        deps = dependency_slice(ids)
        counts_local = np.asarray(events["dependency_count"][ids], dtype=np.int64)
        dep_kinds = np.asarray(kind_values[deps], dtype=np.uint16) if deps.size else np.empty(0, dtype=np.uint16)
        if dep_kinds.size and bool((dep_kinds != int(expected)).any()):
            raise TraceValidationError(f"{kind.name} has an invalid dependency kind")
        return deps

    relation_dependencies = check_dependencies(
        PrimitiveKind.RELATION, relation_ids, PrimitiveKind.RELATION_CANDIDATE
    )
    relation_dep_counts = np.asarray(events["dependency_count"][relation_ids], dtype=np.int64)
    if relation_dep_counts.size and bool((relation_dep_counts != 1).any()):
        raise TraceValidationError("relation must depend on exactly one candidate")
    if relation_ids.size:
        relation_candidate_ids = relation_dependencies
        if bool((events["gaussian_id"][relation_ids] != events["gaussian_id"][relation_candidate_ids]).any()):
            raise TraceValidationError("relation Gaussian does not match its candidate")

    relation_query = np.asarray(events["query_id"][relation_ids], dtype=np.int64)
    close_query = np.asarray(events["query_id"][close_ids], dtype=np.int64)
    if close_query.size and not np.array_equal(close_query, np.arange(close_query.size, dtype=np.int64)):
        raise TraceValidationError("query close IDs are not dense")
    relation_query_counts = np.bincount(
        relation_query - int(relation_query.min()) if relation_query.size else np.empty(0, dtype=np.int64),
        minlength=int(close_query.size),
    )
    close_dependencies = dependency_slice(close_ids)
    close_dep_counts = np.asarray(events["dependency_count"][close_ids], dtype=np.int64)
    if close_dep_counts.size and bool((close_dep_counts != relation_query_counts).any()):
        raise TraceValidationError("query close does not depend on every relation")
    if close_dependencies.size:
        close_dep_kinds = kind_values[close_dependencies]
        if bool((close_dep_kinds != int(PrimitiveKind.RELATION)).any()):
            raise TraceValidationError("query close has a non-relation dependency")
        dep_query = np.asarray(events["query_id"][close_dependencies], dtype=np.int64)
        repeated_queries = np.repeat(close_query, close_dep_counts)
        if bool((dep_query != repeated_queries).any()):
            raise TraceValidationError("query close relation belongs to a different query")

    forward_dependencies = dependency_slice(forward_ids)
    forward_dep_counts = np.asarray(events["dependency_count"][forward_ids], dtype=np.int64)
    if forward_dep_counts.size and bool((forward_dep_counts != 2).any()):
        raise TraceValidationError("forward must depend on relation and cache return")
    if forward_dependencies.size:
        reshaped = forward_dependencies.reshape(-1, 2)
        if bool((kind_values[reshaped[:, 0]] != int(PrimitiveKind.RELATION)).any()) or bool((kind_values[reshaped[:, 1]] != int(PrimitiveKind.CACHE_RETURN)).any()):
            raise TraceValidationError("forward dependency kinds are invalid")
        if bool((events["relation_id"][forward_ids] != events["relation_id"][reshaped[:, 0]]).any()):
            raise TraceValidationError("forward relation IDs do not match relation dependencies")
        if bool((events["relation_id"][forward_ids] != events["relation_id"][reshaped[:, 1]]).any()):
            raise TraceValidationError("forward relation IDs do not match cache returns")

    request_dependencies = check_dependencies(
        PrimitiveKind.CACHE_REQUEST, request_ids, PrimitiveKind.RELATION
    )
    return_dependencies = check_dependencies(
        PrimitiveKind.CACHE_RETURN, return_ids, PrimitiveKind.CACHE_REQUEST
    )
    if request_ids.size and bool((events["relation_id"][request_ids] != events["relation_id"][relation_ids]).any()):
        raise TraceValidationError("cache request relation IDs do not match relations")
    if return_ids.size and bool((events["relation_id"][return_ids] != events["relation_id"][request_ids]).any()):
        raise TraceValidationError("cache return relation IDs do not match requests")
    if return_dependencies.size:
        expected_requests = request_ids
        if not np.array_equal(return_dependencies, expected_requests):
            raise TraceValidationError("cache return does not depend on its request")

    reduction_query = np.asarray(events["query_id"][reduction_ids], dtype=np.int64)
    if reduction_query.size and not np.array_equal(reduction_query, close_query):
        raise TraceValidationError("query reduction IDs do not match query closes")
    reduction_dependencies = dependency_slice(reduction_ids)
    reduction_dep_counts = np.asarray(events["dependency_count"][reduction_ids], dtype=np.int64)
    if reduction_dependencies.size:
        reduction_kinds = kind_values[reduction_dependencies]
        if not bool(np.isin(
            reduction_kinds,
            [int(PrimitiveKind.FORWARD), int(PrimitiveKind.QUERY_CLOSE)],
        ).all()):
            raise TraceValidationError("query reduction has an invalid dependency kind")
        close_dep_mask = reduction_kinds == int(PrimitiveKind.QUERY_CLOSE)
        if bool(close_dep_mask.sum() != reduction_ids.size):
            raise TraceValidationError("query reduction does not depend on exactly one query close")
    if consumer_ids.size != close_ids.size:
        raise TraceValidationError("consumer count does not match query count")
    consumer_query = np.asarray(events["query_id"][consumer_ids], dtype=np.int64)
    if consumer_query.size and not np.array_equal(consumer_query, close_query):
        raise TraceValidationError("consumer IDs do not match query closes")
    consumer_dependencies = dependency_slice(consumer_ids)
    if consumer_dependencies.size and bool((kind_values[consumer_dependencies] != int(PrimitiveKind.QUERY_REDUCTION)).any()):
        raise TraceValidationError("consumer has a non-reduction dependency")

    adjoint_dependencies = check_dependencies(
        PrimitiveKind.ADJOINT, adjoint_ids, PrimitiveKind.CONSUMER
    )
    if adjoint_dependencies.size:
        if bool((events["query_id"][adjoint_ids] != events["query_id"][adjoint_dependencies]).any()):
            raise TraceValidationError("adjoint consumer query does not match relation query")
    gradient_dependencies = check_dependencies(
        PrimitiveKind.GRADIENT_REDUCTION, gradient_ids, PrimitiveKind.ADJOINT
    )
    if gradient_dependencies.size and bool((events["relation_id"][gradient_ids] != events["relation_id"][gradient_dependencies]).any()):
        raise TraceValidationError("gradient relation IDs do not match adjoints")

    _validate_capture_audit(trace, counts)
    query_count = int(close_ids.size)
    gaussian_count = int(initial_gaussian_count or 0)
    if initial_gaussian_count is None:
        gaussian_values = np.asarray(events["gaussian_id"], dtype=np.int64)
        used = gaussian_values[gaussian_values >= 0]
        gaussian_count = int(used.max()) + 1 if used.size else 0
    return TraceValidationReport(
        event_count=event_count,
        dependency_count=dependency_count,
        counts=counts,
        query_count=query_count,
        gaussian_count=gaussian_count,
        update_count=0,
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
    if trace.metadata.get("capture_audit_schema_version") != "gala-r2-capture-audit-v4":
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
    transfer_names = {
        "relation_record_device_batches", "relation_record_d2h_batches",
    }
    if not transfer_names.issubset(audit):
        raise TraceValidationError("relation record transfer audit totals are missing")
    device_batches = int(audit["relation_record_device_batches"])
    d2h_batches = int(audit["relation_record_d2h_batches"])
    candidate_count = int(audit.get("cuda_relation_candidates", 0))
    if (
        device_batches > captured_kernels
        or device_batches > candidate_count
        or (device_batches == 0) != (d2h_batches == 0)
        or d2h_batches > device_batches
    ):
        raise TraceValidationError("relation record transfer audit totals are inconsistent")

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
