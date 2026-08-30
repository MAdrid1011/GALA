"""Dependency-closed query packets for explicitly scoped quick cycle validation."""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from typing import Callable, Iterator

import numpy as np

from gala_sim.clamp.events import PrimitiveKind, dependency_dtype, event_dtype
from gala_sim.mechanisms import CANONICAL_VARIANT_POLICIES

from .model import Trace
from .validator import validate_trace
from .virtual import (
    MASK_WORD_BITS,
    RASTER_BLOCK,
    RASTER_TEMPLATE_ID,
    VOXEL_BLOCK,
    VOXEL_TEMPLATE_ID,
    VirtualQueryEventExpander,
    VirtualTracePacket,
)


QUERY_PACKET_SAMPLE_SCHEMA_VERSION = "gala-query-packet-sample-v2"
QUERY_PACKET_SAMPLE_SCHEMA_VERSIONS = frozenset({
    "gala-query-packet-sample-v1",
    QUERY_PACKET_SAMPLE_SCHEMA_VERSION,
})


@dataclass(frozen=True)
class QueryRange:
    start: int
    count: int

    def __post_init__(self) -> None:
        maximum = np.iinfo(np.int64).max
        if (
            self.start < 0 or self.count <= 0
            or self.start > maximum or self.count > maximum - self.start
        ):
            raise ValueError("query ranges require a non-negative start and positive count")

    @property
    def end(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class TraceSampleConfig:
    query_ranges: tuple[QueryRange, ...]
    max_events: int
    max_dependencies: int
    scan_events: int
    scan_backend: str = "auto"

    def __post_init__(self) -> None:
        if not self.query_ranges:
            raise ValueError("at least one query range is required")
        if min(self.max_events, self.max_dependencies, self.scan_events) <= 0:
            raise ValueError("sample and scan limits must be positive")
        if self.scan_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("scan_backend must be auto, cpu, or cuda")


@dataclass(frozen=True)
class QueryPacketSampleConfig:
    """Real query supports rebased into a bounded cross-iteration packet sample."""

    query_ranges: tuple[QueryRange, ...]
    scan_events: int
    scan_backend: str = "auto"
    query_lanes: int = 8
    ssim_radius: int = 0

    def __post_init__(self) -> None:
        if len(self.query_ranges) < 2:
            raise ValueError("query packet samples require at least two query ranges")
        if self.scan_events <= 0 or not 0 < self.query_lanes <= 8:
            raise ValueError("query packet scan and lane parameters must be positive")
        if self.ssim_radius < 0:
            raise ValueError("query packet SSIM radius must be non-negative")
        if self.scan_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("scan_backend must be auto, cpu, or cuda")
        ordered = sorted(self.query_ranges, key=lambda item: item.start)
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("query packet ranges must not overlap")


def real_query_packet_sample(
    trace: Trace,
    config: QueryPacketSampleConfig,
    *,
    source_identity: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> Trace:
    """Rebuild selected real relation supports without an all-iteration barrier.

    This is a query-scheduler microbenchmark, not a dependency-closed or formal
    trace.  Every relation and Gaussian support comes from the source trace.
    Query domains and state versions are rebased so consecutive iterations can
    exercise history without pulling an optimizer barrier's unrelated fan-in.
    """

    resolved_backend, scanner = _resolve_scan_backend(
        config.scan_backend,
        primitive_kinds=(PrimitiveKind.RELATION, PrimitiveKind.CONSUMER),
    )
    range_array = np.asarray(
        [(item.start, item.end) for item in config.query_ranges], dtype=np.int64,
    )
    selected_parts: list[np.ndarray] = []
    selected_id_parts: list[np.ndarray] = []
    for start in range(0, trace.event_count, config.scan_events):
        end = min(start + config.scan_events, trace.event_count)
        rows = trace.events[start:end]
        if scanner is None:
            queries = rows["query_id"]
            selected = np.zeros(rows.size, dtype=bool)
            for query_range in config.query_ranges:
                selected |= (queries >= query_range.start) & (queries < query_range.end)
            selected &= (
                (rows["primitive_kind"] == int(PrimitiveKind.RELATION))
                | (rows["primitive_kind"] == int(PrimitiveKind.CONSUMER))
            )
            positions = np.flatnonzero(selected)
        else:
            positions = scanner(rows, range_array)
        if positions.size:
            selected_parts.append(np.array(rows[positions], copy=True))
            selected_id_parts.append(positions.astype(np.uint64, copy=False) + start)
        _release_mmap_pages(trace.events)
        if progress is not None:
            progress("query_packet_scan", end, trace.event_count)
    if not selected_parts:
        raise ValueError("selected query packet ranges contain no relations or consumers")

    selected_rows = np.concatenate(selected_parts)
    selected_ids = np.concatenate(selected_id_parts)
    packets: list[VirtualTracePacket] = []
    reports: list[dict[str, object]] = []
    source_versions: list[int] = []
    maximum_gaussian = -1
    for query_range in config.query_ranges:
        in_range = (
            (selected_rows["query_id"] >= query_range.start)
            & (selected_rows["query_id"] < query_range.end)
        )
        rows = selected_rows[in_range]
        row_ids = selected_ids[in_range]
        relations = rows[rows["primitive_kind"] == int(PrimitiveKind.RELATION)]
        relation_source_ids = row_ids[
            rows["primitive_kind"] == int(PrimitiveKind.RELATION)
        ]
        consumers = rows[rows["primitive_kind"] == int(PrimitiveKind.CONSUMER)]
        if relations.size == 0 or consumers.size != query_range.count:
            raise ValueError(
                "query packet range needs relations and exactly one consumer per query"
            )
        expected_queries = np.arange(query_range.start, query_range.end, dtype=np.int64)
        if not np.array_equal(np.sort(consumers["query_id"]), expected_queries):
            raise ValueError("query packet consumers do not cover the selected range")
        identity_fields = ("iteration_id", "template_id")
        identity = tuple(int(relations[field][0]) for field in identity_fields)
        if any(
            not bool((rows[field] == value).all())
            for field, value in zip(identity_fields, identity, strict=True)
        ):
            raise ValueError("query packet range crosses an iteration or query domain")
        iteration_id, template_id = identity
        relation_versions = np.unique(relations["state_version"])
        relation_field_masks = np.unique(relations["field_mask"])
        if relation_versions.size != 1 or relation_field_masks.size != 1:
            raise ValueError("query packet relations cross a state or field domain")
        state_version = int(relation_versions[0])
        field_mask = int(relation_field_masks[0])
        loss_flags = np.unique(consumers["flags"])
        if loss_flags.size != 1 or int(loss_flags[0]) == 0:
            raise ValueError("query packet consumers have inconsistent loss flags")
        query_shape = _rebased_query_shape(template_id, query_range.count)

        dependency_counts = np.asarray(relations["dependency_count"], dtype=np.uint64)
        if not bool((dependency_counts == 1).all()):
            raise ValueError("source relations must have one candidate dependency")
        candidate_ids = _gather_values(
            trace.dependencies,
            np.asarray(relations["dependency_begin"], dtype=np.uint64),
            dependency_counts,
        )
        unique_candidate_ids, candidate_inverse = np.unique(
            candidate_ids, return_inverse=True,
        )
        candidate_rows = np.asarray(trace.events[unique_candidate_ids])
        if not bool((
            candidate_rows["primitive_kind"]
            == int(PrimitiveKind.RELATION_CANDIDATE)
        ).all()):
            raise ValueError("source relation lacks a relation-candidate dependency")
        if not np.array_equal(
            candidate_rows["gaussian_id"][candidate_inverse], relations["gaussian_id"],
        ):
            raise ValueError("source relation and candidate Gaussian IDs disagree")

        local_queries = np.asarray(
            relations["query_id"] - query_range.start, dtype=np.int64,
        )
        local_query_capacity = int(np.prod(
            RASTER_BLOCK if template_id == RASTER_TEMPLATE_ID else VOXEL_BLOCK,
            dtype=np.int64,
        ))
        masks = np.zeros(
            (unique_candidate_ids.size, local_query_capacity // MASK_WORD_BITS),
            dtype=np.dtype("<u4"),
        )
        bits = np.left_shift(
            np.uint32(1), (local_queries % MASK_WORD_BITS).astype(np.uint32),
        )
        np.bitwise_or.at(
            masks,
            (candidate_inverse, local_queries // MASK_WORD_BITS),
            bits,
        )
        point_ids = np.asarray(candidate_rows["gaussian_id"], dtype=np.int64)
        point_keys = np.bitwise_and(
            np.asarray(candidate_rows["address_token"], dtype=np.uint64),
            np.uint64((1 << 32) - 1),
        )
        maximum_gaussian = max(maximum_gaussian, int(point_ids.max(initial=-1)))
        source_versions.append(state_version)
        packets.append(VirtualTracePacket(
            iteration_id=iteration_id,
            template_id=template_id,
            query_base=query_range.start,
            query_shape=query_shape,
            point_ids=point_ids,
            point_keys=point_keys,
            masks=masks,
            state_version=state_version,
            field_mask=field_mask,
            loss_flags=int(loss_flags[0]),
            ssim_radius=(
                config.ssim_radius if template_id == RASTER_TEMPLATE_ID else 0
            ),
            backward_confirmed=True,
        ))
        reports.append({
            "query_base": query_range.start,
            "query_start": query_range.start,
            "query_count": query_range.count,
            "query_shape": list(query_shape),
            "iteration_id": iteration_id,
            "template_id": template_id,
            "source_state_version": state_version,
            "source_consumer_state_versions": [
                int(value) for value in np.unique(consumers["state_version"])
            ],
            "source_relation_count": int(relations.size),
            "relation_count": int(relations.size),
            "source_relation_event_min": int(relation_source_ids.min()),
            "source_relation_event_max": int(relation_source_ids.max()),
            "candidate_count": int(unique_candidate_ids.size),
            "loss_flags": int(loss_flags[0]),
        })
        _release_mmap_pages(trace.dependencies)
        _release_mmap_pages(trace.events)

    ordered = sorted(
        zip(packets, reports, source_versions, strict=True),
        key=lambda item: (
            item[0].iteration_id, item[0].template_id, item[0].query_base,
        ),
    )
    packets = [item[0] for item in ordered]
    reports = [item[1] for item in ordered]
    source_versions = [item[2] for item in ordered]

    iteration_ids = sorted({packet.iteration_id for packet in packets})
    if len(iteration_ids) < 2 or any(
        right != left + 1 for left, right in zip(iteration_ids, iteration_ids[1:])
    ):
        raise ValueError("query packet ranges must come from consecutive iterations")

    layouts: list[tuple[tuple[int, int, tuple[int, ...]], ...]] = []
    for iteration_id in iteration_ids:
        iteration_packets = [
            packet for packet in packets if packet.iteration_id == iteration_id
        ]
        template_bases = {
            template_id: min(
                packet.query_base for packet in iteration_packets
                if packet.template_id == template_id
            )
            for template_id in {packet.template_id for packet in iteration_packets}
        }
        layouts.append(tuple(
            (
                packet.template_id,
                packet.query_base - template_bases[packet.template_id],
                packet.query_shape,
            )
            for packet in iteration_packets
        ))
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("query packet history requires matching logical domains")

    expander = VirtualQueryEventExpander(
        max_events=config.scan_events,
        relation_query_lanes=config.query_lanes,
    )
    event_parts: list[np.ndarray] = []
    dependency_parts: list[np.ndarray] = []
    dependency_offset = 0
    for packet in packets:
        for event_packet in expander.expand(packet):
            event_rows = event_packet.events.copy()
            event_rows["dependency_begin"] += dependency_offset
            event_parts.append(event_rows)
            dependency_parts.append(event_packet.dependencies)
            dependency_offset += event_packet.dependencies.size
    events = np.concatenate(event_parts).astype(event_dtype(), copy=False)
    dependencies = np.concatenate(dependency_parts).astype(
        dependency_dtype(), copy=False,
    )
    metadata = {
        "schema_version": "gala-clamp-events-v2",
        "model": trace.metadata.get("model", "unknown"),
        "dataset": trace.metadata.get("dataset", "unknown"),
        "initial_gaussian_count": maximum_gaussian + 1,
        "state_record_bytes": int(trace.metadata.get("state_record_bytes", 128)),
        "trace_sample": {
            "schema_version": QUERY_PACKET_SAMPLE_SCHEMA_VERSION,
            "result_scope": "quick_cycle_validation",
            "formal_performance_eligible": False,
            "quality_eligible": False,
            "selection": "real_relation_supported_query_packets",
            "eligible_policies": [
                "base", "query", "residency", "full",
                "query_oracle", "residency_oracle",
                *CANONICAL_VARIANT_POLICIES,
            ],
            "source_identity": source_identity,
            "source_event_count": trace.event_count,
            "source_dependency_count": int(trace.dependencies.size),
            "source_initial_gaussian_count": trace.metadata.get(
                "initial_gaussian_count"
            ),
            "source_state_versions": source_versions,
            "state_versions_preserved": True,
            "boundary_condition": "prior_selected_packet_completes_before_next_iteration",
            "query_lanes": config.query_lanes,
            "ssim_radius": config.ssim_radius,
            "scan_events": config.scan_events,
            "scan_backend": resolved_backend,
            "packets": reports,
            "sample_event_count": int(events.size),
            "sample_dependency_count": int(dependencies.size),
        },
    }
    sample = Trace(
        events,
        dependencies,
        np.empty(0, dtype=np.dtype("<f4")),
        metadata,
    )
    validate_trace(sample)
    return sample


def _rebased_query_shape(template_id: int, query_count: int) -> tuple[int, ...]:
    if template_id == RASTER_TEMPLATE_ID:
        if query_count > RASTER_BLOCK[1]:
            raise ValueError("raster query packet range exceeds one physical row")
        return 1, query_count
    if template_id == VOXEL_TEMPLATE_ID:
        if query_count > VOXEL_BLOCK[2]:
            raise ValueError("voxel query packet range exceeds one physical row")
        return 1, 1, query_count
    raise ValueError(f"unsupported query packet template {template_id}")


def dependency_closed_query_sample(
    trace: Trace,
    config: TraceSampleConfig,
    *,
    source_identity: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> Trace:
    """Extract selected query terminals and every transitive predecessor.

    The result is for quick cycle validation only. It deliberately keeps the
    original query, Gaussian, relation, address, byte-count and timing fields,
    while rebasing event/dependency/payload offsets into a standalone trace.
    """

    resolved_backend, cuda_scanner = _resolve_scan_backend(config.scan_backend)
    range_array = np.asarray(
        [(item.start, item.end) for item in config.query_ranges], dtype=np.int64,
    )
    targets: list[np.ndarray] = []
    terminal_kinds = np.asarray([
        int(PrimitiveKind.CONSUMER), int(PrimitiveKind.GRADIENT_REDUCTION),
    ], dtype=trace.events.dtype.fields["primitive_kind"][0])
    for start in range(0, trace.event_count, config.scan_events):
        end = min(start + config.scan_events, trace.event_count)
        rows = trace.events[start:end]
        if cuda_scanner is None:
            queries = rows["query_id"]
            selected = np.zeros(rows.size, dtype=bool)
            for query_range in config.query_ranges:
                selected |= (queries >= query_range.start) & (queries < query_range.end)
            selected &= (
                (rows["primitive_kind"] == terminal_kinds[0])
                | (rows["primitive_kind"] == terminal_kinds[1])
            )
            positions = np.flatnonzero(selected)
        else:
            positions = cuda_scanner(rows, range_array)
        if positions.size:
            targets.append(positions.astype(np.uint64, copy=False) + start)
        _release_mmap_pages(trace.events)
        if progress is not None:
            progress("terminal_scan", end, trace.event_count)
    if not targets:
        raise ValueError("selected query ranges contain no consumer or gradient terminals")

    terminal_ids = np.concatenate(targets)
    selected_ids = {int(value) for value in terminal_ids}
    if len(selected_ids) > config.max_events:
        raise ValueError("selected query terminals exceed max_events")
    frontier = terminal_ids
    while frontier.size:
        next_ids: list[int] = []
        for start in range(0, frontier.size, config.scan_events):
            rows = trace.events[frontier[start:start + config.scan_events]]
            for dependency_batch in _gather_value_batches(
                trace.dependencies,
                np.asarray(rows["dependency_begin"], dtype=np.uint64),
                np.asarray(rows["dependency_count"], dtype=np.uint64),
                max_values=config.max_dependencies,
            ):
                for raw_dependency in np.unique(dependency_batch):
                    dependency = int(raw_dependency)
                    if dependency not in selected_ids:
                        selected_ids.add(dependency)
                        next_ids.append(dependency)
                        if len(selected_ids) > config.max_events:
                            raise ValueError(
                                "dependency closure exceeds max_events; reduce the query "
                                "ranges or raise the explicit quick-validation limit"
                            )
        if len(selected_ids) > config.max_events:
            raise ValueError(
                "dependency closure exceeds max_events; reduce the query ranges "
                "or raise the explicit quick-validation limit"
            )
        frontier = np.asarray(next_ids, dtype=np.uint64)
        _release_mmap_pages(trace.events)
        _release_mmap_pages(trace.dependencies)
        if progress is not None:
            progress("dependency_closure", len(selected_ids), config.max_events)

    source_event_ids = np.asarray(sorted(selected_ids), dtype=np.uint64)
    rows = np.array(trace.events[source_event_ids], copy=True)
    dependency_counts = np.asarray(rows["dependency_count"], dtype=np.uint64)
    dependency_begins = np.asarray(rows["dependency_begin"], dtype=np.uint64)
    source_dependencies = _gather_values(
        trace.dependencies, dependency_begins, dependency_counts,
        max_values=config.max_dependencies,
    )
    mapped_dependencies = np.searchsorted(source_event_ids, source_dependencies).astype(
        dependency_dtype(), copy=False,
    )
    if (
        source_dependencies.size
        and (
            bool((mapped_dependencies >= source_event_ids.size).any())
            or not np.array_equal(source_event_ids[mapped_dependencies], source_dependencies)
        )
    ):
        raise RuntimeError("query sample dependency closure is incomplete")

    payload_counts = np.asarray(rows["payload_length"], dtype=np.uint64)
    payload_begins = np.asarray(rows["payload_offset"], dtype=np.uint64)
    payload = _gather_values(trace.payload, payload_begins, payload_counts)

    rows["event_id"] = np.arange(rows.size, dtype=np.uint64)
    rows["dependency_begin"] = _exclusive_prefix(dependency_counts)
    rows["payload_offset"] = _exclusive_prefix(payload_counts)
    metadata = {
        key: value for key, value in trace.metadata.items()
        if key not in {
            "capture_audit", "capture_audit_schema_version", "trace_storage_format",
        }
    }
    metadata["trace_sample"] = {
        "schema_version": "gala-query-sample-v1",
        "result_scope": "quick_cycle_validation",
        "formal_performance_eligible": False,
        "quality_eligible": False,
        "selection": "dependency_closed_query_terminals",
        "query_ranges": [
            {"start": item.start, "count": item.count} for item in config.query_ranges
        ],
        "source_identity": source_identity,
        "source_event_count": trace.event_count,
        "source_dependency_count": int(trace.dependencies.size),
        "selected_terminal_count": int(terminal_ids.size),
        "sample_event_count": int(rows.size),
        "sample_dependency_count": int(mapped_dependencies.size),
        "max_events": config.max_events,
        "max_dependencies": config.max_dependencies,
        "scan_events": config.scan_events,
        "scan_backend": resolved_backend,
    }
    sample = Trace(
        np.asarray(rows, dtype=event_dtype()),
        np.asarray(mapped_dependencies, dtype=dependency_dtype()),
        np.asarray(payload, dtype=np.dtype("<f4")),
        metadata,
    )
    validate_trace(sample)
    return sample


def _gather_values(
    source: np.ndarray, begins: np.ndarray, counts: np.ndarray,
    *, max_values: int | None = None,
) -> np.ndarray:
    total = int(counts.sum(dtype=np.uint64))
    if total == 0:
        return np.empty(0, dtype=source.dtype)
    if total > np.iinfo(np.intp).max:
        raise ValueError("sample dependency or payload count exceeds platform index range")
    if max_values is not None and total > max_values:
        raise ValueError("sample dependency frontier exceeds max_dependencies")
    repeat_counts = counts.astype(np.intp, copy=False)
    prefixes = _exclusive_prefix(counts)
    offsets = np.arange(total, dtype=np.uint64) - np.repeat(prefixes, repeat_counts)
    indices = np.repeat(begins, repeat_counts) + offsets
    return np.asarray(source[indices], dtype=source.dtype)


def _gather_value_batches(
    source: np.ndarray, begins: np.ndarray, counts: np.ndarray,
    *, max_values: int,
) -> Iterator[np.ndarray]:
    """Yield dependency values without treating duplicate frontier edges as output."""

    if max_values <= 0:
        raise ValueError("sample dependency batch limit must be positive")
    start = 0
    while start < counts.size:
        first_count = int(counts[start])
        if first_count > max_values:
            raise ValueError("sample dependency frontier exceeds max_dependencies")
        end = start + 1
        total = first_count
        while end < counts.size:
            count = int(counts[end])
            if count > max_values:
                raise ValueError("sample dependency frontier exceeds max_dependencies")
            if total + count > max_values:
                break
            total += count
            end += 1
        yield _gather_values(source, begins[start:end], counts[start:end])
        start = end


def _exclusive_prefix(counts: np.ndarray) -> np.ndarray:
    prefixes = np.empty(counts.size, dtype=np.uint64)
    if counts.size:
        prefixes[0] = 0
        if counts.size > 1:
            np.cumsum(counts[:-1], out=prefixes[1:])
    return prefixes


def _release_mmap_pages(array: object) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


def _resolve_scan_backend(
    requested: str,
    *,
    primitive_kinds: tuple[PrimitiveKind, PrimitiveKind] = (
        PrimitiveKind.CONSUMER,
        PrimitiveKind.GRADIENT_REDUCTION,
    ),
) -> tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray] | None]:
    if requested == "cpu":
        return "cpu", None
    try:
        import torch
        from gala_sim.adapters.buffer_decoder import (
            load_buffer_decoder, scan_trace_terminals_cuda,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        decoder = load_buffer_decoder()
    except (ImportError, OSError, RuntimeError) as error:
        if requested == "cuda":
            raise RuntimeError("CUDA trace scanner is unavailable") from error
        return "cpu", None

    dtype = event_dtype()
    primitive_offset = int(dtype.fields["primitive_kind"][1])
    query_offset = int(dtype.fields["query_id"][1])

    def scan(rows: np.ndarray, query_ranges: np.ndarray) -> np.ndarray:
        return scan_trace_terminals_cuda(
            decoder, rows, int(rows.size), int(dtype.itemsize),
            primitive_offset, query_offset, query_ranges,
            int(primitive_kinds[0]), int(primitive_kinds[1]),
        )

    return "cuda", scan
