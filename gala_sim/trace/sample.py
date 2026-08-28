"""Dependency-closed query packets for explicitly scoped quick cycle validation."""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from typing import Callable, Iterator

import numpy as np

from gala_sim.clamp.events import PrimitiveKind, dependency_dtype, event_dtype

from .model import Trace
from .validator import validate_trace


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
            int(PrimitiveKind.CONSUMER), int(PrimitiveKind.GRADIENT_REDUCTION),
        )

    return "cuda", scan
