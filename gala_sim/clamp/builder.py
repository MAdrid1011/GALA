"""Append-only trace builder used by model adapters and unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from array import array
import json
from typing import Iterable
from pathlib import Path

import numpy as np

from .events import EVENT_SCHEMA_VERSION, TraceEvent, dependency_dtype, event_dtype


@dataclass(frozen=True)
class TraceChunkManifest:
    """Manifest for a capture that remains in bounded columnar chunks."""

    root: Path
    event_count: int
    dependency_count: int
    payload_count: int
    metadata: dict[str, object]
    event_chunks: tuple[str, ...]
    dependency_chunks: tuple[str, ...]
    payload_chunks: tuple[str, ...]
    storage_format: str = "npy_chunks"


@dataclass
class TraceBuilder:
    """Build a trace without Python callbacks in the eventual hot path."""

    _rows: list[tuple[int, ...]] = field(default_factory=list)
    _dependencies: list[int] = field(default_factory=list)
    _payload: list[float] = field(default_factory=list)
    _next_event_id: int = 0

    def emit(self, event: TraceEvent | None = None, *, dependencies: Iterable[int] = (),
             payload: Iterable[float] = ()) -> int:
        event = event or TraceEvent()
        deps = tuple(int(item) for item in dependencies)
        payload_values = tuple(float(item) for item in payload)
        values = dict(event.values)
        values["event_id"] = self._next_event_id
        values["dependency_begin"] = len(self._dependencies)
        values["dependency_count"] = len(deps)
        values["payload_offset"] = len(self._payload)
        values["payload_length"] = len(payload_values)
        self._dependencies.extend(deps)
        self._payload.extend(payload_values)
        self._rows.append(TraceEvent(**values).as_tuple())
        self._next_event_id += 1
        return self._next_event_id - 1

    def finish(self, *, metadata: dict[str, object] | None = None,
               materialize: bool = True):
        from gala_sim.trace.model import Trace

        if not materialize:
            raise ValueError("TraceBuilder cannot finish without materializing its in-memory rows")
        rows = np.asarray(self._rows, dtype=event_dtype())
        if not self._rows:
            rows = np.empty(0, dtype=event_dtype())
        return Trace(
            events=rows,
            dependencies=np.asarray(self._dependencies, dtype=dependency_dtype()),
            payload=np.asarray(self._payload, dtype=np.float32),
            metadata={"schema_version": EVENT_SCHEMA_VERSION, **(metadata or {})},
        )


@dataclass
class ChunkedTraceBuilder:
    """Write events directly into bounded structured chunks for real captures."""

    chunk_events: int
    chunk_root: Path | None = None
    stream_only: bool = False
    _chunks: list[np.ndarray] = field(default_factory=list)
    _chunk_paths: list[Path] = field(default_factory=list, init=False)
    _current: np.ndarray | None = field(default=None, init=False, repr=False)
    _current_size: int = field(default=0, init=False, repr=False)
    _dependencies: array = field(default_factory=lambda: array("Q"))
    _payload: array = field(default_factory=lambda: array("f"))
    _current_dependencies: array = field(default_factory=lambda: array("Q"), init=False, repr=False)
    _current_payload: array = field(default_factory=lambda: array("f"), init=False, repr=False)
    _dependency_count: int = field(default=0, init=False, repr=False)
    _payload_count: int = field(default=0, init=False, repr=False)
    _dependency_chunk_paths: list[Path] = field(default_factory=list, init=False, repr=False)
    _payload_chunk_paths: list[Path] = field(default_factory=list, init=False, repr=False)
    _next_event_id: int = 0

    def __post_init__(self) -> None:
        if self.chunk_events <= 0:
            raise ValueError("trace chunk capacity must be positive")
        if self.stream_only and self.chunk_root is None:
            raise ValueError("stream-only mode requires a chunk root")
        if self.chunk_root is not None:
            self.chunk_root = Path(self.chunk_root)
            self.chunk_root.mkdir(parents=True, exist_ok=True)
            if self.stream_only:
                for suffix in ("events.raw", "dependencies.raw", "payload.raw"):
                    (self.chunk_root.parent / suffix).unlink(missing_ok=True)

    @property
    def next_event_id(self) -> int:
        return self._next_event_id

    def _flush_current(self) -> None:
        if self._current is None:
            return
        if self.chunk_root is None:
            self._chunks.append(self._current[:self._current_size].copy())
        elif self.stream_only:
            root = self.chunk_root.parent
            with (root / "events.raw").open("ab") as output:
                output.write(self._current[:self._current_size].tobytes())
            with (root / "dependencies.raw").open("ab") as output:
                output.write(np.asarray(self._current_dependencies, dtype=dependency_dtype()).tobytes())
            with (root / "payload.raw").open("ab") as output:
                output.write(np.asarray(self._current_payload, dtype=np.dtype("<f4")).tobytes())
        else:
            chunk_index = len(self._chunk_paths)
            path = self.chunk_root / f"events_{chunk_index:08d}.npy"
            np.save(path, self._current[:self._current_size], allow_pickle=False)
            self._chunk_paths.append(path)
            dependency_path = self.chunk_root / f"dependencies_{chunk_index:08d}.npy"
            payload_path = self.chunk_root / f"payload_{chunk_index:08d}.npy"
            np.save(
                dependency_path,
                np.asarray(self._current_dependencies, dtype=dependency_dtype()),
                allow_pickle=False,
            )
            np.save(
                payload_path,
                np.asarray(self._current_payload, dtype=np.dtype("<f4")),
                allow_pickle=False,
            )
            self._dependency_chunk_paths.append(dependency_path)
            self._payload_chunk_paths.append(payload_path)
        self._current = None
        self._current_size = 0
        self._current_dependencies = array("Q")
        self._current_payload = array("f")

    def emit(self, event: TraceEvent | None = None, *, dependencies: Iterable[int] = (),
             payload: Iterable[float] = ()) -> int:
        event = event or TraceEvent()
        deps = tuple(int(item) for item in dependencies)
        payload_values = tuple(float(item) for item in payload)
        if self._current is not None and self._current_size == self.chunk_events:
            self._flush_current()
        if self._current is None:
            self._current = np.empty(self.chunk_events, dtype=event_dtype())
            self._current_size = 0
        values = dict(event.values)
        if self.chunk_root is None:
            dependency_begin = len(self._dependencies)
            payload_offset = len(self._payload)
        else:
            dependency_begin = self._dependency_count
            payload_offset = self._payload_count
        values.update({
            "event_id": self._next_event_id,
            "dependency_begin": dependency_begin,
            "dependency_count": len(deps),
            "payload_offset": payload_offset,
            "payload_length": len(payload_values),
        })
        if self.chunk_root is None:
            self._dependencies.extend(deps)
            self._payload.extend(payload_values)
        else:
            self._current_dependencies.extend(deps)
            self._current_payload.extend(payload_values)
            self._dependency_count += len(deps)
            self._payload_count += len(payload_values)
        self._current[self._current_size] = TraceEvent(**values).as_tuple()
        self._current_size += 1
        self._next_event_id += 1
        return self._next_event_id - 1

    def emit_batch(
        self,
        events: np.ndarray,
        *,
        dependencies: np.ndarray | None = None,
        dependency_counts: np.ndarray | None = None,
        payload: np.ndarray | None = None,
        payload_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Append structured events and their row-major ragged columns."""

        if events.dtype != event_dtype() or events.ndim != 1:
            raise ValueError("event batch does not use the frozen structured schema")
        event_count = int(events.size)
        dependencies = (
            np.empty(0, dtype=dependency_dtype())
            if dependencies is None else np.asarray(dependencies)
        )
        payload = (
            np.empty(0, dtype=np.dtype("<f4"))
            if payload is None else np.asarray(payload)
        )
        if dependencies.dtype != dependency_dtype() or dependencies.ndim != 1:
            raise ValueError("event batch dependencies must use uint64 schema")
        if payload.dtype != np.dtype("<f4") or payload.ndim != 1:
            raise ValueError("event batch payload must use little-endian float32 schema")
        dependency_counts = (
            np.zeros(event_count, dtype=np.int64)
            if dependency_counts is None else np.asarray(dependency_counts, dtype=np.int64)
        )
        payload_counts = (
            np.zeros(event_count, dtype=np.int64)
            if payload_counts is None else np.asarray(payload_counts, dtype=np.int64)
        )
        if dependency_counts.shape != (event_count,) or payload_counts.shape != (event_count,):
            raise ValueError("event batch count arrays have an invalid shape")
        if bool((dependency_counts < 0).any()) or bool((payload_counts < 0).any()):
            raise ValueError("event batch count arrays cannot be negative")
        if int(dependency_counts.sum()) != int(dependencies.size):
            raise ValueError("event batch dependency counts do not match dependencies")
        if int(payload_counts.sum()) != int(payload.size):
            raise ValueError("event batch payload counts do not match payload")
        dependency_prefix = np.empty(event_count + 1, dtype=np.int64)
        dependency_prefix[0] = 0
        np.cumsum(dependency_counts, out=dependency_prefix[1:])
        payload_prefix = np.empty(event_count + 1, dtype=np.int64)
        payload_prefix[0] = 0
        np.cumsum(payload_counts, out=payload_prefix[1:])
        event_ids = np.arange(
            self._next_event_id, self._next_event_id + event_count, dtype=np.int64
        )
        dependency_base = (
            len(self._dependencies) if self.chunk_root is None else self._dependency_count
        )
        payload_base = len(self._payload) if self.chunk_root is None else self._payload_count

        if self.chunk_root is None:
            self._dependencies.frombytes(dependencies.tobytes())
            self._payload.frombytes(payload.tobytes())
        start = 0
        while start < event_count:
            if self._current is None:
                self._current = np.empty(self.chunk_events, dtype=event_dtype())
                self._current_size = 0
            capacity = self.chunk_events - self._current_size
            end = min(start + capacity, event_count)
            rows = np.array(events[start:end], copy=True)
            rows["event_id"] = event_ids[start:end]
            rows["dependency_begin"] = dependency_base + dependency_prefix[start:end]
            rows["dependency_count"] = dependency_counts[start:end]
            rows["payload_offset"] = payload_base + payload_prefix[start:end]
            rows["payload_length"] = payload_counts[start:end]
            dep_begin = int(dependency_prefix[start])
            dep_end = int(dependency_prefix[end])
            payload_begin = int(payload_prefix[start])
            payload_end = int(payload_prefix[end])
            if self.chunk_root is not None:
                self._current_dependencies.frombytes(
                    dependencies[dep_begin:dep_end].tobytes()
                )
                self._current_payload.frombytes(
                    payload[payload_begin:payload_end].tobytes()
                )
                self._dependency_count += dep_end - dep_begin
                self._payload_count += payload_end - payload_begin
            batch_size = end - start
            self._current[self._current_size:self._current_size + batch_size] = rows
            self._current_size += batch_size
            if self._current_size == self.chunk_events:
                self._flush_current()
            start = end
        self._next_event_id += event_count
        return event_ids

    def finish(self, *, metadata: dict[str, object] | None = None,
               materialize: bool = True):
        from gala_sim.trace.model import Trace

        if self.stream_only and materialize:
            raise ValueError("stream-only builder cannot materialize merged arrays")
        if self.chunk_root is not None:
            self._flush_current()
            event_metadata = {"schema_version": EVENT_SCHEMA_VERSION, **(metadata or {})}
            if not materialize:
                if self.stream_only:
                    root = self.chunk_root.parent
                    manifest = TraceChunkManifest(
                        root=root,
                        event_count=self._next_event_id,
                        dependency_count=self._dependency_count,
                        payload_count=self._payload_count,
                        metadata=event_metadata,
                        event_chunks=("events.raw",),
                        dependency_chunks=("dependencies.raw",),
                        payload_chunks=("payload.raw",),
                        storage_format="raw_columns",
                    )
                else:
                    manifest = TraceChunkManifest(
                        root=self.chunk_root,
                        event_count=sum(
                            int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
                            for path in self._chunk_paths
                        ),
                        dependency_count=self._dependency_count,
                        payload_count=self._payload_count,
                        metadata=event_metadata,
                        event_chunks=tuple(path.name for path in self._chunk_paths),
                        dependency_chunks=tuple(path.name for path in self._dependency_chunk_paths),
                        payload_chunks=tuple(path.name for path in self._payload_chunk_paths),
                    )
                manifest_path = self.chunk_root.parent / "chunk_manifest.json"
                manifest_path.write_text(
                    json.dumps({
                        "schema_version": "gala-trace-chunks-v1",
                        "event_schema_version": EVENT_SCHEMA_VERSION,
                        "event_count": manifest.event_count,
                        "dependency_count": manifest.dependency_count,
                        "payload_count": manifest.payload_count,
                        "metadata": manifest.metadata,
                        "event_chunks": list(manifest.event_chunks),
                        "dependency_chunks": list(manifest.dependency_chunks),
                        "payload_chunks": list(manifest.payload_chunks),
                        "storage_format": manifest.storage_format,
                    }, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                return manifest
            events_path = self.chunk_root.parent / "events.npy"
            dependencies_path = self.chunk_root.parent / "dependencies.npy"
            payload_path = self.chunk_root.parent / "payload.npy"
            event_chunks = [np.load(path, mmap_mode="r", allow_pickle=False)
                            for path in self._chunk_paths]
            total_events = sum(int(chunk.shape[0]) for chunk in event_chunks)
            events = _merge_chunk_arrays(
                self._chunk_paths, events_path, event_dtype(), total_events
            )
            dependencies = _merge_chunk_arrays(
                self._dependency_chunk_paths, dependencies_path,
                dependency_dtype(), self._dependency_count,
            )
            payload = _merge_chunk_arrays(
                self._payload_chunk_paths, payload_path,
                np.dtype("<f4"), self._payload_count,
            )
            for path in (*self._chunk_paths, *self._dependency_chunk_paths,
                         *self._payload_chunk_paths):
                path.unlink(missing_ok=True)
            try:
                self.chunk_root.rmdir()
            except OSError:
                pass
        else:
            self._flush_current()
            if not self._chunks:
                events = np.empty(0, dtype=event_dtype())
            elif len(self._chunks) == 1:
                events = self._chunks[0].copy()
            else:
                events = np.concatenate(self._chunks)
            dependencies = np.frombuffer(self._dependencies, dtype=dependency_dtype()).copy()
            payload = np.frombuffer(self._payload, dtype=np.dtype("<f4")).copy()
        return Trace(
            events=events,
            dependencies=dependencies,
            payload=payload,
            metadata={"schema_version": EVENT_SCHEMA_VERSION, **(metadata or {})},
        )


def _merge_chunk_arrays(
    paths: list[Path], output_path: Path, dtype: np.dtype, total_size: int
) -> np.ndarray:
    """Merge bounded `.npy` chunks into a memory-mapped final column."""
    if not paths:
        np.save(output_path, np.empty(0, dtype=dtype), allow_pickle=False)
        return np.load(output_path, mmap_mode="r", allow_pickle=False)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=dtype, shape=(total_size,)
    )
    begin = 0
    for path in paths:
        chunk = np.load(path, mmap_mode="r", allow_pickle=False)
        end = begin + int(chunk.shape[0])
        output[begin:end] = chunk
        begin = end
    output.flush()
    del output
    return np.load(output_path, mmap_mode="r", allow_pickle=False)
