"""Chunk sink boundary for device-to-host trace transfer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gala_sim.clamp.events import event_dtype


class DeviceTraceSink(Protocol):
    def push(self, events: np.ndarray, dependencies: np.ndarray, payload: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass
class NumpyChunkSink:
    """Preallocated host staging sink; adapters submit complete chunks only."""

    chunk_events: int
    max_inflight_chunks: int
    _events: list[np.ndarray] | None = None
    _dependencies: list[np.ndarray] | None = None
    _payload: list[np.ndarray] | None = None
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.chunk_events <= 0 or self.max_inflight_chunks <= 0:
            raise ValueError("trace chunk limits must be positive")
        self._events = []
        self._dependencies = []
        self._payload = []

    def push(self, events: np.ndarray, dependencies: np.ndarray, payload: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("trace sink is closed")
        if events.dtype != event_dtype() or events.ndim != 1 or events.size > self.chunk_events:
            raise ValueError("trace chunk does not match the configured structured schema")
        if dependencies.dtype != np.dtype("<u8") or dependencies.ndim != 1:
            raise ValueError("trace chunk dependencies must use uint64 schema")
        if payload.dtype != np.dtype("<f4") or payload.ndim != 1:
            raise ValueError("trace chunk payload must use little-endian float32 schema")
        for row in events:
            dep_begin = int(row["dependency_begin"])
            dep_end = dep_begin + int(row["dependency_count"])
            payload_begin = int(row["payload_offset"])
            payload_end = payload_begin + int(row["payload_length"])
            if dep_begin < 0 or dep_end > dependencies.size:
                raise ValueError("trace chunk dependency range is out of bounds")
            if payload_begin < 0 or payload_end > payload.size:
                raise ValueError("trace chunk payload range is out of bounds")
        if len(self._events or []) >= self.max_inflight_chunks:
            raise BufferError("trace sink staging capacity is full")
        self._events.append(np.array(events, copy=True))
        self._dependencies.append(np.array(dependencies, copy=True))
        self._payload.append(np.array(payload, copy=True))

    def close(self) -> None:
        self._closed = True

    def collect(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.close()
        events = []
        dependency_parts = []
        payload_parts = []
        dependency_offset = 0
        payload_offset = 0
        for chunk_events, chunk_dependencies, chunk_payload in zip(
            self._events or [], self._dependencies or [], self._payload or []
        ):
            rebased_events = chunk_events.copy()
            rebased_events["dependency_begin"] += dependency_offset
            rebased_events["payload_offset"] += payload_offset
            events.append(rebased_events)
            dependency_parts.append(chunk_dependencies)
            payload_parts.append(chunk_payload)
            dependency_offset += int(chunk_dependencies.size)
            payload_offset += int(chunk_payload.size)
        return (
            np.concatenate(events or [np.empty(0, dtype=event_dtype())]),
            np.concatenate(dependency_parts or [np.empty(0, dtype=np.dtype("<u8"))]),
            np.concatenate(payload_parts or [np.empty(0, dtype=np.dtype("<f4"))]),
        )
