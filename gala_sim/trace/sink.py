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
        if len(self._events or []) >= self.max_inflight_chunks:
            raise BufferError("trace sink staging capacity is full")
        self._events.append(np.array(events, copy=True))
        self._dependencies.append(np.array(dependencies, copy=True))
        self._payload.append(np.array(payload, copy=True))

    def close(self) -> None:
        self._closed = True

    def collect(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.close()
        return (
            np.concatenate(self._events or [np.empty(0, dtype=event_dtype())]),
            np.concatenate(self._dependencies or [np.empty(0, dtype=np.dtype("<u8"))]),
            np.concatenate(self._payload or [np.empty(0, dtype=np.dtype("<f4"))]),
        )
