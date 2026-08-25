"""Append-only trace builder used by model adapters and unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .events import EVENT_SCHEMA_VERSION, TraceEvent, dependency_dtype, event_dtype


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

    def finish(self, *, metadata: dict[str, object] | None = None):
        from gala_sim.trace.model import Trace

        rows = np.asarray(self._rows, dtype=event_dtype())
        if not self._rows:
            rows = np.empty(0, dtype=event_dtype())
        return Trace(
            events=rows,
            dependencies=np.asarray(self._dependencies, dtype=dependency_dtype()),
            payload=np.asarray(self._payload, dtype=np.float32),
            metadata={"schema_version": EVENT_SCHEMA_VERSION, **(metadata or {})},
        )
