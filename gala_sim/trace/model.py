"""In-memory representation of a validated structured trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gala_sim.clamp.events import EVENT_SCHEMA_VERSION, dependency_dtype, event_dtype


TRACE_SCHEMA_VERSION = "gala-trace-v1"


@dataclass(frozen=True)
class Trace:
    events: np.ndarray
    dependencies: np.ndarray
    payload: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.events.dtype != event_dtype():
            raise ValueError("trace events do not use the frozen schema")
        if self.dependencies.dtype != dependency_dtype():
            raise ValueError("trace dependencies do not use uint64 schema")
        if self.payload.dtype != np.dtype("<f4"):
            raise ValueError("trace payload must use little-endian float32")
        if self.metadata.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ValueError("trace metadata has an unsupported event schema")

    @property
    def event_count(self) -> int:
        return int(self.events.shape[0])

    def dependency_ids(self, row: np.void) -> np.ndarray:
        begin = int(row["dependency_begin"])
        end = begin + int(row["dependency_count"])
        return self.dependencies[begin:end]

    def payload_values(self, row: np.void) -> np.ndarray:
        begin = int(row["payload_offset"])
        end = begin + int(row["payload_length"])
        return self.payload[begin:end]
