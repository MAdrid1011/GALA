"""Future-visible plans used by resource-constrained cycle Oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.trace.model import Trace

if TYPE_CHECKING:
    from gala_sim.timing.packets import RelationPacketPlan


@dataclass(frozen=True)
class FutureTracePlan:
    """Immutable future information derived from one complete trace.

    The plan never changes readiness or resource availability.  Query issue
    uses ``critical_cycles`` only to order already-ready tasks.  Residency
    uses cache-request positions only when choosing among currently evictable
    records.
    """

    critical_cycles: np.ndarray
    cache_request_positions: dict[tuple[int, int], tuple[int, ...]]
    cache_request_ordinals: dict[int, int]

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        event_service_cycles: np.ndarray,
        dependent_offsets: np.ndarray,
        dependents: np.ndarray,
        packet_plan: "RelationPacketPlan | None" = None,
    ) -> "FutureTracePlan":
        event_count = trace.event_count
        service = np.asarray(event_service_cycles, dtype=np.uint64)
        if service.shape != (event_count,):
            raise ValueError("event service-cycle vector does not cover the trace")
        offsets = np.asarray(dependent_offsets, dtype=np.uint64)
        if offsets.shape != (event_count + 1,):
            raise ValueError("dependent offsets do not cover the trace")

        critical = service.copy()
        for event_id in range(event_count - 1, -1, -1):
            begin = int(offsets[event_id])
            end = int(offsets[event_id + 1])
            if begin != end:
                critical[event_id] += np.max(critical[dependents[begin:end]])
        critical.setflags(write=False)

        positions: dict[tuple[int, int], list[int]] = {}
        ordinals: dict[int, int] = {}
        for row in trace.events:
            if PrimitiveKind(int(row["primitive_kind"])) is not PrimitiveKind.CACHE_REQUEST:
                continue
            event_id = int(row["event_id"])
            if packet_plan is not None and not packet_plan.is_stage_head(event_id):
                continue
            key = (int(row["gaussian_id"]), int(row["state_version"]))
            requests = positions.setdefault(key, [])
            ordinals[event_id] = len(requests)
            requests.append(event_id)
        return cls(
            critical_cycles=critical,
            cache_request_positions={
                key: tuple(event_ids) for key, event_ids in positions.items()
            },
            cache_request_ordinals=ordinals,
        )

    def query_priority(self, event_id: int) -> tuple[int, int]:
        """Heap key that favors the longest remaining dependency path."""

        return -int(self.critical_cycles[event_id]), event_id

    def next_cache_use(self, key: tuple[int, int], *, after_event: int) -> int | None:
        """Return the first real future request for ``key`` after an event."""

        requests = self.cache_request_positions.get(key, ())
        if not requests:
            return None
        ordinal = self.cache_request_ordinals.get(after_event)
        if ordinal is not None and requests[ordinal] == after_event:
            next_ordinal = ordinal + 1
            return requests[next_ordinal] if next_ordinal < len(requests) else None
        position = int(np.searchsorted(requests, after_event, side="right"))
        return requests[position] if position < len(requests) else None
