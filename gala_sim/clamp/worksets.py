"""Exact semantic worksets derived from real cache-request events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gala_sim.clamp.events import PrimitiveKind

if TYPE_CHECKING:
    from gala_sim.trace.model import Trace
    from gala_sim.timing.packets import RelationPacketPlan


WORKSET_DTYPE = np.dtype([
    ("event_id", "<u8"),
    ("gaussian_id", "<i8"),
    ("state_version", "<u4"),
    ("ordinal", "<u4"),
    ("total_uses", "<u4"),
    ("remaining_uses", "<u4"),
    ("last_use", "?"),
], align=False)


@dataclass(frozen=True)
class SemanticWorksets:
    """Typed per-request use and release hints with no averaged counts."""

    requests: np.ndarray
    _positions: dict[int, int] = field(repr=False)

    @classmethod
    def from_trace(
        cls, trace: Trace, packet_plan: "RelationPacketPlan | None" = None,
    ) -> "SemanticWorksets":
        mask = trace.events["primitive_kind"] == int(PrimitiveKind.CACHE_REQUEST)
        if packet_plan is not None:
            event_ids = np.flatnonzero(mask)
            physical_heads = np.asarray([
                packet_plan.is_stage_head(int(event_id)) for event_id in event_ids
            ], dtype=np.bool_)
            mask[event_ids] = physical_heads
        source = trace.events[mask]
        requests = np.empty(source.size, dtype=WORKSET_DTYPE)
        if source.size == 0:
            return cls(requests, {})
        requests["event_id"] = source["event_id"]
        requests["gaussian_id"] = source["gaussian_id"]
        requests["state_version"] = source["state_version"]
        order = np.lexsort((source["event_id"], source["state_version"], source["gaussian_id"]))
        gaussians = source["gaussian_id"][order]
        versions = source["state_version"][order]
        group_start = np.empty(source.size, dtype=np.bool_)
        group_start[0] = True
        group_start[1:] = (
            (gaussians[1:] != gaussians[:-1])
            | (versions[1:] != versions[:-1])
        )
        starts = np.flatnonzero(group_start)
        ends = np.r_[starts[1:], source.size]
        for start, end in zip(starts, ends, strict=True):
            group_order = order[start:end]
            total = int(end - start)
            requests["ordinal"][group_order] = np.arange(total, dtype=np.uint32)
            requests["total_uses"][group_order] = total
            requests["remaining_uses"][group_order] = np.arange(
                total, 0, -1, dtype=np.uint32
            )
            requests["last_use"][group_order] = False
            requests["last_use"][group_order[-1]] = True
        requests.setflags(write=False)
        return cls(requests, {
            int(event_id): position
            for position, event_id in enumerate(requests["event_id"])
        })

    def for_event(self, event_id: int) -> np.void:
        position = self._positions.get(event_id)
        if position is None:
            raise KeyError(f"cache request {event_id} has no unique semantic workset entry")
        return self.requests[position]

    @property
    def key_count(self) -> int:
        if self.requests.size == 0:
            return 0
        keys = np.stack((
            self.requests["gaussian_id"],
            self.requests["state_version"].astype(np.int64),
        ), axis=1)
        return int(np.unique(keys, axis=0).shape[0])
