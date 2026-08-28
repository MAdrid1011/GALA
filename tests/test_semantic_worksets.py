from __future__ import annotations

from gala_sim.clamp import (
    PrimitiveKind, ResourceClass, SemanticWorksets, TraceBuilder, TraceEvent,
)


def test_semantic_worksets_preserve_every_real_request_and_release() -> None:
    builder = TraceBuilder()
    event_ids = []
    for query_id, gaussian_id in ((0, 7), (1, 8), (2, 7)):
        event_ids.append(builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=query_id,
            gaussian_id=gaussian_id, state_version=3, data_bytes=64,
            address_token=gaussian_id * 64,
            resource_class=int(ResourceClass.CACHE),
        )))
    worksets = SemanticWorksets.from_trace(builder.finish())
    first = worksets.for_event(event_ids[0])
    last = worksets.for_event(event_ids[2])
    assert int(first["total_uses"]) == 2
    assert int(first["remaining_uses"]) == 2
    assert bool(first["last_use"]) is False
    assert int(last["remaining_uses"]) == 1
    assert bool(last["last_use"]) is True
    assert worksets.key_count == 2
