from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from gala_sim.clamp.events import (
    PrimitiveKind,
    decode_relation_packet_flags,
    dependency_dtype,
    event_dtype,
)
from gala_sim.timing import RelationPacketPlan, RelationPacketPlanError
from gala_sim.cli import main
from gala_sim.trace import (
    QueryDomain,
    Trace,
    TraceReader,
    TraceWriter,
    VirtualQueryEventExpander,
    VirtualTracePacket,
    derive_quick_relation_packets,
    validate_packet_derivation,
)


def _legacy_quick_trace(*, partial_backward: bool = False) -> Trace:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=20,
        query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        loss_flags=1,
        backward_confirmed=True,
    )
    packets = tuple(VirtualQueryEventExpander(
        max_events=2, relation_query_lanes=8,
    ).expand(source))
    events = np.concatenate([packet.events for packet in packets])
    dependencies = np.concatenate([packet.dependencies for packet in packets])
    if partial_backward:
        remove = np.isin(
            events["primitive_kind"],
            [int(PrimitiveKind.ADJOINT), int(PrimitiveKind.GRADIENT_REDUCTION)],
        ) & (events["query_id"] == 27)
        events, dependencies = _filter_events(events, dependencies, ~remove)
    events = events.copy()
    packetized = np.isin(events["primitive_kind"], [
        int(PrimitiveKind.RELATION),
        int(PrimitiveKind.QUERY_CLOSE),
        int(PrimitiveKind.CACHE_REQUEST),
        int(PrimitiveKind.CACHE_RETURN),
        int(PrimitiveKind.FORWARD),
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.GRADIENT_REDUCTION),
    ])
    events["flags"][packetized] = 0
    return Trace(
        np.asarray(events, dtype=event_dtype()),
        np.asarray(dependencies, dtype=dependency_dtype()),
        np.empty(0, dtype=np.dtype("<f4")),
        {
            "schema_version": "gala-clamp-events-v2",
            "trace_sample": {
                "result_scope": "quick_cycle_validation",
                "formal_performance_eligible": False,
                "quality_eligible": False,
            },
        },
    )


def _filter_events(
    events: np.ndarray, dependencies: np.ndarray, keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    old_ids = np.flatnonzero(keep)
    mapping = np.full(events.size, -1, dtype=np.int64)
    mapping[old_ids] = np.arange(old_ids.size, dtype=np.int64)
    rows = events[old_ids].copy()
    dependency_parts = []
    dependency_counts = []
    for row in rows:
        begin = int(row["dependency_begin"])
        values = dependencies[begin:begin + int(row["dependency_count"])]
        mapped = mapping[values]
        assert np.all(mapped >= 0)
        dependency_parts.append(mapped.astype(dependency_dtype(), copy=False))
        dependency_counts.append(mapped.size)
    counts = np.asarray(dependency_counts, dtype=np.uint64)
    rows["event_id"] = np.arange(rows.size, dtype=np.uint64)
    rows["dependency_begin"] = 0
    if rows.size > 1:
        np.cumsum(counts[:-1], out=rows["dependency_begin"][1:])
    rows["dependency_count"] = counts
    return rows, np.concatenate(dependency_parts).astype(
        dependency_dtype(), copy=False,
    )


def test_quick_packet_derivation_changes_only_flags() -> None:
    source = _legacy_quick_trace()

    derived = derive_quick_relation_packets(
        source, (QueryDomain(1, 20, (1, 8)),), query_lanes=8,
    )

    validate_packet_derivation(source, derived)
    plan = RelationPacketPlan.from_trace(derived, query_lanes=8)
    assert plan.relation_packet_count == 1
    relation_flags = derived.events["flags"][
        derived.events["primitive_kind"] == int(PrimitiveKind.RELATION)
    ]
    assert [decode_relation_packet_flags(int(flags)) for flags in relation_flags] == [
        (0, 0b10000001), (7, 0b10000001),
    ]
    assert derived.metadata["relation_packet_derivation"][
        "formal_performance_eligible"
    ] is False


def test_partial_backward_packets_are_confined_to_explicit_quick_derivation() -> None:
    source = _legacy_quick_trace(partial_backward=True)
    derived = derive_quick_relation_packets(
        source, (QueryDomain(1, 20, (1, 8)),), query_lanes=8,
    )

    plan = RelationPacketPlan.from_trace(derived, query_lanes=8)
    adjoint = next(
        stage for stage in plan.stages if stage.kind is PrimitiveKind.ADJOINT
    )
    assert adjoint.lanes == (0,)
    assert adjoint.lane_mask == 1
    assert derived.metadata["relation_packet_derivation"][
        "backward_subset_stage_packets"
    ] == 2

    formal_metadata = dict(derived.metadata)
    formal_metadata.pop("trace_sample")
    formal = Trace(
        derived.events, derived.dependencies, derived.payload, formal_metadata,
    )
    with pytest.raises(RelationPacketPlanError, match="changed physical packet metadata"):
        RelationPacketPlan.from_trace(formal, query_lanes=8)


def test_packet_derivation_rejects_formal_trace() -> None:
    source = _legacy_quick_trace()
    formal = Trace(
        source.events,
        source.dependencies,
        source.payload,
        {"schema_version": "gala-clamp-events-v2"},
    )

    with pytest.raises(ValueError, match="restricted to sampled quick traces"):
        derive_quick_relation_packets(
            formal, (QueryDomain(1, 20, (1, 8)),), query_lanes=8,
        )


def test_cli_packetizes_to_a_distinct_quick_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "derived"
    TraceWriter().write(_legacy_quick_trace(), source, validate=False)

    assert main([
        "trace-packetize",
        "--trace", str(source),
        "--output", str(output),
        "--query-domain", "1:20:1x8",
        "--query-lanes", "8",
    ]) == 0

    report = __import__("json").loads(capsys.readouterr().out)
    assert report["physical_relation_packets"] == 1
    derived = TraceReader().read(output, validate=False)
    validate_packet_derivation(_legacy_quick_trace(), derived)
