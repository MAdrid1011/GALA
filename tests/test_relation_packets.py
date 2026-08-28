from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from gala_sim.clamp import (
    PrimitiveKind, ResourceClass, SemanticWorksets, TraceBuilder, TraceEvent,
)
from gala_sim.clamp.events import EVENT_SCHEMA_VERSION, dependency_dtype
from gala_sim.config import load_config
from gala_sim.timing import (
    BufferedVirtualCycleConsumer, CycleConfig, CycleEngine, RelationPacketPlan,
    RelationPacketPlanError, RelationWindowDescriptor, analyze_cycle_lower_bounds,
)
from gala_sim.timing.modules import (
    OwnerGradientTracker, QueryReplayTracker, RelationWindowTracker,
)
from gala_sim.trace import Trace, VirtualQueryEventExpander, VirtualTracePacket


def _virtual_trace(source: VirtualTracePacket, *, query_lanes: int = 8) -> Trace:
    rows = []
    dependencies = []
    dependency_offset = 0
    for packet in VirtualQueryEventExpander(
        max_events=2, relation_query_lanes=query_lanes,
    ).expand(source):
        part = packet.events.copy()
        part["dependency_begin"] += dependency_offset
        rows.append(part)
        dependencies.append(packet.dependencies)
        dependency_offset += packet.dependencies.size
    return Trace(
        np.concatenate(rows),
        np.concatenate(dependencies).astype(dependency_dtype(), copy=False),
        np.empty(0, dtype=np.dtype("<f4")),
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "trace_sample": {
                "result_scope": "quick_cycle_validation",
                "packets": [{
                    "query_base": source.query_base,
                    "query_count": source.query_count,
                    "candidate_count": source.candidate_count,
                }],
            },
        },
    )


def _virtual_event_packets(
    source: VirtualTracePacket, *, event_base: int = 0,
) -> tuple:
    return tuple(VirtualQueryEventExpander(
        max_events=2, relation_query_lanes=8, next_event_id=event_base,
    ).expand(source))


def test_sparse_relation_packet_has_one_physical_stage_per_chain_kind() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    trace = _virtual_trace(VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    ))

    plan = RelationPacketPlan.from_trace(trace, query_lanes=8)

    assert plan.relation_packet_count == 1
    for kind in (
        PrimitiveKind.RELATION,
        PrimitiveKind.CACHE_REQUEST,
        PrimitiveKind.CACHE_RETURN,
        PrimitiveKind.FORWARD,
        PrimitiveKind.ADJOINT,
        PrimitiveKind.GRADIENT_REDUCTION,
    ):
        assert plan.physical_stage_count(kind) == 1
        stage = next(stage for stage in plan.stages if stage.kind is kind)
        assert stage.lanes == (0, 7)
        assert stage.lane_mask == 0b10000001
        assert len(stage.event_ids) == 2
    assert plan.physical_stage_count(PrimitiveKind.QUERY_CLOSE) == 1


def test_atomic_virtual_packet_plan_preserves_global_event_ids() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )

    plan = RelationPacketPlan.from_event_packets(
        _virtual_event_packets(source, event_base=100), query_lanes=8,
    )

    assert plan.event_id_base == 100
    relation_stage = next(
        stage for stage in plan.stages if stage.kind is PrimitiveKind.RELATION
    )
    assert min(relation_stage.event_ids) >= 100
    assert plan.stage_for_event(relation_stage.event_ids[0]) == relation_stage


def test_relation_window_keeps_four_reference_classes_until_final_adjoint() -> None:
    descriptor = RelationWindowDescriptor(
        window_id=0,
        relation_stage_heads=frozenset({1}),
        producer_event_ids=frozenset({0, 1}),
        forward_stage_heads=frozenset({2}),
        consumer_event_ids=frozenset({3}),
        adjoint_event_ids=frozenset({4}),
    )
    tracker = RelationWindowTracker(
        window_capacity=1, relation_capacity=1, relation_banks=1,
    )
    tracker.register(descriptor)

    tracker.issue(0, PrimitiveKind.RELATION_CANDIDATE,
                  physical_stage_head=True, cycle=0)
    tracker.issue(1, PrimitiveKind.RELATION,
                  physical_stage_head=True, cycle=1)
    for event_id in (0, 1, 2, 3):
        tracker.complete_event(event_id)
    assert tracker.snapshot()["relation_windows_live"] == 1
    assert tracker.snapshot()["relation_records_live"] == 1

    tracker.complete_event(4)
    assert tracker.snapshot() == {
        "window_allocations": 1,
        "window_releases": 1,
        "window_peak_occupancy": 1,
        "relation_records_appended": 1,
        "relation_store_peak_records": 1,
        "relation_windows_live": 0,
        "relation_records_live": 0,
    }


def test_relation_window_full_stage_barrier_waits_for_all_prior_work() -> None:
    descriptor = RelationWindowDescriptor(
        window_id=0,
        relation_stage_heads=frozenset({1}),
        producer_event_ids=frozenset({0, 1}),
        forward_stage_heads=frozenset({2, 3}),
        consumer_event_ids=frozenset({4, 5}),
        adjoint_event_ids=frozenset({6}),
    )
    tracker = RelationWindowTracker(
        window_capacity=1, relation_capacity=1, relation_banks=1,
    )
    tracker.register(descriptor)
    assert tracker.full_stage_blocking_reason(
        0, PrimitiveKind.RELATION_CANDIDATE,
    ) is None
    tracker.issue(0, PrimitiveKind.RELATION_CANDIDATE,
                  physical_stage_head=True, cycle=0)

    assert tracker.full_stage_blocking_reason(
        4, PrimitiveKind.CONSUMER,
    ) == "base_forward_stage"
    tracker.complete_event(2)
    assert tracker.full_stage_blocking_reason(
        4, PrimitiveKind.CONSUMER,
    ) == "base_forward_stage"
    tracker.complete_event(3)
    assert tracker.full_stage_blocking_reason(4, PrimitiveKind.CONSUMER) is None

    # Once the complete forward stage retires, adjoint replay can drain behind
    # each consumer; waiting for every consumer would exceed a bounded replay
    # queue for larger windows.
    assert tracker.full_stage_blocking_reason(
        6, PrimitiveKind.ADJOINT,
    ) is None
    tracker.complete_event(4)
    tracker.complete_event(5)
    assert tracker.full_stage_blocking_reason(6, PrimitiveKind.ADJOINT) is None


def test_relation_window_and_record_store_have_independent_backpressure() -> None:
    def descriptor(window_id: int, base: int) -> RelationWindowDescriptor:
        return RelationWindowDescriptor(
            window_id=window_id,
            relation_stage_heads=frozenset({base + 1}),
            producer_event_ids=frozenset({base, base + 1}),
            forward_stage_heads=frozenset({base + 2}),
            consumer_event_ids=frozenset({base + 3}),
            adjoint_event_ids=frozenset({base + 4}),
        )

    windows = RelationWindowTracker(
        window_capacity=1, relation_capacity=2, relation_banks=2,
    )
    windows.register(descriptor(0, 0))
    windows.register(descriptor(1, 10))
    windows.issue(0, PrimitiveKind.RELATION_CANDIDATE,
                  physical_stage_head=True, cycle=0)
    assert windows.blocking_reason(
        10, PrimitiveKind.RELATION_CANDIDATE,
        physical_stage_head=True, cycle=0,
    ) == "relation_window_capacity"

    records = RelationWindowTracker(
        window_capacity=2, relation_capacity=1, relation_banks=2,
    )
    records.register(descriptor(0, 0))
    records.register(descriptor(1, 10))
    records.issue(0, PrimitiveKind.RELATION_CANDIDATE,
                  physical_stage_head=True, cycle=0)
    records.issue(1, PrimitiveKind.RELATION,
                  physical_stage_head=True, cycle=1)
    records.issue(10, PrimitiveKind.RELATION_CANDIDATE,
                  physical_stage_head=True, cycle=1)
    assert records.blocking_reason(
        11, PrimitiveKind.RELATION,
        physical_stage_head=True, cycle=2,
    ) == "relation_store_capacity"


def test_query_replay_queue_releases_only_after_last_adjoint_dispatch() -> None:
    rows = np.empty(3, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = [0, 1, 2]
    rows["iteration_id"] = 7
    rows["query_id"] = 9
    rows["primitive_kind"] = [
        int(PrimitiveKind.CONSUMER),
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.ADJOINT),
    ]
    tracker = QueryReplayTracker(capacity=1)
    tracker.register_rows(rows)

    tracker.reserve_consumer(0)
    tracker.dispatch_adjoint((1,))
    assert tracker.snapshot()["replay_queue_live_entries"] == 1
    tracker.dispatch_adjoint((2,))
    assert tracker.snapshot() == {
        "replay_queue_reservations": 1,
        "replay_queue_releases": 1,
        "replay_queue_peak_entries": 1,
        "replay_queue_live_entries": 0,
    }


def test_owner_gradient_slots_are_partitioned_by_pod_and_owner_cluster() -> None:
    rows = np.empty(4, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(4)
    rows["iteration_id"] = [1, 1, 2, 2]
    # Gaussian 1 maps to Pod 1, local owner cluster 1 in both epochs.
    rows["gaussian_id"] = 1
    rows["state_version"] = [0, 0, 1, 1]
    rows["relation_id"] = [10, 10, 20, 20]
    rows["primitive_kind"] = [
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.GRADIENT_REDUCTION),
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.GRADIENT_REDUCTION),
    ]
    tracker = OwnerGradientTracker(
        pods=4, clusters_per_pod=5, slots_per_cluster=2,
    )
    tracker.register_rows(rows)

    tracker.reserve_adjoint((0,))
    tracker.reserve_adjoint((2,))
    assert tracker.snapshot()["owner_gradient_peak_slots_per_cluster"] == 2
    deadlock = tracker.deadlock_snapshot(
        pending_adjoint_event_ids=(0, 2),
        remaining_dependencies=np.asarray([0, 2, 0, 0]),
    )
    assert deadlock["active_by_cluster"] == {
        6: ((1, 1, 0), (2, 1, 1)),
    }
    assert deadlock["pending_adjoint_count"] == 2
    assert deadlock["active_gradient_reductions"] == (
        {
            "key": (1, 1, 0), "cluster": 6, "event_count": 1,
            "ready_count": 0, "event_sample": ((1, 2),),
        },
        {
            "key": (2, 1, 1), "cluster": 6, "event_count": 1,
            "ready_count": 1, "event_sample": ((3, 0),),
        },
    )
    tracker.complete_gradient((1,))
    assert tracker.snapshot()["owner_gradient_live_slots"] == 1
    tracker.complete_gradient((3,))
    assert tracker.snapshot() == {
        "owner_gradient_slot_reservations": 2,
        "owner_gradient_slot_releases": 2,
        "owner_gradient_peak_slots_per_cluster": 2,
        "owner_gradient_live_slots": 0,
    }


def test_owner_gradient_slot_tracks_only_in_flight_adjoint_relations() -> None:
    rows = np.empty(4, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(4)
    rows["iteration_id"] = 1
    rows["gaussian_id"] = 7
    rows["state_version"] = 0
    rows["relation_id"] = [10, 11, 10, 11]
    rows["primitive_kind"] = [
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.GRADIENT_REDUCTION),
        int(PrimitiveKind.GRADIENT_REDUCTION),
    ]
    tracker = OwnerGradientTracker(
        pods=4, clusters_per_pod=5, slots_per_cluster=2,
    )
    tracker.register_rows(rows)

    tracker.reserve_adjoint((0, 1))
    tracker.reserve_adjoint((0, 1))
    assert tracker.remaining_gradient_by_key[(1, 7, 0)] == 2
    tracker.complete_gradient((2,))
    assert tracker.snapshot()["owner_gradient_live_slots"] == 1
    tracker.complete_gradient((3,))
    assert tracker.snapshot()["owner_gradient_live_slots"] == 0


def test_query_close_packets_follow_rows_and_partial_width() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    masks[0, 16 // 32] |= np.uint32(1 << (16 % 32))
    trace = _virtual_trace(VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=100, query_shape=(2, 10),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    ))

    plan = RelationPacketPlan.from_trace(trace, query_lanes=8)
    close_stages = [
        stage for stage in plan.stages
        if stage.kind is PrimitiveKind.QUERY_CLOSE
    ]

    assert [stage.query_base for stage in close_stages] == [100, 108, 110, 118]
    assert [stage.lane_mask for stage in close_stages] == [0xFF, 0x03, 0xFF, 0x03]


def test_wide_packet_plan_rejects_trace_without_packet_metadata() -> None:
    builder = TraceBuilder()
    relation = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0,
        gaussian_id=4, relation_id=7,
        resource_class=int(ResourceClass.RELATION),
    ))
    trace = builder.finish()

    with pytest.raises(RelationPacketPlanError, match="lacks packet metadata"):
        RelationPacketPlan.from_trace(trace, query_lanes=8)

    scalar = RelationPacketPlan.from_trace(trace, query_lanes=1)
    assert scalar.stage_for_event(relation) is not None


class _Memory:
    def submit(
        self, *, address: int, size_bytes: int, is_write: bool,
        arrival_cycle: int,
    ) -> int:
        return arrival_cycle + size_bytes // 64 + int(is_write)


def test_offline_cycle_engine_charges_sparse_relation_chain_per_physical_packet() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    trace = _virtual_trace(VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    ))
    config = CycleConfig.from_gala(load_config(
        Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    ), _Memory())

    result = CycleEngine(config).run(trace)

    assert result.module_counters["relation_constructor"]["accepted"] == 3
    assert result.module_counters["semantic_cache"]["accepted"] == 2
    assert result.module_counters["shared_sram"]["accepted"] == 2
    assert result.module_counters["compute_pod"]["accepted"] == 3
    assert result.module_counters["bidirectional_query"]["accepted"] == 19
    assert len(result.memory_requests) == 0
    assert result.module_counters["semantic_cache"]["memory_requests"] == 1
    forward_ids = trace.events["event_id"][
        trace.events["primitive_kind"] == int(PrimitiveKind.FORWARD)
    ].tolist()
    assert len(forward_ids) == 2
    assert result.completion_cycles[int(forward_ids[1])] - result.completion_cycles[
        int(forward_ids[0])
    ] == 3


def test_online_and_offline_packet_replay_have_exact_cycles_and_completions() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    trace = _virtual_trace(source)
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    offline = CycleEngine(CycleConfig.from_gala(
        load_config(config_path), _Memory(),
    )).run(trace, collect_compute_telemetry=True)
    session = CycleEngine(CycleConfig.from_gala(
        load_config(config_path), _Memory(),
    )).online_session(
        max_events=2, max_frontier_events=64,
        initial_gaussian_count=5, retain_completion_cycles=True,
        collect_compute_telemetry=True,
    )

    session.accept_query_packet(source)
    session.close_iteration(1)
    online = session.finish()

    assert online.total_cycles == offline.total_cycles
    assert online.completion_cycles == offline.completion_cycles
    assert online.module_counters == offline.module_counters
    assert online.compute_telemetry == offline.compute_telemetry


def test_semantic_worksets_count_physical_packet_readers_not_logical_lanes() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    trace = _virtual_trace(source)
    plan = RelationPacketPlan.from_trace(trace, query_lanes=8)
    worksets = SemanticWorksets.from_trace(trace, plan)
    assert worksets.requests.size == 1
    assert int(worksets.requests["total_uses"][0]) == 1
    assert bool(worksets.requests["last_use"][0])

    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    offline = CycleEngine(CycleConfig.from_gala(
        load_config(config_path), _Memory(),
    ), policy="variant:0101").run(trace)
    sink = BufferedVirtualCycleConsumer(CycleEngine(CycleConfig.from_gala(
        load_config(config_path), _Memory(),
    ), policy="variant:0101").online_session(
        max_events=2, max_frontier_events=64, initial_gaussian_count=5,
    ))
    sink.accept_query_packet(source)
    sink.close_iteration(1)
    online = sink.finish()

    for result in (offline, online):
        assert result.module_counters["semantic_cache"]["workset_uses"] == 1
        assert result.module_counters["semantic_cache"]["workset_releases"] == 1


def test_cycle_bounds_report_physical_packets_and_lane_utilization() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    trace = _virtual_trace(source)
    config = CycleConfig.from_gala(load_config(
        Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    ), _Memory())
    engine = CycleEngine(config)
    base = engine.run(trace)

    report = analyze_cycle_lower_bounds(
        CycleEngine(CycleConfig.from_gala(load_config(
            Path(__file__).parents[1] / "configs/architecture/gala.yaml"
        ), _Memory())),
        trace,
        base_asic_cycles=base.total_cycles,
        targets={"query": 1.01, "residency": 1.01, "full": 1.01},
    )

    diagnostics = report.capacity_diagnostics["relation_packets"]
    assert diagnostics == {
        "logical_relation_lane_events": 2,
        "physical_relation_packets": 1,
        "query_lanes_per_packet": 8,
        "mean_active_lanes": 2.0,
        "lane_utilization": 0.25,
    }
    query = next(item for item in report.scenarios if item.scenario == "query")
    assert query.memory_request_lower_bound == 1
    assert query.memory_byte_lower_bound == 128


def test_base_and_oracles_share_identical_physical_packet_work() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    trace = _virtual_trace(VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    ))
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"

    def run(policy: str):
        return CycleEngine(CycleConfig.from_gala(
            load_config(config_path), _Memory(),
        ), policy=policy).run(trace)

    results = (run("base"), run("query_oracle"), run("residency_oracle"))
    expected_counts = results[0].event_counts
    expected_physical = {
        module: results[0].module_counters[module]["accepted"]
        for module in (
            "relation_constructor", "semantic_cache", "shared_sram", "compute_pod",
        )
    }
    for result in results[1:]:
        assert result.event_counts == expected_counts
        assert {
            module: result.module_counters[module]["accepted"]
            for module in expected_physical
        } == expected_physical
