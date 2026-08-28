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
    RelationPacketPlanError, analyze_cycle_lower_bounds,
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
        {"schema_version": EVENT_SCHEMA_VERSION},
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
    assert result.module_counters["compute_pod"]["accepted"] == 2
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
    )).run(trace)
    session = CycleEngine(CycleConfig.from_gala(
        load_config(config_path), _Memory(),
    )).online_session(
        max_events=2, max_frontier_events=64,
        initial_gaussian_count=5, retain_completion_cycles=True,
    )

    session.accept_query_packet(source)
    session.close_iteration(1)
    online = session.finish()

    assert online.total_cycles == offline.total_cycles
    assert online.completion_cycles == offline.completion_cycles
    assert online.module_counters == offline.module_counters


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
