from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import PrimitiveKind, ResourceClass, TraceBuilder, TraceEvent
from gala_sim.config import load_config
from gala_sim.timing import (
    ComputeStage,
    ComputePathProfile,
    ComputeTemplateProfile,
    CycleConfig,
    CycleConfigurationError,
    CycleEngine,
    ModuleTiming,
)
from gala_sim.timing.engine import _ReadyCandidateQueue
from gala_sim.timing.modules import (
    ComputePod, CounterBlock, OwnerGradientTracker, QueryReplayTracker,
)
from gala_sim.timing.packets import PhysicalPacketStage


class _Memory:
    def submit(self, *, address: int, size_bytes: int, is_write: bool,
               arrival_cycle: int) -> int:
        return arrival_cycle


def _production_config() -> CycleConfig:
    root = Path(__file__).parents[1]
    return CycleConfig.from_gala(
        load_config(root / "configs/architecture/gala.yaml"), _Memory()
    )


def test_production_compute_profiles_have_audited_path_latencies() -> None:
    config = _production_config()
    assert config.compute_templates is not None
    assert config.compute_resource_capacities == {
        "pods": 4,
        "clusters_per_pod": 5,
        "clusters": 20,
        "cluster_issue": 40,
        "fma_groups": 80,
        "transcendental_lanes": 40,
        "reduction_trees": 20,
        "microcontext_slots": 80,
        "feedback_lanes": 60,
    }
    assert config.query_reduction_banks == 64
    assert config.query_partial_sum_groups_per_bank == 4
    assert config.query_loss_fma_lanes == 32
    assert config.query_loss_queries_per_cycle == 16
    assert config.query_adjoint_replay_lanes == 8
    assert config.query_replay_queue_entries == 256
    assert config.query_relation_window_entries == 256
    assert config.query_relation_store_records == 16_384
    assert config.query_volume_banks == 16
    assert config.query_relation_store_banks == 16
    assert config.relation_support_lanes == 8
    assert config.owner_gradient_slots_per_cluster == 2
    assert CycleEngine(config)._module_issue_ports("bidirectional_query") == 312
    assert CycleEngine(config)._module_issue_ports("relation_constructor") == 24
    assert config.compute_templates[1].latency_for("forward") == 17
    assert config.compute_templates[1].latency_for("adjoint") == 27
    assert config.compute_templates[2].latency_for("forward") == 27
    assert config.compute_templates[2].latency_for("adjoint") == 47
    assert config.compute_templates[1].latency_for("gradient_reduction") == 4
    raster_forward = config.compute_templates[1].path_for("forward")
    voxel_forward = config.compute_templates[2].path_for("forward")
    raster_adjoint = config.compute_templates[1].path_for("adjoint")
    voxel_adjoint = config.compute_templates[2].path_for("adjoint")
    assert [raster_forward.packet_completion_offset(lane) for lane in range(8)] == [
        17, 17, 18, 18, 19, 19, 20, 20,
    ]
    assert [voxel_forward.packet_completion_offset(lane) for lane in range(8)] == list(
        range(27, 35)
    )
    assert {raster_adjoint.packet_completion_offset(lane) for lane in range(8)} == {34}
    assert {voxel_adjoint.packet_completion_offset(lane) for lane in range(8)} == {68}


def test_production_compute_profile_rejects_unknown_template() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, relation_id=0, reduction_key=0, template_id=3,
        resource_class=int(ResourceClass.COMPUTE),
    ))
    with pytest.raises(CycleConfigurationError, match="no ComputePod profile"):
        CycleEngine(replace(
            _production_config(), relation_query_lanes=1,
        )).run(builder.finish())


def test_query_and_compute_paths_follow_bidirectional_hardware_contract() -> None:
    assert CycleEngine._stages_for(PrimitiveKind.FORWARD) == (
        "fusion_issue", "compute_pod", "bidirectional_query",
    )
    assert CycleEngine._stages_for(PrimitiveKind.CONSUMER) == (
        "fusion_issue", "bidirectional_query",
    )
    assert CycleEngine._stages_for(PrimitiveKind.ADJOINT) == (
        "fusion_issue", "bidirectional_query", "compute_pod",
    )
    assert CycleEngine._stages_for(PrimitiveKind.GRADIENT_REDUCTION) == (
        "compute_pod",
    )


def test_query_reduction_banks_issue_independently_and_serialize_aliases() -> None:
    def run(query_ids: tuple[int, int]):
        builder = TraceBuilder()
        for query_id in query_ids:
            builder.emit(TraceEvent(
                primitive_kind=int(PrimitiveKind.QUERY_REDUCTION),
                query_id=query_id,
                reduction_key=query_id,
                resource_class=int(ResourceClass.QUERY),
            ))
        config = replace(_production_config(), relation_query_lanes=1)
        return CycleEngine(config).run(builder.finish(), validate_input=False)

    independent = run((0, 1))
    # Four interleaved partial groups share a bank; query 256 wraps to the
    # same bank/group slot as query 0.
    aliased = run((0, 256))

    assert independent.completion_cycles[0] == independent.completion_cycles[1]
    assert aliased.completion_cycles[1] == aliased.completion_cycles[0] + 1
    assert aliased.module_counters["bidirectional_query"]["bank_conflicts"] > 0


def test_packet_forward_reserves_only_the_arriving_query_lane() -> None:
    engine = CycleEngine(_production_config())
    lanes = [0] * engine._module_issue_ports("bidirectional_query")
    dtype = TraceBuilder().finish().events.dtype
    row = np.zeros((), dtype=dtype)
    row["primitive_kind"] = int(PrimitiveKind.FORWARD)
    row["query_id"] = 0
    packet = PhysicalPacketStage(
        stage_id=0,
        kind=PrimitiveKind.FORWARD,
        event_ids=(10, 11),
        lanes=(0, 1),
        lane_mask=0b11,
        query_base=0,
        relation_packet_id=0,
    )

    # The second packet lane has not arrived yet and its busy reduction slot
    # cannot block the first lane from issuing.
    lanes[1] = 1
    assert engine._query_resource_allocation(
        lanes, row, PrimitiveKind.FORWARD, packet, 0,
    ) == (0,)


def test_query_datapaths_allocate_loss_replay_and_query_volume_ports() -> None:
    engine = CycleEngine(_production_config())
    lanes = [0] * engine._module_issue_ports("bidirectional_query")
    dtype = TraceBuilder().finish().events.dtype

    def row(kind: PrimitiveKind, query_id: int) -> object:
        value = np.zeros((), dtype=dtype)
        value["primitive_kind"] = int(kind)
        value["query_id"] = query_id
        return value

    consumer = engine._query_resource_allocation(
        lanes, row(PrimitiveKind.CONSUMER, 3), PrimitiveKind.CONSUMER, None, 0,
    )
    assert consumer == (256, 283, 299)
    for lane in consumer:
        lanes[lane] = 1

    # A second query can use another loss slot and another SRAM bank.
    assert engine._query_resource_allocation(
        lanes, row(PrimitiveKind.CONSUMER, 4), PrimitiveKind.CONSUMER, None, 0,
    ) == (257, 284, 300)
    # An adjoint read to query 3 conflicts with the consumer read port, while
    # the independent write port remains legal in the same cycle.
    assert engine._query_resource_allocation(
        lanes, row(PrimitiveKind.ADJOINT, 3), PrimitiveKind.ADJOINT, None, 0,
    ) is None
    read_busy_only = [0] * len(lanes)
    read_busy_only[283] = 1
    assert engine._query_resource_allocation(
        read_busy_only, row(PrimitiveKind.QUERY_REDUCTION, 3),
        PrimitiveKind.QUERY_REDUCTION, None, 0,
    ) == (12, 299)

    # A full replay packet waits when any of the eight lanes are busy; that is
    # backpressure, not an invalid packet-width error.
    replay_busy = [0] * len(lanes)
    replay_busy[272] = 1
    packet = type("Packet", (), {"query_base": 0, "lanes": tuple(range(8))})()
    assert engine._query_resource_allocation(
        replay_busy, row(PrimitiveKind.ADJOINT, 0),
        PrimitiveKind.ADJOINT, packet, 0,
    ) is None


def test_compute_pod_reserves_twenty_independent_cluster_issue_slots() -> None:
    profile = ComputeTemplateProfile(1, {
        "forward": ComputePathProfile((
            ComputeStage("TRANSFORM", latency=3, fma_groups=4),
        )),
    })
    pod = ComputePod(
        "compute_pod",
        ModuleTiming(latency=1, initiation_interval=1, queue_capacity=256,
                     ports=1, banks=4),
        CounterBlock(),
        template_profiles={1: profile},
        resource_capacities={
            "clusters": 20, "cluster_issue": 20, "fma_groups": 80,
            "transcendental_lanes": 40, "reduction_trees": 20,
            "microcontext_slots": 640, "feedback_lanes": 60,
        },
    )
    plans = []
    for _ in range(20):
        plan = pod.reservation_plan(1, PrimitiveKind.FORWARD, 0)
        assert pod.can_reserve(plan, 0)
        pod.reserve(plan)
        plans.append(plan)
    blocked = pod.reservation_plan(1, PrimitiveKind.FORWARD, 0)
    assert not pod.can_reserve(blocked, 0)
    assert {
        int(resource.partition(":")[2])
        for plan in plans for resource, cycle, _ in plan
        if resource.startswith("cluster_issue:") and cycle == 0
    } == set(range(20))


def test_compute_pod_retires_resource_points_without_full_table_scan() -> None:
    profile = ComputeTemplateProfile(1, {
        "forward": ComputePathProfile((
            ComputeStage("TRANSFORM", latency=2, fma_groups=1),
        )),
    })
    pod = ComputePod(
        "compute_pod",
        ModuleTiming(latency=1, initiation_interval=1, queue_capacity=16,
                     ports=1, banks=1),
        CounterBlock(),
        template_profiles={1: profile},
        resource_capacities={
            "clusters": 1, "cluster_issue": 1, "fma_groups": 1,
            "microcontext_slots": 1,
        },
    )
    first = pod.reservation_plan(1, PrimitiveKind.FORWARD, 0)
    assert pod.can_reserve(first, 0)
    pod.reserve(first)
    assert not pod.can_reserve(first, 0)
    assert pod.can_reserve(first, 3)
    pod._discard_retired(3)
    assert not pod._resource_use
    assert not pod._retirement_buckets
    assert not pod._retirement_cycles


def test_ready_selection_finds_reusable_owner_epoch_beyond_fifo_head() -> None:
    engine = CycleEngine(_production_config())
    rows = np.empty(258, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(258)
    rows["iteration_id"] = 1
    rows["primitive_kind"] = int(PrimitiveKind.ADJOINT)
    rows["state_version"] = 0
    rows["relation_id"] = np.arange(258)
    # All keys target owner cluster zero. The final event reuses the first
    # active epoch after 255 blocked keys.
    rows["gaussian_id"] = np.arange(258) * 20
    rows[257]["gaussian_id"] = rows[0]["gaussian_id"]
    tracker = OwnerGradientTracker(
        pods=4, clusters_per_pod=5, slots_per_cluster=2,
    )
    tracker.register_rows(rows)
    tracker.reserve_adjoint((0,))
    tracker.reserve_adjoint((1,))
    queue = _ReadyCandidateQueue()
    for event_id in range(2, 258):
        queue.push((event_id, 2))

    selected = engine._pop_ready_candidates(
        queue, "compute_pod", row_for=rows.__getitem__,
        physical_stage_for=lambda _event_id: None,
        owner_gradients=tracker,
        query_replay=None,
    )

    assert selected == [(257, 2)]
    assert len(queue) == 255


def test_ready_index_matches_full_scan_across_owner_state_changes() -> None:
    rows = np.empty(14, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(14)
    rows["iteration_id"] = 1
    rows["primitive_kind"] = int(PrimitiveKind.ADJOINT)
    rows["state_version"] = 0
    rows["relation_id"] = np.arange(14)
    rows["gaussian_id"] = np.arange(14) * 20
    rows[12]["gaussian_id"] = rows[0]["gaussian_id"]
    rows[13]["primitive_kind"] = int(PrimitiveKind.FORWARD)
    tracker = OwnerGradientTracker(
        pods=4, clusters_per_pod=5, slots_per_cluster=2,
    )
    tracker.register_rows(rows)
    first_key = tracker.key_for_adjoint(0)
    second_key = tracker.key_for_adjoint(1)
    tracker.active_by_cluster[0] = {first_key, second_key}
    queue = _ReadyCandidateQueue()
    remaining = [(event_id, 2) for event_id in range(2, 14)]
    for candidate in remaining:
        queue.push(candidate)

    def acceptable(candidate: tuple[int, int]) -> bool:
        event_id, _stage = candidate
        if PrimitiveKind(int(rows[event_id]["primitive_kind"])) is not PrimitiveKind.ADJOINT:
            return True
        return not tracker.blocks_adjoint((event_id,))

    for active in (
        {first_key, second_key},
        {tracker.key_for_adjoint(3)},
        set(),
    ):
        if active:
            tracker.active_by_cluster[0] = active
        else:
            tracker.active_by_cluster.clear()
        expected = [candidate for candidate in sorted(remaining) if acceptable(candidate)][:3]
        selected = queue.pop_acceptable(
            3, "compute_pod", row_for=rows.__getitem__,
            physical_stage_for=lambda _event_id: None,
            owner_gradients=tracker,
            query_replay=None,
        )
        assert selected == expected
        for candidate in selected:
            remaining.remove(candidate)
    assert list(queue) == sorted(remaining)


def test_ready_index_checks_multi_owner_packet_with_original_capacity_rule() -> None:
    rows = np.empty(5, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(5)
    rows["iteration_id"] = 1
    rows["primitive_kind"] = int(PrimitiveKind.ADJOINT)
    rows["state_version"] = 0
    rows["relation_id"] = np.arange(5)
    rows["gaussian_id"] = np.arange(5) * 20
    tracker = OwnerGradientTracker(
        pods=4, clusters_per_pod=5, slots_per_cluster=2,
    )
    tracker.register_rows(rows)
    tracker.reserve_adjoint((0,))
    packet = PhysicalPacketStage(
        stage_id=0, kind=PrimitiveKind.ADJOINT, event_ids=(1, 2),
        lanes=(0, 1), lane_mask=0b11, query_base=0, relation_packet_id=0,
    )
    queue = _ReadyCandidateQueue()
    queue.push((1, 2))
    queue.push((3, 2))

    selected = queue.pop_acceptable(
        3, "compute_pod", row_for=rows.__getitem__,
        physical_stage_for=lambda event_id: packet if event_id in (1, 2) else None,
        owner_gradients=tracker,
        query_replay=None,
    )

    assert selected == [(3, 2)]
    assert list(queue) == [(1, 2)]


def test_ready_index_reaches_adjoint_behind_full_replay_queue() -> None:
    rows = np.empty(4, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(4)
    rows["iteration_id"] = 1
    rows["query_id"] = [0, 1, 0, 1]
    rows["primitive_kind"] = [
        int(PrimitiveKind.CONSUMER),
        int(PrimitiveKind.CONSUMER),
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.ADJOINT),
    ]
    replay = QueryReplayTracker(capacity=1)
    replay.register_rows(rows)
    replay.reserve_consumer(0)
    queue = _ReadyCandidateQueue()
    queue.push((1, 1))
    queue.push((2, 1))

    selected = queue.pop_acceptable(
        1, "bidirectional_query", row_for=rows.__getitem__,
        physical_stage_for=lambda _event_id: None,
        owner_gradients=None, query_replay=replay,
    )
    assert selected == [(2, 1)]

    replay.dispatch_adjoint((2,))
    selected = queue.pop_acceptable(
        1, "bidirectional_query", row_for=rows.__getitem__,
        physical_stage_for=lambda _event_id: None,
        owner_gradients=None, query_replay=replay,
    )
    assert selected == [(1, 1)]


def test_ready_queue_compacts_stale_requeue_indexes() -> None:
    rows = np.empty(300, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(rows.size)
    rows["primitive_kind"] = int(PrimitiveKind.FORWARD)
    queue = _ReadyCandidateQueue()
    for event_id in range(rows.size):
        queue.push((event_id, 1))
    queue._configure(
        "compute_pod", row_for=rows.__getitem__,
        physical_stage_for=lambda _event_id: None,
        owner_gradients=None, query_replay=None,
    )

    retained = (rows.size - 1, 1)
    for token, candidate in tuple(queue._candidate_by_token.items()):
        if candidate != retained:
            queue._remove((candidate, token))
    # Requeue the high-ID candidate while lower entries are gone.  Without
    # compaction this leaves every old token in the general heap.
    for _ in range(256):
        token = next(
            token for token, candidate in queue._candidate_by_token.items()
            if candidate == retained
        )
        queue._remove((retained, token))
        queue.push(retained)

    assert len(queue) == 1
    assert len(queue._general) <= 2
    assert queue.pop_acceptable(
        1, "compute_pod", row_for=rows.__getitem__,
        physical_stage_for=lambda _event_id: None,
        owner_gradients=None, query_replay=None,
    ) == [retained]


def test_compute_telemetry_is_exact_and_does_not_change_cycles() -> None:
    profile = ComputeTemplateProfile(1, {
        "gradient_reduction": ComputePathProfile((
            ComputeStage("TRANSFORM", latency=3, fma_groups=1),
        )),
    })
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=8, ports=1, banks=1,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache",
            "compute_pod", "bidirectional_query", "reconstruction_update",
            "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=1,
        relation_seed_fifo_entries=1, candidate_lanes=2,
        compute_templates={1: profile},
        compute_resource_capacities={
            "clusters": 1,
            "cluster_issue": 2,
            "fma_groups": 1,
            "transcendental_lanes": 1,
            "reduction_trees": 1,
            "microcontext_slots": 2,
            "feedback_lanes": 1,
        },
    )
    builder = TraceBuilder()
    for gaussian_id in range(2):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.GRADIENT_REDUCTION),
            gaussian_id=gaussian_id, template_id=1,
            resource_class=int(ResourceClass.COMPUTE),
        ))
    trace = builder.finish()

    plain = CycleEngine(config).run(trace, validate_input=False)
    diagnosed = CycleEngine(config).run(
        trace, validate_input=False, collect_compute_telemetry=True,
    )

    assert diagnosed.total_cycles == plain.total_cycles
    assert diagnosed.completion_cycles == plain.completion_cycles
    conflict = next(
        stall for stall in diagnosed.stalls
        if stall.reason == "compute_resource"
    )
    assert (
        conflict.resource,
        conflict.pod,
        conflict.cluster,
        conflict.resource_cycle,
        conflict.resource_in_use,
        conflict.resource_demand,
        conflict.resource_capacity,
    ) == ("fma_groups", None, 0, 0, 1, 1, 1)
    telemetry = diagnosed.compute_telemetry
    assert telemetry is not None
    assert telemetry.cluster_count == 1
    assert len(telemetry.event_timings) == 2
    assert telemetry.event_timings[0].dependency_ready_cycle == 0
    assert telemetry.event_timings[0].compute_issue_cycle == 0
    assert telemetry.event_timings[0].finish_cycle == 3
    assert all(
        len(run.active_microcontexts) == telemetry.cluster_count
        for run in telemetry.cluster_occupancy_runs
    )
    assert max(
        run.active_microcontexts[0]
        for run in telemetry.cluster_occupancy_runs
    ) == 1


def test_production_compute_route_stays_in_resident_pod_and_owner_cluster() -> None:
    engine = CycleEngine(_production_config())
    compute = engine.modules["compute_pod"]
    assert isinstance(compute, ComputePod)
    dtype = TraceBuilder().finish().events.dtype
    row = np.zeros((), dtype=dtype)
    row["gaussian_id"] = 7
    row["template_id"] = 1

    pod, owner = engine._compute_route(row, PrimitiveKind.FORWARD)
    assert (pod, owner) == (3, None)
    forward = compute.reservation_plan(
        1, PrimitiveKind.FORWARD, 0, pod=pod, cluster_hint=owner,
    )
    forward_clusters = {
        int(resource.partition(":")[2])
        for resource, _cycle, _demand in forward
        if resource.startswith("cluster_issue:")
    }
    assert len(forward_clusters) == 1
    assert forward_clusters <= set(range(15, 20))

    pod, owner = engine._compute_route(row, PrimitiveKind.GRADIENT_REDUCTION)
    assert (pod, owner) == (3, 17)
    gradient = compute.reservation_plan(
        1, PrimitiveKind.GRADIENT_REDUCTION, 0,
        pod=pod, cluster_hint=owner,
    )
    assert any(
        resource == "cluster_issue:17" for resource, _cycle, _demand in gradient
    )


def test_compute_pod_tracks_each_stage_resource_window() -> None:
    profile = ComputeTemplateProfile(2, {
        "adjoint": ComputePathProfile((
            ComputeStage("TRANSFORM", latency=2, fma_groups=4),
            ComputeStage("EVALUATE", latency=3, transcendental_lanes=2),
            ComputeStage("COMBINE", latency=4, reduction_trees=1,
                         feedback_lanes=3),
        ), cluster_issue_slots=2, cluster_issue_cycles=3),
    })
    pod = ComputePod(
        "compute_pod",
        ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8,
                     ports=1, banks=1), CounterBlock(),
        template_profiles={2: profile},
        resource_capacities={
            "clusters": 1, "cluster_issue": 2, "fma_groups": 4,
            "transcendental_lanes": 2, "reduction_trees": 1,
            "microcontext_slots": 32, "feedback_lanes": 3,
        },
    )
    plan = pod.reservation_plan(2, PrimitiveKind.ADJOINT, 7)
    assert pod.can_reserve(plan, 7)
    pod.reserve(plan)
    assert ("fma_groups:0", 7, 4) in plan
    assert ("fma_groups:0", 8, 4) not in plan
    assert ("transcendental_lanes:0", 9, 2) in plan
    assert ("reduction_trees:0", 12, 1) in plan
    assert ("feedback_lanes:0", 12, 3) in plan
    assert ("microcontext_slots:0", 9, 1) in plan
    assert ("microcontext_slots:0", 10, 1) not in plan
    assert not pod.can_reserve(
        pod.reservation_plan(2, PrimitiveKind.ADJOINT, 7), 7
    )


def test_fully_pipelined_fma_releases_issue_resource_before_result() -> None:
    profile = ComputeTemplateProfile(1, {
        "forward": ComputePathProfile((
            ComputeStage("TRANSFORM", latency=5, fma_groups=1),
        )),
    })
    pod = ComputePod(
        "compute_pod",
        ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8,
                     ports=1, banks=1), CounterBlock(),
        template_profiles={1: profile},
        resource_capacities={
            "clusters": 1, "cluster_issue": 1, "fma_groups": 1,
            "transcendental_lanes": 1, "reduction_trees": 1,
            "microcontext_slots": 8, "feedback_lanes": 1,
        },
    )
    first = pod.reservation_plan(1, PrimitiveKind.FORWARD, 10)
    pod.reserve(first)
    second = pod.reservation_plan(1, PrimitiveKind.FORWARD, 11)

    assert ("fma_groups:0", 10, 1) in first
    assert ("fma_groups:0", 11, 1) not in first
    assert pod.can_reserve(second, 11)
    assert pod.service_cycles_for(1, PrimitiveKind.FORWARD) == 5


def test_microcontext_releases_after_packet_admission_before_result() -> None:
    profile = ComputeTemplateProfile(1, {
        "forward": ComputePathProfile((
            ComputeStage("TRANSFORM", latency=9, fma_groups=1),
        ), cluster_issue_cycles=2, packet_first_result_latency=9,
           packet_last_result_offset=10),
    })
    pod = ComputePod(
        "compute_pod",
        ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8,
                     ports=1, banks=1), CounterBlock(),
        template_profiles={1: profile},
        resource_capacities={
            "clusters": 1, "cluster_issue": 1, "fma_groups": 2,
            "transcendental_lanes": 1, "reduction_trees": 1,
            "microcontext_slots": 1, "feedback_lanes": 1,
        },
    )
    first = pod.reservation_plan(1, PrimitiveKind.FORWARD, 4)
    pod.reserve(first)

    assert ("microcontext_slots:0", 4, 1) in first
    assert ("microcontext_slots:0", 5, 1) in first
    assert ("microcontext_slots:0", 6, 1) not in first
    assert not pod.can_reserve(
        pod.reservation_plan(1, PrimitiveKind.FORWARD, 5), 5,
    )
    assert pod.can_reserve(
        pod.reservation_plan(1, PrimitiveKind.FORWARD, 6), 6,
    )
    assert profile.paths["forward"].packet_completion_offset(0) == 9
