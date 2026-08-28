from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from gala_sim.timing.modules import ComputePod, CounterBlock


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
        "clusters": 20,
        "cluster_issue": 40,
        "fma_groups": 80,
        "transcendental_lanes": 40,
        "reduction_trees": 20,
        "microcontext_slots": 80,
        "feedback_lanes": 60,
    }
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


def test_consumer_uses_query_loss_path_without_compute_profile() -> None:
    assert CycleEngine._stages_for(PrimitiveKind.CONSUMER) == (
        "fusion_issue", "bidirectional_query",
    )
    assert CycleEngine._stages_for(PrimitiveKind.GRADIENT_REDUCTION) == (
        "bidirectional_query",
    )


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
    assert ("microcontext_slots:0", 15, 1) in plan
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
