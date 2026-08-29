from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from gala_sim.config import load_config

from gala_sim.clamp.builder import TraceBuilder
from gala_sim.clamp.events import PrimitiveKind, TraceEvent
from gala_sim.timing import (
    ComputePathProfile, ComputeStage, ComputeTemplateProfile,
    CycleConfig, CycleEngine, ModuleTiming, analyze_cycle_lower_bounds,
)


class _Memory:
    def submit(self, *, address, size_bytes, is_write, arrival_cycle):
        return arrival_cycle + 10

    def metadata(self):
        return {"read_latency_cycles": 7, "config_sha256": "a" * 64}


def _config() -> CycleConfig:
    timing = ModuleTiming(
        latency=2, initiation_interval=1, queue_capacity=8, ports=2, banks=2,
    )
    modules = {
        name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache",
            "compute_pod", "bidirectional_query", "reconstruction_update",
            "shared_sram",
        )
    }
    modules["semantic_cache"] = ModuleTiming(
        latency=4, initiation_interval=1, queue_capacity=8, ports=1, banks=2,
    )
    return CycleConfig(
        modules=modules,
        memory=_Memory(),
        clock_frequency_hz=100,
        relation_seed_fifo_entries=8,
        candidate_lanes=3,
        cache_instances=1,
        cache_capacity_per_instance=2,
        cache_directory_banks=2,
        cache_sector_bytes=64,
        cache_multicast_destinations=1,
        fusion_forward_ports=1,
        fusion_consumer_ports=1,
        fusion_adjoint_ports=1,
        memory_peak_bandwidth_bytes_per_second=100,
    )


def _trace():
    builder = TraceBuilder()
    previous = None
    for query_id, gaussian_id in enumerate((0, 0, 1)):
        request = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
            query_id=query_id,
            gaussian_id=gaussian_id,
            state_version=0,
            address_token=(gaussian_id + 1) * 64,
            data_bytes=64,
        ), dependencies=[] if previous is None else [previous])
        previous = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_RETURN),
            query_id=query_id,
            gaussian_id=gaussian_id,
            state_version=0,
            address_token=(gaussian_id + 1) * 64,
            data_bytes=64,
        ), dependencies=[request])
    return builder.finish()


def test_cycle_bounds_preserve_dependency_resource_and_compulsory_memory_floors() -> None:
    report = analyze_cycle_lower_bounds(
        CycleEngine(_config(), policy="base"),
        _trace(),
        base_asic_cycles=100,
        targets={"query": 2.0, "residency": 2.0, "full": 4.0},
    )
    scenarios = {item.scenario: item for item in report.scenarios}

    assert scenarios["query"].memory_request_lower_bound == 3
    assert scenarios["query"].memory_byte_lower_bound == 192
    assert scenarios["residency"].memory_request_lower_bound == 2
    assert scenarios["residency"].memory_byte_lower_bound == 128
    assert scenarios["query"].minimum_cycles == 192
    assert scenarios["residency"].minimum_cycles == 128
    query_sram = next(
        item for item in scenarios["query"].components
        if item.name == "shared_sram.banks"
    )
    residency_sram = next(
        item for item in scenarios["residency"].components
        if item.name == "shared_sram.banks"
    )
    assert query_sram.evidence["write_accesses_per_bank"] == [[1, 2]]
    assert residency_sram.evidence["write_accesses_per_bank"] == [[1, 1]]
    query_sram_issue = next(
        item for item in scenarios["query"].components
        if item.name == "shared_sram.issue_ports"
    )
    residency_sram_issue = next(
        item for item in scenarios["residency"].components
        if item.name == "shared_sram.issue_ports"
    )
    assert query_sram_issue.evidence["physical_stage_count"] == 6
    assert residency_sram_issue.evidence["physical_stage_count"] == 5
    dependency = next(
        item for item in scenarios["query"].components
        if item.name == "trace.dependency_critical_path"
    )
    assert dependency.cycles == 36


def test_cycle_bounds_only_rule_out_targets_proven_below_a_necessary_floor() -> None:
    report = analyze_cycle_lower_bounds(
        CycleEngine(_config(), policy="base"),
        _trace(),
        base_asic_cycles=400,
        targets={"query": 2.0, "residency": 2.0, "full": 4.0},
    )
    reachability = {item.scenario: item for item in report.reachability}

    assert reachability["query"].status == "not_ruled_out_by_necessary_lower_bound"
    assert reachability["full"].status == "ruled_out_by_necessary_lower_bound"
    assert reachability["full"].minimum_uncovered_cycles == 128
    assert reachability["full"].maximum_coverable_cycles == 272


def test_query_loss_bound_uses_registered_query_issue_width() -> None:
    builder = TraceBuilder()
    for query_id in range(17):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CONSUMER),
            query_id=query_id,
        ))
    config = CycleConfig.from_gala(
        load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml"),
        _Memory(),
    )
    report = analyze_cycle_lower_bounds(
        CycleEngine(config),
        builder.finish(),
        base_asic_cycles=100,
        targets={"query": 2.0, "residency": 2.0, "full": 2.0},
    )
    component = next(
        item for item in report.scenarios[0].components
        if item.name == "bidirectional_query.loss_fma_lanes"
    )
    assert component.cycles == 5
    assert component.evidence["queries_per_cycle"] == 16


def test_query_reduction_bound_counts_physical_banks_not_partial_entries() -> None:
    builder = TraceBuilder()
    for query_id in (0, 64, 1):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.QUERY_REDUCTION),
            query_id=query_id,
            reduction_key=query_id,
        ))
    config = CycleConfig.from_gala(
        load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml"),
        _Memory(),
    )
    report = analyze_cycle_lower_bounds(
        CycleEngine(config), builder.finish(), base_asic_cycles=100,
        targets={"query": 2.0, "residency": 2.0, "full": 2.0},
    )
    component = next(
        item for item in report.scenarios[0].components
        if item.name == "bidirectional_query.reduction_banks"
    )

    assert component.evidence["reduction_banks"] == 64
    assert component.evidence["partial_sum_groups_per_bank"] == 4
    assert component.evidence["reduction_issue_ports"] == 64
    assert component.evidence["busiest_bank_events"] == 2


def test_fusion_bank_bound_uses_physical_query_state_banks() -> None:
    builder = TraceBuilder()
    for query_id in range(64):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CONSUMER),
            query_id=query_id, consumer_id=query_id,
            address_token=0,
        ))
    config = replace(
        _config(), relation_query_lanes=1, fusion_query_state_banks=8,
    )
    report = analyze_cycle_lower_bounds(
        CycleEngine(config), builder.finish(),
        base_asic_cycles=100,
        targets={"query": 2.0, "residency": 2.0, "full": 2.0},
    )
    component = next(
        item for item in report.scenarios[0].components
        if item.name == "fusion_issue.banks"
    )

    assert component.cycles == 9
    assert component.evidence["banks_per_partition"] == 8


def test_shared_sram_maps_aligned_records_to_distinct_record_banks() -> None:
    builder = TraceBuilder()
    rows = []
    for gaussian_id in range(3):
        rows.append(builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
            gaussian_id=gaussian_id,
            address_token=gaussian_id * 128,
            data_bytes=128,
        )))
    trace = builder.finish()
    config = replace(
        _config(),
        modules={
            **_config().modules,
            "shared_sram": ModuleTiming(
                latency=2, initiation_interval=1, queue_capacity=8,
                ports=2, banks=16,
            ),
        },
    )
    engine = CycleEngine(config)
    assert [
        engine._shared_sram_bank(trace.events[event_id]) for event_id in rows
    ] == [0, 1, 2]


def test_shared_sram_read_and_write_ports_are_independent_per_bank() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
        gaussian_id=0, address_token=0, data_bytes=128,
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN),
        gaussian_id=0, address_token=0, data_bytes=128,
    ))
    config = replace(_config(), modules={
        **_config().modules,
        "shared_sram": ModuleTiming(
            latency=2, initiation_interval=1, queue_capacity=8,
            ports=2, banks=16,
        ),
    })
    result = CycleEngine(config).run(builder.finish(), validate_input=False)
    assert result.module_counters["shared_sram"]["port_stalls"] == 0
    assert result.module_counters["shared_sram"]["bank_conflicts"] == 0


def test_adjoint_replay_bound_is_work_conserving_across_physical_lanes() -> None:
    builder = TraceBuilder()
    for _ in range(8):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.ADJOINT),
            query_id=0,
        ))
    config = CycleConfig.from_gala(
        load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml"),
        _Memory(),
    )
    config = replace(
        config,
        relation_query_lanes=1,
        compute_templates=None,
        compute_resource_capacities=None,
    )
    report = analyze_cycle_lower_bounds(
        CycleEngine(config), builder.finish(), base_asic_cycles=100,
        targets={"query": 2.0, "residency": 2.0, "full": 2.0},
    )
    component = next(
        item for item in report.scenarios[0].components
        if item.name == "bidirectional_query.adjoint_replay_lanes"
    )
    assert component.cycles == 4
    assert component.evidence["total_replay_lane_work"] == 8
    assert component.evidence["work_conserving_lanes"] == 8


def test_compute_microcontext_bound_counts_packet_admission_cycles() -> None:
    profile = ComputeTemplateProfile(1, {
        "gradient_reduction": ComputePathProfile((
            ComputeStage("COMBINE", latency=10, reduction_trees=1),
        ), cluster_issue_cycles=2, packet_first_result_latency=10,
           packet_last_result_offset=10),
    })
    builder = TraceBuilder()
    for gaussian_id in range(3):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.GRADIENT_REDUCTION),
            gaussian_id=gaussian_id, template_id=1,
        ))
    config = replace(
        _config(), relation_query_lanes=1,
        compute_templates={1: profile},
        compute_resource_capacities={
            "pods": 1, "clusters_per_pod": 1, "clusters": 1,
            "cluster_issue": 1, "fma_groups": 1,
            "transcendental_lanes": 1, "reduction_trees": 1,
            "microcontext_slots": 1, "feedback_lanes": 1,
        },
    )
    report = analyze_cycle_lower_bounds(
        CycleEngine(config), builder.finish(),
        base_asic_cycles=100,
        targets={"query": 2.0, "residency": 2.0, "full": 2.0},
    )
    component = next(
        item for item in report.scenarios[0].components
        if item.name == "compute_pod.microcontext_slots"
    )

    assert component.cycles == 6
    assert component.evidence["total_slot_cycles_or_demands"] == 6


def test_semantic_cache_instances_have_independent_ports_and_banks() -> None:
    def cache_returns(gaussian_ids: tuple[int, int]):
        builder = TraceBuilder()
        for event_id, gaussian_id in enumerate(gaussian_ids):
            builder.emit(TraceEvent(
                primitive_kind=int(PrimitiveKind.CACHE_RETURN),
                query_id=event_id,
                gaussian_id=gaussian_id,
                state_version=0,
                address_token=(gaussian_id + 1) * 64,
                data_bytes=64,
            ))
        return builder.finish()

    config = replace(_config(), cache_instances=2)
    distributed = CycleEngine(config, policy="base").run(
        cache_returns((0, 1)), validate_input=False
    )
    same_instance = CycleEngine(config, policy="base").run(
        cache_returns((0, 2)), validate_input=False
    )

    assert distributed.total_cycles == 6
    assert same_instance.total_cycles == 7


def test_blocked_cache_queue_cannot_starve_independent_relation_constructor() -> None:
    builder = TraceBuilder()
    cache_returns: list[int] = []
    for query_id in range(64):
        request = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
            query_id=query_id,
            gaussian_id=0,
            state_version=0,
            address_token=64,
            data_bytes=64,
        ))
        cache_returns.append(builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_RETURN),
            query_id=query_id,
            gaussian_id=0,
            state_version=0,
            address_token=64,
            data_bytes=64,
        ), dependencies=[request]))
    relation = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=100,
    ))

    result = CycleEngine(_config(), policy="base").run(
        builder.finish(), validate_input=False
    )

    assert result.completion_cycles[relation] == 2
    assert result.completion_cycles[relation] < min(
        result.completion_cycles[event_id] for event_id in cache_returns
    )
