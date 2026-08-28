from __future__ import annotations

from dataclasses import replace

from gala_sim.clamp.builder import TraceBuilder
from gala_sim.clamp.events import PrimitiveKind, TraceEvent
from gala_sim.timing import (
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
