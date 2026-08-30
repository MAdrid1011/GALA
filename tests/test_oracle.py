from __future__ import annotations

from dataclasses import replace

import numpy as np

from gala_sim.clamp import ReductionDomain, TaskKind, TaskPacket
from gala_sim.clamp.builder import TraceBuilder
from gala_sim.clamp.events import PrimitiveKind, TraceEvent
from gala_sim.timing import CycleConfig, CycleConfigurationError, CycleEngine, ModuleTiming
from gala_sim.timing.engine import _DependencyIndex
from gala_sim.timing.oracle import FutureTracePlan


class _Memory:
    def submit(self, *, address, size_bytes, is_write, arrival_cycle):
        return arrival_cycle + 10


def _config() -> CycleConfig:
    timing = ModuleTiming(
        latency=2, initiation_interval=1, queue_capacity=8, ports=3, banks=8,
    )
    return CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache",
            "compute_pod", "bidirectional_query", "reconstruction_update",
            "shared_sram",
        )},
        memory=_Memory(),
        clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8,
        candidate_lanes=3,
        fusion_forward_ports=1,
        fusion_consumer_ports=1,
        fusion_adjoint_ports=1,
    )


def _plan(trace, service_cycles):
    dependencies = _DependencyIndex.from_trace(trace)
    return FutureTracePlan.from_trace(
        trace,
        event_service_cycles=np.asarray(service_cycles, dtype=np.uint64),
        dependent_offsets=dependencies.offsets,
        dependents=dependencies.dependents,
    )


def test_future_trace_plan_prioritizes_longest_real_dependency_path() -> None:
    builder = TraceBuilder()
    short = builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION)))
    long = builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION)))
    middle = builder.emit(
        TraceEvent(primitive_kind=int(PrimitiveKind.RELATION)), dependencies=[long]
    )
    builder.emit(
        TraceEvent(primitive_kind=int(PrimitiveKind.QUERY_CLOSE)),
        dependencies=[middle],
    )
    plan = _plan(builder.finish(), [1, 1, 4, 2])

    assert plan.critical_cycles.tolist() == [1, 7, 6, 2]
    assert plan.query_priority(long) < plan.query_priority(short)


def test_future_trace_plan_tracks_each_real_cache_reuse() -> None:
    builder = TraceBuilder()
    first = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
        gaussian_id=4, state_version=2,
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
        gaussian_id=8, state_version=2,
    ))
    second = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
        gaussian_id=4, state_version=2,
    ))
    plan = _plan(builder.finish(), [1, 1, 1])

    key = (4, 2)
    assert plan.next_cache_use(key, after_event=first) == second
    assert plan.next_cache_use(key, after_event=second) is None


def test_query_oracle_uses_future_path_without_dropping_events() -> None:
    builder = TraceBuilder()
    short = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, reduction_key=0,
    ))
    long = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=1,
        gaussian_id=1, reduction_key=1,
    ))
    previous = long
    for _ in range(4):
        previous = builder.emit(
            TraceEvent(
                primitive_kind=int(PrimitiveKind.QUERY_REDUCTION), query_id=1,
                reduction_key=1,
            ),
            dependencies=[previous],
        )
    trace = builder.finish()

    base = CycleEngine(_config(), policy="base").run(
        trace, validate_input=False
    )
    actual = CycleEngine(_config(), policy="query").run(
        trace, validate_input=False
    )
    oracle = CycleEngine(_config(), policy="query_oracle").run(
        trace, validate_input=False
    )

    assert actual.completion_cycles[short] < actual.completion_cycles[long]
    assert oracle.completion_cycles[long] < oracle.completion_cycles[short]
    assert oracle.total_cycles < actual.total_cycles
    assert oracle.event_counts == actual.event_counts
    assert oracle.module_counters["fusion_issue"]["accepted"] == 2
    assert oracle.oracle_status == "portfolio_best_known_not_proven_upper_bound"
    assert oracle.oracle_portfolio is not None
    assert oracle.oracle_portfolio.winner == "future"
    assert [(member.name, member.status, member.total_cycles)
            for member in oracle.oracle_portfolio.members] == [
        ("base", "passed", base.total_cycles),
        ("actual", "passed", actual.total_cycles),
        ("future", "passed", oracle.total_cycles),
    ]
    assert oracle.oracle_member_results is not None
    assert set(oracle.oracle_member_results) == {"base", "actual", "future"}
    assert oracle.oracle_member_results["actual"].stalls == actual.stalls


def test_oracle_portfolio_can_decline_a_harmful_mechanism() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.QUERY_REDUCTION), query_id=0,
        reduction_key=0,
    ))
    trace = builder.finish()

    result = CycleEngine(_config(), policy="residency_oracle").run(
        trace, validate_input=False,
    )

    assert result.oracle_portfolio is not None
    assert result.oracle_portfolio.winner == "base"
    assert result.total_cycles == next(
        member.total_cycles for member in result.oracle_portfolio.members
        if member.name == "base"
    )


def test_query_oracle_selects_exact_legal_combination_from_bounded_queue() -> None:
    builder = TraceBuilder()
    forward_ids = [
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
            gaussian_id=query_id, reduction_key=query_id,
            address_token=query_id,
        ))
        for query_id in range(3)
    ]
    consumer = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=9,
        gaussian_id=9, reduction_key=9, address_token=7,
    ))
    trace = builder.finish()
    critical = np.asarray([100, 90, 80, 1], dtype=np.uint64)
    critical.setflags(write=False)
    future = FutureTracePlan(critical, {}, {})
    engine = CycleEngine(_config(), policy="query_oracle")

    selected = engine._select_query_oracle_candidates(
        trace, (*forward_ids, consumer), future
    )

    assert set(selected) == {forward_ids[0], consumer}


def test_query_oracle_compatibility_graph_matches_exhaustive_selection() -> None:
    kinds = (
        TaskKind.FORWARD, TaskKind.FORWARD, TaskKind.ADJOINT,
        TaskKind.CONSUMER, TaskKind.ADJOINT, TaskKind.FORWARD,
        TaskKind.CONSUMER, TaskKind.ADJOINT, TaskKind.FORWARD,
    )
    banks = (0, 0, 1, 2, 3, 1, 4, 5, 6)
    packets = [TaskPacket(
        event_id=index, query_id=index, gaussian_id=index,
        reduction_key=(0 if index in {1, 4} else index),
        resource=index, state_version=0, template_id=0,
        address_token=index, task_kind=kind,
        reduction_domain=(
            ReductionDomain.GAUSSIAN
            if kind is TaskKind.ADJOINT else ReductionDomain.QUERY
        ),
        target_resource=(7 if index in {2, 6} else index),
    ) for index, kind in enumerate(kinds)]
    candidates = [
        (20 - index, banks[index], packet)
        for index, packet in enumerate(packets)
    ]
    limits = {
        TaskKind.FORWARD: 2,
        TaskKind.CONSUMER: 1,
        TaskKind.ADJOINT: 2,
    }
    best_score = 0
    expected: tuple[int, ...] = ()

    def exhaustive(
        index: int, score: int, selected: tuple[int, ...],
        counts: dict[TaskKind, int], keys: frozenset[tuple[ReductionDomain, int]],
        targets: frozenset[int], occupied_banks: frozenset[int],
    ) -> None:
        nonlocal best_score, expected
        if score > best_score:
            best_score = score
            expected = tuple(packets[item].event_id for item in selected)
        if len(selected) == 3 or index == len(candidates):
            return
        weight, bank, packet = candidates[index]
        target = packet.target_resource
        if (
            counts[packet.task_kind] < limits[packet.task_kind]
            and packet.conflict_keys.isdisjoint(keys)
            and bank not in occupied_banks
            and (target is None or target not in targets)
        ):
            next_counts = dict(counts)
            next_counts[packet.task_kind] += 1
            exhaustive(
                index + 1, score + weight, (*selected, index), next_counts,
                keys.union(packet.conflict_keys),
                targets if target is None else targets.union((target,)),
                occupied_banks.union((bank,)),
            )
        exhaustive(
            index + 1, score, selected, counts, keys, targets, occupied_banks,
        )

    exhaustive(
        0, 0, (), {kind: 0 for kind in TaskKind},
        frozenset(), frozenset(), frozenset(),
    )

    assert CycleEngine._maximum_weight_query_candidates(
        candidates, source_limits=limits,
    ) == expected


def test_query_oracle_same_bank_queue_has_bounded_exact_selection() -> None:
    candidates = [(
        256 - event_id,
        0,
        TaskPacket(
            event_id=event_id, query_id=event_id, gaussian_id=event_id,
            reduction_key=event_id, resource=event_id, state_version=0,
            template_id=0, address_token=event_id,
            task_kind=TaskKind.FORWARD,
        ),
    ) for event_id in range(128)]

    assert CycleEngine._maximum_weight_query_candidates(
        candidates,
        source_limits={
            TaskKind.FORWARD: 2,
            TaskKind.CONSUMER: 0,
            TaskKind.ADJOINT: 2,
        },
    ) == (0,)


def test_query_oracle_borrows_idle_consumer_slot_under_frozen_port_limits() -> None:
    builder = TraceBuilder()
    forward_ids = [
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
            gaussian_id=query_id, reduction_key=query_id,
            address_token=query_id,
        ))
        for query_id in range(3)
    ]
    adjoint = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.ADJOINT), query_id=7,
        gaussian_id=7, reduction_key=7, address_token=7,
    ))
    trace = builder.finish()
    critical = np.asarray([100, 90, 80, 70], dtype=np.uint64)
    critical.setflags(write=False)
    future = FutureTracePlan(critical, {}, {})
    config = replace(
        _config(), fusion_forward_ports=2, fusion_adjoint_ports=2,
    )
    engine = CycleEngine(config, policy="query_oracle")

    selected = engine._select_query_oracle_candidates(
        trace, (*forward_ids, adjoint), future,
    )

    assert selected == (forward_ids[0], forward_ids[1], adjoint)


def test_query_oracle_keeps_consumer_candidate_slot_when_nonempty() -> None:
    builder = TraceBuilder()
    first_forward = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, reduction_key=0, address_token=0,
    ))
    second_forward = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=1,
        gaussian_id=1, reduction_key=1, address_token=1,
    ))
    consumer = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=2,
        gaussian_id=2, reduction_key=2, address_token=2,
    ))
    trace = builder.finish()
    critical = np.asarray([100, 90, 1], dtype=np.uint64)
    critical.setflags(write=False)
    future = FutureTracePlan(critical, {}, {})
    config = replace(
        _config(), fusion_forward_ports=2, fusion_adjoint_ports=2,
    )
    engine = CycleEngine(config, policy="query_oracle")

    selected = engine._select_query_oracle_candidates(
        trace, (first_forward, second_forward, consumer), future,
    )

    assert selected == (first_forward, consumer)


def test_query_oracle_uses_runtime_consumer_credit_owner_for_readiness() -> None:
    builder = TraceBuilder()
    consumer = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=9,
        gaussian_id=9, reduction_key=9, address_token=9,
    ))
    trace = builder.finish()
    critical = np.asarray([10], dtype=np.uint64)
    critical.setflags(write=False)
    future = FutureTracePlan(critical, {}, {})
    engine = CycleEngine(_config(), policy="query_oracle")
    engine.issue_scheduler.enable_exact_readiness()
    engine.issue_scheduler.allocate((1, 2))
    ready = engine.issue_scheduler.states[1]
    ready.generator_closed = True
    ready.reduction_ready = True
    ready.consumer_count = 1
    blocked = engine.issue_scheduler.states[2]
    blocked.consumer_count = 1

    selected = engine._select_query_oracle_candidates(
        trace, (consumer,), future,
        consumer_owner_for_event={consumer: 1},
    )

    assert selected == (consumer,)


def test_query_oracle_does_not_reorder_non_fusion_modules() -> None:
    builder = TraceBuilder()
    first = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0,
    ))
    second = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=1,
    ))
    trace = builder.finish()
    critical = np.asarray([1, 10], dtype=np.uint64)
    critical.setflags(write=False)
    future = FutureTracePlan(critical, {}, {})
    engine = CycleEngine(_config(), policy="query_oracle")

    assert engine._ordered_candidates(
        trace, [first, second], future_plan=future
    ) == [first, second]


def test_query_oracle_uses_independent_per_source_fifo_capacity() -> None:
    builder = TraceBuilder()
    first_forward = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, reduction_key=0, address_token=0,
    ))
    second_forward = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=1,
        gaussian_id=1, reduction_key=1, address_token=1,
    ))
    consumer = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=9,
        gaussian_id=9, reduction_key=9, address_token=9,
    ))
    previous = consumer
    for _ in range(4):
        previous = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.QUERY_REDUCTION),
            query_id=9, reduction_key=9,
        ), dependencies=[previous])
    trace = builder.finish()
    result = CycleEngine(
        replace(_config(), candidate_fifo_entries=1),
        policy="query_oracle",
    ).run(trace, validate_input=False)
    assert result.oracle_member_results is not None
    future = result.oracle_member_results["future"]

    assert future.completion_cycles[first_forward] < future.completion_cycles[second_forward]
    assert future.completion_cycles[consumer] < future.completion_cycles[second_forward]


def test_online_oracle_refuses_to_claim_future_visibility() -> None:
    for policy in ("query_oracle", "residency_oracle"):
        with np.testing.assert_raises_regex(
            CycleConfigurationError, "requires a complete trace"
        ):
            CycleEngine(_config(), policy=policy).online_session(
                max_events=8, initial_gaussian_count=1
            )


def _cache_trace(keys: tuple[int, ...]):
    builder = TraceBuilder()
    previous = None
    for query_id, gaussian_id in enumerate(keys):
        dependencies = [] if previous is None else [previous]
        request = builder.emit(
            TraceEvent(
                primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
                query_id=query_id,
                gaussian_id=gaussian_id,
                state_version=0,
                address_token=(gaussian_id + 1) * 64,
                data_bytes=64,
            ),
            dependencies=dependencies,
        )
        previous = builder.emit(
            TraceEvent(
                primitive_kind=int(PrimitiveKind.CACHE_RETURN),
                query_id=query_id,
                gaussian_id=gaussian_id,
                state_version=0,
                address_token=(gaussian_id + 1) * 64,
                data_bytes=64,
            ),
            dependencies=[request],
        )
    return builder.finish()


def _cache_config() -> CycleConfig:
    config = _config()
    return CycleConfig(
        modules=config.modules,
        memory=config.memory,
        clock_frequency_hz=config.clock_frequency_hz,
        relation_seed_fifo_entries=config.relation_seed_fifo_entries,
        candidate_lanes=config.candidate_lanes,
        fusion_forward_ports=config.fusion_forward_ports,
        fusion_consumer_ports=config.fusion_consumer_ports,
        fusion_adjoint_ports=config.fusion_adjoint_ports,
        cache_instances=1,
        cache_capacity_per_instance=2,
        cache_directory_banks=2,
        cache_sector_bytes=64,
        cache_multicast_destinations=1,
    )


def test_residency_oracle_uses_belady_eviction_and_refills_real_state() -> None:
    trace = _cache_trace((0, 1, 2, 0, 2, 1))
    base = CycleEngine(_cache_config(), policy="base").run(
        trace, validate_input=False
    )
    oracle = CycleEngine(_cache_config(), policy="residency_oracle").run(
        trace, validate_input=False
    )

    cache = oracle.module_counters["semantic_cache"]
    assert base.module_counters["semantic_cache"]["memory_requests"] == 6
    assert cache["memory_requests"] == 4
    assert cache["oracle_evictions"] == 1
    assert cache["directory_misses"] == 4
    assert cache["directory_hits"] == 2
    assert oracle.total_cycles < base.total_cycles
    assert oracle.event_counts == base.event_counts
    assert cache["accepted"] == base.module_counters["semantic_cache"]["accepted"] == 12


def test_residency_oracle_is_not_worse_when_workset_fits() -> None:
    trace = _cache_trace((0, 1, 0, 1))
    actual = CycleEngine(_cache_config(), policy="residency").run(
        trace, validate_input=False
    )
    oracle = CycleEngine(_cache_config(), policy="residency_oracle").run(
        trace, validate_input=False
    )

    assert oracle.total_cycles <= actual.total_cycles
    assert oracle.event_counts == actual.event_counts
    assert oracle.module_counters["semantic_cache"]["memory_requests"] == 2
