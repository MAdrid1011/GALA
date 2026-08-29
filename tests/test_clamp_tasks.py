from __future__ import annotations

import pytest

from gala_sim.clamp import (
    FusionIssueScheduler, ReductionDomain, TaskKind, TaskPacket,
)


def _task(
    event_id: int, kind: TaskKind, reduction_key: int,
    domain: ReductionDomain = ReductionDomain.QUERY,
    query_id: int | None = None,
) -> TaskPacket:
    return TaskPacket(
        event_id, 1 if query_id is None else query_id,
        event_id + 1, reduction_key, 1, 0, 1, event_id, kind,
        domain,
    )


def test_fusion_issue_enforces_candidate_and_port_contract() -> None:
    scheduler = FusionIssueScheduler(candidate_lanes=3, forward_ports=1,
                                     consumer_ports=1, adjoint_ports=1)
    decision = scheduler.issue([
        _task(0, TaskKind.CONSUMER, 1),
        _task(1, TaskKind.ADJOINT, 2),
        _task(2, TaskKind.FORWARD, 3),
        _task(3, TaskKind.ADJOINT, 4),
    ])
    assert len(decision.accepted) == 3
    assert len(decision.rejected) == 1
    assert decision.reason == "conflict_or_port"


def test_query_state_history_is_only_used_after_round_boundary() -> None:
    scheduler = FusionIssueScheduler(candidate_lanes=3, forward_ports=1,
                                     consumer_ports=1, adjoint_ports=1)
    scheduler.relation_accept((1,))
    state = scheduler.states[1]
    assert state.forecast() == (1, 0, 0)
    state.add_round_overlap(2)
    state.begin_round()
    assert state.forecast()[0] >= 1


def test_query_history_survives_release_and_changes_next_iteration_priority() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1,
    )
    scheduler.set_strict_lifecycle()
    for _ in range(3):
        scheduler.relation_accept(
            (1,), iteration_id=0, history_keys=((1, 0),),
        )
    scheduler.relation_accept(
        (2,), iteration_id=0, adjoint_query_ids=(2,),
        history_keys=((1, 1),),
    )
    scheduler.producer_close((1, 2))
    scheduler.reduction_writeback((1, 2))
    scheduler.forward_retire((1, 1, 1, 2))
    scheduler.adjoint_retire((2,))
    assert scheduler.release_completed() == 2

    scheduler.relation_accept(
        (101, 102), iteration_id=1, adjoint_query_ids=(102,),
        history_keys=((1, 0), (1, 1)),
    )
    scheduler.producer_close((102,))
    scheduler.reduction_writeback((102,))
    scheduler.forward_retire((102,))
    high_previous_load = TaskPacket(
        0, 101, 1, 101, 1, 0, 1, 0, TaskKind.FORWARD,
        target_resource=9,
    )
    stable_load = TaskPacket(
        1, 102, 2, 102, 1, 0, 1, 1, TaskKind.ADJOINT,
        target_resource=9,
    )

    assert scheduler.states[101].forecast()[0] == 3
    assert scheduler.states[101].exact_remaining(TaskKind.FORWARD) == 1
    assert scheduler.select(
        (high_previous_load, stable_load), use_load_rules=False,
    ).accepted == (high_previous_load,)
    assert scheduler.select(
        (high_previous_load, stable_load), use_load_rules=True,
    ).accepted == (stable_load,)
    assert scheduler.history_snapshot() == {
        "query_history_restored": 2,
        "query_history_candidate_evaluations": 2,
        "query_load_rule_evaluations": 1,
        "query_load_rule_selection_changes": 1,
    }


def test_forecast_is_pure_and_issue_uses_typed_conflict_keys() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1, adjoint_ports=1,
    )
    query = _task(0, TaskKind.FORWARD, 7, ReductionDomain.QUERY)
    gaussian = _task(1, TaskKind.ADJOINT, 7, ReductionDomain.GAUSSIAN)
    scheduler.forecast([query, gaussian])
    assert scheduler.states == {}
    decision = scheduler.issue([query, gaussian])
    assert decision.accepted == (query, gaussian)


def test_issue_rejects_same_domain_reduction_conflict() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1, adjoint_ports=1,
    )
    decision = scheduler.issue([
        _task(0, TaskKind.FORWARD, 7),
        _task(1, TaskKind.CONSUMER, 7),
    ])
    assert len(decision.accepted) == 1
    assert len(decision.rejected) == 1


def test_issue_rejects_overlap_with_any_query_in_physical_packet() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=2, consumer_ports=1, adjoint_ports=1,
    )
    packet = TaskPacket(
        0, 1, 1, 1, 1, 0, 1, 0, TaskKind.FORWARD,
        ReductionDomain.QUERY, (1, 7),
    )
    overlaps_lane_seven = _task(1, TaskKind.FORWARD, 7)

    decision = scheduler.issue([packet, overlaps_lane_seven])

    assert decision.accepted == (packet,)
    assert decision.rejected == (overlaps_lane_seven,)


def test_select_is_uncommitted_and_commit_occurs_once() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1, adjoint_ports=1,
    )
    task = _task(0, TaskKind.FORWARD, 7)
    scheduler.observe_arrival([task])
    assert scheduler.select([task]).accepted == (task,)
    assert scheduler.select([task]).accepted == (task,)
    scheduler.commit_issued([task])
    with pytest.raises(ValueError, match="more than once"):
        scheduler.commit_issued([task])


def test_query_lifecycle_updates_exact_f_c_a_counters() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1,
    )
    scheduler.relation_accept((10, 11), adjoint_query_ids=(10,))
    scheduler.producer_close((10, 11))
    scheduler.forward_retire((10, 11))
    scheduler.reduction_writeback((10, 11))
    assert scheduler.successor_credit(10)

    assert scheduler.states[10].forecast() == (0, 1, 1)
    assert scheduler.states[11].forecast() == (0, 0, 0)
    assert scheduler.states[10].exact_ready(TaskKind.CONSUMER)
    assert scheduler.states[10].exact_ready(TaskKind.ADJOINT)

    assert scheduler.successor_dispatch(10)
    scheduler.adjoint_retire((10,))
    assert scheduler.states[10].forecast() == (0, 0, 0)


def test_score_uses_all_physical_packet_lanes_and_authoritative_key() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1,
    )
    scheduler.relation_accept((1, 2, 3))
    scheduler.relation_accept((3,))
    packet = TaskPacket(
        0, 1, 7, 1, 1, 0, 1, 0, TaskKind.FORWARD,
        conflict_query_ids=(1, 2, 3),
    )

    score = scheduler.score(packet)

    assert (score.released_work, score.completed_queries, score.remaining_work) == (
        2, 0, 1,
    )


def test_exact_readiness_blocks_consumer_until_close_and_reduction_writeback() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1,
    )
    scheduler.enable_exact_readiness()
    scheduler.relation_accept((1,))
    scheduler.forward_retire((1,))
    scheduler.successor_credit(1)
    consumer = _task(0, TaskKind.CONSUMER, 1, query_id=1)
    assert scheduler.select((consumer,)).accepted == ()

    scheduler.producer_close((1,))
    scheduler.reduction_writeback((1,))
    assert scheduler.select((consumer,)).accepted == (consumer,)


def test_target_resource_conflict_blocks_otherwise_independent_heads() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=2, consumer_ports=1,
        adjoint_ports=1,
    )
    first = TaskPacket(
        0, 1, 1, 1, 1, 0, 1, 0, TaskKind.FORWARD,
        target_resource=9,
    )
    second = TaskPacket(
        1, 2, 2, 2, 1, 0, 1, 1, TaskKind.FORWARD,
        target_resource=9,
    )

    decision = scheduler.issue((first, second))

    assert decision.accepted == (first,)
    assert decision.rejected == (second,)


def test_waiting_age_breaks_an_authoritative_score_tie() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1,
    )
    scheduler.set_strict_lifecycle()
    scheduler.relation_accept((1, 2))
    older = _task(0, TaskKind.FORWARD, 1, query_id=1)
    younger = _task(1, TaskKind.FORWARD, 2, query_id=2)
    scheduler.observe_arrival((older,), arrival_cycle=3)
    scheduler.observe_arrival((younger,), arrival_cycle=7)
    scheduler.set_clock(10)

    assert scheduler.forecast((younger, older)) == [older, younger]


def test_query_state_uses_and_reuses_physical_eight_lane_slots() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1, query_state_entries=2, query_state_lanes=8,
    )
    scheduler.relation_accept((0, 7, 8))
    assert scheduler.physical_location(0) == (0, 0)
    assert scheduler.physical_location(7) == (0, 7)
    assert scheduler.physical_location(8) == (1, 0)
    with pytest.raises(ValueError, match="table is full"):
        scheduler.relation_accept((16,))

    scheduler.producer_close((0, 7))
    scheduler.forward_retire((0, 7))
    scheduler.reduction_writeback((0, 7))
    assert scheduler.release_completed() == 2
    scheduler.relation_accept((16,))

    assert scheduler.physical_location(16) == (0, 0)
    assert scheduler.state_table_snapshot() == {
        "query_state_entries_live": 2,
        "query_state_entries_peak": 2,
        "query_state_entries_released": 1,
        "query_state_lanes_live": 2,
    }
