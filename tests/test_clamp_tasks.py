from __future__ import annotations

import pytest

from gala_sim.clamp import (
    FusionIssueScheduler, ReductionDomain, TaskKind, TaskPacket,
)


def _task(
    event_id: int, kind: TaskKind, reduction_key: int,
    domain: ReductionDomain = ReductionDomain.QUERY,
) -> TaskPacket:
    return TaskPacket(
        event_id, 1, event_id + 1, reduction_key, 1, 0, 1, event_id, kind,
        domain,
    )


def test_fusion_issue_enforces_candidate_and_port_contract() -> None:
    scheduler = FusionIssueScheduler(candidate_lanes=3, forward_ports=1,
                                     consumer_ports=1, adjoint_ports=1)
    decision = scheduler.issue([
        _task(0, TaskKind.FORWARD, 1),
        _task(1, TaskKind.FORWARD, 2),
        _task(2, TaskKind.CONSUMER, 3),
        _task(3, TaskKind.ADJOINT, 4),
    ])
    assert len(decision.accepted) == 3
    assert len(decision.rejected) == 1
    assert decision.reason == "conflict_or_port"


def test_query_state_history_is_only_used_after_round_boundary() -> None:
    scheduler = FusionIssueScheduler(candidate_lanes=3, forward_ports=1,
                                     consumer_ports=1, adjoint_ports=1)
    scheduler.observe_arrival([_task(0, TaskKind.FORWARD, 1)])
    state = scheduler.states[1]
    assert state.forecast() == (1, 0, 0)
    state.add_round_overlap(2)
    state.begin_round()
    assert state.forecast()[0] >= 1


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
