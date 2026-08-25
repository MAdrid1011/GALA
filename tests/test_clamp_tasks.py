from __future__ import annotations

from gala_sim.clamp import FusionIssueScheduler, TaskKind, TaskPacket


def _task(event_id: int, kind: TaskKind, reduction_key: int) -> TaskPacket:
    return TaskPacket(event_id, 1, event_id + 1, reduction_key, 1, 0, 1, event_id, kind)


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
    scheduler.forecast([_task(0, TaskKind.FORWARD, 1)])
    state = scheduler.states[1]
    assert state.forecast() == (1, 0, 0)
    state.add_round_overlap(2)
    state.begin_round()
    assert state.forecast()[0] >= 1
