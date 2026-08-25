"""CLAMP task packets and overlap-guided issue state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class TaskKind(IntEnum):
    FORWARD = 1
    CONSUMER = 2
    ADJOINT = 3


@dataclass(frozen=True)
class TaskPacket:
    event_id: int
    query_id: int
    gaussian_id: int
    reduction_key: int
    resource: int
    state_version: int
    template_id: int
    address_token: int
    task_kind: TaskKind

    def __post_init__(self) -> None:
        if min(self.event_id, self.query_id, self.gaussian_id, self.reduction_key,
               self.state_version, self.template_id, self.address_token) < 0:
            raise ValueError("task packet identifiers must be non-negative")


@dataclass
class QueryState:
    forward_count: int = 0
    consumer_count: int = 0
    adjoint_count: int = 0
    current_round: int = 0
    previous_round: int = 0
    history_valid: bool = False
    gaussian_done: int = 0
    output_ready: int = 0

    def observe(self, task: TaskPacket) -> None:
        if task.task_kind is TaskKind.FORWARD:
            self.forward_count += 1
        elif task.task_kind is TaskKind.CONSUMER:
            self.consumer_count += 1
        elif task.task_kind is TaskKind.ADJOINT:
            self.adjoint_count += 1
        else:  # pragma: no cover - IntEnum guards this path
            raise ValueError("unknown task kind")

    def forecast(self) -> tuple[int, int, int]:
        history_delta = max(self.previous_round - self.current_round, 0) if self.history_valid else 0
        return (
            self.forward_count + history_delta,
            self.consumer_count,
            self.adjoint_count + history_delta,
        )

    def begin_round(self) -> None:
        self.previous_round = self.current_round
        self.current_round = 0
        self.history_valid = True

    def add_round_overlap(self, overlap_count: int) -> None:
        if overlap_count < 0:
            raise ValueError("overlap count must be non-negative")
        self.current_round += overlap_count


@dataclass(frozen=True)
class IssueDecision:
    accepted: tuple[TaskPacket, ...]
    rejected: tuple[TaskPacket, ...]
    reason: str | None


@dataclass
class FusionIssueScheduler:
    """Three-lane forecast/conflict/issue scheduler with explicit port limits."""

    candidate_lanes: int
    forward_ports: int
    consumer_ports: int
    adjoint_ports: int
    states: dict[int, QueryState] = field(default_factory=dict)
    _observed: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.candidate_lanes, self.forward_ports, self.consumer_ports, self.adjoint_ports) <= 0:
            raise ValueError("fusion issue widths must be positive")

    def forecast(self, candidates: Iterable[TaskPacket]) -> list[TaskPacket]:
        candidates = list(candidates)
        for task in candidates:
            if task.event_id not in self._observed:
                self.states.setdefault(task.query_id, QueryState()).observe(task)
                self._observed.add(task.event_id)
        return sorted(candidates, key=self._sort_key)

    def issue(self, candidates: Iterable[TaskPacket], *, occupied_keys: set[int] | None = None) -> IssueDecision:
        ordered = self.forecast(candidates)
        occupied_keys = set(occupied_keys or ())
        accepted: list[TaskPacket] = []
        rejected: list[TaskPacket] = []
        used_ports = {TaskKind.FORWARD: 0, TaskKind.CONSUMER: 0, TaskKind.ADJOINT: 0}
        for task in ordered[: self.candidate_lanes]:
            limit = {
                TaskKind.FORWARD: self.forward_ports,
                TaskKind.CONSUMER: self.consumer_ports,
                TaskKind.ADJOINT: self.adjoint_ports,
            }[task.task_kind]
            if used_ports[task.task_kind] >= limit or task.reduction_key in occupied_keys:
                rejected.append(task)
                continue
            accepted.append(task)
            occupied_keys.add(task.reduction_key)
            used_ports[task.task_kind] += 1
        rejected.extend(ordered[self.candidate_lanes:])
        reason = "conflict_or_port" if rejected else None
        return IssueDecision(tuple(accepted), tuple(rejected), reason)

    def _sort_key(self, task: TaskPacket) -> tuple[int, int, int, int]:
        forecast = self.states[task.query_id].forecast()
        priority = {
            TaskKind.FORWARD: forecast[0],
            TaskKind.CONSUMER: forecast[1],
            TaskKind.ADJOINT: forecast[2],
        }[task.task_kind]
        return (priority, task.query_id, task.reduction_key, task.event_id)
