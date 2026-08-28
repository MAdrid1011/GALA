"""CLAMP task packets and overlap-guided issue state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class TaskKind(IntEnum):
    FORWARD = 1
    CONSUMER = 2
    ADJOINT = 3


class ReductionDomain(IntEnum):
    QUERY = 1
    GAUSSIAN = 2


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
    reduction_domain: ReductionDomain = ReductionDomain.QUERY
    conflict_query_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if min(self.event_id, self.query_id, self.gaussian_id, self.reduction_key,
               self.state_version, self.template_id, self.address_token) < 0:
            raise ValueError("task packet identifiers must be non-negative")
        if not isinstance(self.reduction_domain, ReductionDomain):
            raise ValueError("task packet reduction domain is invalid")
        if any(query_id < 0 for query_id in self.conflict_query_ids):
            raise ValueError("task packet conflict query IDs must be non-negative")
        if len(set(self.conflict_query_ids)) != len(self.conflict_query_ids):
            raise ValueError("task packet conflict query IDs must be unique")

    @property
    def conflict_key(self) -> tuple[ReductionDomain, int]:
        return self.reduction_domain, self.reduction_key

    @property
    def conflict_keys(self) -> frozenset[tuple[ReductionDomain, int]]:
        if self.reduction_domain is ReductionDomain.QUERY and self.conflict_query_ids:
            return frozenset(
                (ReductionDomain.QUERY, query_id)
                for query_id in self.conflict_query_ids
            )
        return frozenset((self.conflict_key,))


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
    _issued: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.candidate_lanes, self.forward_ports, self.consumer_ports, self.adjoint_ports) <= 0:
            raise ValueError("fusion issue widths must be positive")

    def observe_arrival(self, candidates: Iterable[TaskPacket]) -> None:
        """Record each task once when it reaches an issue input queue."""

        for task in candidates:
            if task.event_id not in self._observed:
                self.states.setdefault(task.query_id, QueryState()).observe(task)
                self._observed.add(task.event_id)

    def forecast(self, candidates: Iterable[TaskPacket]) -> list[TaskPacket]:
        """Order candidates without mutating scheduler state."""

        candidates = list(candidates)
        return sorted(candidates, key=self._sort_key)

    def issue(
        self,
        candidates: Iterable[TaskPacket],
        *,
        occupied_keys: set[tuple[ReductionDomain, int]] | None = None,
    ) -> IssueDecision:
        candidates = list(candidates)
        self.observe_arrival(candidates)
        decision = self.select(candidates, occupied_keys=occupied_keys)
        self.commit_issued(decision.accepted)
        return decision

    def select(
        self,
        candidates: Iterable[TaskPacket],
        *,
        occupied_keys: set[tuple[ReductionDomain, int]] | None = None,
    ) -> IssueDecision:
        """Choose conflict-free candidates without committing issue state."""

        return self.select_in_order(
            self.forecast(candidates), occupied_keys=occupied_keys
        )

    def select_in_order(
        self,
        candidates: Iterable[TaskPacket],
        *,
        occupied_keys: set[tuple[ReductionDomain, int]] | None = None,
    ) -> IssueDecision:
        """Apply fixed lane, port, and conflict limits to a supplied order."""

        ordered = list(candidates)
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
            if (
                used_ports[task.task_kind] >= limit
                or not task.conflict_keys.isdisjoint(occupied_keys)
            ):
                rejected.append(task)
                continue
            accepted.append(task)
            occupied_keys.update(task.conflict_keys)
            used_ports[task.task_kind] += 1
        rejected.extend(ordered[self.candidate_lanes:])
        reason = "conflict_or_port" if rejected else None
        return IssueDecision(tuple(accepted), tuple(rejected), reason)

    def commit_issued(self, tasks: Iterable[TaskPacket]) -> None:
        for task in tasks:
            if task.event_id in self._issued:
                raise ValueError(f"task {task.event_id} was issued more than once")
            self._issued.add(task.event_id)

    def _sort_key(self, task: TaskPacket) -> tuple[int, int, int, int]:
        forecast = self.states.get(task.query_id, QueryState()).forecast()
        priority = {
            TaskKind.FORWARD: forecast[0],
            TaskKind.CONSUMER: forecast[1],
            TaskKind.ADJOINT: forecast[2],
        }[task.task_kind]
        return (priority, task.query_id, task.reduction_key, task.event_id)
