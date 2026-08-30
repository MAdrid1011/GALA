"""CLAMP task packets and overlap-guided issue state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import heapq
from typing import Iterable


HistoryQueryKey = tuple[int, int]


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
    workset_hit: bool = False
    age: int = 0
    target_resource: int | None = None
    consumer_owner_ids: tuple[int, ...] = ()

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
        if self.age < 0:
            raise ValueError("task packet age must be non-negative")
        if self.target_resource is not None and self.target_resource < 0:
            raise ValueError("task packet target resource must be non-negative")
        if any(query_id < 0 for query_id in self.consumer_owner_ids):
            raise ValueError("task packet consumer owner IDs must be non-negative")
        if len(set(self.consumer_owner_ids)) != len(self.consumer_owner_ids):
            raise ValueError("task packet consumer owner IDs must be unique")

    @property
    def query_lane_ids(self) -> tuple[int, ...]:
        """Logical query lanes affected by this physical candidate."""

        return self.conflict_query_ids or (self.query_id,)

    @property
    def readiness_query_ids(self) -> tuple[int, ...]:
        """Query states whose results and producer status gate this task."""

        if self.task_kind is TaskKind.CONSUMER and self.consumer_owner_ids:
            return self.consumer_owner_ids
        return self.query_lane_ids

    @property
    def credit_owner_id(self) -> int | None:
        """The dependency owner receiving this consumer's one credit."""

        if self.consumer_owner_ids:
            return self.consumer_owner_ids[0]
        return self.query_id if self.task_kind is TaskKind.CONSUMER else None

    @property
    def conflict_key(self) -> tuple[ReductionDomain, int]:
        return self.reduction_domain, self.reduction_key

    @property
    def conflict_keys(self) -> frozenset[tuple[ReductionDomain, int]]:
        keys = {self.conflict_key}
        if self.conflict_query_ids:
            keys.update(
                (ReductionDomain.QUERY, query_id)
                for query_id in self.conflict_query_ids
            )
        return frozenset(keys)


@dataclass
class QueryState:
    """Live F/C/A and overlap-history state for one logical query lane."""

    forward_count: int = 0
    consumer_count: int = 0
    adjoint_count: int = 0
    current_round: int = 0
    previous_round: int = 0
    history_valid: bool = False
    generator_closed: bool = False
    reduction_ready: bool = False
    completion_emitted: bool = False

    def _adjust(self, field_name: str, delta: int) -> None:
        value = int(getattr(self, field_name)) + int(delta)
        if value < 0:
            raise ValueError(f"query {field_name} counter underflows")
        setattr(self, field_name, value)

    def accept_relation(self, *, needs_adjoint: bool, support_delta: int = 1) -> None:
        self._adjust("forward_count", 1)
        if needs_adjoint:
            self._adjust("adjoint_count", 1)
        self._adjust("current_round", support_delta)

    def retire_forward(self) -> None:
        self._adjust("forward_count", -1)

    def credit_successor(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("successor credit must be positive")
        self._adjust("consumer_count", count)
        if self.consumer_count > 255:
            raise ValueError("query consumer counter exceeds eight-bit sidecar")

    def dispatch_successor(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("successor dispatch must be positive")
        self._adjust("consumer_count", -count)

    def retire_adjoint(self) -> None:
        self._adjust("adjoint_count", -1)

    def mark_generator_closed(self) -> None:
        if self.generator_closed:
            raise ValueError("query generator closed more than once")
        self.generator_closed = True

    def mark_reduction_ready(self) -> None:
        if self.reduction_ready:
            raise ValueError("query reduction became ready more than once")
        self.reduction_ready = True

    def forecast(self) -> tuple[int, int, int]:
        history_delta = (
            max(self.previous_round - self.current_round, 0)
            if self.history_valid else 0
        )
        return (
            self.forward_count + history_delta,
            self.consumer_count,
            self.adjoint_count + history_delta,
        )

    def exact_remaining(self, kind: TaskKind) -> int:
        return {
            TaskKind.FORWARD: self.forward_count,
            TaskKind.CONSUMER: self.consumer_count,
            TaskKind.ADJOINT: self.adjoint_count,
        }[kind]

    def predicted_remaining(self, kind: TaskKind) -> int:
        predicted = self.forecast()
        return {
            TaskKind.FORWARD: predicted[0],
            TaskKind.CONSUMER: predicted[1],
            TaskKind.ADJOINT: predicted[2],
        }[kind]

    def exact_ready(self, kind: TaskKind) -> bool:
        if kind is TaskKind.FORWARD:
            return self.forward_count > 0
        common = (
            self.generator_closed
            and self.reduction_ready
            and self.forward_count == 0
        )
        if kind is TaskKind.CONSUMER:
            return common and self.consumer_count > 0
        return common and self.adjoint_count > 0

    @property
    def releasable(self) -> bool:
        return (
            self.generator_closed
            and self.reduction_ready
            and self.forward_count == 0
            and self.consumer_count == 0
            and self.adjoint_count == 0
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
class IssueScore:
    released_work: int
    completed_queries: int
    remaining_work: int
    workset_hit: bool
    age: int
    source: TaskKind


@dataclass(frozen=True)
class IssueDecision:
    accepted: tuple[TaskPacket, ...]
    rejected: tuple[TaskPacket, ...]
    reason: str | None


@dataclass
class FusionIssueScheduler:
    """Three FIFO-head forecast/conflict/issue scheduler."""

    candidate_lanes: int
    forward_ports: int
    consumer_ports: int
    adjoint_ports: int
    query_state_entries: int | None = None
    query_state_lanes: int = 1
    states: dict[int, QueryState] = field(default_factory=dict)
    _observed: set[int] = field(default_factory=set, init=False, repr=False)
    _issued: set[int] = field(default_factory=set, init=False, repr=False)
    _round_robin_head: TaskKind = field(
        default=TaskKind.FORWARD, init=False, repr=False,
    )
    enforce_exact_readiness: bool = field(default=False, init=False)
    strict_lifecycle: bool = field(default=False, init=False, repr=False)
    _arrival_cycle: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _clock: int = field(default=0, init=False, repr=False)
    _peak_state_packs: int = field(default=0, init=False, repr=False)
    _released_state_packs: int = field(default=0, init=False, repr=False)
    _pack_to_slot: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _slot_to_pack: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _live_queries_by_pack: dict[int, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    _release_candidates: set[int] = field(default_factory=set, init=False, repr=False)
    _free_slots: list[int] = field(default_factory=list, init=False, repr=False)
    _next_dynamic_slot: int = field(default=0, init=False, repr=False)
    _iteration_id: int | None = field(default=None, init=False, repr=False)
    _state_history_key: dict[int, HistoryQueryKey] = field(
        default_factory=dict, init=False, repr=False,
    )
    _support_current: dict[HistoryQueryKey, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    _support_previous: dict[HistoryQueryKey, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    _history_restored_queries: int = field(default=0, init=False, repr=False)
    _load_rule_evaluations: int = field(default=0, init=False, repr=False)
    _load_rule_selection_changes: int = field(default=0, init=False, repr=False)
    _history_candidate_evaluations: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.candidate_lanes, self.forward_ports, self.consumer_ports,
               self.adjoint_ports) <= 0:
            raise ValueError("fusion issue widths must be positive")
        if self.query_state_entries is not None and self.query_state_entries <= 0:
            raise ValueError("query-state capacity must be positive")
        if self.query_state_lanes <= 0:
            raise ValueError("query-state lane count must be positive")
        self._reset_slots()

    def _reset_slots(self) -> None:
        self._pack_to_slot.clear()
        self._slot_to_pack.clear()
        self._live_queries_by_pack.clear()
        self._release_candidates.clear()
        self._free_slots = (
            list(range(self.query_state_entries))
            if self.query_state_entries is not None else []
        )
        self._next_dynamic_slot = 0

    def reset(self) -> None:
        self.states.clear()
        self._observed.clear()
        self._issued.clear()
        self._round_robin_head = TaskKind.FORWARD
        self.enforce_exact_readiness = False
        self.strict_lifecycle = False
        self._arrival_cycle.clear()
        self._clock = 0
        self._peak_state_packs = 0
        self._released_state_packs = 0
        self._iteration_id = None
        self._state_history_key.clear()
        self._support_current.clear()
        self._support_previous.clear()
        self._history_restored_queries = 0
        self._load_rule_evaluations = 0
        self._load_rule_selection_changes = 0
        self._history_candidate_evaluations = 0
        self._reset_slots()

    def set_clock(self, cycle: int) -> None:
        if cycle < 0:
            raise ValueError("scheduler clock must be non-negative")
        self._clock = int(cycle)

    def enable_exact_readiness(self, enabled: bool = True) -> None:
        self.enforce_exact_readiness = bool(enabled)

    def set_strict_lifecycle(self, enabled: bool = True) -> None:
        self.strict_lifecycle = bool(enabled)

    @property
    def has_previous_support(self) -> bool:
        """Whether the immediately preceding iteration supplied support history."""

        return bool(self._support_previous)

    @property
    def peak_current_support(self) -> int:
        """Largest real support count observed for one live history identity."""

        return max(self._support_current.values(), default=0)

    def allocate(
        self,
        query_ids: Iterable[int],
        *,
        history_keys: Iterable[HistoryQueryKey] | None = None,
    ) -> None:
        query_ids = tuple(query_ids)
        resolved_history_keys = (
            tuple(
                self._state_history_key.get(query_id, (0, query_id))
                for query_id in query_ids
            )
            if history_keys is None else tuple(history_keys)
        )
        if len(resolved_history_keys) != len(query_ids):
            raise ValueError("query IDs and history keys must have equal length")
        self.release_completed()
        for query_id, history_key in zip(
            query_ids, resolved_history_keys, strict=True,
        ):
            if query_id < 0:
                raise ValueError("query ID must be non-negative")
            if len(history_key) != 2 or min(history_key) < 0:
                raise ValueError("query history key must be a non-negative pair")
            if query_id in self.states:
                if self._state_history_key[query_id] != history_key:
                    raise ValueError("live query changed its history identity")
                continue
            logical_pack = query_id // self.query_state_lanes
            if logical_pack not in self._pack_to_slot:
                if self.query_state_entries is None:
                    slot = self._next_dynamic_slot
                    self._next_dynamic_slot += 1
                else:
                    if not self._free_slots:
                        raise ValueError("query-state table is full")
                    slot = heapq.heappop(self._free_slots)
                self._pack_to_slot[logical_pack] = slot
                self._slot_to_pack[slot] = logical_pack
            previous = self._support_previous.get(history_key)
            self.states[query_id] = QueryState(
                current_round=self._support_current.get(history_key, 0),
                previous_round=previous or 0,
                history_valid=previous is not None,
            )
            self._live_queries_by_pack[logical_pack] = (
                self._live_queries_by_pack.get(logical_pack, 0) + 1
            )
            self._state_history_key[query_id] = history_key
            if previous is not None:
                self._history_restored_queries += 1
            self._peak_state_packs = max(
                self._peak_state_packs, len(self._pack_to_slot),
            )

    def _active_state_packs(self) -> set[int]:
        return set(self._pack_to_slot)

    def physical_location(self, query_id: int) -> tuple[int, int]:
        """Return the allocated physical Query State slot and lane."""

        logical_pack = query_id // self.query_state_lanes
        try:
            slot = self._pack_to_slot[logical_pack]
        except KeyError as error:
            raise ValueError(f"query {query_id} has no live query-state slot") from error
        return slot, query_id % self.query_state_lanes

    def relation_accept(
        self,
        query_ids: Iterable[int],
        *,
        adjoint_query_ids: Iterable[int] = (),
        support_delta: int = 1,
        iteration_id: int | None = None,
        history_keys: Iterable[HistoryQueryKey] | None = None,
    ) -> None:
        if iteration_id is not None:
            self.begin_iteration(iteration_id)
        query_ids = tuple(query_ids)
        adjoint_ids = set(adjoint_query_ids)
        resolved_history_keys = (
            tuple((0, query_id) for query_id in query_ids)
            if history_keys is None else tuple(history_keys)
        )
        self.allocate(query_ids, history_keys=resolved_history_keys)
        for query_id, history_key in zip(
            query_ids, resolved_history_keys, strict=True,
        ):
            self.states[query_id].accept_relation(
                needs_adjoint=query_id in adjoint_ids,
                support_delta=support_delta,
            )
            self._support_current[history_key] = (
                self._support_current.get(history_key, 0) + support_delta
            )

    def begin_iteration(self, iteration_id: int) -> None:
        """Roll observed support into the history sidecar at a real boundary."""

        if iteration_id < 0:
            raise ValueError("iteration ID must be non-negative")
        if self._iteration_id is None:
            self._iteration_id = iteration_id
            return
        if iteration_id < self._iteration_id:
            raise ValueError("query history iteration moves backwards")
        if iteration_id == self._iteration_id:
            return
        self.release_completed()
        if self.states:
            raise ValueError("query history advances with live F/C/A state")
        self._support_previous = (
            dict(self._support_current)
            if iteration_id == self._iteration_id + 1 else {}
        )
        self._support_current.clear()
        self._iteration_id = iteration_id

    def producer_close(self, query_ids: Iterable[int]) -> None:
        query_ids = tuple(query_ids)
        self.allocate(query_ids)
        for query_id in query_ids:
            state = self.states[query_id]
            state.mark_generator_closed()
            self._update_release_candidate(query_id, state)

    def reduction_writeback(self, query_ids: Iterable[int]) -> None:
        query_ids = tuple(query_ids)
        self.allocate(query_ids)
        for query_id in query_ids:
            state = self.states[query_id]
            state.mark_reduction_ready()
            self._update_release_candidate(query_id, state)

    def forward_retire(self, query_ids: Iterable[int]) -> None:
        for query_id in query_ids:
            state = self.states.get(query_id)
            if state is not None:
                state.retire_forward()
                self._update_release_candidate(query_id, state)

    def successor_credit(self, query_id: int, count: int = 1) -> bool:
        state = self.states.get(query_id)
        if state is None:
            return False
        state.credit_successor(count)
        self._update_release_candidate(query_id, state)
        return True

    def successor_dispatch(self, query_id: int, count: int = 1) -> bool:
        state = self.states.get(query_id)
        if state is None:
            return False
        state.dispatch_successor(count)
        self._update_release_candidate(query_id, state)
        return True

    def adjoint_retire(self, query_ids: Iterable[int]) -> None:
        for query_id in query_ids:
            state = self.states.get(query_id)
            if state is not None:
                state.retire_adjoint()
                self._update_release_candidate(query_id, state)

    def roll_history(self, query_ids: Iterable[int] | None = None) -> None:
        """Commit this round's observed support counts into previous history."""

        states = self.states.values() if query_ids is None else (
            self.states[query_id] for query_id in query_ids
        )
        for state in states:
            state.begin_round()

    def release_completed(self) -> int:
        """Release query entries only after every live counter has drained."""

        releasable = sorted(
            query_id for query_id in self._release_candidates
            if (state := self.states.get(query_id)) is not None and state.releasable
        )
        for query_id in releasable:
            del self.states[query_id]
            del self._state_history_key[query_id]
            self._release_candidates.discard(query_id)
            logical_pack = query_id // self.query_state_lanes
            live_queries = self._live_queries_by_pack[logical_pack] - 1
            if live_queries < 0:
                raise ValueError("query-state pack occupancy underflows")
            if live_queries:
                self._live_queries_by_pack[logical_pack] = live_queries
                continue
            del self._live_queries_by_pack[logical_pack]
            slot = self._pack_to_slot.pop(logical_pack)
            del self._slot_to_pack[slot]
            if self.query_state_entries is not None:
                heapq.heappush(self._free_slots, slot)
            self._released_state_packs += 1
        return len(releasable)

    def _update_release_candidate(self, query_id: int, state: QueryState) -> None:
        if state.releasable:
            self._release_candidates.add(query_id)
        else:
            self._release_candidates.discard(query_id)

    def state_table_snapshot(self) -> dict[str, int]:
        return {
            "query_state_entries_live": len(self._pack_to_slot),
            "query_state_entries_peak": self._peak_state_packs,
            "query_state_entries_released": self._released_state_packs,
            "query_state_lanes_live": len(self.states),
        }

    def history_snapshot(self) -> dict[str, int]:
        return {
            "query_history_restored": self._history_restored_queries,
            "query_history_candidate_evaluations": self._history_candidate_evaluations,
            "query_load_rule_evaluations": self._load_rule_evaluations,
            "query_load_rule_selection_changes": self._load_rule_selection_changes,
        }

    def observe_arrival(
        self, candidates: Iterable[TaskPacket], *, arrival_cycle: int | None = None,
    ) -> None:
        """Track candidate identity; counters are updated by lifecycle events."""

        for task in candidates:
            if task.event_id in self._observed:
                continue
            self._arrival_cycle[task.event_id] = (
                self._clock if arrival_cycle is None else int(arrival_cycle)
            )
            if not self.strict_lifecycle:
                self.allocate((task.query_id,))
                state = self.states[task.query_id]
                state._adjust({
                    TaskKind.FORWARD: "forward_count",
                    TaskKind.CONSUMER: "consumer_count",
                    TaskKind.ADJOINT: "adjoint_count",
                }[task.task_kind], 1)
            self._observed.add(task.event_id)

    def live_counter_totals(self) -> tuple[int, int, int]:
        return (
            sum(state.forward_count for state in self.states.values()),
            sum(state.consumer_count for state in self.states.values()),
            sum(state.adjoint_count for state in self.states.values()),
        )

    def require_drained(self) -> None:
        totals = self.live_counter_totals()
        if totals != (0, 0, 0):
            raise ValueError(
                "query lifecycle did not drain F/C/A: "
                f"F={totals[0]}, C={totals[1]}, A={totals[2]}"
            )

    def score(
        self, task: TaskPacket, *, use_history_prediction: bool = True,
    ) -> IssueScore:
        released_work = 0
        completed_queries = 0
        remaining_work = 0
        for query_id in task.query_lane_ids:
            state = self.states.get(query_id)
            if state is None:
                continue
            remaining = (
                state.predicted_remaining(task.task_kind)
                if use_history_prediction
                else state.exact_remaining(task.task_kind)
            )
            releases_lane = remaining == 1
            if releases_lane:
                released_work += 1
            if remaining > 0:
                remaining -= 1
            remaining_work += remaining
            if releases_lane and (
                task.task_kind is not TaskKind.FORWARD
                or (state.generator_closed and state.reduction_ready)
            ):
                completed_queries += 1
        return IssueScore(
            released_work=released_work,
            completed_queries=completed_queries,
            remaining_work=remaining_work,
            workset_hit=task.workset_hit,
            age=max(
                task.age,
                self._clock - self._arrival_cycle.get(task.event_id, self._clock),
            ),
            source=task.task_kind,
        )

    def forecast(
        self,
        candidates: Iterable[TaskPacket],
        *,
        use_history_prediction: bool = True,
    ) -> list[TaskPacket]:
        """Order FIFO heads using the authoritative runtime comparison key."""

        return sorted(
            candidates,
            key=lambda task: self._sort_key(
                task, use_history_prediction=use_history_prediction,
            ),
        )

    def compiler_admission_key(self, task: TaskPacket) -> tuple[int, ...]:
        """Rank ready compiler work using exact current F/C/A state."""

        return self._sort_key(task, use_history_prediction=False)

    def issue(
        self,
        candidates: Iterable[TaskPacket],
        *,
        occupied_keys: set[tuple[ReductionDomain, int]] | None = None,
        occupied_targets: set[int] | None = None,
        use_load_rules: bool = True,
        use_history_prediction: bool = True,
    ) -> IssueDecision:
        candidates = list(candidates)
        self.observe_arrival(candidates)
        decision = self.select(
            candidates,
            occupied_keys=occupied_keys,
            occupied_targets=occupied_targets,
            use_load_rules=use_load_rules,
            use_history_prediction=use_history_prediction,
        )
        self.commit_issued(decision.accepted)
        return decision

    def select(
        self,
        candidates: Iterable[TaskPacket],
        *,
        occupied_keys: set[tuple[ReductionDomain, int]] | None = None,
        occupied_targets: set[int] | None = None,
        use_load_rules: bool = True,
        use_history_prediction: bool = True,
    ) -> IssueDecision:
        """Choose ready conflict-free FIFO heads without committing state."""

        candidates = list(candidates)
        if not use_load_rules and not use_history_prediction:
            return self.select_in_order(
                candidates,
                occupied_keys=occupied_keys,
                occupied_targets=occupied_targets,
            )
        if use_load_rules:
            self._load_rule_evaluations += 1
        if use_history_prediction:
            self._history_candidate_evaluations += sum(
                any(
                    (state := self.states.get(query_id)) is not None
                    and state.history_valid
                    for query_id in task.query_lane_ids
                )
                for task in candidates
            )
        ordered = self.forecast(
            candidates, use_history_prediction=use_history_prediction,
        )
        baseline = self.select_in_order(
            candidates,
            occupied_keys=occupied_keys,
            occupied_targets=occupied_targets,
        )
        selected = self.select_in_order(
            ordered, occupied_keys=occupied_keys, occupied_targets=occupied_targets,
        )
        if use_load_rules and tuple(task.event_id for task in selected.accepted) != tuple(
            task.event_id for task in baseline.accepted
        ):
            self._load_rule_selection_changes += 1
        return selected

    def select_in_order(
        self,
        candidates: Iterable[TaskPacket],
        *,
        occupied_keys: set[tuple[ReductionDomain, int]] | None = None,
        occupied_targets: set[int] | None = None,
    ) -> IssueDecision:
        """Apply exact readiness, lane, port, and conflict limits."""

        ordered = list(candidates)
        occupied_keys = set(occupied_keys or ())
        occupied_targets = set(occupied_targets or ())
        accepted: list[TaskPacket] = []
        rejected: list[TaskPacket] = []
        used_ports = {
            TaskKind.FORWARD: 0,
            TaskKind.CONSUMER: 0,
            TaskKind.ADJOINT: 0,
        }
        for task in ordered[: self.candidate_lanes]:
            limit = {
                TaskKind.FORWARD: self.forward_ports,
                TaskKind.CONSUMER: self.consumer_ports,
                TaskKind.ADJOINT: self.adjoint_ports,
            }[task.task_kind]
            tracked_states = [
                self.states.get(query_id) for query_id in task.readiness_query_ids
            ]
            exact_ready = (
                not self.enforce_exact_readiness
                or all(
                    state is not None and state.exact_ready(task.task_kind)
                    for state in tracked_states
                )
            )
            if (
                not exact_ready
                or used_ports[task.task_kind] >= limit
                or not task.conflict_keys.isdisjoint(occupied_keys)
                or (
                    task.target_resource is not None
                    and task.target_resource in occupied_targets
                )
            ):
                rejected.append(task)
                continue
            accepted.append(task)
            occupied_keys.update(task.conflict_keys)
            if task.target_resource is not None:
                occupied_targets.add(task.target_resource)
            used_ports[task.task_kind] += 1
        rejected.extend(ordered[self.candidate_lanes:])
        reason = "conflict_or_port" if rejected else None
        return IssueDecision(tuple(accepted), tuple(rejected), reason)

    def commit_issued(self, tasks: Iterable[TaskPacket]) -> None:
        first_kind: TaskKind | None = None
        for task in tasks:
            if task.event_id in self._issued:
                raise ValueError(f"task {task.event_id} was issued more than once")
            self._issued.add(task.event_id)
            if first_kind is None:
                first_kind = task.task_kind
        if first_kind is not None:
            self._round_robin_head = TaskKind((int(first_kind) % len(TaskKind)) + 1)

    def _sort_key(
        self, task: TaskPacket, *, use_history_prediction: bool = True,
    ) -> tuple[int, int, int, int, int, int, int]:
        if not self.strict_lifecycle:
            priority = {
                TaskKind.FORWARD: self.states.get(task.query_id, QueryState()).forward_count,
                TaskKind.CONSUMER: self.states.get(task.query_id, QueryState()).consumer_count,
                TaskKind.ADJOINT: self.states.get(task.query_id, QueryState()).adjoint_count,
            }[task.task_kind]
            return (priority, task.query_id, task.reduction_key, 0, 0, 0, task.event_id)
        score = self.score(
            task, use_history_prediction=use_history_prediction,
        )
        distance = (
            int(score.source) - int(self._round_robin_head)
        ) % len(TaskKind)
        return (
            -score.released_work,
            -score.completed_queries,
            score.remaining_work,
            -int(score.workset_hit),
            -score.age,
            distance,
            task.event_id,
        )
