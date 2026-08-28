"""Dependency-aware, event-jumping cycle executor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import heapq
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping

import numpy as np

from gala_sim.clamp import (
    FusionIssueScheduler, ReductionDomain, SemanticWorksets, TaskKind, TaskPacket,
)
from gala_sim.clamp.events import (
    PrimitiveKind, ResourceClass, TraceEvent, dependency_dtype, event_dtype,
)
from gala_sim.trace.model import Trace
from gala_sim.trace.virtual import (
    VirtualEventPacket,
    VirtualEventStreamValidator,
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualTraceLifecycleValidator,
    VirtualQueryEventExpander,
    VirtualTracePacket,
)
from gala_sim.trace.io import TraceReader
from gala_sim.clamp.builder import ChunkedTraceBuilder
from gala_sim.trace.validator import validate_trace

from .config import CycleConfig, ModuleTiming
from .memory import MemoryRequestRecord
from .modules import (
    BidirectionalQueryUnit,
    CacheBackpressure,
    ComputePod,
    CounterBlock,
    FusionIssueUnit,
    GaussianSemanticCache,
    RelationConstructor,
    ReconstructionUpdateUnit,
    SharedSram,
    StallRecord,
    CacheLookup,
    SemanticCacheState,
)


class CycleConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MechanismSelection:
    query_load_rules: bool
    semantic_worksets: bool
    overlap_guided_issue: bool
    semantic_residency: bool
    query_oracle: bool = False
    residency_oracle: bool = False


@dataclass(frozen=True)
class CycleResult:
    total_cycles: int
    module_counters: dict[str, dict[str, int]]
    stalls: tuple[StallRecord, ...]
    completion_cycles: dict[int, int]
    event_counts: dict[str, int]
    policy: str
    oracle_status: str
    memory_requests: tuple[MemoryRequestRecord, ...]


@dataclass(frozen=True)
class CycleProgress:
    phase: str
    completed_events: int
    total_events: int
    completed_iterations: int
    total_iterations: int
    last_completed_iteration: int | None
    simulated_cycles: int
    elapsed_seconds: float
    last_completed_iteration_events: int | None = None
    last_completed_iteration_cycles: int | None = None
    last_completed_iteration_elapsed_seconds: float | None = None


@dataclass
class _StallAccumulator:
    cycle: int
    module: str
    reason: str
    event_ids: list[int]
    count: int = 1


@dataclass
class _InFlight:
    completion_cycle: int
    event_id: int
    module: str


@dataclass
class _DependencyIndex:
    """Compressed reverse edges and mutable unsatisfied counts for one trace."""

    remaining: np.ndarray
    offsets: np.ndarray
    dependents: np.ndarray

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        progress: Callable[[str, int, int], None] | None = None,
        progress_interval_events: int | None = None,
        progress_interval_seconds: float | None = None,
    ) -> "_DependencyIndex":
        event_count = trace.event_count
        remaining = np.asarray(
            trace.events["dependency_count"], dtype=np.uint32
        ).copy()
        reverse_counts = np.zeros(event_count, dtype=np.uint64)
        if trace.dependencies.size:
            scan_size = max(progress_interval_events or trace.dependencies.size, 1)
            next_time = (
                time.monotonic() + progress_interval_seconds
                if progress_interval_seconds is not None else None
            )
            for start in range(0, int(trace.dependencies.size), scan_size):
                end = min(start + scan_size, int(trace.dependencies.size))
                np.add.at(reverse_counts, trace.dependencies[start:end], 1)
                if progress is not None and next_time is not None and time.monotonic() >= next_time:
                    progress("dependency_count", end, int(trace.dependencies.size))
                    next_time = time.monotonic() + progress_interval_seconds
        offsets = np.empty(event_count + 1, dtype=np.uint64)
        offsets[0] = 0
        np.cumsum(reverse_counts, out=offsets[1:])
        dependents = np.empty(trace.dependencies.size, dtype=np.uint64)
        cursors = offsets[:-1].copy()
        started_at = time.monotonic()
        next_time = (
            started_at + progress_interval_seconds
            if progress_interval_seconds is not None else None
        )
        for row_index, row in enumerate(trace.events, start=1):
            event_id = int(row["event_id"])
            for raw_dependency in trace.dependency_ids(row):
                dependency = int(raw_dependency)
                position = int(cursors[dependency])
                dependents[position] = event_id
                cursors[dependency] += 1
            if (
                progress is not None
                and next_time is not None
                and time.monotonic() >= next_time
            ):
                progress("dependency_fill", row_index, event_count)
                now = time.monotonic()
                next_time = now + progress_interval_seconds
        if progress is not None:
            progress("dependency_fill", event_count, event_count)
        return cls(remaining, offsets, dependents)

    def for_event(self, event_id: int):
        begin = int(self.offsets[event_id])
        end = int(self.offsets[event_id + 1])
        return self.dependents[begin:end]


def _iteration_index_from_metadata(
    trace: Trace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Use capture-provided exact iteration totals when available."""

    raw_counts = trace.metadata.get("iteration_event_counts")
    if raw_counts is None:
        return None
    if not isinstance(raw_counts, Mapping) or not raw_counts:
        raise CycleConfigurationError("iteration_event_counts metadata is malformed")
    pairs: list[tuple[int, int]] = []
    for raw_iteration, raw_count in raw_counts.items():
        try:
            iteration = int(raw_iteration)
            count = int(raw_count)
        except (TypeError, ValueError) as error:
            raise CycleConfigurationError(
                "iteration_event_counts metadata is malformed"
            ) from error
        if iteration < 0 or count <= 0:
            raise CycleConfigurationError("iteration_event_counts metadata is malformed")
        pairs.append((iteration, count))
    pairs.sort()
    if len({iteration for iteration, _ in pairs}) != len(pairs):
        raise CycleConfigurationError("iteration_event_counts metadata has duplicate IDs")
    if sum(count for _, count in pairs) != trace.event_count:
        raise CycleConfigurationError(
            "iteration_event_counts metadata does not cover the trace"
        )
    iteration_ids = np.asarray([iteration for iteration, _ in pairs], dtype=np.uint32)
    iteration_totals = np.asarray([count for _, count in pairs], dtype=np.uint64)
    iteration_positions = np.full(int(iteration_ids[-1]) + 1, -1, dtype=np.int64)
    iteration_positions[iteration_ids] = np.arange(iteration_ids.size)
    iteration_completed = np.zeros(iteration_ids.size, dtype=np.uint64)
    iteration_done = np.zeros(iteration_ids.size, dtype=np.bool_)
    return (
        iteration_ids, iteration_totals, iteration_positions,
        iteration_completed, iteration_done,
    )


class CycleEngine:
    """Run the same trace for Base, Oracle, or mechanism-specific policies."""

    def __init__(self, config: CycleConfig, *, policy: str = "base") -> None:
        valid_variants = {f"variant:{bits:04b}" for bits in range(16)}
        if policy not in {"base", "query_oracle", "residency_oracle", "query", "residency", "full"} | valid_variants:
            raise ValueError(f"unknown cycle policy: {policy}")
        self.config = config
        self.policy = policy
        self.selection = self._selection_for_policy(policy)
        self.issue_scheduler = FusionIssueScheduler(
            candidate_lanes=config.candidate_lanes,
            forward_ports=config.fusion_forward_ports or config.modules["fusion_issue"].ports,
            consumer_ports=config.fusion_consumer_ports or config.modules["fusion_issue"].ports,
            adjoint_ports=config.fusion_adjoint_ports or config.modules["fusion_issue"].ports,
        )
        counters = {name: CounterBlock() for name in config.modules}
        self.modules = {
            "relation_constructor": RelationConstructor("relation_constructor", config.modules["relation_constructor"], counters["relation_constructor"]),
            "fusion_issue": FusionIssueUnit("fusion_issue", config.modules["fusion_issue"], counters["fusion_issue"]),
            "semantic_cache": GaussianSemanticCache("semantic_cache", config.modules["semantic_cache"], counters["semantic_cache"]),
            "compute_pod": ComputePod("compute_pod", config.modules["compute_pod"], counters["compute_pod"]),
            "bidirectional_query": BidirectionalQueryUnit("bidirectional_query", config.modules["bidirectional_query"], counters["bidirectional_query"]),
            "reconstruction_update": ReconstructionUpdateUnit("reconstruction_update", config.modules["reconstruction_update"], counters["reconstruction_update"]),
            "shared_sram": SharedSram("shared_sram", config.modules["shared_sram"], counters["shared_sram"]),
        }
        self._stalls: list[_StallAccumulator] = []
        self._stall_index: dict[tuple[int, str, str], int] = {}
        self._stall_cycle: int | None = None

    def _record_stall(self, cycle: int, module: str, reason: str, event_id: int) -> None:
        if self._stall_cycle != cycle:
            self._stall_index.clear()
            self._stall_cycle = cycle
        key = (cycle, module, reason)
        index = self._stall_index.get(key)
        if index is None:
            self._stall_index[key] = len(self._stalls)
            self._stalls.append(_StallAccumulator(cycle, module, reason, [event_id]))
            return
        previous = self._stalls[index]
        if len(previous.event_ids) < self.config.candidate_lanes:
            previous.event_ids.append(event_id)
        previous.count += 1

    def _stall_records(self) -> tuple[StallRecord, ...]:
        return tuple(
            StallRecord(item.cycle, item.module, item.reason,
                        tuple(item.event_ids), item.count)
            for item in self._stalls
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _stages_for(kind: PrimitiveKind) -> tuple[str, ...]:
        if kind in {PrimitiveKind.RELATION_CANDIDATE, PrimitiveKind.RELATION, PrimitiveKind.QUERY_CLOSE}:
            return ("relation_constructor",)
        if kind in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN}:
            return ("semantic_cache", "shared_sram")
        if kind in {
            PrimitiveKind.UPDATE_BEGIN,
            PrimitiveKind.UPDATE_END,
            PrimitiveKind.UPDATE_COMMIT,
            PrimitiveKind.SET_MODIFICATION,
        }:
            return ("reconstruction_update",)
        if kind is PrimitiveKind.QUERY_REDUCTION:
            return ("bidirectional_query",)
        if kind is PrimitiveKind.GRADIENT_REDUCTION:
            return ("bidirectional_query", "compute_pod")
        if kind is PrimitiveKind.CONSUMER:
            return ("fusion_issue", "compute_pod", "bidirectional_query")
        if kind is PrimitiveKind.ADJOINT:
            return ("fusion_issue", "compute_pod", "bidirectional_query")
        return ("fusion_issue", "compute_pod")

    def _ordered_candidates(self, trace: Trace, candidates: list[int]) -> list[int]:
        base_order = candidates
        if self.selection.semantic_residency:
            base_order = sorted(base_order, key=lambda event_id: (
                0 if PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN} else 1,
                int(trace.events[event_id]["gaussian_id"]), event_id,
            ))
        if self.selection.query_load_rules:
            base_order = sorted(base_order, key=lambda event_id: (
                int(trace.events[event_id]["query_id"]),
                int(trace.events[event_id]["relation_id"]), event_id,
            ))
        return base_order

    @staticmethod
    def _task_packet(trace: Trace, event_id: int) -> TaskPacket:
        row = trace.events[event_id]
        kind = PrimitiveKind(int(row["primitive_kind"]))
        task_kind = {
            PrimitiveKind.FORWARD: TaskKind.FORWARD,
            PrimitiveKind.CONSUMER: TaskKind.CONSUMER,
            PrimitiveKind.ADJOINT: TaskKind.ADJOINT,
        }[kind]
        query_id = int(row["query_id"])
        gaussian_id = int(row["gaussian_id"])
        reduction_key = int(row["reduction_key"])
        if task_kind is TaskKind.ADJOINT:
            domain = ReductionDomain.GAUSSIAN
            semantic_key = gaussian_id
        else:
            domain = ReductionDomain.QUERY
            semantic_key = reduction_key if reduction_key >= 0 else query_id
        if semantic_key < 0:
            raise CycleConfigurationError(
                f"{kind.name} event {event_id} lacks a semantic reduction key"
            )
        return TaskPacket(
            event_id=event_id,
            query_id=max(query_id, 0),
            gaussian_id=max(gaussian_id, 0),
            reduction_key=semantic_key,
            resource=int(row["resource_class"]),
            reduction_domain=domain,
            state_version=int(row["state_version"]),
            template_id=int(row["template_id"]),
            address_token=int(row["address_token"]),
            task_kind=task_kind,
        )

    @staticmethod
    def _selection_for_policy(policy: str) -> MechanismSelection:
        aliases = {
            "base": "0000",
            "query": "1010",
            "residency": "0101",
            "full": "1111",
            "query_oracle": "1010",
            "residency_oracle": "0101",
        }
        bits = aliases.get(policy, policy.removeprefix("variant:"))
        return MechanismSelection(
            query_load_rules=bits[0] == "1",
            semantic_worksets=bits[1] == "1",
            overlap_guided_issue=bits[2] == "1",
            semantic_residency=bits[3] == "1",
            query_oracle=policy == "query_oracle",
            residency_oracle=policy == "residency_oracle",
        )

    def _fusion_port_limit(self, kind: PrimitiveKind) -> tuple[str, int]:
        timing_ports = self.config.modules["fusion_issue"].ports
        if kind is PrimitiveKind.FORWARD:
            return "forward", self.config.fusion_forward_ports or timing_ports
        if kind is PrimitiveKind.CONSUMER:
            return "consumer", self.config.fusion_consumer_ports or timing_ports
        if kind is PrimitiveKind.ADJOINT:
            return "adjoint", self.config.fusion_adjoint_ports or timing_ports
        raise CycleConfigurationError(f"invalid fusion task kind: {kind.name}")

    def _residency_states(self) -> dict[int, SemanticCacheState]:
        if self.config.cache_instances is None:
            raise CycleConfigurationError(
                "semantic residency requires cache resource parameters"
            )
        if any(value is None for value in (
            self.config.cache_capacity_per_instance,
            self.config.cache_directory_banks,
            self.config.cache_sector_bytes,
        )):
            raise CycleConfigurationError(
                "semantic residency requires complete cache resource parameters"
            )
        return {
            instance: SemanticCacheState.create(
                capacity=self.config.cache_capacity_per_instance,
                directory_banks=self.config.cache_directory_banks,
                sector_bytes=self.config.cache_sector_bytes,
            )
            for instance in range(self.config.cache_instances)
        }

    def _cache_instance(self, gaussian_id: int) -> int:
        instances = self.config.cache_instances
        if instances is None:
            raise CycleConfigurationError("cache instance count is not configured")
        return gaussian_id % instances

    def run(
        self,
        trace: Trace,
        *,
        validate_input: bool = True,
        progress: Callable[[CycleProgress], None] | None = None,
        progress_interval_events: int | None = None,
        progress_interval_seconds: float | None = None,
    ) -> CycleResult:
        """Run a trace; only callers that just validated it may disable validation."""
        progress_values = (
            progress is not None,
            progress_interval_events is not None,
            progress_interval_seconds is not None,
        )
        if any(progress_values) and not all(progress_values):
            raise ValueError("cycle progress callback and intervals must be provided together")
        if progress_interval_events is not None and progress_interval_events <= 0:
            raise ValueError("cycle progress interval must be positive")
        if progress_interval_seconds is not None and progress_interval_seconds <= 0:
            raise ValueError("cycle progress seconds must be positive")
        started_at = time.monotonic()

        def run_phase(phase: str, operation, *, total_iterations: int = 0):
            if progress is None or progress_interval_seconds is None:
                return operation()
            progress(CycleProgress(
                phase=phase,
                completed_events=0,
                total_events=trace.event_count,
                completed_iterations=0,
                total_iterations=total_iterations,
                last_completed_iteration=None,
                simulated_cycles=0,
                elapsed_seconds=time.monotonic() - started_at,
            ))
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(operation)
                while True:
                    try:
                        return future.result(timeout=progress_interval_seconds)
                    except FutureTimeoutError:
                        progress(CycleProgress(
                            phase=phase,
                            completed_events=0,
                            total_events=trace.event_count,
                            completed_iterations=0,
                            total_iterations=total_iterations,
                            last_completed_iteration=None,
                            simulated_cycles=0,
                            elapsed_seconds=time.monotonic() - started_at,
                        ))

        if validate_input:
            run_phase("validation", lambda: validate_trace(trace))
        if not self.config.modules:
            raise CycleConfigurationError("cycle modules are not configured")
        completed: dict[int, int] = {}
        iteration_ids = np.array([], dtype=np.uint32)
        iteration_totals = np.array([], dtype=np.uint64)
        iteration_positions = np.array([], dtype=np.int64)
        iteration_completed = np.array([], dtype=np.uint64)
        iteration_done = np.array([], dtype=np.bool_)
        contiguous_iteration_position = 0
        contiguous_iteration_events = 0
        completed_iterations = 0
        last_completed_iteration: int | None = None
        last_completed_iteration_events: int | None = None
        last_completed_iteration_cycles: int | None = None
        last_completed_iteration_elapsed_seconds: float | None = None
        metadata_iteration_index = _iteration_index_from_metadata(trace)
        if metadata_iteration_index is not None:
            (
                iteration_ids, iteration_totals, iteration_positions,
                iteration_completed, iteration_done,
            ) = metadata_iteration_index
        elif progress is not None:
            iteration_values = np.asarray(trace.events["iteration_id"])
            max_iteration = int(iteration_values.max()) if iteration_values.size else 0
            iteration_totals = np.zeros(max_iteration + 1, dtype=np.uint64)
            scan_events = max(progress_interval_events or 1, 1)
            next_iteration_progress_time = time.monotonic() + progress_interval_seconds
            for start in range(0, int(iteration_values.size), scan_events):
                chunk = iteration_values[start:start + scan_events]
                iteration_totals += np.bincount(
                    chunk.astype(np.int64, copy=False), minlength=max_iteration + 1
                ).astype(np.uint64, copy=False)
                if time.monotonic() >= next_iteration_progress_time:
                    progress(CycleProgress(
                        phase="iteration_index",
                        completed_events=0,
                        total_events=trace.event_count,
                        completed_iterations=0,
                        total_iterations=0,
                        last_completed_iteration=None,
                        simulated_cycles=0,
                        elapsed_seconds=time.monotonic() - started_at,
                    ))
                    next_iteration_progress_time = time.monotonic() + progress_interval_seconds
            iteration_ids = np.flatnonzero(iteration_totals).astype(np.uint32)
            compact_totals = iteration_totals[iteration_ids]
            iteration_positions = np.full(max_iteration + 1, -1, dtype=np.int64)
            iteration_positions[iteration_ids] = np.arange(iteration_ids.size)
            iteration_totals = compact_totals
            iteration_completed = np.zeros(iteration_ids.size, dtype=np.uint64)
            iteration_done = np.zeros(iteration_ids.size, dtype=np.bool_)

        def dependency_index_progress(stage: str, current: int, total: int) -> None:
            if progress is None:
                return
            progress(CycleProgress(
                phase=f"dependency_index_{stage}",
                completed_events=0,
                total_events=trace.event_count,
                completed_iterations=0,
                total_iterations=int(iteration_ids.size),
                last_completed_iteration=None,
                simulated_cycles=0,
                elapsed_seconds=time.monotonic() - started_at,
            ))

        dependency_index = run_phase(
            "dependency_index",
            lambda: _DependencyIndex.from_trace(
                trace,
                progress=dependency_index_progress if progress is not None else None,
                progress_interval_events=progress_interval_events,
                progress_interval_seconds=progress_interval_seconds,
            ),
            total_iterations=int(iteration_ids.size),
        )

        def build_ready() -> list[tuple[int, int]]:
            return [
                (int(event_id), 0)
                for event_id in np.flatnonzero(dependency_index.remaining == 0)
            ]

        initial_ready = run_phase(
            "ready_queue", build_ready, total_iterations=int(iteration_ids.size)
        )
        ready: list[tuple[int, int]] = []
        fusion_pending: list[tuple[int, TaskKind]] = []
        fusion_inputs: dict[TaskKind, list[int]] = {
            TaskKind.FORWARD: [],
            TaskKind.CONSUMER: [],
            TaskKind.ADJOINT: [],
        }

        def fusion_kind(event_id: int, stage: int) -> TaskKind | None:
            if not self.selection.overlap_guided_issue or stage != 0:
                return None
            kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
            return {
                PrimitiveKind.FORWARD: TaskKind.FORWARD,
                PrimitiveKind.CONSUMER: TaskKind.CONSUMER,
                PrimitiveKind.ADJOINT: TaskKind.ADJOINT,
            }.get(kind)

        def push_ready(event_id: int, stage: int) -> None:
            task_kind = fusion_kind(event_id, stage)
            if task_kind is None:
                heapq.heappush(ready, (event_id, stage))
            else:
                heapq.heappush(fusion_pending, (event_id, task_kind))

        def requeue_candidate(event_id: int, stage: int) -> None:
            task_kind = fusion_kind(event_id, stage)
            if task_kind is None:
                heapq.heappush(ready, (event_id, stage))
            else:
                heapq.heappush(fusion_inputs[task_kind], event_id)

        for event_id, stage in initial_ready:
            push_ready(event_id, stage)
        remaining_events = len(trace.events)
        in_flight: list[tuple[int, int, int, str]] = []
        module_busy_until = {
            name: [0] * module.timing.ports
            for name, module in self.modules.items()
        }
        fusion_busy_until = {name: 0 for name in ("forward", "consumer", "adjoint")}
        module_inflight = {name: 0 for name in self.modules}
        relation_seed_inflight = 0
        bank_busy: dict[tuple[str, int, int], int] = {}
        residency_enabled = self.selection.semantic_residency
        cache_states = (
            self._residency_states()
            if residency_enabled and self.config.cache_instances is not None
            else {}
        )
        async_memory = callable(getattr(self.config.memory, "submit_async", None))
        cache_fill_done: dict[tuple[int, tuple[int, int]], int] = {}
        cache_fill_request: dict[tuple[int, tuple[int, int]], int] = {}
        memory_waiters: dict[int, list[tuple[int, int, str, int]]] = {}
        cache_event_state: dict[int, tuple[SemanticCacheState, tuple[int, int], CacheLookup]] = {}
        cache_keys_by_version: dict[
            int, list[tuple[SemanticCacheState, tuple[int, int]]]
        ] = {}
        semantic_worksets = (
            SemanticWorksets.from_trace(trace)
            if self.selection.semantic_worksets else None
        )
        closed_versions: set[int] = set()
        memory_requests = 0
        cycle = 0
        next_progress_event = progress_interval_events
        next_progress_time = (
            started_at + progress_interval_seconds
            if progress_interval_seconds is not None else None
        )
        last_progress_completed = -1
        ready_scan_window = max(
            self.config.candidate_lanes,
            sum(module.timing.ports for module in self.modules.values()),
        )
        while remaining_events or in_flight:
            # Bank reservations are scoped to one cycle.  Drop old entries so
            # long traces do not retain one dictionary item per cycle.
            bank_busy = {
                key: value for key, value in bank_busy.items()
                if key[1] >= cycle
            }
            progressed = False
            if async_memory:
                self.config.memory.advance(cycle)  # type: ignore[attr-defined]
                for memory_record in self.config.memory.pop_completions():  # type: ignore[attr-defined]
                    waiters = memory_waiters.pop(memory_record.request_id, None)
                    if not waiters or memory_record.completion_cycle is None:
                        raise CycleConfigurationError(
                            "Ramulator completion has no pending cycle event"
                        )
                    for event_id, stage, module_name, service_finish in waiters:
                        completion = max(service_finish, memory_record.completion_cycle)
                        if memory_record.completion_cycle > service_finish:
                            self.modules[module_name].counters.memory_wait_cycles += (
                                memory_record.completion_cycle - service_finish
                            )
                        heapq.heappush(
                            in_flight, (completion, event_id, stage, module_name)
                        )
                    progressed = True
            while in_flight and in_flight[0][0] <= cycle:
                finish, event_id, stage, module_name = heapq.heappop(in_flight)
                self.modules[module_name].complete(event_id, finish)
                module_inflight[module_name] -= 1
                if module_inflight[module_name] < 0:
                    raise CycleConfigurationError(
                        f"negative in-flight count for {module_name}"
                    )
                stages = self._stages_for(PrimitiveKind(int(trace.events[event_id]["primitive_kind"])))
                kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                if kind is PrimitiveKind.CACHE_REQUEST and stage == 0 and event_id in cache_event_state:
                    state, key, lookup = cache_event_state[event_id]
                    if lookup is CacheLookup.MISS:
                        workset = (
                            semantic_worksets.for_event(event_id)
                            if semantic_worksets is not None else None
                        )
                        state.fill_complete(
                            key,
                            remaining_uses=(
                                int(workset["remaining_uses"])
                                if workset is not None else 1
                            ),
                        )
                        if int(trace.events[event_id]["state_version"]) in closed_versions:
                            state.close(key)
                if cache_states and kind is PrimitiveKind.CACHE_RETURN and stage == len(stages) - 1:
                    request_ids = trace.dependency_ids(trace.events[event_id])
                    if len(request_ids) != 1 or int(request_ids[0]) not in cache_event_state:
                        raise CycleConfigurationError(
                            f"cache return {event_id} has no captured request state"
                        )
                    state, key, _ = cache_event_state[int(request_ids[0])]
                    state.complete_read(key)
                    if semantic_worksets is not None:
                        request_event_id = int(request_ids[0])
                        if bool(semantic_worksets.for_event(request_event_id)["last_use"]):
                            state.close(key)
                    if int(trace.events[event_id]["state_version"]) in closed_versions:
                        state.close(key)
                if kind is PrimitiveKind.UPDATE_END and stage == len(stages) - 1:
                    state_version = int(trace.events[event_id]["state_version"])
                    if int(trace.events[event_id]["field_mask"]) != 0:
                        closed_versions.add(state_version)
                        for state, key in cache_keys_by_version.get(state_version, []):
                            state.close(key)
                if stage + 1 < len(stages):
                    push_ready(event_id, stage + 1)
                else:
                    completed[event_id] = finish
                    remaining_events -= 1
                    if progress is not None:
                        iteration_id = int(trace.events[event_id]["iteration_id"])
                        iteration_position = int(iteration_positions[iteration_id])
                        iteration_completed[iteration_position] += 1
                        if (
                            iteration_completed[iteration_position]
                            == iteration_totals[iteration_position]
                        ):
                            iteration_done[iteration_position] = True
                            while (
                                contiguous_iteration_position < iteration_done.size
                                and iteration_done[contiguous_iteration_position]
                            ):
                                completed_iterations += 1
                                last_completed_iteration = int(
                                    iteration_ids[contiguous_iteration_position]
                                )
                                contiguous_iteration_events += int(
                                    iteration_totals[contiguous_iteration_position]
                                )
                                last_completed_iteration_events = contiguous_iteration_events
                                last_completed_iteration_cycles = finish
                                last_completed_iteration_elapsed_seconds = (
                                    time.monotonic() - started_at
                                )
                                contiguous_iteration_position += 1
                    for raw_dependent in dependency_index.for_event(event_id):
                        dependent = int(raw_dependent)
                        remaining = int(dependency_index.remaining[dependent])
                        if remaining <= 0:
                            raise CycleConfigurationError(
                                f"dependency count underflow at event {dependent}"
                            )
                        dependency_index.remaining[dependent] = remaining - 1
                        if remaining == 1:
                            push_ready(dependent, 0)
                if (PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                        is PrimitiveKind.RELATION_CANDIDATE):
                    relation_seed_inflight -= 1
                    if relation_seed_inflight < 0:
                        raise CycleConfigurationError("negative relation seed FIFO occupancy")
                progressed = True
            if (
                progress is not None
                and next_progress_event is not None
                and next_progress_time is not None
                and (
                    len(completed) >= next_progress_event
                    or time.monotonic() >= next_progress_time
                )
            ):
                now = time.monotonic()
                progress(CycleProgress(
                    phase="replay",
                    completed_events=len(completed),
                    total_events=trace.event_count,
                    completed_iterations=completed_iterations,
                    total_iterations=int(iteration_ids.size),
                    last_completed_iteration=last_completed_iteration,
                    simulated_cycles=cycle,
                    elapsed_seconds=now - started_at,
                    last_completed_iteration_events=last_completed_iteration_events,
                    last_completed_iteration_cycles=last_completed_iteration_cycles,
                    last_completed_iteration_elapsed_seconds=(
                        last_completed_iteration_elapsed_seconds
                    ),
                ))
                last_progress_completed = len(completed)
                next_progress_event = len(completed) + progress_interval_events
                next_progress_time = now + progress_interval_seconds
            if self.selection.overlap_guided_issue:
                fusion_capacity = self.modules["fusion_issue"].timing.queue_capacity
                fusion_occupancy = sum(len(queue) for queue in fusion_inputs.values())
                while fusion_pending and fusion_occupancy < fusion_capacity:
                    event_id, task_kind = heapq.heappop(fusion_pending)
                    heapq.heappush(fusion_inputs[task_kind], event_id)
                    self.issue_scheduler.observe_arrival((
                        self._task_packet(trace, event_id),
                    ))
                    fusion_occupancy += 1
                if fusion_pending:
                    self.modules["fusion_issue"].counters.queue_stalls += 1
                    self._record_stall(
                        cycle, "fusion_issue", "input_queue_capacity",
                        fusion_pending[0][0],
                    )
            candidates: list[tuple[int, int]] = []
            for _ in range(ready_scan_window):
                if not ready:
                    break
                candidates.append(heapq.heappop(ready))
            if self.selection.overlap_guided_issue:
                for task_kind in (
                    TaskKind.FORWARD, TaskKind.CONSUMER, TaskKind.ADJOINT,
                ):
                    if fusion_inputs[task_kind]:
                        candidates.append((heapq.heappop(fusion_inputs[task_kind]), 0))
            fusion_issued = 0
            fusion_port_issued: dict[str, int] = {}
            issued_modules: dict[str, int] = {}
            ordered_ids = self._ordered_candidates(trace, [item[0] for item in candidates])
            ordered = [(event_id, dict(candidates)[event_id]) for event_id in ordered_ids]
            fusion_packets: dict[int, TaskPacket] = {}
            fusion_selected: set[int] = set()
            if self.selection.overlap_guided_issue:
                for event_id, stage in ordered:
                    kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                    if stage == 0 and kind in {
                        PrimitiveKind.FORWARD,
                        PrimitiveKind.CONSUMER,
                        PrimitiveKind.ADJOINT,
                    }:
                        fusion_packets[event_id] = self._task_packet(trace, event_id)
                decision = self.issue_scheduler.select(fusion_packets.values())
                fusion_selected = {task.event_id for task in decision.accepted}
                fusion_order = iter(
                    task.event_id for task in (*decision.accepted, *decision.rejected)
                )
                ordered = [
                    (next(fusion_order), stage)
                    if event_id in fusion_packets else (event_id, stage)
                    for event_id, stage in ordered
                ]
            for event_id, stage in ordered:
                row = trace.events[event_id]
                kind = PrimitiveKind(int(row["primitive_kind"]))
                stages = self._stages_for(kind)
                module_name = stages[stage]
                module = self.modules[module_name]
                timing = module.timing
                if (
                    module_name == "fusion_issue" and stage == 0
                    and self.selection.overlap_guided_issue
                    and event_id not in fusion_selected
                ):
                    module.counters.port_stalls += 1
                    self._record_stall(
                        cycle, module_name, "scheduler_conflict_or_port", event_id
                    )
                    requeue_candidate(event_id, stage)
                    continue
                if module_name == "fusion_issue" and stage == 0:
                    issue_limit = (
                        self.config.candidate_lanes
                        if self.selection.overlap_guided_issue else 1
                    )
                    if fusion_issued >= issue_limit:
                        module.counters.port_stalls += 1
                        reason = (
                            "candidate_width"
                            if self.selection.overlap_guided_issue
                            else "base_single_issue"
                        )
                        self._record_stall(cycle, module_name, reason, event_id)
                        requeue_candidate(event_id, stage)
                        continue
                if not module.accepts_kind(kind):
                    raise CycleConfigurationError(f"{module_name} does not accept {kind.name}")
                fusion_port_name: str | None = None
                if (
                    module_name == "fusion_issue" and stage == 0
                    and self.selection.overlap_guided_issue
                ):
                    fusion_port_name, fusion_port_limit = self._fusion_port_limit(kind)
                    if fusion_port_issued.get(fusion_port_name, 0) >= fusion_port_limit:
                        module.counters.port_stalls += 1
                        self._record_stall(cycle, module_name, "port", event_id)
                        requeue_candidate(event_id, stage)
                        continue
                elif issued_modules.get(module_name, 0) >= timing.ports:
                    module.counters.port_stalls += 1
                    self._record_stall(cycle, module_name, "port", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                module_lane: int | None = None
                if fusion_port_name is not None:
                    busy_until = fusion_busy_until[fusion_port_name]
                else:
                    module_lanes = module_busy_until[module_name]
                    module_lane = next(
                        (index for index, value in enumerate(module_lanes)
                         if value <= cycle),
                        None,
                    )
                    if module_lane is None:
                        module.counters.queue_stalls += 1
                        self._record_stall(cycle, module_name, "initiation_interval", event_id)
                        requeue_candidate(event_id, stage)
                        continue
                    busy_until = module_lanes[module_lane]
                if busy_until > cycle:
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "initiation_interval", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                if module_inflight[module_name] >= timing.queue_capacity:
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "queue_capacity", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                if (kind is PrimitiveKind.RELATION_CANDIDATE
                        and relation_seed_inflight >= self.config.relation_seed_fifo_entries):
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "seed_fifo", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                bank_key = (module_name, cycle, module.bank(int(row["address_token"])))
                if bank_key in bank_busy:
                    module.counters.bank_conflicts += 1
                    self._record_stall(cycle, module_name, "bank", event_id)
                    heapq.heappush(ready, (event_id, stage))
                    continue
                if kind is PrimitiveKind.CACHE_REQUEST and stage == 0:
                    completion: int | None
                    data_bytes = int(row["data_bytes"])
                    if data_bytes <= 0:
                        raise CycleConfigurationError(
                            f"{kind.name} event {event_id} has no explicit transfer size"
                        )
                    if cache_states:
                        instance = self._cache_instance(int(row["gaussian_id"]))
                        state = cache_states[instance]
                        key = (int(row["gaussian_id"]), int(row["state_version"]))
                        try:
                            workset = (
                                semantic_worksets.for_event(event_id)
                                if semantic_worksets is not None else None
                            )
                            lookup = state.request(
                                key,
                                remaining_uses=(
                                    int(workset["remaining_uses"])
                                    if workset is not None else 1
                                ),
                                workset_total_uses=(
                                    int(workset["total_uses"])
                                    if workset is not None else None
                                ),
                            )
                        except CacheBackpressure:
                            module.counters.queue_stalls += 1
                            self._record_stall(cycle, module_name, "cache_capacity", event_id)
                            requeue_candidate(event_id, stage)
                            continue
                        cache_event_state[event_id] = (state, key, lookup)
                        state_version = int(row["state_version"])
                        if state_version in closed_versions:
                            raise CycleConfigurationError(
                                f"cache request {event_id} targets a closed state version"
                            )
                        cache_keys_by_version.setdefault(state_version, []).append((state, key))
                        if lookup is CacheLookup.HIT:
                            completion = cycle + module.service_cycles()
                        else:
                            if lookup is CacheLookup.MISS:
                                memory_requests += 1
                                if async_memory:
                                    request_id = self.config.memory.submit_async(  # type: ignore[attr-defined]
                                        address=int(row["address_token"]),
                                        size_bytes=data_bytes,
                                        is_write=False,
                                        arrival_cycle=cycle,
                                    )
                                    cache_fill_request[(instance, key)] = request_id
                                else:
                                    memory_done = self.config.memory.submit(
                                        address=int(row["address_token"]),
                                        size_bytes=data_bytes,
                                        is_write=False,
                                        arrival_cycle=cycle,
                                    )
                                    cache_fill_done[(instance, key)] = memory_done
                            else:
                                if async_memory:
                                    request_id = cache_fill_request.get((instance, key), -1)
                                else:
                                    memory_done = cache_fill_done.get((instance, key), -1)
                                if (async_memory and request_id < 0) or (
                                    not async_memory and memory_done < 0
                                ):
                                    raise CycleConfigurationError(
                                        f"merged cache request {event_id} has no fill completion"
                                    )
                            if async_memory:
                                memory_waiters.setdefault(request_id, []).append((
                                    event_id, stage, module_name,
                                    cycle + module.service_cycles(),
                                ))
                                completion = None
                            else:
                                if memory_done > cycle + timing.latency:
                                    module.counters.memory_wait_cycles += memory_done - cycle - timing.latency
                                completion = max(cycle + module.service_cycles(), memory_done)
                    else:
                        memory_requests += 1
                        if async_memory:
                            request_id = self.config.memory.submit_async(  # type: ignore[attr-defined]
                                address=int(row["address_token"]),
                                size_bytes=data_bytes,
                                is_write=False,
                                arrival_cycle=cycle,
                            )
                            memory_waiters[request_id] = [(
                                event_id, stage, module_name,
                                cycle + module.service_cycles(),
                            )]
                            completion = None
                        else:
                            memory_done = self.config.memory.submit(
                                address=int(row["address_token"]),
                                size_bytes=data_bytes,
                                is_write=False,
                                arrival_cycle=cycle,
                            )
                            if memory_done > cycle + timing.latency:
                                module.counters.memory_wait_cycles += memory_done - cycle - timing.latency
                            completion = max(cycle + module.service_cycles(), memory_done)
                else:
                    completion = cycle + module.service_cycles()
                if completion is not None:
                    heapq.heappush(in_flight, (completion, event_id, stage, module_name))
                if fusion_port_name is not None:
                    fusion_busy_until[fusion_port_name] = cycle + timing.initiation_interval
                else:
                    assert module_lane is not None
                    module_busy_until[module_name][module_lane] = (
                        cycle + timing.initiation_interval
                    )
                bank_busy[bank_key] = cycle
                module_inflight[module_name] += 1
                if kind is PrimitiveKind.RELATION_CANDIDATE:
                    relation_seed_inflight += 1
                module.counters.accepted += 1
                module.counters.busy_cycles += timing.latency
                issued_modules[module_name] = issued_modules.get(module_name, 0) + 1
                if module_name == "fusion_issue" and stage == 0:
                    fusion_issued += 1
                    if self.selection.overlap_guided_issue:
                        self.issue_scheduler.commit_issued((fusion_packets[event_id],))
                    if fusion_port_name is not None:
                        fusion_port_issued[fusion_port_name] = (
                            fusion_port_issued.get(fusion_port_name, 0) + 1
                        )
                progressed = True
            if not progressed:
                memory_wakeup = (
                    self.config.memory.next_wakeup()  # type: ignore[attr-defined]
                    if async_memory else None
                )
                next_points = [point for point in (in_flight[0][0] if in_flight else None,
                                                   min((value for lanes in module_busy_until.values()
                                                        for value in lanes if value > cycle), default=None),
                                                   min((point for point in fusion_busy_until.values()
                                                        if point > cycle), default=None),
                                                   memory_wakeup)
                               if point is not None and point > cycle]
                if not next_points:
                    if ready and any(key[1] == cycle for key in bank_busy):
                        cycle += 1
                        continue
                    blocked = tuple(event_id for event_id, _ in ready[:8])
                    raise CycleConfigurationError(f"deadlock at cycle {cycle}, pending={blocked}")
                cycle = min(next_points)
            else:
                cycle += 1
        if (
            progress is not None
            and len(completed) > 0
            and len(completed) != last_progress_completed
        ):
            progress(CycleProgress(
                phase="replay",
                completed_events=len(completed),
                total_events=trace.event_count,
                completed_iterations=completed_iterations,
                total_iterations=int(iteration_ids.size),
                last_completed_iteration=last_completed_iteration,
                simulated_cycles=max(completed.values(), default=0),
                elapsed_seconds=time.monotonic() - started_at,
                last_completed_iteration_events=last_completed_iteration_events,
                last_completed_iteration_cycles=last_completed_iteration_cycles,
                last_completed_iteration_elapsed_seconds=(
                    last_completed_iteration_elapsed_seconds
                ),
            ))
        module_counters = {name: module.counters.as_dict() for name, module in self.modules.items()}
        if cache_states:
            cache_totals: dict[str, int] = {key: 0 for key in next(iter(cache_states.values())).counters}
            for state in cache_states.values():
                for key, value in state.counters.items():
                    cache_totals[key] += value
            module_counters["semantic_cache"].update(cache_totals)
        module_counters["semantic_cache"]["memory_requests"] = memory_requests
        module_counters["semantic_cache"].update({
            "workset_keys": semantic_worksets.key_count if semantic_worksets else 0,
            "workset_uses": int(semantic_worksets.requests.size) if semantic_worksets else 0,
            "workset_releases": (
                int(semantic_worksets.requests["last_use"].sum())
                if semantic_worksets else 0
            ),
        })
        audit_records = getattr(self.config.memory, "audit_records", None)
        memory_request_records = tuple(audit_records()) if callable(audit_records) else ()
        return CycleResult(
            total_cycles=max(completed.values(), default=0),
            module_counters=module_counters,
            stalls=self._stall_records(),
            completion_cycles=completed,
            event_counts={kind.name: int((trace.events["primitive_kind"] == int(kind)).sum())
                          for kind in PrimitiveKind},
            policy=self.policy,
            oracle_status=("heuristic_unproven" if self.policy.endswith("_oracle") else "not_applicable"),
            memory_requests=memory_request_records,
        )

    def online_session(
        self,
        *,
        max_events: int,
        max_frontier_events: int | None = None,
        initial_gaussian_count: int = 0,
        semantic_workset_totals: Mapping[tuple[int, int], int] | None = None,
        retain_completion_cycles: bool = False,
        progress: Callable[[CycleProgress], None] | None = None,
        progress_interval_seconds: float | None = None,
    ) -> "CycleReplaySession":
        """Create a persistent packet consumer without trace-column staging."""

        return CycleReplaySession(
            self,
            max_events=max_events,
            max_frontier_events=max_frontier_events,
            initial_gaussian_count=initial_gaussian_count,
            semantic_workset_totals=semantic_workset_totals,
            retain_completion_cycles=retain_completion_cycles,
            progress=progress,
            progress_interval_seconds=progress_interval_seconds,
        )

    def run_virtual(
        self,
        packets: Iterable[VirtualTracePacket],
        *,
        trace_root: Path,
        max_events: int,
        max_total_events: int,
        validate_input: bool = True,
        metadata: dict[str, object] | None = None,
        progress: Callable[[CycleProgress], None] | None = None,
        progress_interval_events: int | None = None,
        progress_interval_seconds: float | None = None,
    ) -> CycleResult:
        """Run a bounded development replay of virtual work packets.

        The expander and writer are shared for the entire packet stream. Event
        IDs therefore remain global, while raw columns keep resident memory
        bounded by ``max_events``. The resulting mmap trace is replayed once,
        so cache state, dependency readiness, fusion history, memory waiters,
        and the configured Ramulator backend are never reset at packet edges.

        This is intentionally a quick-validation API. Both event bounds are
        mandatory so a full 30,000-iteration capture cannot accidentally
        expand into a multi-GB staging trace. Formal performance runs must use
        a prevalidated canonical trace and :meth:`run` directly. ``trace_root``
        is an explicit durable staging directory and is not removed after a run.
        """
        if max_events <= 0:
            raise ValueError("virtual cycle event batch size must be positive")
        if max_total_events <= 0:
            raise ValueError("virtual cycle total event bound must be positive")
        root = Path(trace_root)
        existing_outputs = (
            root / "chunk_manifest.json",
            root / "metadata.json",
            root / "events.raw",
            root / "dependencies.raw",
            root / "payload.raw",
            root / ".virtual_chunks",
        )
        if any(path.exists() for path in existing_outputs):
            raise CycleConfigurationError(
                "virtual quick replay requires a new empty trace_root"
            )
        builder = ChunkedTraceBuilder(
            chunk_events=max_events,
            chunk_root=root / ".virtual_chunks",
            stream_only=True,
        )
        expander = VirtualQueryEventExpander(max_events=max_events)
        stream_validator = VirtualEventStreamValidator()
        source_count = 0
        expanded_events = 0
        try:
            for source in packets:
                source_count += 1
                # Reject before the expander allocates query-sized reduction
                # frontiers.  This is the exact event count of the current
                # loss/backward chain: candidates + 6 relation stages + 3
                # query stages.
                projected_events = (
                    source.candidate_count
                    + 6 * source.logical_relation_count
                    + 3 * source.query_count
                )
                if expanded_events + projected_events > max_total_events:
                    raise CycleConfigurationError(
                        "virtual quick replay exceeded max_total_events; "
                        "formal replay requires a prevalidated canonical trace"
                    )
                for packet in expander.expand(source):
                    expanded_events += packet.event_count
                    stream_validator.accept(packet)
                    self._append_virtual_event_packet(builder, packet)
            terminal = VirtualEventPacket(
                packet_id=stream_validator.next_packet_id,
                global_event_start=stream_validator.next_event_id,
                events=np.empty(0, dtype=event_dtype()),
                dependencies=np.empty(0, dtype=dependency_dtype()),
                final_packet=True,
                frontier_complete=True,
            )
            stream_validator.accept(terminal)
            stream_validator.finalize()
            builder.finish(
                metadata={
                    **(metadata or {}),
                    "result_scope": "quick_cycle_validation",
                    "formal_performance_eligible": False,
                    "virtual_source_packets": source_count,
                    "virtual_max_events": max_events,
                    "virtual_max_total_events": max_total_events,
                },
                materialize=False,
            )
        except BaseException:
            # A partial raw stream must never be mistaken for a replayable
            # trace: no manifest is written until the global frontier closes.
            raise
        trace = TraceReader().read(root, validate=validate_input, mmap_mode="r")
        return self.run(
            trace,
            validate_input=False,
            progress=progress,
            progress_interval_events=progress_interval_events,
            progress_interval_seconds=progress_interval_seconds,
        )

    @staticmethod
    def _append_virtual_event_packet(
        builder: ChunkedTraceBuilder, packet: VirtualEventPacket
    ) -> None:
        if packet.global_event_start != builder.next_event_id:
            raise CycleConfigurationError(
                "virtual event packet IDs are not contiguous with the cycle stream"
            )
        events = np.asarray(packet.events, dtype=packet.events.dtype)
        counts = events["dependency_count"].astype(np.int64, copy=False)
        builder.emit_batch(
            events,
            dependencies=np.asarray(packet.dependencies),
            dependency_counts=counts,
            payload=np.empty(0, dtype=np.dtype("<f4")),
            payload_counts=np.zeros(events.size, dtype=np.int64),
        )


class CycleReplaySession:
    """Incremental cycle consumer for a bounded virtual event stream.

    Unlike :meth:`CycleEngine.run_virtual`, this session never constructs a
    ``Trace`` or writes expanded columns.  Packets register their rows and
    dependency frontier, then immediately advance the same module, cache,
    fusion, and memory state.  Completed rows are released; only unresolved
    frontier rows remain resident.
    """

    def __init__(
        self,
        engine: CycleEngine,
        *,
        max_events: int,
        max_frontier_events: int | None = None,
        initial_gaussian_count: int = 0,
        semantic_workset_totals: Mapping[tuple[int, int], int] | None = None,
        retain_completion_cycles: bool = False,
        progress: Callable[[CycleProgress], None] | None = None,
        progress_interval_seconds: float | None = None,
    ) -> None:
        if max_events <= 0:
            raise ValueError("online virtual event batch size must be positive")
        if max_frontier_events is not None and max_frontier_events <= 0:
            raise ValueError("online resident frontier bound must be positive")
        if initial_gaussian_count < 0:
            raise ValueError("initial Gaussian count must be non-negative")
        if progress_interval_seconds is not None and progress_interval_seconds <= 0:
            raise ValueError("online progress interval must be positive")
        self.engine = engine
        self.max_events = max_events
        self.max_frontier_events = max_frontier_events
        self.retain_completion_cycles = retain_completion_cycles
        self.semantic_workset_totals = dict(semantic_workset_totals or {})
        if any(key[0] < 0 or key[1] < 0 or value <= 0
               for key, value in self.semantic_workset_totals.items()):
            raise ValueError("semantic workset totals must use positive keyed counts")
        self.progress = progress
        self.progress_interval_seconds = progress_interval_seconds
        self._expander = VirtualQueryEventExpander(max_events=max_events)
        self._stream_validator = VirtualEventStreamValidator()
        self._lifecycle = VirtualTraceLifecycleValidator(initial_gaussian_count)
        self._events: dict[int, np.void] = {}
        self._kinds: dict[int, PrimitiveKind] = {}
        self._dependencies: dict[int, tuple[int, ...]] = {}
        self._remaining: dict[int, int] = {}
        self._dependents: dict[int, list[int]] = defaultdict(list)
        self._completed_ids: set[int] = set()
        self._completed_through = -1
        self._completion_cycles: dict[int, int] = {}
        self._ready: list[tuple[int, int]] = []
        self._fusion_pending: list[tuple[int, TaskKind]] = []
        self._fusion_inputs: dict[TaskKind, list[int]] = {
            TaskKind.FORWARD: [], TaskKind.CONSUMER: [], TaskKind.ADJOINT: [],
        }
        self._in_flight: list[tuple[int, int, int, str]] = []
        self._module_busy_until = {
            name: [0] * module.timing.ports
            for name, module in engine.modules.items()
        }
        self._fusion_busy_until = {name: 0 for name in ("forward", "consumer", "adjoint")}
        self._module_inflight = {name: 0 for name in engine.modules}
        self._relation_seed_inflight = 0
        self._bank_busy: dict[tuple[str, int, int], int] = {}
        self._cache_states = (
            engine._residency_states()
            if engine.selection.semantic_residency else {}
        )
        self._cache_fill_done: dict[tuple[int, tuple[int, int]], int] = {}
        self._cache_fill_request: dict[tuple[int, tuple[int, int]], int] = {}
        self._memory_waiters: dict[int, list[tuple[int, int, str, int]]] = {}
        self._cache_event_state: dict[int, tuple[SemanticCacheState, tuple[int, int], CacheLookup]] = {}
        self._workset_seen: dict[tuple[int, int], int] = defaultdict(int)
        self._workset_by_request: dict[int, tuple[int, int, int, bool]] = {}
        self._workset_use_count = 0
        self._workset_release_count = 0
        self._cache_keys_by_version: dict[int, list[tuple[SemanticCacheState, tuple[int, int]]] ] = {}
        self._closed_versions: set[int] = set()
        self._memory_requests = 0
        self._event_counts = {kind.name: 0 for kind in PrimitiveKind}
        self._accepted_events = 0
        self._completed_events = 0
        self._peak_frontier_events = 0
        self._cycle = 0
        self._source_packets = 0
        self._query_packets = 0
        self._closed_iterations = 0
        self._last_lifecycle_event: int | None = None
        self._state_barrier_event: int | None = None
        self._backward_frontier: tuple[int, ...] = ()
        self._finalized = False
        self._started_at = time.monotonic()
        self._last_progress_report = self._started_at

    @property
    def pending_event_count(self) -> int:
        return len(self._events)

    @property
    def peak_frontier_events(self) -> int:
        """Largest number of unresolved event rows retained by this session."""

        return self._peak_frontier_events

    @property
    def resident_completion_markers(self) -> int:
        """Completed IDs not yet compacted into the dense prefix watermark."""

        return len(self._completed_ids)

    @property
    def global_event_id(self) -> int:
        return self._stream_validator.next_event_id

    @property
    def accepted_event_count(self) -> int:
        return self._accepted_events

    @property
    def completed_event_count(self) -> int:
        return self._completed_events

    @property
    def simulated_cycles(self) -> int:
        return self._cycle

    @property
    def source_packet_count(self) -> int:
        return self._source_packets

    @property
    def query_packet_count(self) -> int:
        return self._query_packets

    @property
    def closed_iteration_count(self) -> int:
        return self._closed_iterations

    @property
    def quiescent(self) -> bool:
        """Whether no online event, queue, memory, or cache work remains."""

        return not self._quiescence_violations()

    def _quiescence_violations(self) -> tuple[str, ...]:
        violations: list[str] = []
        if self._events:
            violations.append("events")
        if self._in_flight:
            violations.append("in_flight")
        if self._memory_waiters:
            violations.append("memory_waiters")
        if self._ready:
            violations.append("ready")
        if self._fusion_pending or any(self._fusion_inputs.values()):
            violations.append("fusion_queues")
        if self._dependents:
            violations.append("dependents")
        if any(value for value in self._module_inflight.values()):
            violations.append("module_inflight")
        if self._cache_fill_done or self._cache_fill_request:
            violations.append("cache_fill")
        if self._cache_event_state:
            violations.append("cache_events")
        if self._workset_by_request:
            violations.append("workset_requests")
        if self._relation_seed_inflight:
            violations.append("relation_seed_fifo")
        return tuple(violations)

    def accept_event_packet(
        self, packet: VirtualEventPacket, *, _drain_after: bool = True
    ) -> None:
        self._ensure_open()
        if packet.final_packet:
            raise ValueError("final packet is reserved for finish()")
        if (
            self.max_frontier_events is not None
            and len(self._events) + packet.event_count > self.max_frontier_events
        ):
            raise CycleConfigurationError(
                "online resident frontier exceeds max_frontier_events"
            )
        self._stream_validator.accept(packet)
        for index, row in enumerate(packet.events):
            event_id = int(row["event_id"])
            if (
                event_id in self._events
                or event_id in self._completed_ids
                or event_id <= self._completed_through
            ):
                raise CycleConfigurationError(f"duplicate online event {event_id}")
            dependencies = tuple(int(value) for value in packet.dependency_ids(index))
            unresolved = 0
            for dependency in dependencies:
                if dependency >= event_id:
                    raise CycleConfigurationError(
                        f"online event {event_id} has a forward dependency"
                    )
                if (
                    dependency > self._completed_through
                    and dependency not in self._completed_ids
                ):
                    if dependency not in self._events:
                        raise CycleConfigurationError(
                            f"online event {event_id} references an unknown dependency {dependency}"
                        )
                    unresolved += 1
                    self._dependents[dependency].append(event_id)
            kind = PrimitiveKind(int(row["primitive_kind"]))
            self._events[event_id] = row.copy()
            self._kinds[event_id] = kind
            self._peak_frontier_events = max(
                self._peak_frontier_events, len(self._events)
            )
            self._dependencies[event_id] = dependencies
            self._remaining[event_id] = unresolved
            if kind is PrimitiveKind.CACHE_REQUEST and self.semantic_workset_totals:
                key = (int(row["gaussian_id"]), int(row["state_version"]))
                total = self.semantic_workset_totals.get(key)
                if total is None:
                    raise CycleConfigurationError(
                        f"semantic workset is missing cache key {key}"
                    )
                ordinal = self._workset_seen[key]
                if ordinal >= total:
                    raise CycleConfigurationError(
                        f"semantic workset has too many uses for key {key}"
                    )
                self._workset_by_request[event_id] = (
                    ordinal, total, total - ordinal, ordinal + 1 == total
                )
                self._workset_use_count += 1
                self._workset_seen[key] = ordinal + 1
            self._event_counts[kind.name] += 1
            self._accepted_events += 1
            if unresolved == 0:
                self._push_ready(event_id, 0)
        self._source_packets += 1
        if _drain_after:
            self._drain()
            self._compact_completed_prefix()
            self._report_progress()

    def accept_query_packet(self, packet: VirtualTracePacket) -> None:
        """Expand one point/mask packet lazily into the online consumer."""

        self.accept_query_packets((packet,))

    def accept_query_packets(self, packets: Iterable[VirtualTracePacket]) -> None:
        """Register one capture batch before advancing the online scheduler."""

        self._ensure_open()
        accepted = False
        for packet in packets:
            self._accept_query_packet_without_drain(packet)
            accepted = True
        if not accepted:
            return
        self._drain()
        self._compact_completed_prefix()
        self._report_progress()

    def _accept_query_packet_without_drain(self, packet: VirtualTracePacket) -> None:
        self._lifecycle.accept_packet(packet)
        self._query_packets += 1
        terminal_ids: list[int] = []
        external_dependencies = (
            (self._state_barrier_event,)
            if self._state_barrier_event is not None else ()
        )
        for event_packet in self._expander.expand(
            packet, external_dependencies=external_dependencies
        ):
            terminal_ids.extend(
                int(event_id) for event_id in event_packet.events["event_id"][
                    event_packet.events["primitive_kind"]
                    == int(PrimitiveKind.GRADIENT_REDUCTION)
                ]
            )
            # A CUDA work-buffer packet is the arrival unit.  Its expanded
            # event sub-packets must enter the frontier before the scheduler
            # advances, otherwise each sub-packet edge introduces artificial
            # serialization that is absent from the equivalent trace replay.
            if (
                self.max_frontier_events is not None
                and self.pending_event_count + event_packet.event_count
                > self.max_frontier_events
            ):
                # Keep the online stream bounded.  This is a frontier
                # backpressure point, not a semantic packet boundary.
                self._drain()
                self._compact_completed_prefix()
            self.accept_event_packet(event_packet, _drain_after=False)
            # Large frontier fills can spend substantial wall time registering
            # rows before the next drain; keep the long-run monitor live.
            self._report_progress()
        self._backward_frontier = (*self._backward_frontier, *terminal_ids)

    def register_semantic_workset_totals(
        self, totals: Mapping[tuple[int, int], int]
    ) -> None:
        """Register exact totals before their cache-request events arrive."""

        self._ensure_open()
        for key, value in totals.items():
            normalized_key = (int(key[0]), int(key[1]))
            normalized_value = int(value)
            if normalized_key[0] < 0 or normalized_key[1] < 0 or normalized_value <= 0:
                raise ValueError("semantic workset totals must use positive keyed counts")
            existing = self.semantic_workset_totals.get(normalized_key)
            if existing is not None and existing != normalized_value:
                raise CycleConfigurationError(
                    f"semantic workset total changed for key {normalized_key}"
                )
            self.semantic_workset_totals[normalized_key] = normalized_value

    def accept_lifecycle(self, record: VirtualLifecycleRecord) -> None:
        """Insert a lifecycle event with a dependency on the live frontier.

        The current capture record format does not yet carry the producer's
        exact backward/gradient frontier.  Until that sidecar is present, the
        session uses every currently unresolved event plus the prior lifecycle
        event as a conservative ordering barrier; it never invents a shorter
        dependency set.
        """

        self._ensure_open()
        self._lifecycle.accept_lifecycle(record)
        primitive = {
            VirtualLifecycleKind.UPDATE_BEGIN: PrimitiveKind.UPDATE_BEGIN,
            VirtualLifecycleKind.UPDATE_COMMIT: PrimitiveKind.UPDATE_COMMIT,
            VirtualLifecycleKind.UPDATE_END: PrimitiveKind.UPDATE_END,
            VirtualLifecycleKind.PRUNE: PrimitiveKind.SET_MODIFICATION,
            VirtualLifecycleKind.CLONE: PrimitiveKind.SET_MODIFICATION,
            VirtualLifecycleKind.SPLIT: PrimitiveKind.SET_MODIFICATION,
        }[record.kind]
        base_dependencies = (
            tuple(record.dependency_ids)
            if record.dependency_ids else (
                self._backward_frontier or tuple(self._events)
            )
        )
        gaussian_ids = (
            tuple(record.active_ids)
            if record.kind is VirtualLifecycleKind.UPDATE_COMMIT
            and record.all_active and record.active_ids
            else ((record.parent_id, *record.child_ids)
                  if record.kind in {
                      VirtualLifecycleKind.CLONE, VirtualLifecycleKind.SPLIT,
                  } and record.child_ids else (record.gaussian_id,))
        )
        for gaussian_id in gaussian_ids:
            event_id = self._stream_validator.next_event_id
            dependency_ids = base_dependencies
            if self._last_lifecycle_event is not None:
                dependency_ids = (*dependency_ids, self._last_lifecycle_event)
            row = np.empty(1, dtype=event_dtype())
            row[:] = TraceEvent().as_tuple()
            row["event_id"] = event_id
            row["iteration_id"] = record.iteration_id
            row["primitive_kind"] = int(primitive)
            row["gaussian_id"] = gaussian_id if gaussian_id >= 0 else 0
            row["state_version"] = record.state_version
            row["resource_class"] = int(ResourceClass.UPDATE)
            row["field_mask"] = record.field_mask
            row["template_id"] = 0
            row["flags"] = (
                int(record.kind)
                if record.kind in {
                    VirtualLifecycleKind.PRUNE,
                    VirtualLifecycleKind.CLONE,
                    VirtualLifecycleKind.SPLIT,
                }
                else record.transaction_kind
            )
            row["dependency_begin"] = 0
            row["dependency_count"] = len(dependency_ids)
            event_packet = VirtualEventPacket(
                packet_id=self._stream_validator.next_packet_id,
                global_event_start=event_id,
                events=row,
                dependencies=np.asarray(dependency_ids, dtype=dependency_dtype()),
            )
            self._last_lifecycle_event = event_id
            self.accept_event_packet(event_packet)
        if record.kind is VirtualLifecycleKind.UPDATE_END:
            self._state_barrier_event = self._last_lifecycle_event
            self._backward_frontier = ()

    def close_iteration(self, iteration_id: int) -> None:
        self._ensure_open()
        self._lifecycle.close_iteration(iteration_id)
        self._closed_iterations += 1
        self._drain()

    def retire_semantic_workset_totals(
        self, keys: Iterable[tuple[int, int]]
    ) -> None:
        """Release exact workset bookkeeping after a quiescent iteration."""

        self._ensure_open()
        if self._events or self._in_flight or self._memory_waiters:
            raise CycleConfigurationError(
                "cannot retire semantic worksets before online replay is quiescent"
            )
        for raw_key in keys:
            key = (int(raw_key[0]), int(raw_key[1]))
            self.semantic_workset_totals.pop(key, None)
            self._workset_seen.pop(key, None)

    def finish(self) -> CycleResult:
        self._ensure_open()
        self._drain()
        violations = self._quiescence_violations()
        if violations:
            raise CycleConfigurationError(
                "online replay is not quiescent at finish: " + ",".join(violations)
            )
        terminal = VirtualEventPacket(
            packet_id=self._stream_validator.next_packet_id,
            global_event_start=self._stream_validator.next_event_id,
            events=np.empty(0, dtype=event_dtype()),
            dependencies=np.empty(0, dtype=dependency_dtype()),
            final_packet=True,
            frontier_complete=True,
        )
        self._stream_validator.accept(terminal)
        self._stream_validator.finalize()
        ledgers = self._lifecycle.finalize()
        if self._query_packets == 0 or not ledgers:
            raise CycleConfigurationError(
                "online replay requires at least one closed query iteration"
            )
        if self.semantic_workset_totals:
            observed = dict(self._workset_seen)
            if observed != self.semantic_workset_totals:
                raise CycleConfigurationError(
                    "online semantic workset totals do not match cache requests"
                )
        self._finalized = True
        self._report_progress(force=True)
        counters = {
            name: module.counters.as_dict() for name, module in self.engine.modules.items()
        }
        if self._cache_states:
            cache_totals = {
                key: 0 for key in next(iter(self._cache_states.values())).counters
            }
            for state in self._cache_states.values():
                for key, value in state.counters.items():
                    cache_totals[key] += value
            counters["semantic_cache"].update(cache_totals)
        counters["semantic_cache"]["memory_requests"] = self._memory_requests
        counters["semantic_cache"]["workset_keys"] = len(self.semantic_workset_totals)
        counters["semantic_cache"]["workset_uses"] = self._workset_use_count
        counters["semantic_cache"]["workset_releases"] = self._workset_release_count
        audit_records = getattr(self.engine.config.memory, "audit_records", None)
        memory_records = tuple(audit_records()) if callable(audit_records) else ()
        return CycleResult(
            total_cycles=self._cycle,
            module_counters=counters,
            stalls=self.engine._stall_records(),
            completion_cycles=(dict(self._completion_cycles)
                               if self.retain_completion_cycles else {}),
            event_counts=dict(self._event_counts),
            policy=self.engine.policy,
            oracle_status=("heuristic_unproven"
                           if self.engine.policy.endswith("_oracle")
                           else "not_applicable"),
            memory_requests=memory_records,
        )

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("online replay session is already finalized")

    def _compact_completed_prefix(self) -> None:
        """Collapse a quiescent dense packet's completed IDs to one watermark."""

        if self._events or self._in_flight or self._memory_waiters:
            return
        expected_end = self._stream_validator.next_event_id - 1
        if expected_end <= self._completed_through:
            return
        expected_start = self._completed_through + 1
        expected_count = expected_end - expected_start + 1
        if (
            len(self._completed_ids) != expected_count
            or min(self._completed_ids, default=expected_start) != expected_start
            or max(self._completed_ids, default=expected_end) != expected_end
        ):
            raise CycleConfigurationError(
                "online completed event prefix is not dense at quiescence"
            )
        self._completed_ids.clear()
        self._completed_through = expected_end

    def _push_ready(self, event_id: int, stage: int) -> None:
        task_kind = self._fusion_kind(event_id, stage)
        if task_kind is None:
            heapq.heappush(self._ready, (event_id, stage))
        else:
            heapq.heappush(self._fusion_pending, (event_id, task_kind))

    def _requeue(self, event_id: int, stage: int) -> None:
        task_kind = self._fusion_kind(event_id, stage)
        if task_kind is None:
            heapq.heappush(self._ready, (event_id, stage))
        else:
            heapq.heappush(self._fusion_inputs[task_kind], event_id)

    def _fusion_kind(self, event_id: int, stage: int) -> TaskKind | None:
        if not self.engine.selection.overlap_guided_issue or stage != 0:
            return None
        kind = self._kinds[event_id]
        return {
            PrimitiveKind.FORWARD: TaskKind.FORWARD,
            PrimitiveKind.CONSUMER: TaskKind.CONSUMER,
            PrimitiveKind.ADJOINT: TaskKind.ADJOINT,
        }.get(kind)

    def _task_packet(self, event_id: int) -> TaskPacket:
        row = self._events[event_id]
        kind = self._kinds[event_id]
        task_kind = {
            PrimitiveKind.FORWARD: TaskKind.FORWARD,
            PrimitiveKind.CONSUMER: TaskKind.CONSUMER,
            PrimitiveKind.ADJOINT: TaskKind.ADJOINT,
        }[kind]
        query_id = max(int(row["query_id"]), 0)
        gaussian_id = max(int(row["gaussian_id"]), 0)
        reduction_key = int(row["reduction_key"])
        domain = ReductionDomain.GAUSSIAN if task_kind is TaskKind.ADJOINT else ReductionDomain.QUERY
        semantic_key = gaussian_id if domain is ReductionDomain.GAUSSIAN else (
            reduction_key if reduction_key >= 0 else query_id
        )
        if semantic_key < 0:
            raise CycleConfigurationError(f"event {event_id} lacks a reduction key")
        return TaskPacket(
            event_id=event_id, query_id=query_id, gaussian_id=gaussian_id,
            reduction_key=semantic_key, resource=int(row["resource_class"]),
            reduction_domain=domain, state_version=int(row["state_version"]),
            template_id=int(row["template_id"]), address_token=int(row["address_token"]),
            task_kind=task_kind,
        )

    def _ordered(self, candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
        def key(item: tuple[int, int]) -> tuple[int, int, int]:
            event_id, _stage = item
            row = self._events[event_id]
            return int(row["query_id"]), int(row["relation_id"]), event_id
        if self.engine.selection.query_load_rules:
            return sorted(candidates, key=key)
        if self.engine.selection.semantic_residency:
            return sorted(candidates, key=lambda item: (
                0 if PrimitiveKind(int(self._events[item[0]]["primitive_kind"]))
                in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN} else 1,
                int(self._events[item[0]]["gaussian_id"]), item[0],
            ))
        return candidates

    def _drain(self) -> None:
        async_memory = callable(getattr(self.engine.config.memory, "submit_async", None))
        ready_scan_window = max(
            self.engine.config.candidate_lanes,
            sum(module.timing.ports for module in self.engine.modules.values()),
        )
        while self._ready or self._in_flight or self._memory_waiters:
            # Bank reservations are scoped to one cycle.  Drop old entries so
            # long packet streams do not retain one dictionary item per cycle.
            self._bank_busy = {
                key: value for key, value in self._bank_busy.items()
                if key[1] >= self._cycle
            }
            self._report_progress()
            self._cycle_fusion_issued = 0
            self._cycle_fusion_ports: dict[str, int] = {}
            self._cycle_module_issued: dict[str, int] = {}
            progressed = False
            if async_memory:
                self.engine.config.memory.advance(self._cycle)  # type: ignore[attr-defined]
                for record in self.engine.config.memory.pop_completions():  # type: ignore[attr-defined]
                    waiters = self._memory_waiters.pop(record.request_id, None)
                    if not waiters or record.completion_cycle is None:
                        raise CycleConfigurationError("Ramulator completion has no online waiter")
                    for event_id, stage, module_name, service_finish in waiters:
                        completion = max(service_finish, record.completion_cycle)
                        if record.completion_cycle > service_finish:
                            self.engine.modules[module_name].counters.memory_wait_cycles += (
                                record.completion_cycle - service_finish
                            )
                        heapq.heappush(self._in_flight, (completion, event_id, stage, module_name))
                    progressed = True
            while self._in_flight and self._in_flight[0][0] <= self._cycle:
                self._complete_stage(*heapq.heappop(self._in_flight))
                progressed = True
            if self.engine.selection.overlap_guided_issue:
                capacity = self.engine.modules["fusion_issue"].timing.queue_capacity
                occupancy = sum(len(queue) for queue in self._fusion_inputs.values())
                while self._fusion_pending and occupancy < capacity:
                    event_id, task_kind = heapq.heappop(self._fusion_pending)
                    heapq.heappush(self._fusion_inputs[task_kind], event_id)
                    self.engine.issue_scheduler.observe_arrival((self._task_packet(event_id),))
                    occupancy += 1
            candidates: list[tuple[int, int]] = []
            for _ in range(ready_scan_window):
                if not self._ready:
                    break
                candidates.append(heapq.heappop(self._ready))
            if self.engine.selection.overlap_guided_issue:
                for task_kind in (TaskKind.FORWARD, TaskKind.CONSUMER, TaskKind.ADJOINT):
                    if self._fusion_inputs[task_kind]:
                        candidates.append((heapq.heappop(self._fusion_inputs[task_kind]), 0))
            if candidates:
                ordered = self._ordered(candidates)
                fusion_packets = {
                    event_id: self._task_packet(event_id)
                    for event_id, stage in ordered
                    if self.engine.selection.overlap_guided_issue and stage == 0
                    and PrimitiveKind(int(self._events[event_id]["primitive_kind"]))
                    in {PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT}
                }
                selected: set[int] = set()
                if fusion_packets:
                    decision = self.engine.issue_scheduler.select(fusion_packets.values())
                    selected = {task.event_id for task in decision.accepted}
                    order = iter(task.event_id for task in (*decision.accepted, *decision.rejected))
                    ordered = [
                        (next(order), stage) if event_id in fusion_packets else (event_id, stage)
                        for event_id, stage in ordered
                    ]
                for event_id, stage in ordered:
                    if self._try_issue(event_id, stage, selected, async_memory):
                        progressed = True
            if not progressed:
                memory_wakeup = (
                    self.engine.config.memory.next_wakeup()  # type: ignore[attr-defined]
                    if async_memory else None
                )
                next_points = [point for point in (
                    self._in_flight[0][0] if self._in_flight else None,
                    min((value for lanes in self._module_busy_until.values()
                         for value in lanes if value > self._cycle), default=None),
                    min((value for value in self._fusion_busy_until.values() if value > self._cycle), default=None),
                    memory_wakeup,
                ) if point is not None and point > self._cycle]
                if not next_points:
                    if self._ready and any(
                        key[1] == self._cycle for key in self._bank_busy
                    ):
                        self._cycle += 1
                        continue
                    if self._ready or self._memory_waiters:
                        raise CycleConfigurationError("deadlock in online cycle replay")
                    break
                self._cycle = min(next_points)
            else:
                self._cycle += 1

    def _try_issue(self, event_id: int, stage: int, selected: set[int], async_memory: bool) -> bool:
        row = self._events[event_id]
        kind = self._kinds[event_id]
        stages = self.engine._stages_for(kind)
        module_name = stages[stage]
        module = self.engine.modules[module_name]
        timing = module.timing
        if (module_name == "fusion_issue" and stage == 0
                and self.engine.selection.overlap_guided_issue and event_id not in selected):
            module.counters.port_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "scheduler_conflict_or_port", event_id)
            self._requeue(event_id, stage)
            return False
        if module_name == "fusion_issue" and stage == 0:
            limit = self.engine.config.candidate_lanes if self.engine.selection.overlap_guided_issue else 1
            issued = getattr(self, "_cycle_fusion_issued", 0)
            if issued >= limit:
                module.counters.port_stalls += 1
                self.engine._record_stall(self._cycle, module_name,
                                           "candidate_width" if self.engine.selection.overlap_guided_issue
                                           else "base_single_issue", event_id)
                self._requeue(event_id, stage)
                return False
        if not module.accepts_kind(kind):
            raise CycleConfigurationError(f"{module_name} does not accept {kind.name}")
        port_name: str | None = None
        if module_name == "fusion_issue" and stage == 0 and self.engine.selection.overlap_guided_issue:
            port_name, limit = self.engine._fusion_port_limit(kind)
            used = getattr(self, "_cycle_fusion_ports", {}).get(port_name, 0)
            if used >= limit:
                module.counters.port_stalls += 1
                self.engine._record_stall(self._cycle, module_name, "port", event_id)
                self._requeue(event_id, stage)
                return False
        elif getattr(self, "_cycle_module_issued", {}).get(module_name, 0) >= timing.ports:
            module.counters.port_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "port", event_id)
            self._requeue(event_id, stage)
            return False
        module_lane: int | None = None
        if port_name:
            busy_until = self._fusion_busy_until[port_name]
        else:
            module_lanes = self._module_busy_until[module_name]
            module_lane = next(
                (index for index, value in enumerate(module_lanes)
                 if value <= self._cycle),
                None,
            )
            if module_lane is None:
                module.counters.queue_stalls += 1
                self.engine._record_stall(
                    self._cycle, module_name, "initiation_interval", event_id
                )
                self._requeue(event_id, stage)
                return False
            busy_until = module_lanes[module_lane]
        if busy_until > self._cycle:
            module.counters.queue_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "initiation_interval", event_id)
            self._requeue(event_id, stage)
            return False
        if self._module_inflight[module_name] >= timing.queue_capacity:
            module.counters.queue_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "queue_capacity", event_id)
            self._requeue(event_id, stage)
            return False
        if kind is PrimitiveKind.RELATION_CANDIDATE and self._relation_seed_inflight >= self.engine.config.relation_seed_fifo_entries:
            module.counters.queue_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "seed_fifo", event_id)
            self._requeue(event_id, stage)
            return False
        bank_key = (module_name, self._cycle, module.bank(int(row["address_token"])))
        if bank_key in self._bank_busy:
            module.counters.bank_conflicts += 1
            self.engine._record_stall(self._cycle, module_name, "bank", event_id)
            self._requeue(event_id, stage)
            return False
        completion: int | None = self._issue_memory_if_needed(
            event_id, stage, kind, module_name, timing, async_memory
        )
        if completion is not None:
            heapq.heappush(self._in_flight, (completion, event_id, stage, module_name))
        if port_name:
            self._fusion_busy_until[port_name] = self._cycle + timing.initiation_interval
        else:
            assert module_lane is not None
            self._module_busy_until[module_name][module_lane] = (
                self._cycle + timing.initiation_interval
            )
        self._bank_busy[bank_key] = self._cycle
        self._module_inflight[module_name] += 1
        if kind is PrimitiveKind.RELATION_CANDIDATE:
            self._relation_seed_inflight += 1
        module.counters.accepted += 1
        module.counters.busy_cycles += timing.latency
        self._cycle_module_issued[module_name] = self._cycle_module_issued.get(module_name, 0) + 1
        if module_name == "fusion_issue" and stage == 0:
            self._cycle_fusion_issued = getattr(self, "_cycle_fusion_issued", 0) + 1
            if self.engine.selection.overlap_guided_issue:
                self.engine.issue_scheduler.commit_issued((self._task_packet(event_id),))
            if port_name:
                self._cycle_fusion_ports[port_name] = self._cycle_fusion_ports.get(port_name, 0) + 1
        return True

    def _issue_memory_if_needed(self, event_id: int, stage: int, kind: PrimitiveKind,
                                module_name: str, timing: ModuleTiming,
                                async_memory: bool) -> int | None:
        if kind is not PrimitiveKind.CACHE_REQUEST or stage != 0:
            return self._cycle + timing.latency
        row = self._events[event_id]
        data_bytes = int(row["data_bytes"])
        if data_bytes <= 0:
            raise CycleConfigurationError(f"CACHE_REQUEST {event_id} has no transfer size")
        if not self._cache_states:
            self._memory_requests += 1
            if async_memory:
                request_id = self.engine.config.memory.submit_async(
                    address=int(row["address_token"]), size_bytes=data_bytes,
                    is_write=False, arrival_cycle=self._cycle,
                )
                self._memory_waiters[request_id] = [(event_id, stage, module_name,
                                                     self._cycle + timing.latency)]
                return None
            memory_done = self.engine.config.memory.submit(
                address=int(row["address_token"]), size_bytes=data_bytes,
                is_write=False, arrival_cycle=self._cycle,
            )
            return max(self._cycle + timing.latency, memory_done)
        instance = self.engine._cache_instance(int(row["gaussian_id"]))
        key = (int(row["gaussian_id"]), int(row["state_version"]))
        state = self._cache_states[instance]
        workset = self._workset_by_request.get(event_id)
        lookup = state.request(
            key,
            remaining_uses=workset[2] if workset is not None else 1,
            workset_total_uses=workset[1] if workset is not None else None,
        )
        self._cache_event_state[event_id] = (state, key, lookup)
        self._cache_keys_by_version.setdefault(key[1], []).append((state, key))
        if lookup is CacheLookup.HIT:
            return self._cycle + timing.latency
        self._memory_requests += 1
        if lookup is CacheLookup.MISS:
            if async_memory:
                request_id = self.engine.config.memory.submit_async(
                    address=int(row["address_token"]), size_bytes=data_bytes,
                    is_write=False, arrival_cycle=self._cycle,
                )
                self._cache_fill_request[(instance, key)] = request_id
            else:
                memory_done = self.engine.config.memory.submit(
                    address=int(row["address_token"]), size_bytes=data_bytes,
                    is_write=False, arrival_cycle=self._cycle,
                )
                self._cache_fill_done[(instance, key)] = memory_done
        request_id = (self._cache_fill_request.get((instance, key), -1)
                      if async_memory else self._cache_fill_done.get((instance, key), -1))
        if request_id < 0:
            raise CycleConfigurationError(f"merged cache request {event_id} has no fill completion")
        if async_memory:
            self._memory_waiters.setdefault(request_id, []).append(
                (event_id, stage, module_name, self._cycle + timing.latency)
            )
            return None
        return max(self._cycle + timing.latency, request_id)

    def _complete_stage(self, finish: int, event_id: int, stage: int, module_name: str) -> None:
        module = self.engine.modules[module_name]
        module.complete(event_id, finish)
        self._module_inflight[module_name] -= 1
        row = self._events[event_id]
        kind = self._kinds[event_id]
        stages = self.engine._stages_for(kind)
        if (
            kind is PrimitiveKind.RELATION_CANDIDATE
            and stage == len(stages) - 1
        ):
            self._relation_seed_inflight -= 1
            if self._relation_seed_inflight < 0:
                raise CycleConfigurationError(
                    "negative relation seed FIFO occupancy"
                )
        if kind is PrimitiveKind.CACHE_REQUEST and stage == 0 and event_id in self._cache_event_state:
            state, key, lookup = self._cache_event_state[event_id]
            if lookup is CacheLookup.MISS:
                pending = state.pending.get(key)
                if pending is None:
                    raise CycleConfigurationError(
                        f"cache fill for {event_id} has no pending state"
                    )
                state.fill_complete(
                    key, remaining_uses=int(pending["remaining_uses"])
                )
                instance = self.engine._cache_instance(int(row["gaussian_id"]))
                self._cache_fill_request.pop((instance, key), None)
                self._cache_fill_done.pop((instance, key), None)
        if kind is PrimitiveKind.CACHE_RETURN and stage == len(stages) - 1 and self._cache_states:
            deps = self._dependencies[event_id]
            request_ids = [dependency for dependency in deps if dependency in self._cache_event_state]
            if len(request_ids) != 1:
                raise CycleConfigurationError(f"cache return {event_id} has no unique request")
            request_id = request_ids[0]
            state, key, _lookup = self._cache_event_state[request_id]
            state.complete_read(key)
            workset = self._workset_by_request.get(request_id)
            if workset is None or workset[3]:
                state.close(key)
            if workset is not None:
                if workset[3]:
                    self._workset_release_count += 1
                self._workset_by_request.pop(request_id, None)
            self._cache_event_state.pop(request_id, None)
        if kind is PrimitiveKind.UPDATE_END and stage == len(stages) - 1:
            version = int(row["state_version"])
            if int(row["field_mask"]) != 0:
                self._closed_versions.add(version)
                for state, key in self._cache_keys_by_version.pop(version, []):
                    state.close(key)
        if stage + 1 < len(stages):
            self._push_ready(event_id, stage + 1)
            return
        self._completed_ids.add(event_id)
        self._completed_events += 1
        if self.retain_completion_cycles:
            self._completion_cycles[event_id] = finish
        for dependent in self._dependents.pop(event_id, []):
            remaining = self._remaining[dependent] - 1
            if remaining < 0:
                raise CycleConfigurationError(f"dependency count underflow at event {dependent}")
            self._remaining[dependent] = remaining
            if remaining == 0:
                self._push_ready(dependent, 0)
        self._events.pop(event_id, None)
        self._kinds.pop(event_id, None)
        self._dependencies.pop(event_id, None)
        self._remaining.pop(event_id, None)

    def _report_progress(self, *, force: bool = False) -> None:
        if self.progress is None:
            return
        now = time.monotonic()
        if (not force and self.progress_interval_seconds is not None
                and now - self._last_progress_report < self.progress_interval_seconds):
            return
        self.progress(CycleProgress(
            phase="online_replay",
            completed_events=self._completed_events,
            total_events=self._accepted_events,
            completed_iterations=0,
            total_iterations=0,
            last_completed_iteration=None,
            simulated_cycles=self._cycle,
            elapsed_seconds=now - self._started_at,
        ))
        self._last_progress_report = now


class BufferedVirtualCycleConsumer:
    """Feed one complete iteration of virtual packets into an online session.

    Semantic workset totals are unknowable until all queries for the current
    state version have arrived.  This adapter therefore buffers only the
    compact point list/key/mask packets for one iteration, derives exact
    per-key totals, registers them, and forwards the packets before the first
    lifecycle record can enter the cycle session.
    """

    def __init__(self, session: CycleReplaySession) -> None:
        self.session = session
        self._packets: list[VirtualTracePacket] = []
        self._iteration: int | None = None
        self._iteration_workset_keys: set[tuple[int, int]] = set()
        self._finished = False
        self.result: CycleResult | None = None

    def accept_query_packet(self, packet: VirtualTracePacket) -> None:
        if self._finished:
            raise RuntimeError("buffered virtual consumer is finalized")
        if self._iteration is None:
            self._iteration = packet.iteration_id
        if packet.iteration_id != self._iteration:
            raise ValueError("buffered virtual consumer changed iteration without a close")
        self._packets.append(packet)

    def accept_lifecycle(self, record: VirtualLifecycleRecord) -> None:
        if self._iteration is None:
            self._iteration = record.iteration_id
        if record.iteration_id != self._iteration:
            raise ValueError("buffered virtual consumer lifecycle changed iteration")
        self._flush_packets()
        self.session.accept_lifecycle(record)

    def close_iteration(self, iteration_id: int) -> None:
        if self._iteration != iteration_id:
            raise ValueError("buffered virtual consumer iteration close does not match")
        self._flush_packets()
        self.session.close_iteration(iteration_id)
        self.session.retire_semantic_workset_totals(self._iteration_workset_keys)
        self._iteration_workset_keys.clear()
        self._iteration = None

    def finish(self) -> CycleResult:
        if self._finished:
            raise RuntimeError("buffered virtual consumer is already finalized")
        self._flush_packets()
        if self._iteration_workset_keys:
            self.session.retire_semantic_workset_totals(self._iteration_workset_keys)
            self._iteration_workset_keys.clear()
        self._finished = True
        self.result = self.session.finish()
        return self.result

    def _flush_packets(self) -> None:
        if not self._packets:
            return
        totals: dict[tuple[int, int], int] = defaultdict(int)
        for packet in self._packets:
            counts = packet.relation_counts_by_candidate
            for gaussian_id, count in zip(packet.point_ids, counts, strict=True):
                if int(count):
                    key = (int(gaussian_id), int(packet.state_version))
                    totals[key] += int(count)
        self._iteration_workset_keys.update(totals)
        self.session.register_semantic_workset_totals(totals)
        packets = tuple(self._packets)
        self._packets.clear()
        self.session.accept_query_packets(packets)
