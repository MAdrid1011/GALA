"""Dependency-aware, event-jumping cycle executor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import defaultdict, deque
from dataclasses import dataclass, replace
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
from .oracle import FutureTracePlan
from .packets import PhysicalPacketStage, RelationPacketPlan, RelationWindowPlan
from .telemetry import ComputeTelemetry, ComputeTelemetryCollector
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
    RelationWindowTracker,
    QueryReplayTracker,
    OwnerGradientTracker,
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

    @property
    def query_scheduler_enabled(self) -> bool:
        """Whether the three bounded F/C/A source FIFOs feed the scheduler."""

        return self.query_load_rules or self.overlap_guided_issue


@dataclass(frozen=True)
class OraclePortfolioMember:
    name: str
    policy: str
    status: str
    total_cycles: int | None
    failure_reason: str | None


@dataclass(frozen=True)
class OraclePortfolio:
    winner: str
    members: tuple[OraclePortfolioMember, ...]


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
    oracle_portfolio: OraclePortfolio | None = None
    compute_telemetry: ComputeTelemetry | None = None
    oracle_member_results: dict[str, "CycleResult"] | None = None


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
    resource: str | None = None
    pod: int | None = None
    cluster: int | None = None
    resource_cycle: int | None = None
    resource_in_use: int | None = None
    resource_demand: int | None = None
    resource_capacity: int | None = None


@dataclass
class _InFlight:
    completion_cycle: int
    event_id: int
    module: str


class _ReadyCandidateQueue:
    """Heap-index ready work without changing candidate ordering semantics."""

    def __init__(self) -> None:
        self._next_token = 0
        self._candidate_by_token: dict[int, tuple[int, int]] = {}
        self._general: list[tuple[tuple[int, int], int]] = []
        self._replay_consumers: list[tuple[tuple[int, int], int]] = []
        self._simple_by_cluster: dict[
            int, list[tuple[tuple[int, int], int]]
        ] = defaultdict(list)
        self._simple_by_key: dict[
            tuple[int, int, int], list[tuple[tuple[int, int], int]]
        ] = defaultdict(list)
        self._simple_location_by_token: dict[
            int, tuple[int, tuple[int, int, int]]
        ] = {}
        self._complex: list[tuple[tuple[int, int], int]] = []
        self._complex_event_ids: dict[int, tuple[int, ...]] = {}
        self._configured = False
        self._module_name = ""
        self._row_for: Callable[[int], np.void] | None = None
        self._physical_stage_for: Callable[[int], PhysicalPacketStage | None] | None = None
        self._owner_gradients: OwnerGradientTracker | None = None
        self._query_replay: QueryReplayTracker | None = None

    def __bool__(self) -> bool:
        return bool(self._candidate_by_token)

    def __len__(self) -> int:
        return len(self._candidate_by_token)

    def __iter__(self):
        return iter(sorted(self._candidate_by_token.values()))

    def __getitem__(self, index):
        return sorted(self._candidate_by_token.values())[index]

    def push(self, candidate: tuple[int, int]) -> None:
        token = self._next_token
        self._next_token += 1
        self._candidate_by_token[token] = candidate
        if self._configured:
            self._index(token, candidate)
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """Bound lazy-deletion overhead after repeated resource requeues.

        Rejected ready work is issued again with a fresh token so the indexed
        heaps preserve their existing ordering contract.  Long traces can
        otherwise retain many stale entries in each secondary index.  Rebuild
        only after the stale population is materially larger than the live
        candidate set; this keeps normal short-queue behavior unchanged.
        """

        # Counting every secondary heap is itself linear in the number of
        # indexed keys, so inspect the stale population at a coarse cadence.
        # The token is monotonic and therefore also covers queues that are
        # repeatedly drained and refilled.
        if self._next_token % 256:
            return
        live = len(self._candidate_by_token)
        heap_entries = (
            len(self._general)
            + len(self._replay_consumers)
            + len(self._complex)
            + sum(len(heap) for heap in self._simple_by_cluster.values())
            + sum(len(heap) for heap in self._simple_by_key.values())
        )
        threshold = max(128, live * 4 + 64)
        if heap_entries <= threshold:
            return

        def retain(heap: list[tuple[tuple[int, int], int]]) -> None:
            heap[:] = [
                entry for entry in heap if entry[1] in self._candidate_by_token
            ]
            heapq.heapify(heap)

        retain(self._general)
        retain(self._replay_consumers)
        retain(self._complex)
        for heaps in (self._simple_by_cluster, self._simple_by_key):
            empty_keys = []
            for key, heap in heaps.items():
                retain(heap)
                if not heap:
                    empty_keys.append(key)
            for key in empty_keys:
                del heaps[key]

    def _index(self, token: int, candidate: tuple[int, int]) -> None:
        assert self._row_for is not None
        assert self._physical_stage_for is not None
        owner_gradients = self._owner_gradients
        query_replay = self._query_replay
        event_id, _stage = candidate
        row = self._row_for(event_id)
        kind = PrimitiveKind(int(row["primitive_kind"]))
        if (
            query_replay is not None
            and self._module_name == "bidirectional_query"
            and kind is PrimitiveKind.CONSUMER
            and query_replay.requires_consumer_slot(event_id)
        ):
            heapq.heappush(self._replay_consumers, (candidate, token))
            return
        if (
            owner_gradients is None
            or self._module_name != "compute_pod"
            or kind is not PrimitiveKind.ADJOINT
        ):
            heapq.heappush(self._general, (candidate, token))
            return
        physical_stage = self._physical_stage_for(event_id)
        event_ids = (
            physical_stage.event_ids
            if physical_stage is not None else (event_id,)
        )
        keys = tuple(dict.fromkeys(
            owner_gradients.key_for_adjoint(member) for member in event_ids
        ))
        if len(keys) == 1:
            key = keys[0]
            cluster = owner_gradients.cluster_for_key(key)
            heapq.heappush(self._simple_by_cluster[cluster], (candidate, token))
            heapq.heappush(self._simple_by_key[key], (candidate, token))
            self._simple_location_by_token[token] = (cluster, key)
            return
        heapq.heappush(self._complex, (candidate, token))
        self._complex_event_ids[token] = event_ids

    def _configure(
        self,
        module_name: str,
        *,
        row_for: Callable[[int], np.void],
        physical_stage_for: Callable[[int], PhysicalPacketStage | None],
        owner_gradients: OwnerGradientTracker | None,
        query_replay: QueryReplayTracker | None,
    ) -> None:
        if self._configured:
            if module_name != self._module_name:
                raise CycleConfigurationError("ready queue changed module ownership")
            self._row_for = row_for
            self._physical_stage_for = physical_stage_for
            return
        self._configured = True
        self._module_name = module_name
        self._row_for = row_for
        self._physical_stage_for = physical_stage_for
        self._owner_gradients = owner_gradients
        self._query_replay = query_replay
        for token, candidate in self._candidate_by_token.items():
            self._index(token, candidate)

    def _peek(
        self, heap: list[tuple[tuple[int, int], int]],
    ) -> tuple[tuple[int, int], int] | None:
        while heap and heap[0][1] not in self._candidate_by_token:
            heapq.heappop(heap)
        return heap[0] if heap else None

    def _complex_candidate(self) -> tuple[tuple[int, int], int] | None:
        owner_gradients = self._owner_gradients
        if owner_gradients is None:
            return self._peek(self._complex)
        best: tuple[tuple[int, int], int] | None = None
        for entry in self._complex:
            candidate, token = entry
            if token not in self._candidate_by_token:
                continue
            if owner_gradients.blocks_adjoint(self._complex_event_ids[token]):
                continue
            if best is None or candidate < best[0]:
                best = entry
        return best

    def _earliest_acceptable(self) -> tuple[tuple[int, int], int] | None:
        choices: list[tuple[tuple[int, int], int]] = []
        general = self._peek(self._general)
        if general is not None:
            choices.append(general)
        query_replay = self._query_replay
        if (
            query_replay is not None
            and len(query_replay.active_queries) < query_replay.capacity
        ):
            consumer = self._peek(self._replay_consumers)
            if consumer is not None:
                choices.append(consumer)
        owner_gradients = self._owner_gradients
        if owner_gradients is not None:
            for cluster, heap in self._simple_by_cluster.items():
                active = owner_gradients.active_by_cluster.get(cluster, set())
                if len(active) < owner_gradients.slots_per_cluster:
                    entry = self._peek(heap)
                    if entry is not None:
                        choices.append(entry)
                    continue
                for key in active:
                    entry = self._peek(self._simple_by_key[key])
                    if entry is not None:
                        choices.append(entry)
        complex_entry = self._complex_candidate()
        if complex_entry is not None:
            choices.append(complex_entry)
        return min(choices, default=None)

    def _remove(self, entry: tuple[tuple[int, int], int]) -> None:
        _candidate, token = entry
        location = self._simple_location_by_token.pop(token, None)
        if location is not None:
            cluster, key = location
            key_heap = self._simple_by_key[key]
            if self._peek(key_heap) == entry:
                heapq.heappop(key_heap)
            self._peek(key_heap)
            if not key_heap:
                del self._simple_by_key[key]
            cluster_heap = self._simple_by_cluster[cluster]
            if self._peek(cluster_heap) == entry:
                heapq.heappop(cluster_heap)
            self._peek(cluster_heap)
            if not cluster_heap:
                del self._simple_by_cluster[cluster]
        elif self._peek(self._general) == entry:
            heapq.heappop(self._general)
        elif self._peek(self._replay_consumers) == entry:
            heapq.heappop(self._replay_consumers)
        self._complex_event_ids.pop(token, None)
        del self._candidate_by_token[token]

    def pop_acceptable(
        self,
        width: int,
        module_name: str,
        *,
        row_for: Callable[[int], np.void],
        physical_stage_for: Callable[[int], PhysicalPacketStage | None],
        owner_gradients: OwnerGradientTracker | None,
        query_replay: QueryReplayTracker | None,
    ) -> list[tuple[int, int]]:
        self._configure(
            module_name, row_for=row_for,
            physical_stage_for=physical_stage_for,
            owner_gradients=owner_gradients,
            query_replay=query_replay,
        )
        selected: list[tuple[int, int]] = []
        for _ in range(min(width, len(self))):
            entry = self._earliest_acceptable()
            if entry is None:
                break
            candidate, _token = entry
            self._remove(entry)
            selected.append(candidate)
        return selected


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

    def __init__(
        self,
        config: CycleConfig,
        *,
        policy: str = "base",
        _oracle_portfolio_member: bool = False,
    ) -> None:
        valid_variants = {f"variant:{bits:04b}" for bits in range(16)}
        if policy not in {"base", "query_oracle", "residency_oracle", "query", "residency", "full"} | valid_variants:
            raise ValueError(f"unknown cycle policy: {policy}")
        self.config = config
        self.policy = policy
        self._oracle_portfolio_member = _oracle_portfolio_member
        self.selection = self._selection_for_policy(policy)
        self.issue_scheduler = FusionIssueScheduler(
            candidate_lanes=config.candidate_lanes,
            forward_ports=config.fusion_forward_ports or config.modules["fusion_issue"].ports,
            consumer_ports=config.fusion_consumer_ports or config.modules["fusion_issue"].ports,
            adjoint_ports=config.fusion_adjoint_ports or config.modules["fusion_issue"].ports,
            query_state_entries=config.query_state_entries,
            query_state_lanes=config.relation_query_lanes,
        )
        counters = {name: CounterBlock() for name in config.modules}
        self.modules = {
            "relation_constructor": RelationConstructor("relation_constructor", config.modules["relation_constructor"], counters["relation_constructor"]),
            "fusion_issue": FusionIssueUnit("fusion_issue", config.modules["fusion_issue"], counters["fusion_issue"]),
            "semantic_cache": GaussianSemanticCache("semantic_cache", config.modules["semantic_cache"], counters["semantic_cache"]),
            "compute_pod": ComputePod(
                "compute_pod", config.modules["compute_pod"], counters["compute_pod"],
                template_profiles=(dict(config.compute_templates)
                                   if config.compute_templates is not None else None),
                resource_capacities=(dict(config.compute_resource_capacities)
                                     if config.compute_resource_capacities is not None else None),
            ),
            "bidirectional_query": BidirectionalQueryUnit("bidirectional_query", config.modules["bidirectional_query"], counters["bidirectional_query"]),
            "reconstruction_update": ReconstructionUpdateUnit("reconstruction_update", config.modules["reconstruction_update"], counters["reconstruction_update"]),
            "shared_sram": SharedSram("shared_sram", config.modules["shared_sram"], counters["shared_sram"]),
        }
        self._stalls: list[_StallAccumulator] = []
        self._stall_index: dict[tuple[object, ...], int] = {}
        self._stall_cycle: int | None = None
        self._query_history_bases: dict[tuple[int, int], int] = {}

    def _register_query_history_base(
        self, iteration_id: int, template_id: int, query_base: int,
    ) -> None:
        if min(iteration_id, template_id, query_base) < 0:
            raise CycleConfigurationError("query history domain is negative")
        domain = (int(iteration_id), int(template_id))
        previous = self._query_history_bases.get(domain)
        if previous is None or query_base < previous:
            self._query_history_bases[domain] = int(query_base)

    def _register_query_history_plan(
        self, trace: Trace, packet_plan: RelationPacketPlan,
    ) -> None:
        self._query_history_bases.clear()
        for stage in packet_plan.stages:
            event_index = stage.head_event_id - packet_plan.event_id_base
            if not 0 <= event_index < trace.event_count:
                raise CycleConfigurationError(
                    "physical query packet is outside its trace"
                )
            row = trace.events[event_index]
            if int(row["query_id"]) < 0:
                continue
            self._register_query_history_base(
                int(row["iteration_id"]), int(row["template_id"]),
                stage.query_base,
            )

    def _query_history_key(self, row: np.void) -> tuple[int, int]:
        query_id = int(row["query_id"])
        template_id = int(row["template_id"])
        domain = (int(row["iteration_id"]), template_id)
        query_base = self._query_history_bases.get(domain)
        if query_base is None:
            raise CycleConfigurationError(
                "query history domain was not registered before lifecycle replay"
            )
        local_query_id = query_id - query_base
        if local_query_id < 0:
            raise CycleConfigurationError("query precedes its history domain base")
        return template_id, local_query_id

    def _record_stall(
        self,
        cycle: int,
        module: str,
        reason: str,
        event_id: int,
        *,
        resource: str | None = None,
        pod: int | None = None,
        cluster: int | None = None,
        resource_cycle: int | None = None,
        resource_in_use: int | None = None,
        resource_demand: int | None = None,
        resource_capacity: int | None = None,
    ) -> None:
        if self._stall_cycle != cycle:
            self._stall_index.clear()
            self._stall_cycle = cycle
        key = (
            cycle, module, reason, resource, pod, cluster, resource_cycle,
            resource_in_use, resource_demand, resource_capacity,
        )
        index = self._stall_index.get(key)
        if index is None:
            self._stall_index[key] = len(self._stalls)
            self._stalls.append(_StallAccumulator(
                cycle, module, reason, [event_id], 1, resource, pod, cluster,
                resource_cycle, resource_in_use, resource_demand,
                resource_capacity,
            ))
            return
        previous = self._stalls[index]
        if len(previous.event_ids) < self.config.candidate_lanes:
            previous.event_ids.append(event_id)
        previous.count += 1

    def _stall_records(self) -> tuple[StallRecord, ...]:
        return tuple(
            StallRecord(
                item.cycle, item.module, item.reason, tuple(item.event_ids),
                item.count, item.resource, item.pod, item.cluster,
                item.resource_cycle, item.resource_in_use,
                item.resource_demand, item.resource_capacity,
            )
            for item in self._stalls
        )

    def _new_compute_telemetry(self) -> ComputeTelemetryCollector:
        capacities = self.config.compute_resource_capacities or {}
        clusters = int(capacities.get("clusters", 1))
        clusters_per_pod = int(capacities.get("clusters_per_pod", clusters))
        return ComputeTelemetryCollector(
            cluster_count=clusters, clusters_per_pod=clusters_per_pod,
        )

    def _record_compute_resource_stall(
        self,
        *,
        cycle: int,
        event_id: int,
        pod: int | None,
        compute: ComputePod,
        plan: tuple[tuple[str, int, int], ...],
    ) -> None:
        blocker = compute.first_blocking_resource(plan, cycle)
        if blocker is None:
            raise CycleConfigurationError(
                "ComputePod rejected a reservation without a blocking resource"
            )
        resource, point, in_use, demand, capacity = blocker
        cluster = (
            int(resource.partition(":")[2]) if ":" in resource else None
        )
        self._record_stall(
            cycle, "compute_pod", "compute_resource", event_id,
            resource=resource.partition(":")[0], pod=pod, cluster=cluster,
            resource_cycle=point, resource_in_use=in_use,
            resource_demand=demand, resource_capacity=capacity,
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
            return ("compute_pod",)
        if kind is PrimitiveKind.CONSUMER:
            return ("fusion_issue", "bidirectional_query")
        if kind is PrimitiveKind.ADJOINT:
            return ("fusion_issue", "bidirectional_query", "compute_pod")
        if kind is PrimitiveKind.FORWARD:
            return ("fusion_issue", "compute_pod", "bidirectional_query")
        return ("fusion_issue", "compute_pod")

    def _ordered_candidates(
        self,
        trace: Trace,
        candidates: list[int],
        *,
        future_plan: FutureTracePlan | None = None,
    ) -> list[int]:
        base_order = candidates
        if self.selection.query_oracle:
            if future_plan is None:
                raise CycleConfigurationError(
                    "query Oracle candidate ordering requires a future plan"
                )
            # Future visibility is confined to Fusion candidate selection.
            # Reordering unrelated modules changes arbitration outside the
            # mechanism whose upper bound this Oracle is meant to measure.
        return base_order

    @staticmethod
    def _task_packet(
        trace: Trace,
        event_id: int,
        packet_stage: PhysicalPacketStage | None = None,
        *,
        consumer_owner_ids: tuple[int, ...] = (),
        workset_hit: bool = False,
        age: int = 0,
    ) -> TaskPacket:
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
        if kind is PrimitiveKind.CONSUMER and not consumer_owner_ids:
            consumer_owner_ids = CycleEngine._consumer_owner_ids(trace, event_id)
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
            target_resource=(
                (int(row["resource_class"]) << 56)
                | int(row["address_token"])
            ),
            consumer_owner_ids=consumer_owner_ids,
            workset_hit=workset_hit,
            age=age,
            conflict_query_ids=(
                tuple(
                    int(trace.events[member]["query_id"])
                    for member in packet_stage.event_ids
                )
                if task_kind in {TaskKind.FORWARD, TaskKind.ADJOINT}
                and packet_stage is not None
                else ()
            ),
        )

    @staticmethod
    def _consumer_owner_ids(trace: Trace, event_id: int) -> tuple[int, ...]:
        """Return the real query results whose last arrival releases a consumer."""

        row = trace.events[event_id]
        owners = {
            int(trace.events[int(dependency)]["query_id"])
            for dependency in trace.dependency_ids(row)
            if PrimitiveKind(
                int(trace.events[int(dependency)]["primitive_kind"])
            ) is PrimitiveKind.QUERY_REDUCTION
            and int(trace.events[int(dependency)]["query_id"]) >= 0
        }
        return tuple(sorted(owners))

    @staticmethod
    def _workset_hit(
        cache_states: Mapping[int, SemanticCacheState], row: np.void,
    ) -> bool:
        """Report a hit only when the keyed active record is actually resident."""

        if not cache_states or int(row["gaussian_id"]) < 0:
            return False
        key = (int(row["gaussian_id"]), int(row["state_version"]))
        return any(key in state.active for state in cache_states.values())

    def _update_query_state_for_completion(
        self,
        row: np.void,
        kind: PrimitiveKind,
        adjoint_relation_ids: set[int],
    ) -> None:
        """Commit one logical relation/query lifecycle event to live F/C/A."""

        if not self.issue_scheduler.strict_lifecycle:
            return
        query_id = int(row["query_id"])
        if query_id < 0:
            return
        try:
            if kind is PrimitiveKind.RELATION:
                relation_id = int(row["relation_id"])
                self.issue_scheduler.relation_accept(
                    (query_id,),
                    adjoint_query_ids=(
                        (query_id,) if relation_id in adjoint_relation_ids else ()
                    ),
                    iteration_id=int(row["iteration_id"]),
                    history_keys=(self._query_history_key(row),),
                )
            elif kind is PrimitiveKind.QUERY_CLOSE:
                self.issue_scheduler.producer_close((query_id,))
            elif kind is PrimitiveKind.FORWARD:
                self.issue_scheduler.forward_retire((query_id,))
            elif kind is PrimitiveKind.QUERY_REDUCTION:
                self.issue_scheduler.reduction_writeback((query_id,))
            elif kind is PrimitiveKind.ADJOINT:
                self.issue_scheduler.adjoint_retire((query_id,))
        except ValueError as error:
            raise CycleConfigurationError(
                f"invalid query lifecycle at {kind.name} event "
                f"{int(row['event_id'])}: {error}"
            ) from error

    @staticmethod
    def _selection_for_policy(policy: str) -> MechanismSelection:
        if policy == "query_oracle":
            return MechanismSelection(False, False, True, False, query_oracle=True)
        if policy == "residency_oracle":
            return MechanismSelection(False, True, False, True, residency_oracle=True)
        aliases = {
            "base": "0000",
            "query": "1010",
            "residency": "0101",
            "full": "1111",
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

    def _future_plan(
        self,
        trace: Trace,
        dependency_index: _DependencyIndex,
        packet_plan: RelationPacketPlan,
    ) -> FutureTracePlan:
        service_cycles = np.empty(trace.event_count, dtype=np.uint64)
        for event_id, row in enumerate(trace.events):
            kind = PrimitiveKind(int(row["primitive_kind"]))
            physical_stage = packet_plan.stage_for_event(event_id)
            total = 0
            for module_name in self._stages_for(kind):
                if (
                    physical_stage is not None
                    and module_name == "compute_pod"
                    and kind is PrimitiveKind.FORWARD
                    and self.config.compute_templates is not None
                ):
                    compute = self.modules[module_name]
                    assert isinstance(compute, ComputePod)
                    path = compute.path_for(int(row["template_id"]), kind)
                    lane = physical_stage.lanes[
                        physical_stage.event_ids.index(event_id)
                    ]
                    total += path.packet_completion_offset(lane)
                else:
                    total += self._physical_service_cycles(
                        module_name, row, kind, physical_stage,
                    )
            service_cycles[event_id] = total
        return FutureTracePlan.from_trace(
            trace,
            event_service_cycles=service_cycles,
            dependent_offsets=dependency_index.offsets,
            dependents=dependency_index.dependents,
            packet_plan=packet_plan,
        )

    def _select_query_oracle_candidates(
        self,
        trace: Trace,
        event_ids: Iterable[int],
        future_plan: FutureTracePlan,
        packet_plan: RelationPacketPlan | None = None,
        consumer_owner_for_event: Mapping[int, int] | None = None,
    ) -> tuple[int, ...]:
        """Maximize future weight for the frozen three-candidate contract."""

        if (
            self.config.candidate_lanes != 3
            or self.config.fusion_consumer_ports != 1
            or self.config.fusion_forward_ports not in {1, 2}
            or self.config.fusion_adjoint_ports not in {1, 2}
        ):
            raise CycleConfigurationError(
                "query Oracle requires the frozen three-candidate contract with "
                "one consumer port and one or two forward/adjoint ports"
            )
        by_kind: dict[TaskKind, list[tuple[int, int, TaskPacket]]] = {
            TaskKind.FORWARD: [],
            TaskKind.CONSUMER: [],
            TaskKind.ADJOINT: [],
        }
        for event_id in event_ids:
            packet = self._task_packet(
                trace, event_id,
                packet_plan.stage_for_event(event_id) if packet_plan is not None else None,
                consumer_owner_ids=(
                    (consumer_owner_for_event[event_id],)
                    if consumer_owner_for_event is not None
                    and event_id in consumer_owner_for_event else ()
                ),
            )
            weight = int(future_plan.critical_cycles[event_id])
            bank = self._fusion_query_state_bank(trace.events[event_id])
            by_kind[packet.task_kind].append((weight, bank, packet))
        for candidates in by_kind.values():
            candidates.sort(key=lambda item: (-item[0], item[2].event_id))

        source_limits = {
            TaskKind.FORWARD: 1,
            TaskKind.CONSUMER: 1,
            TaskKind.ADJOINT: 1,
        }
        if not by_kind[TaskKind.CONSUMER]:
            source_limits[TaskKind.CONSUMER] = 0
            source_limits[TaskKind.FORWARD] = self.config.fusion_forward_ports
            source_limits[TaskKind.ADJOINT] = self.config.fusion_adjoint_ports

        candidates = sorted(
            (
                (weight, bank, packet)
                for kind_candidates in by_kind.values()
                for weight, bank, packet in kind_candidates
            ),
            key=lambda item: (-item[0], item[2].event_id),
        )
        best_score = -1
        best_ids: tuple[int, ...] = ()

        def search(
            index: int,
            score: int,
            selected: tuple[int, ...],
            used_sources: dict[TaskKind, int],
            occupied_keys: frozenset[tuple[ReductionDomain, int]],
            occupied_targets: frozenset[int],
            occupied_banks: frozenset[int],
        ) -> None:
            nonlocal best_score, best_ids
            if score > best_score:
                best_score = score
                best_ids = selected
            remaining_slots = self.config.candidate_lanes - len(selected)
            if remaining_slots == 0 or index == len(candidates):
                return
            optimistic = score + sum(
                weight for weight, _bank, _packet in
                candidates[index:index + remaining_slots]
            )
            if optimistic <= best_score:
                return

            weight, bank, packet = candidates[index]
            tracked_states = (
                self.issue_scheduler.states.get(query_id)
                for query_id in packet.readiness_query_ids
            )
            exact_ready = (
                not self.issue_scheduler.enforce_exact_readiness
                or all(
                    state is not None
                    and state.exact_ready(packet.task_kind)
                    for state in tracked_states
                )
            )
            target = packet.target_resource
            if (
                exact_ready
                and used_sources[packet.task_kind]
                < source_limits[packet.task_kind]
                and packet.conflict_keys.isdisjoint(occupied_keys)
                and bank not in occupied_banks
                and (target is None or target not in occupied_targets)
            ):
                next_sources = dict(used_sources)
                next_sources[packet.task_kind] += 1
                search(
                    index + 1,
                    score + weight,
                    (*selected, packet.event_id),
                    next_sources,
                    occupied_keys.union(packet.conflict_keys),
                    (
                        occupied_targets
                        if target is None else occupied_targets.union((target,))
                    ),
                    occupied_banks.union((bank,)),
                )
            search(
                index + 1, score, selected, used_sources,
                occupied_keys, occupied_targets, occupied_banks,
            )

        search(
            0,
            0,
            (),
            {kind: 0 for kind in TaskKind},
            frozenset(),
            frozenset(),
            frozenset(),
        )
        return best_ids

    def _fusion_port_limit(self, kind: PrimitiveKind) -> tuple[str, int]:
        timing_ports = self.config.modules["fusion_issue"].ports
        if kind is PrimitiveKind.FORWARD:
            return "forward", self.config.fusion_forward_ports or timing_ports
        if kind is PrimitiveKind.CONSUMER:
            return "consumer", self.config.fusion_consumer_ports or timing_ports
        if kind is PrimitiveKind.ADJOINT:
            return "adjoint", self.config.fusion_adjoint_ports or timing_ports
        raise CycleConfigurationError(f"invalid fusion task kind: {kind.name}")

    def _pop_fusion_input_candidates(
        self,
        inputs: Mapping[TaskKind, deque[int]],
        *,
        borrow_cursor: int,
    ) -> tuple[list[int], dict[int, tuple[TaskKind, int]], int]:
        """Pop real FIFO heads, lending otherwise idle candidate lanes.

        Every nonempty source retains one first-look slot.  A remaining slot may
        inspect another head only when that source has a second physical output
        port.  The rotating cursor prevents a continuously nonempty Forward or
        Adjoint FIFO from monopolizing the single commonly idle Consumer slot.
        """

        source_order = (
            TaskKind.FORWARD, TaskKind.CONSUMER, TaskKind.ADJOINT,
        )
        port_limits = {
            TaskKind.FORWARD: self.issue_scheduler.forward_ports,
            TaskKind.CONSUMER: self.issue_scheduler.consumer_ports,
            TaskKind.ADJOINT: self.issue_scheduler.adjoint_ports,
        }
        selected: list[int] = []
        selected_per_source = {kind: 0 for kind in source_order}
        pop_ranks: dict[int, tuple[TaskKind, int]] = {}

        def take(kind: TaskKind) -> None:
            event_id = inputs[kind].popleft()
            rank = selected_per_source[kind]
            selected_per_source[kind] += 1
            selected.append(event_id)
            pop_ranks[event_id] = (kind, rank)

        for kind in source_order:
            if len(selected) >= self.config.candidate_lanes:
                break
            if inputs[kind]:
                take(kind)

        borrow_order = (TaskKind.FORWARD, TaskKind.ADJOINT, TaskKind.CONSUMER)
        while len(selected) < self.config.candidate_lanes:
            chosen_index: int | None = None
            for offset in range(len(borrow_order)):
                index = (borrow_cursor + offset) % len(borrow_order)
                kind = borrow_order[index]
                if (
                    inputs[kind]
                    and selected_per_source[kind] < port_limits[kind]
                ):
                    chosen_index = index
                    break
            if chosen_index is None:
                break
            take(borrow_order[chosen_index])
            borrow_cursor = (chosen_index + 1) % len(borrow_order)
        return selected, pop_ranks, borrow_cursor

    def _module_issue_ports(self, module_name: str) -> int:
        """Return the actual issue slots represented by a module instance.

        The legacy ``compute_pod`` timing port is an aggregate fixture value.
        Production profiles expose the frozen 20-cluster organization, so the
        scheduler must not collapse those clusters behind that fixture port.
        """

        if module_name == "compute_pod" and self.config.compute_templates is not None:
            capacities = self.config.compute_resource_capacities or {}
            issue_slots = capacities.get("cluster_issue")
            if issue_slots is None or issue_slots <= 0:
                raise CycleConfigurationError(
                    "compute template profiles require positive cluster issue capacity"
                )
            return int(issue_slots)
        if module_name == "semantic_cache" and self.config.cache_instances is not None:
            return (
                self.modules[module_name].timing.ports
                * self.config.cache_instances
            )
        if (
            module_name == "relation_constructor"
            and self.config.relation_support_lanes is not None
            and self.config.query_relation_store_banks is not None
        ):
            return (
                self.config.relation_support_lanes
                + self.config.query_relation_store_banks
            )
        if module_name == "bidirectional_query" and self._has_query_resources():
            assert self.config.query_reduction_banks is not None
            assert self.config.query_partial_sum_groups_per_bank is not None
            assert self.config.query_loss_queries_per_cycle is not None
            assert self.config.query_adjoint_replay_lanes is not None
            assert self.config.query_volume_banks is not None
            return (
                self.config.query_reduction_banks
                + self.config.query_loss_queries_per_cycle
                + self.config.query_adjoint_replay_lanes
                + 2 * self.config.query_volume_banks
            )
        if module_name == "shared_sram":
            # SharedSRAM is a banked 1R1W array.  ``ModuleTiming.ports`` is a
            # legacy aggregate fixture value and must not collapse sixteen
            # independent Banks into two chip-wide ports.
            banks = self.modules[module_name].timing.banks
            return banks * (
                self.config.shared_sram_read_ports_per_bank
                + self.config.shared_sram_write_ports_per_bank
            )
        return self.modules[module_name].timing.ports

    def _has_query_resources(self) -> bool:
        return self.config.query_reduction_banks is not None

    def _query_resource_allocation(
        self,
        module_lanes: list[int],
        row: np.void,
        kind: PrimitiveKind,
        physical_stage: PhysicalPacketStage | None,
        cycle: int,
    ) -> tuple[int, ...] | None:
        """Allocate the frozen, independent query-unit datapaths."""

        if not self._has_query_resources():
            raise CycleConfigurationError("query resource allocation is unavailable")
        banks = self.config.query_reduction_banks
        groups = self.config.query_partial_sum_groups_per_bank
        loss_slots = self.config.query_loss_queries_per_cycle
        replay_lanes = self.config.query_adjoint_replay_lanes
        volume_banks = self.config.query_volume_banks
        assert (
            banks is not None
            and groups is not None
            and loss_slots is not None
            and replay_lanes is not None
            and volume_banks is not None
        )
        # ``groups`` are active partial entries within one physical bank.  They
        # provide feedback storage, not independent issue ports; the C++
        # network accepts at most one input per bank per cycle.
        reduction_slots = banks
        loss_base = reduction_slots
        replay_base = loss_base + loss_slots
        volume_read_base = replay_base + replay_lanes
        volume_write_base = volume_read_base + volume_banks

        def first_free(begin: int, count: int, needed: int) -> tuple[int, ...] | None:
            available = tuple(
                lane for lane in range(begin, begin + count)
                if module_lanes[lane] <= cycle
            )
            return available[:needed] if len(available) >= needed else None

        def reduction_slot(query_id: int) -> int:
            return query_id & (banks - 1)

        if kind is PrimitiveKind.FORWARD:
            # ComputePod emits packet lanes independently.  By the time a
            # FORWARD event reaches this stage, only this logical lane is
            # issuing into the query reduction fabric; reserving every lane
            # in its source packet would multiply bank occupancy.
            allocation = (reduction_slot(int(row["query_id"])),)
        elif kind is PrimitiveKind.QUERY_REDUCTION:
            query_id = int(row["query_id"])
            allocation = (
                reduction_slot(query_id),
                volume_write_base + query_id % volume_banks,
            )
        elif kind is PrimitiveKind.CONSUMER:
            loss = first_free(loss_base, loss_slots, 1)
            if loss is None:
                return None
            query_id = int(row["query_id"])
            allocation = (
                *loss,
                volume_read_base + query_id % volume_banks,
                volume_write_base + query_id % volume_banks,
            )
        elif kind is PrimitiveKind.ADJOINT:
            query_ids = (
                tuple(
                    physical_stage.query_base + lane
                    for lane in physical_stage.lanes
                )
                if physical_stage is not None else (int(row["query_id"]),)
            )
            if len(query_ids) > replay_lanes:
                raise CycleConfigurationError("adjoint packet exceeds replay lanes")
            replay = first_free(replay_base, replay_lanes, len(query_ids))
            if replay is None:
                return None
            reads = tuple(
                volume_read_base + bank
                for bank in dict.fromkeys(query_id % volume_banks for query_id in query_ids)
            )
            allocation = (*replay, *reads)
        else:
            raise CycleConfigurationError(
                f"{kind.name} has no bidirectional-query datapath"
            )
        return (
            allocation
            if all(module_lanes[lane] <= cycle for lane in allocation)
            else None
        )

    def _module_partition_count(self, module_name: str) -> int:
        if module_name == "semantic_cache" and self.config.cache_instances is not None:
            return self.config.cache_instances
        if (
            module_name == "relation_constructor"
            and self.config.relation_support_lanes is not None
            and self.config.query_relation_store_banks is not None
        ):
            return 1 + self.config.query_relation_store_banks
        return 1

    def _module_partition(self, module_name: str, row: np.void) -> int:
        if module_name == "semantic_cache" and self.config.cache_instances is not None:
            return self._cache_instance(int(row["gaussian_id"]))
        if (
            module_name == "relation_constructor"
            and self.config.relation_support_lanes is not None
            and self.config.query_relation_store_banks is not None
        ):
            banks = self.config.query_relation_store_banks
            kind = PrimitiveKind(int(row["primitive_kind"]))
            if kind is PrimitiveKind.RELATION_CANDIDATE:
                return 0
            query_id = int(row["query_id"])
            if query_id < 0:
                bank = int(row["address_token"]) % banks
            else:
                bank = (query_id // self.config.relation_query_lanes) % banks
            return 1 + bank
        return 0

    def _module_partition_issue_limit(
        self, module_name: str, partition: int | None = None,
    ) -> int:
        if module_name == "semantic_cache" and self.config.cache_instances is not None:
            return self.modules[module_name].timing.ports
        if (
            module_name == "relation_constructor"
            and self.config.relation_support_lanes is not None
            and self.config.query_relation_store_banks is not None
        ):
            if partition is None:
                return self._module_issue_ports(module_name)
            return self.config.relation_support_lanes if partition == 0 else 1
        if module_name == "shared_sram":
            return self._module_issue_ports(module_name)
        return self._module_issue_ports(module_name)

    def _module_lane_indices(self, module_name: str, row: np.void) -> range:
        if module_name == "semantic_cache" and self.config.cache_instances is not None:
            ports = self.modules[module_name].timing.ports
            begin = self._module_partition(module_name, row) * ports
            return range(begin, begin + ports)
        if (
            module_name == "relation_constructor"
            and self.config.relation_support_lanes is not None
            and self.config.query_relation_store_banks is not None
        ):
            partition = self._module_partition(module_name, row)
            if partition == 0:
                return range(self.config.relation_support_lanes)
            lane = self.config.relation_support_lanes + partition - 1
            return range(lane, lane + 1)
        if module_name == "shared_sram":
            banks = self.modules[module_name].timing.banks
            bank = self._shared_sram_bank(row)
            kind = PrimitiveKind(int(row["primitive_kind"]))
            if self._shared_sram_direction(kind) == "read":
                begin = bank * self.config.shared_sram_read_ports_per_bank
            else:
                begin = (
                    banks * self.config.shared_sram_read_ports_per_bank
                    + bank * self.config.shared_sram_write_ports_per_bank
                )
            width = (
                self.config.shared_sram_read_ports_per_bank
                if self._shared_sram_direction(kind) == "read"
                else self.config.shared_sram_write_ports_per_bank
            )
            return range(begin, begin + width)
        return range(self._module_issue_ports(module_name))

    def _module_bank_partition(
        self, module_name: str, row: np.void,
    ) -> tuple[int, int]:
        if module_name == "fusion_issue":
            return 0, self._fusion_query_state_bank(row)
        if (
            module_name == "relation_constructor"
            and self.config.relation_support_lanes is not None
            and self.config.query_relation_store_banks is not None
        ):
            return self._module_partition(module_name, row), 0
        if module_name == "shared_sram":
            return self._module_partition(module_name, row), self._shared_sram_bank(row)
        return (
            self._module_partition(module_name, row),
            self.modules[module_name].bank(int(row["address_token"])),
        )

    def _shared_sram_direction(self, kind: PrimitiveKind) -> str:
        if kind is PrimitiveKind.CACHE_REQUEST:
            return "write"
        if kind is PrimitiveKind.CACHE_RETURN:
            return "read"
        raise CycleConfigurationError(
            f"{kind.name} has no SharedSRAM access direction"
        )

    def _shared_sram_address_key(
        self, row: np.void, cycle: int,
    ) -> tuple[int, int]:
        """Return the cycle-scoped physical address key used for read/write replay."""

        return cycle, int(row["address_token"])

    def _shared_sram_bank(self, row: np.void) -> int:
        """Map an Active Gaussian record to its physical global SRAM Bank.

        Captured cache addresses are byte addresses of 128-byte records.  The
        authoritative design stores two 64-byte sectors per record, so using
        a 64-byte line here would incorrectly force every record onto even
        Banks.  Small fixture tokens that are not record-aligned retain the
        generic mapping for backwards-compatible unit tests.
        """

        banks = self.modules["shared_sram"].timing.banks
        token = int(row["address_token"])
        record_bytes = 2 * (self.config.cache_sector_bytes or 1)
        if (
            record_bytes > 1
            and int(row["data_bytes"]) >= record_bytes
            and token % record_bytes == 0
        ):
            return (token // record_bytes) % banks
        return self.modules["shared_sram"].bank(token)

    def _module_bank_reservation_keys(
        self, module_name: str, row: np.void, cycle: int,
    ) -> tuple[tuple[object, ...], ...]:
        partition, bank = self._module_bank_partition(module_name, row)
        base = (module_name, cycle, partition, bank)
        if module_name != "shared_sram":
            return (base,)
        direction = self._shared_sram_direction(
            PrimitiveKind(int(row["primitive_kind"]))
        )
        return (base + (direction,),)

    def _fusion_query_state_bank(self, row: np.void) -> int:
        banks = self.config.fusion_query_state_banks
        if banks is None:
            return self.modules["fusion_issue"].bank(int(row["address_token"]))
        query_id = int(row["query_id"])
        if query_id < 0:
            raise CycleConfigurationError("fusion task has no query identity")
        return (query_id // self.config.relation_query_lanes) % banks

    def _ready_scan_window(self, module_name: str) -> int:
        width = max(
            self.config.candidate_lanes,
            self._module_partition_issue_limit(module_name),
        )
        if (
            module_name == "semantic_cache"
            and self.selection.semantic_residency
            and self.config.cache_multicast_destinations is not None
        ):
            width = max(width, self.config.cache_multicast_destinations)
        return width

    def _pop_ready_candidates(
        self,
        queue: _ReadyCandidateQueue,
        module_name: str,
        *,
        row_for,
        physical_stage_for,
        owner_gradients: OwnerGradientTracker | None,
        query_replay: QueryReplayTracker | None,
    ) -> list[tuple[int, int]]:
        """Pop the earliest work whose owner-gradient epoch is acceptable."""

        width = self._ready_scan_window(module_name)
        return queue.pop_acceptable(
            width, module_name, row_for=row_for,
            physical_stage_for=physical_stage_for,
            owner_gradients=owner_gradients,
            query_replay=query_replay,
        )

    def _uses_generic_module_limits(self, module_name: str) -> bool:
        return not (
            (module_name == "compute_pod" and self.config.compute_templates is not None)
            or (module_name == "bidirectional_query" and self._has_query_resources())
        )

    def _uses_generic_module_bank_limits(self, module_name: str) -> bool:
        return (
            self._uses_generic_module_limits(module_name)
            and not (
                module_name == "relation_constructor"
                and self.config.relation_support_lanes is not None
                and self.config.query_relation_store_banks is not None
            )
        )

    def _is_lane_granular_stage(self, kind: PrimitiveKind, stage: int) -> bool:
        stages = self._stages_for(kind)
        return (
            self._has_query_resources()
            and kind is PrimitiveKind.FORWARD
            and 0 <= stage < len(stages)
            and stages[stage] == "bidirectional_query"
        )

    def _stage_event_ids(
        self,
        event_id: int,
        kind: PrimitiveKind,
        stage: int,
        physical_stage: PhysicalPacketStage | None,
    ) -> tuple[int, ...]:
        if physical_stage is None or self._is_lane_granular_stage(kind, stage):
            return (event_id,)
        return physical_stage.event_ids

    def _service_cycles(
        self, module_name: str, row: np.void, kind: PrimitiveKind,
    ) -> int:
        module = self.modules[module_name]
        if isinstance(module, ComputePod):
            try:
                return module.service_cycles_for(int(row["template_id"]), kind)
            except KeyError as error:
                raise CycleConfigurationError(str(error)) from error
        return module.service_cycles()

    def _compute_route(
        self, row: np.void, kind: PrimitiveKind,
    ) -> tuple[int | None, int | None]:
        """Route Gaussian work through its resident Pod and owner cluster."""

        capacities = self.config.compute_resource_capacities
        if capacities is None or "pods" not in capacities:
            return None, None
        gaussian_id = int(row["gaussian_id"])
        if gaussian_id < 0:
            raise CycleConfigurationError(
                f"{kind.name} ComputePod event has no Gaussian identity"
            )
        pods = int(capacities["pods"])
        clusters_per_pod = int(capacities["clusters_per_pod"])
        pod = gaussian_id % pods
        owner_cluster = pod * clusters_per_pod + gaussian_id % clusters_per_pod
        return (
            pod,
            owner_cluster
            if kind is PrimitiveKind.GRADIENT_REDUCTION else None,
        )

    def _physical_service_cycles(
        self,
        module_name: str,
        row: np.void,
        kind: PrimitiveKind,
        packet_stage: PhysicalPacketStage | None,
    ) -> int:
        """Return resource-retirement latency for one physical task."""

        if (
            packet_stage is not None
            and module_name == "compute_pod"
            and kind in {PrimitiveKind.FORWARD, PrimitiveKind.ADJOINT}
            and self.config.compute_templates is not None
        ):
            module = self.modules[module_name]
            assert isinstance(module, ComputePod)
            try:
                path = module.path_for(int(row["template_id"]), kind)
            except KeyError as error:
                raise CycleConfigurationError(str(error)) from error
            return path.packet_last_result_offset or path.latency
        return self._service_cycles(module_name, row, kind)

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

    def _new_relation_window_tracker(self) -> RelationWindowTracker | None:
        values = (
            self.config.query_relation_window_entries,
            self.config.query_relation_store_records,
            self.config.query_relation_store_banks,
        )
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise CycleConfigurationError(
                "relation-window execution resources are incomplete"
            )
        return RelationWindowTracker(
            window_capacity=int(values[0]),
            relation_capacity=int(values[1]),
            relation_banks=int(values[2]),
        )

    def _new_query_replay_tracker(self) -> QueryReplayTracker | None:
        capacity = self.config.query_replay_queue_entries
        return None if capacity is None else QueryReplayTracker(
            int(capacity),
            stage_gated=(
                not self.selection.query_load_rules
                and not self.selection.query_oracle
            ),
        )

    def _new_owner_gradient_tracker(self) -> OwnerGradientTracker | None:
        slots = self.config.owner_gradient_slots_per_cluster
        capacities = self.config.compute_resource_capacities
        if slots is None or capacities is None or "pods" not in capacities:
            return None
        return OwnerGradientTracker(
            pods=int(capacities["pods"]),
            clusters_per_pod=int(capacities["clusters_per_pod"]),
            slots_per_cluster=int(slots),
        )

    def run(
        self,
        trace: Trace,
        *,
        validate_input: bool = True,
        progress: Callable[[CycleProgress], None] | None = None,
        progress_interval_events: int | None = None,
        progress_interval_seconds: float | None = None,
        collect_compute_telemetry: bool = False,
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
        if (
            (self.selection.query_oracle or self.selection.residency_oracle)
            and not self._oracle_portfolio_member
        ):
            return self._run_oracle_portfolio(
                trace,
                validate_input=validate_input,
                progress=progress,
                progress_interval_events=progress_interval_events,
                progress_interval_seconds=progress_interval_seconds,
                collect_compute_telemetry=collect_compute_telemetry,
            )
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
        packet_plan = run_phase(
            "relation_packet_plan",
            lambda: RelationPacketPlan.from_trace(
                trace, query_lanes=self.config.relation_query_lanes,
            ),
        )
        self._register_query_history_plan(trace, packet_plan)
        self.issue_scheduler.reset()
        trace_kinds = set(
            np.unique(trace.events["primitive_kind"]).astype(int).tolist()
        )
        complete_query_lifecycle = all(
            int(kind) in trace_kinds
            for kind in (
                PrimitiveKind.RELATION,
                PrimitiveKind.QUERY_CLOSE,
                PrimitiveKind.FORWARD,
                PrimitiveKind.QUERY_REDUCTION,
            )
        )
        self.issue_scheduler.set_strict_lifecycle(complete_query_lifecycle)
        self.issue_scheduler.enable_exact_readiness(complete_query_lifecycle)
        adjoint_relation_ids = {
            int(row["relation_id"])
            for row in trace.events[
                trace.events["primitive_kind"] == int(PrimitiveKind.ADJOINT)
            ]
            if int(row["relation_id"]) >= 0
        }
        window_plan = run_phase(
            "relation_window_plan",
            lambda: RelationWindowPlan.from_trace(trace, packet_plan),
        )
        relation_windows = self._new_relation_window_tracker()
        query_replay = self._new_query_replay_tracker()
        owner_gradients = self._new_owner_gradient_tracker()
        compute_telemetry = (
            self._new_compute_telemetry() if collect_compute_telemetry else None
        )
        if query_replay is not None:
            try:
                query_replay.register_rows(trace.events)
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if owner_gradients is not None:
            try:
                owner_gradients.register_rows(trace.events)
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if window_plan is not None:
            if relation_windows is None:
                raise CycleConfigurationError(
                    "trace declares relation windows but hardware has no window resources"
                )
            for descriptor in window_plan.descriptors:
                relation_windows.register(descriptor)
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
        future_plan = (
            run_phase(
                "oracle_future_plan",
                lambda: self._future_plan(trace, dependency_index, packet_plan),
                total_iterations=int(iteration_ids.size),
            )
            if self.selection.query_oracle or self.selection.residency_oracle
            else None
        )
        oracle_unissued_cache_requests = (
            {
                key: set(event_ids)
                for key, event_ids in future_plan.cache_request_positions.items()
            }
            if self.selection.residency_oracle and future_plan is not None
            else {}
        )
        oracle_remaining_cache_uses = (
            {key: len(event_ids) for key, event_ids in oracle_unissued_cache_requests.items()}
            if self.selection.residency_oracle else {}
        )

        sample_metadata = trace.metadata.get("trace_sample")
        boundary_conditioned_iterations = bool(
            isinstance(sample_metadata, dict)
            and sample_metadata.get("boundary_condition")
            == "prior_selected_packet_completes_before_next_iteration"
        )
        boundary_iteration_ids = np.array([], dtype=np.uint32)
        boundary_iteration_remaining: dict[int, int] = {}
        boundary_iteration_position = 0
        boundary_deferred_ready: dict[int, list[tuple[int, int]]] = defaultdict(list)
        if boundary_conditioned_iterations:
            raw_ids, raw_counts = np.unique(
                trace.events["iteration_id"], return_counts=True,
            )
            boundary_iteration_ids = raw_ids.astype(np.uint32, copy=False)
            if boundary_iteration_ids.size < 2 or bool((
                np.diff(boundary_iteration_ids.astype(np.int64)) != 1
            ).any()):
                raise CycleConfigurationError(
                    "query packet boundary requires consecutive iterations"
                )
            boundary_iteration_remaining = {
                int(iteration_id): int(count)
                for iteration_id, count in zip(
                    boundary_iteration_ids, raw_counts, strict=True,
                )
            }

        def build_ready() -> list[tuple[int, int]]:
            roots = [
                (int(event_id), 0)
                for event_id in np.flatnonzero(dependency_index.remaining == 0)
            ]
            if not boundary_conditioned_iterations:
                return roots
            first_iteration = int(boundary_iteration_ids[0])
            initial: list[tuple[int, int]] = []
            for item in roots:
                iteration_id = int(trace.events[item[0]]["iteration_id"])
                if iteration_id == first_iteration:
                    initial.append(item)
                else:
                    boundary_deferred_ready[iteration_id].append(item)
            return initial

        initial_ready = run_phase(
            "ready_queue", build_ready, total_iterations=int(iteration_ids.size)
        )
        ready: dict[tuple[str, int], _ReadyCandidateQueue] = defaultdict(
            _ReadyCandidateQueue
        )
        fusion_pending: deque[tuple[int, TaskKind]] = deque()
        fusion_inputs: dict[TaskKind, deque[int]] = {
            TaskKind.FORWARD: deque(),
            TaskKind.CONSUMER: deque(),
            TaskKind.ADJOINT: deque(),
        }
        fusion_oracle_inputs: dict[
            TaskKind, list[tuple[tuple[int, int], int]]
        ] = {
            TaskKind.FORWARD: [],
            TaskKind.CONSUMER: [],
            TaskKind.ADJOINT: [],
        }
        packet_ready_members: dict[int, set[int]] = defaultdict(set)
        packet_ready_stages: set[tuple[int, int]] = set()
        consumer_credit_owner: dict[int, int] = {}
        consumer_reduction_remaining: dict[int, int] = {}
        consumer_last_reduction: dict[int, tuple[int, int, int]] = {}
        cycle = 0

        def observe_consumer_reduction(
            consumer_id: int, reduction_id: int, finish: int, owner: int,
        ) -> None:
            if consumer_id in consumer_credit_owner:
                return
            remaining = consumer_reduction_remaining.get(consumer_id)
            if remaining is None:
                reduction_dependencies = tuple(
                    int(dependency)
                    for dependency in trace.dependency_ids(trace.events[consumer_id])
                    if PrimitiveKind(
                        int(trace.events[int(dependency)]["primitive_kind"])
                    ) is PrimitiveKind.QUERY_REDUCTION
                )
                if not reduction_dependencies:
                    raise CycleConfigurationError(
                        f"consumer {consumer_id} has no query reduction dependency"
                    )
                remaining = len(reduction_dependencies)
            remaining -= 1
            if remaining < 0:
                raise CycleConfigurationError(
                    f"consumer {consumer_id} query dependency count underflow"
                )
            consumer_reduction_remaining[consumer_id] = remaining
            completed = (finish, reduction_id, owner)
            if completed > consumer_last_reduction.get(
                consumer_id, (-1, -1, -1),
            ):
                consumer_last_reduction[consumer_id] = completed
            if remaining != 0:
                return
            credit_owner = consumer_last_reduction[consumer_id][2]
            consumer_credit_owner[consumer_id] = credit_owner
            try:
                if not self.issue_scheduler.successor_credit(credit_owner):
                    raise CycleConfigurationError(
                        f"consumer {consumer_id} credit owner {credit_owner} is not live"
                    )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error

        def fusion_priority(event_id: int) -> tuple[int, int]:
            if self.selection.query_oracle:
                assert future_plan is not None
                return future_plan.query_priority(event_id)
            return event_id, event_id

        def fusion_kind(event_id: int, stage: int) -> TaskKind | None:
            if not self.selection.query_scheduler_enabled or stage != 0:
                return None
            kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
            return {
                PrimitiveKind.FORWARD: TaskKind.FORWARD,
                PrimitiveKind.CONSUMER: TaskKind.CONSUMER,
                PrimitiveKind.ADJOINT: TaskKind.ADJOINT,
            }.get(kind)

        def push_ready(event_id: int, stage: int) -> None:
            physical_stage = packet_plan.stage_for_event(event_id)
            kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
            if compute_telemetry is not None and stage == 0:
                compute_telemetry.mark_dependency_ready(event_id, kind, cycle)
            lane_granular = self._is_lane_granular_stage(kind, stage)
            if physical_stage is not None and not lane_granular:
                if stage == 0:
                    members = packet_ready_members[physical_stage.stage_id]
                    members.add(event_id)
                    if len(members) < len(physical_stage.event_ids):
                        return
                    if len(members) > len(physical_stage.event_ids):
                        raise CycleConfigurationError(
                            f"physical packet stage {physical_stage.stage_id} became ready twice"
                        )
                elif event_id != physical_stage.head_event_id:
                    raise CycleConfigurationError(
                        "only a physical packet head may advance module stages"
                    )
                ready_key = (physical_stage.stage_id, stage)
                if ready_key in packet_ready_stages:
                    return
                packet_ready_stages.add(ready_key)
                event_id = physical_stage.head_event_id
            task_kind = fusion_kind(event_id, stage)
            if task_kind is None:
                module_name = self._stages_for(kind)[stage]
                partition = self._module_partition(
                    module_name, trace.events[event_id]
                )
                ready[(module_name, partition)].push((event_id, stage))
            else:
                fusion_pending.append((event_id, task_kind))

        def requeue_candidate(event_id: int, stage: int) -> None:
            physical_stage = packet_plan.stage_for_event(event_id)
            kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
            if physical_stage is not None and not self._is_lane_granular_stage(kind, stage):
                event_id = physical_stage.head_event_id
            task_kind = fusion_kind(event_id, stage)
            if task_kind is None:
                module_name = self._stages_for(kind)[stage]
                partition = self._module_partition(
                    module_name, trace.events[event_id]
                )
                ready[(module_name, partition)].push((event_id, stage))
            elif self.selection.query_oracle:
                heapq.heappush(
                    fusion_oracle_inputs[task_kind],
                    (fusion_priority(event_id), event_id),
                )
            else:
                popped = fusion_pop_ranks.get(event_id)
                if popped is None:
                    fusion_inputs[task_kind].appendleft(event_id)
                else:
                    _kind, rank = popped
                    queue = fusion_inputs[task_kind]
                    position = next((
                        index for index, queued_id in enumerate(queue)
                        if (
                            (queued := fusion_pop_ranks.get(queued_id)) is None
                            or queued[1] > rank
                        )
                    ), len(queue))
                    queue.insert(position, event_id)

        for event_id, stage in initial_ready:
            push_ready(event_id, stage)
        remaining_events = len(trace.events)
        in_flight: list[tuple[int, int, int, str]] = []
        lane_outputs: list[tuple[int, int, int | None]] = []
        separate_lane_completion: set[int] = set()
        module_busy_until = {
            name: [0] * self._module_issue_ports(name)
            for name in self.modules
        }
        fusion_busy_until = {
            name: [0] * limit for name, limit in (
                ("forward", self.issue_scheduler.forward_ports),
                ("consumer", self.issue_scheduler.consumer_ports),
                ("adjoint", self.issue_scheduler.adjoint_ports),
            )
        }
        fusion_borrow_cursor = 0
        fusion_pop_ranks: dict[int, tuple[TaskKind, int]] = {}
        module_inflight = {
            (name, partition): 0
            for name in self.modules
            for partition in range(self._module_partition_count(name))
        }
        relation_seed_inflight = 0
        bank_busy: dict[tuple[object, ...], int] = {}
        shared_sram_address_busy: dict[tuple[int, int], set[str]] = {}
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
        # A multicast reserves one active read per real destination event.  The
        # follower events remain in the dependency graph, but do not perform a
        # second directory lookup or memory fill.
        cache_multicast_followers: dict[
            int, tuple[SemanticCacheState, tuple[int, int]]
        ] = {}
        cache_keys_by_version: dict[
            int, list[tuple[SemanticCacheState, tuple[int, int]]]
        ] = {}
        semantic_worksets = (
            SemanticWorksets.from_trace(trace, packet_plan)
            if self.selection.semantic_worksets else None
        )
        closed_versions: set[int] = set()
        memory_requests = 0
        next_progress_event = progress_interval_events
        next_progress_time = (
            started_at + progress_interval_seconds
            if progress_interval_seconds is not None else None
        )
        last_progress_completed = -1

        def complete_logical_event(event_id: int, finish: int) -> None:
            nonlocal remaining_events, completed_iterations
            nonlocal contiguous_iteration_position, contiguous_iteration_events
            nonlocal last_completed_iteration, last_completed_iteration_events
            nonlocal last_completed_iteration_cycles
            nonlocal last_completed_iteration_elapsed_seconds
            nonlocal boundary_iteration_position
            if event_id in completed:
                raise CycleConfigurationError(f"event {event_id} completed twice")
            completed[event_id] = finish
            row = trace.events[event_id]
            kind = PrimitiveKind(int(row["primitive_kind"]))
            self._update_query_state_for_completion(
                row, kind, adjoint_relation_ids,
            )
            if compute_telemetry is not None:
                compute_telemetry.mark_finish(event_id, finish)
            if relation_windows is not None and event_id in relation_windows.event_to_window:
                try:
                    relation_windows.complete_event(event_id)
                except ValueError as error:
                    raise CycleConfigurationError(str(error)) from error
            remaining_events -= 1
            if progress is not None:
                iteration_id = int(trace.events[event_id]["iteration_id"])
                iteration_position = int(iteration_positions[iteration_id])
                iteration_completed[iteration_position] += 1
                if iteration_completed[iteration_position] == iteration_totals[iteration_position]:
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
                dependent_kind = PrimitiveKind(
                    int(trace.events[dependent]["primitive_kind"])
                )
                if (
                    kind is PrimitiveKind.QUERY_REDUCTION
                    and dependent_kind is PrimitiveKind.CONSUMER
                    and self.issue_scheduler.strict_lifecycle
                ):
                    owner = int(row["query_id"])
                    if owner < 0:
                        raise CycleConfigurationError(
                            f"consumer {dependent} has an invalid credit owner"
                        )
                    observe_consumer_reduction(
                        dependent, event_id, finish, owner,
                    )
                if remaining == 1:
                    push_ready(dependent, 0)
            if boundary_conditioned_iterations:
                iteration_id = int(row["iteration_id"])
                boundary_remaining = boundary_iteration_remaining[iteration_id] - 1
                boundary_iteration_remaining[iteration_id] = boundary_remaining
                if boundary_remaining == 0:
                    expected = int(
                        boundary_iteration_ids[boundary_iteration_position]
                    )
                    if iteration_id != expected:
                        raise CycleConfigurationError(
                            "query packet iterations completed out of order"
                        )
                    boundary_iteration_position += 1
                    if boundary_iteration_position < boundary_iteration_ids.size:
                        next_iteration = int(
                            boundary_iteration_ids[boundary_iteration_position]
                        )
                        for ready_event, ready_stage in boundary_deferred_ready.pop(
                            next_iteration, ()
                        ):
                            push_ready(ready_event, ready_stage)

        while remaining_events or in_flight or lane_outputs:
            # Bank reservations are scoped to one cycle.  Drop old entries so
            # long traces do not retain one dictionary item per cycle.
            bank_busy = {
                key: value for key, value in bank_busy.items()
                if key[1] >= cycle
            }
            shared_sram_address_busy = {
                key: value for key, value in shared_sram_address_busy.items()
                if key[0] >= cycle
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
                row = trace.events[event_id]
                inflight_key = (
                    module_name, self._module_partition(module_name, row)
                )
                module_inflight[inflight_key] -= 1
                if module_inflight[inflight_key] < 0:
                    raise CycleConfigurationError(
                        f"negative in-flight count for {module_name}"
                    )
                stages = self._stages_for(PrimitiveKind(int(trace.events[event_id]["primitive_kind"])))
                kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                physical_stage = packet_plan.stage_for_event(event_id)
                if (
                    query_replay is not None
                    and module_name == "bidirectional_query"
                    and kind is PrimitiveKind.ADJOINT
                ):
                    logical_adjoint_events = (
                        physical_stage.event_ids
                        if physical_stage is not None else (event_id,)
                    )
                    try:
                        query_replay.dispatch_adjoint(logical_adjoint_events)
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                if (
                    owner_gradients is not None
                    and module_name == "compute_pod"
                    and kind is PrimitiveKind.GRADIENT_REDUCTION
                ):
                    try:
                        owner_gradients.complete_gradient(
                            physical_stage.event_ids
                            if physical_stage is not None else (event_id,)
                        )
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                if kind is PrimitiveKind.CACHE_REQUEST and stage == 0 and event_id in cache_event_state:
                    state, key, lookup = cache_event_state[event_id]
                    if lookup is CacheLookup.MISS:
                        workset = (
                            semantic_worksets.for_event(event_id)
                            if semantic_worksets is not None else None
                        )
                        pending = state.pending.get(key)
                        if pending is None:
                            raise CycleConfigurationError(
                                f"cache fill for {event_id} has no pending state"
                            )
                        waiter_count = int(pending["waiters"])
                        state.fill_complete(
                            key,
                            remaining_uses=int(pending["remaining_uses"]),
                        )
                        max_destinations = self.config.cache_multicast_destinations
                        if max_destinations is not None and max_destinations > 1:
                            remaining_waiters = waiter_count
                            while remaining_waiters > 1:
                                destinations = min(
                                    remaining_waiters, max_destinations
                                )
                                state.begin_multicast(
                                    key,
                                    destinations=destinations,
                                    readers_already_active=True,
                                )
                                remaining_waiters -= destinations
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
                    if self.selection.residency_oracle:
                        remaining_uses = oracle_remaining_cache_uses[key] - 1
                        if remaining_uses < 0:
                            raise CycleConfigurationError(
                                f"negative Oracle remaining-use count for {key}"
                            )
                        oracle_remaining_cache_uses[key] = remaining_uses
                        if remaining_uses == 0:
                            state.close(key)
                    elif semantic_worksets is not None:
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
                skip_shared_sram = (
                    kind is PrimitiveKind.CACHE_REQUEST
                    and stage == 0
                    and event_id in cache_event_state
                    and cache_event_state[event_id][2] is not CacheLookup.MISS
                )
                if skip_shared_sram:
                    # A hit or merged request is already backed by the active
                    # record (or a leader's pending fill); it must not issue a
                    # second Active SRAM write.  The semantic-cache stage is a
                    # physical packet stage, so retire every logical lane;
                    # only the packet head owns the directory/fill state.
                    logical_events = (
                        physical_stage.event_ids
                        if physical_stage is not None else (event_id,)
                    )
                    for logical_event in logical_events:
                        complete_logical_event(logical_event, finish)
                elif stage + 1 < len(stages):
                    if event_id in separate_lane_completion:
                        separate_lane_completion.remove(event_id)
                    else:
                        push_ready(event_id, stage + 1)
                else:
                    if (
                        event_id in separate_lane_completion
                        and module_name == "compute_pod"
                        and kind is PrimitiveKind.FORWARD
                    ):
                        separate_lane_completion.remove(event_id)
                    else:
                        logical_events = (
                            physical_stage.event_ids
                            if physical_stage is not None
                            and not self._is_lane_granular_stage(kind, stage)
                            else (event_id,)
                        )
                        for logical_event in logical_events:
                            complete_logical_event(logical_event, finish)
                if (PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                        is PrimitiveKind.RELATION_CANDIDATE):
                    relation_seed_inflight -= 1
                    if relation_seed_inflight < 0:
                        raise CycleConfigurationError("negative relation seed FIFO occupancy")
                progressed = True
            while lane_outputs and lane_outputs[0][0] <= cycle:
                finish, event_id, next_stage = heapq.heappop(lane_outputs)
                if next_stage is None:
                    complete_logical_event(event_id, finish)
                else:
                    push_ready(event_id, next_stage)
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
            if self.selection.query_scheduler_enabled:
                self.issue_scheduler.set_clock(cycle)
                fusion_capacity = (
                    self.config.candidate_fifo_entries
                    or self.modules["fusion_issue"].timing.queue_capacity
                )
                blocked_pending: list[tuple[int, TaskKind]] = []
                pending_count = len(fusion_pending)
                for _ in range(pending_count):
                    event_id, task_kind = fusion_pending.popleft()
                    if self.selection.query_oracle:
                        if len(fusion_oracle_inputs[task_kind]) >= fusion_capacity:
                            blocked_pending.append((event_id, task_kind))
                            continue
                        heapq.heappush(
                            fusion_oracle_inputs[task_kind],
                            (fusion_priority(event_id), event_id),
                        )
                    else:
                        if len(fusion_inputs[task_kind]) >= fusion_capacity:
                            blocked_pending.append((event_id, task_kind))
                            continue
                        fusion_inputs[task_kind].append(event_id)
                        self.issue_scheduler.observe_arrival((
                            self._task_packet(
                                trace, event_id,
                                packet_plan.stage_for_event(event_id),
                                consumer_owner_ids=(
                                    (consumer_credit_owner[event_id],)
                                    if task_kind is TaskKind.CONSUMER
                                    and event_id in consumer_credit_owner else ()
                                ),
                            ),
                        ), arrival_cycle=cycle)
                fusion_pending.extend(blocked_pending)
                if fusion_pending:
                    self.modules["fusion_issue"].counters.queue_stalls += 1
                    self._record_stall(
                        cycle, "fusion_issue", "input_queue_capacity",
                        fusion_pending[0][0],
                    )
            candidates: list[tuple[int, int]] = []
            fusion_pop_ranks = {}
            for ready_key in tuple(ready):
                queue = ready[ready_key]
                module_name, _partition = ready_key
                candidates.extend(self._pop_ready_candidates(
                    queue, module_name,
                    row_for=lambda event_id: trace.events[event_id],
                    physical_stage_for=packet_plan.stage_for_event,
                    owner_gradients=owner_gradients,
                    query_replay=query_replay,
                ))
                if not queue:
                    del ready[ready_key]
            if self.selection.query_scheduler_enabled:
                if self.selection.query_oracle:
                    assert future_plan is not None
                    selected_ids = set(self._select_query_oracle_candidates(
                        trace,
                        (
                            entry[1]
                            for queue in fusion_oracle_inputs.values()
                            for entry in queue
                        ),
                        future_plan,
                        packet_plan,
                        consumer_credit_owner,
                    ))
                    for queue in fusion_oracle_inputs.values():
                        retained: list[tuple[tuple[int, int], int]] = []
                        while queue:
                            entry = heapq.heappop(queue)
                            if entry[1] in selected_ids:
                                candidates.append((entry[1], 0))
                            else:
                                retained.append(entry)
                        for entry in retained:
                            heapq.heappush(queue, entry)
                else:
                    (
                        fusion_candidate_ids,
                        fusion_pop_ranks,
                        fusion_borrow_cursor,
                    ) = self._pop_fusion_input_candidates(
                        fusion_inputs, borrow_cursor=fusion_borrow_cursor,
                    )
                    candidates.extend(
                        (event_id, 0) for event_id in fusion_candidate_ids
                    )
            fusion_issued = 0
            fusion_port_issued: dict[str, int] = {}
            issued_modules: dict[tuple[str, int], int] = {}
            ordered_ids = self._ordered_candidates(
                trace,
                [item[0] for item in candidates],
                future_plan=future_plan,
            )
            ordered = [(event_id, dict(candidates)[event_id]) for event_id in ordered_ids]
            fusion_packets: dict[int, TaskPacket] = {}
            fusion_selected: set[int] = set()
            if self.selection.query_scheduler_enabled:
                for event_id, stage in ordered:
                    kind = PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                    if stage == 0 and kind in {
                        PrimitiveKind.FORWARD,
                        PrimitiveKind.CONSUMER,
                        PrimitiveKind.ADJOINT,
                    }:
                        fusion_packets[event_id] = self._task_packet(
                            trace, event_id,
                            packet_plan.stage_for_event(event_id),
                            consumer_owner_ids=(
                                (consumer_credit_owner[event_id],)
                                if kind is PrimitiveKind.CONSUMER
                                and event_id in consumer_credit_owner else ()
                            ),
                            workset_hit=self._workset_hit(
                                cache_states, trace.events[event_id],
                            ),
                        )
                decision = (
                    self.issue_scheduler.select_in_order(fusion_packets.values())
                    if self.selection.query_oracle
                    else self.issue_scheduler.select(
                        fusion_packets.values(),
                        use_load_rules=self.selection.query_load_rules,
                    )
                )
                fusion_selected = {task.event_id for task in decision.accepted}
                fusion_order = iter(
                    task.event_id for task in (*decision.accepted, *decision.rejected)
                )
                ordered = [
                    (next(fusion_order), stage)
                    if event_id in fusion_packets else (event_id, stage)
                    for event_id, stage in ordered
                ]
            for ordered_position, (event_id, stage) in enumerate(ordered):
                row = trace.events[event_id]
                kind = PrimitiveKind(int(row["primitive_kind"]))
                physical_stage = packet_plan.stage_for_event(event_id)
                stages = self._stages_for(kind)
                module_name = stages[stage]
                module = self.modules[module_name]
                timing = module.timing
                multicast_follower = (
                    cache_multicast_followers.get(event_id)
                    if kind is PrimitiveKind.CACHE_REQUEST and stage == 0
                    else None
                )
                if multicast_follower is not None:
                    state, key = multicast_follower
                    cache_multicast_followers.pop(event_id, None)
                    cache_event_state[event_id] = (state, key, CacheLookup.HIT)
                    cache_keys_by_version.setdefault(key[1], []).append((state, key))
                    completion = cycle + self._physical_service_cycles(
                        module_name, row, kind, physical_stage,
                    )
                    heapq.heappush(
                        in_flight, (completion, event_id, stage, module_name)
                    )
                    inflight_key = (
                        module_name, self._module_partition(module_name, row)
                    )
                    module_inflight[inflight_key] += 1
                    module.counters.accepted += 1
                    progressed = True
                    continue
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
                module_partition = self._module_partition(module_name, row)
                module_issue_key = (module_name, module_partition)
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
                elif (
                    issued_modules.get(module_issue_key, 0)
                    >= self._module_partition_issue_limit(
                        module_name, module_partition,
                    )
                ):
                    module.counters.port_stalls += 1
                    self._record_stall(cycle, module_name, "port", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                module_lane: int | None = None
                fusion_port_lane: int | None = None
                query_allocation: tuple[int, ...] = ()
                if fusion_port_name is not None:
                    fusion_port_lane = next((
                        index for index, point in enumerate(
                            fusion_busy_until[fusion_port_name]
                        )
                        if point <= cycle
                    ), None)
                    busy_until = (
                        cycle if fusion_port_lane is not None
                        else min(fusion_busy_until[fusion_port_name])
                    )
                elif module_name == "bidirectional_query" and self._has_query_resources():
                    module_lanes = module_busy_until[module_name]
                    allocation = self._query_resource_allocation(
                        module_lanes, row, kind, physical_stage, cycle,
                    )
                    if allocation is None:
                        reason = (
                            "reduction_bank"
                            if kind in {
                                PrimitiveKind.FORWARD,
                                PrimitiveKind.QUERY_REDUCTION,
                            }
                            else "query_datapath"
                        )
                        if reason == "reduction_bank":
                            module.counters.bank_conflicts += 1
                        else:
                            module.counters.port_stalls += 1
                        self._record_stall(cycle, module_name, reason, event_id)
                        requeue_candidate(event_id, stage)
                        continue
                    query_allocation = allocation
                    busy_until = cycle
                else:
                    module_lanes = module_busy_until[module_name]
                    module_lane = next(
                        (index for index in self._module_lane_indices(module_name, row)
                         if module_lanes[index] <= cycle),
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
                if (self._uses_generic_module_limits(module_name)
                        and module_inflight[module_issue_key] >= timing.queue_capacity):
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "queue_capacity", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                if (
                    owner_gradients is not None
                    and kind is PrimitiveKind.ADJOINT
                    and module_name == "compute_pod"
                ):
                    owner_event_ids = (
                        physical_stage.event_ids
                        if physical_stage is not None else (event_id,)
                    )
                    if owner_gradients.blocks_adjoint(owner_event_ids):
                        module.counters.queue_stalls += 1
                        self._record_stall(
                            cycle, module_name,
                            "owner_gradient_slot_capacity", event_id,
                        )
                        requeue_candidate(event_id, stage)
                        continue
                if (
                    query_replay is not None
                    and module_name == "bidirectional_query"
                    and kind is PrimitiveKind.CONSUMER
                    and query_replay.blocks_consumer(event_id)
                ):
                    module.counters.queue_stalls += 1
                    self._record_stall(
                        cycle, module_name, "replay_queue_capacity", event_id
                    )
                    requeue_candidate(event_id, stage)
                    continue
                if (
                    query_replay is not None
                    and module_name == "bidirectional_query"
                    and kind is PrimitiveKind.ADJOINT
                ):
                    replay_event_ids = (
                        physical_stage.event_ids
                        if physical_stage is not None else (event_id,)
                    )
                    try:
                        replay_blocked = query_replay.blocks_adjoint(
                            replay_event_ids
                        )
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                    if replay_blocked:
                        module.counters.queue_stalls += 1
                        self._record_stall(
                            cycle, module_name, "replay_queue_capacity", event_id
                        )
                        requeue_candidate(event_id, stage)
                        continue
                if relation_windows is not None and stage == 0:
                    physical_head = (
                        physical_stage is None
                        or physical_stage.head_event_id == event_id
                    )
                    window_reason = relation_windows.blocking_reason(
                        event_id, kind,
                        physical_stage_head=physical_head,
                        cycle=cycle,
                    )
                    if window_reason is not None:
                        query_module = self.modules["bidirectional_query"]
                        query_module.counters.queue_stalls += 1
                        self._record_stall(
                            cycle, "bidirectional_query", window_reason, event_id
                        )
                        requeue_candidate(event_id, stage)
                        continue
                    if (
                        not self.selection.query_load_rules
                        and not self.selection.query_oracle
                    ):
                        try:
                            stage_reason = relation_windows.full_stage_blocking_reason(
                                event_id, kind,
                            )
                        except ValueError as error:
                            raise CycleConfigurationError(str(error)) from error
                        if stage_reason is not None:
                            query_module = self.modules["bidirectional_query"]
                            query_module.counters.queue_stalls += 1
                            self._record_stall(
                                cycle, "bidirectional_query", stage_reason, event_id,
                            )
                            requeue_candidate(event_id, stage)
                            continue
                if (kind is PrimitiveKind.RELATION_CANDIDATE
                        and relation_seed_inflight >= self.config.relation_seed_fifo_entries):
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "seed_fifo", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                bank_keys = self._module_bank_reservation_keys(
                    module_name, row, cycle,
                )
                shared_sram_direction: str | None = None
                shared_sram_address_key: tuple[int, int] | None = None
                if module_name == "shared_sram":
                    shared_sram_direction = self._shared_sram_direction(kind)
                    shared_sram_address_key = self._shared_sram_address_key(
                        row, cycle,
                    )
                    existing_directions = shared_sram_address_busy.get(
                        shared_sram_address_key, set(),
                    )
                    if any(
                        direction != shared_sram_direction
                        for direction in existing_directions
                    ):
                        module.counters.bank_conflicts += 1
                        self._record_stall(
                            cycle, module_name, "same_address_replay", event_id,
                        )
                        requeue_candidate(event_id, stage)
                        continue
                if (
                    self._uses_generic_module_bank_limits(module_name)
                    and any(key in bank_busy for key in bank_keys)
                ):
                    module.counters.bank_conflicts += 1
                    self._record_stall(cycle, module_name, "bank", event_id)
                    requeue_candidate(event_id, stage)
                    continue
                compute_plan: tuple[tuple[str, int, int], ...] = ()
                if module_name == "compute_pod":
                    try:
                        compute_pod, compute_cluster = self._compute_route(row, kind)
                        compute_plan = module.reservation_plan(  # type: ignore[attr-defined]
                            int(row["template_id"]), kind, cycle,
                            pod=compute_pod, cluster_hint=compute_cluster,
                        )
                    except KeyError as error:
                        raise CycleConfigurationError(str(error)) from error
                    if not module.can_reserve(compute_plan, cycle):  # type: ignore[attr-defined]
                        module.counters.queue_stalls += 1
                        assert isinstance(module, ComputePod)
                        self._record_compute_resource_stall(
                            cycle=cycle, event_id=event_id, pod=compute_pod,
                            compute=module, plan=compute_plan,
                        )
                        requeue_candidate(event_id, stage)
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
                        multicast_followers: tuple[int, ...] = ()
                        if (
                            self.selection.residency_oracle
                            and key not in state.active
                            and key not in state.pending
                            and len(state.active) + len(state.pending) >= state.capacity
                        ):
                            assert future_plan is not None
                            evictable = [
                                candidate_key
                                for candidate_key, record in state.active.items()
                                if int(record["active_reads"]) == 0
                            ]
                            if evictable:
                                def next_unissued(candidate_key: tuple[int, int]) -> int | None:
                                    requests = oracle_unissued_cache_requests.get(
                                        candidate_key, set()
                                    )
                                    return min(requests) if requests else None

                                victim = max(
                                    evictable,
                                    key=lambda candidate_key: (
                                        next_unissued(candidate_key) is None,
                                        next_unissued(candidate_key) or 0,
                                        candidate_key,
                                    ),
                                )
                                state.evict_for_oracle(victim)
                        if self.selection.residency_oracle:
                            oracle_requests = oracle_unissued_cache_requests.get(key)
                            if oracle_requests is None or event_id not in oracle_requests:
                                raise CycleConfigurationError(
                                    f"Oracle cache request {event_id} was issued more than once"
                                )
                            oracle_requests.remove(event_id)
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
                                resident_remaining_uses=(
                                    oracle_remaining_cache_uses[key]
                                    if self.selection.residency_oracle else None
                                ),
                            )
                        except CacheBackpressure:
                            if self.selection.residency_oracle:
                                oracle_unissued_cache_requests[key].add(event_id)
                            module.counters.queue_stalls += 1
                            self._record_stall(cycle, module_name, "cache_capacity", event_id)
                            requeue_candidate(event_id, stage)
                            continue
                        # Scope multicast only over actual ready queue heads in
                        # this scheduling pass.  The configured width includes
                        # the leader, so every added destination is backed by a
                        # real cache-request event and keeps its dependencies.
                        max_destinations = self.config.cache_multicast_destinations
                        if (
                            lookup is CacheLookup.HIT
                            and max_destinations is not None
                            and max_destinations > 1
                            and key in state.active
                        ):
                            candidates_for_multicast: list[int] = []
                            for candidate_id, candidate_stage in ordered[ordered_position + 1:]:
                                if candidate_stage != 0:
                                    continue
                                candidate_row = trace.events[candidate_id]
                                if PrimitiveKind(int(candidate_row["primitive_kind"])) is not PrimitiveKind.CACHE_REQUEST:
                                    continue
                                candidate_key = (
                                    int(candidate_row["gaussian_id"]),
                                    int(candidate_row["state_version"]),
                                )
                                if candidate_key != key:
                                    continue
                                candidates_for_multicast.append(candidate_id)
                                if len(candidates_for_multicast) + 1 >= max_destinations:
                                    break
                            multicast_followers = tuple(candidates_for_multicast)
                        cache_event_state[event_id] = (state, key, lookup)
                        state_version = int(row["state_version"])
                        if state_version in closed_versions:
                            raise CycleConfigurationError(
                                f"cache request {event_id} targets a closed state version"
                            )
                        cache_keys_by_version.setdefault(state_version, []).append((state, key))
                        if lookup is CacheLookup.HIT:
                            completion = cycle + self._service_cycles(module_name, row, kind)
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
                                    cycle + self._physical_service_cycles(
                                        module_name, row, kind, physical_stage,
                                    ),
                                ))
                                completion = None
                            else:
                                if memory_done > cycle + timing.latency:
                                    module.counters.memory_wait_cycles += memory_done - cycle - timing.latency
                                completion = max(
                                    cycle + self._physical_service_cycles(
                                        module_name, row, kind, physical_stage,
                                    ), memory_done,
                                )
                        if multicast_followers:
                            if completion is None:
                                raise CycleConfigurationError(
                                    "active cache hit cannot await a memory fill"
                                )
                            state.begin_multicast(
                                key, destinations=len(multicast_followers)
                            )
                            for follower_id in multicast_followers:
                                cache_multicast_followers[follower_id] = (state, key)
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
                                cycle + self._physical_service_cycles(
                                    module_name, row, kind, physical_stage,
                                ),
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
                            completion = max(
                                cycle + self._physical_service_cycles(
                                    module_name, row, kind, physical_stage,
                                ), memory_done,
                            )
                else:
                    completion = cycle + self._physical_service_cycles(
                        module_name, row, kind, physical_stage,
                    )
                if (
                    physical_stage is not None
                    and module_name == "compute_pod"
                    and kind is PrimitiveKind.FORWARD
                    and self.config.compute_templates is not None
                ):
                    compute = self.modules[module_name]
                    assert isinstance(compute, ComputePod)
                    path = compute.path_for(int(row["template_id"]), kind)
                    for logical_event, lane in zip(
                        physical_stage.event_ids, physical_stage.lanes, strict=True,
                    ):
                        heapq.heappush(
                            lane_outputs,
                            (
                                cycle + path.packet_completion_offset(lane),
                                logical_event,
                                stage + 1 if stage + 1 < len(stages) else None,
                            ),
                        )
                    separate_lane_completion.add(event_id)
                if completion is not None:
                    heapq.heappush(in_flight, (completion, event_id, stage, module_name))
                if fusion_port_name is not None:
                    assert fusion_port_lane is not None
                    fusion_busy_until[fusion_port_name][fusion_port_lane] = (
                        cycle + timing.initiation_interval
                    )
                elif query_allocation:
                    for lane in query_allocation:
                        module_busy_until[module_name][lane] = (
                            cycle + timing.initiation_interval
                        )
                else:
                    assert module_lane is not None
                    module_busy_until[module_name][module_lane] = (
                        cycle + timing.initiation_interval
                    )
                if self._uses_generic_module_bank_limits(module_name):
                    for bank_key in bank_keys:
                        bank_busy[bank_key] = cycle
                if (
                    shared_sram_address_key is not None
                    and shared_sram_direction is not None
                ):
                    shared_sram_address_busy.setdefault(
                        shared_sram_address_key, set(),
                    ).add(shared_sram_direction)
                module_inflight[module_issue_key] += 1
                if kind is PrimitiveKind.RELATION_CANDIDATE:
                    relation_seed_inflight += 1
                if relation_windows is not None and stage == 0:
                    try:
                        relation_windows.issue(
                            event_id, kind,
                            physical_stage_head=(
                                physical_stage is None
                                or physical_stage.head_event_id == event_id
                            ),
                            cycle=cycle,
                        )
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                if (
                    query_replay is not None
                    and module_name == "bidirectional_query"
                    and kind is PrimitiveKind.CONSUMER
                ):
                    try:
                        query_replay.reserve_consumer(event_id)
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                if (
                    query_replay is not None
                    and module_name == "bidirectional_query"
                    and kind is PrimitiveKind.ADJOINT
                ):
                    try:
                        query_replay.reserve_adjoint(
                            physical_stage.event_ids
                            if physical_stage is not None else (event_id,)
                        )
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                if (
                    owner_gradients is not None
                    and kind is PrimitiveKind.ADJOINT
                    and module_name == "compute_pod"
                ):
                    try:
                        owner_gradients.reserve_adjoint(
                            physical_stage.event_ids
                            if physical_stage is not None else (event_id,)
                        )
                    except ValueError as error:
                        raise CycleConfigurationError(str(error)) from error
                module.counters.accepted += 1
                module.counters.busy_cycles += self._physical_service_cycles(
                    module_name, row, kind, physical_stage,
                )
                if compute_plan:
                    module.reserve(compute_plan)  # type: ignore[attr-defined]
                if compute_telemetry is not None:
                    compute_telemetry.mark_issue(
                        self._stage_event_ids(
                            event_id, kind, stage, physical_stage,
                        ),
                        kind, module_name, cycle, compute_plan=compute_plan,
                    )
                issued_modules[module_issue_key] = (
                    issued_modules.get(module_issue_key, 0) + 1
                )
                if module_name == "fusion_issue" and stage == 0:
                    fusion_issued += 1
                    if kind is PrimitiveKind.CONSUMER and self.issue_scheduler.strict_lifecycle:
                        try:
                            owner = consumer_credit_owner.get(event_id)
                            if owner is None or not self.issue_scheduler.successor_dispatch(owner):
                                raise CycleConfigurationError(
                                    f"consumer {event_id} has no committed credit owner"
                                )
                        except ValueError as error:
                            raise CycleConfigurationError(str(error)) from error
                    if self.selection.query_scheduler_enabled:
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
                    min((point for lanes in fusion_busy_until.values()
                         for point in lanes if point > cycle), default=None),
                                                   lane_outputs[0][0] if lane_outputs else None,
                                                   memory_wakeup)
                               if point is not None and point > cycle]
                if not next_points:
                    if ready and (
                        any(key[1] == cycle for key in bank_busy)
                        or relation_windows is not None
                        and relation_windows.has_append_bank_reservation(cycle)
                    ):
                        cycle += 1
                        continue
                    blocked_rows: list[str] = []
                    for queue in ready.values():
                        for event_id, stage in queue[:8]:
                            row = trace.events[event_id]
                            kind = PrimitiveKind(int(row["primitive_kind"]))
                            physical_stage = packet_plan.stage_for_event(event_id)
                            detail = f"{event_id}:{kind.name}:stage={stage}"
                            if relation_windows is not None and stage == 0:
                                try:
                                    window_reason = relation_windows.blocking_reason(
                                        event_id, kind,
                                        physical_stage_head=(
                                            physical_stage is None
                                            or physical_stage.head_event_id == event_id
                                        ),
                                        cycle=cycle,
                                    )
                                except ValueError as error:
                                    window_reason = f"error:{error}"
                                detail += f":window={window_reason}"
                            if (
                                kind in {
                                    PrimitiveKind.FORWARD,
                                    PrimitiveKind.ADJOINT,
                                    PrimitiveKind.GRADIENT_REDUCTION,
                                }
                                and self.config.compute_templates is not None
                            ):
                                compute = self.modules["compute_pod"]
                                assert isinstance(compute, ComputePod)
                                pod, cluster = self._compute_route(row, kind)
                                plan = compute.reservation_plan(
                                    int(row["template_id"]), kind, cycle,
                                    pod=pod, cluster_hint=cluster,
                                )
                                blocker = compute.first_blocking_resource(plan, cycle)
                                detail += f":compute={blocker}"
                            blocked_rows.append(detail)
                            if len(blocked_rows) >= self.config.candidate_lanes:
                                break
                        if len(blocked_rows) >= self.config.candidate_lanes:
                            break
                    window_state = (
                        relation_windows.snapshot()
                        if relation_windows is not None else {}
                    )
                    ready_state: dict[str, object] = {
                        f"{module}:{partition}": len(queue)
                        for (module, partition), queue in ready.items()
                    }
                    ready_state["fusion_pending"] = len(fusion_pending)
                    ready_state["fusion_inputs"] = {
                        task_kind.value: len(queue)
                        for task_kind, queue in fusion_inputs.items()
                    }
                    ready_state["fusion_oracle_inputs"] = {
                        task_kind.name.lower(): len(queue)
                        for task_kind, queue in fusion_oracle_inputs.items()
                    }
                    owner_gradient_state: dict[str, object] = {}
                    if owner_gradients is not None:
                        pending_adjoint_ids: list[int] = []
                        dispatchable_adjoint_ids: list[int] = []
                        for (module, _partition), queue in ready.items():
                            if module != "compute_pod":
                                continue
                            for event_id, stage in sorted(queue):
                                row = trace.events[event_id]
                                if (
                                    PrimitiveKind(int(row["primitive_kind"]))
                                    is not PrimitiveKind.ADJOINT
                                ):
                                    continue
                                physical_stage = packet_plan.stage_for_event(event_id)
                                event_ids = (
                                    physical_stage.event_ids
                                    if physical_stage is not None else (event_id,)
                                )
                                pending_adjoint_ids.append(event_id)
                                if not owner_gradients.blocks_adjoint(event_ids):
                                    dispatchable_adjoint_ids.append(event_id)
                        owner_gradient_state = owner_gradients.deadlock_snapshot(
                            pending_adjoint_event_ids=tuple(pending_adjoint_ids),
                            remaining_dependencies=dependency_index.remaining,
                        )
                        owner_gradient_state["dispatchable_adjoint_count"] = len(
                            dispatchable_adjoint_ids
                        )
                        owner_gradient_state["dispatchable_adjoint_sample"] = tuple(
                            dispatchable_adjoint_ids[:16]
                        )
                    remaining_sample = []
                    for raw_event_id in np.flatnonzero(dependency_index.remaining):
                        event_id = int(raw_event_id)
                        remaining_sample.append((
                            event_id,
                            PrimitiveKind(int(trace.events[event_id]["primitive_kind"])).name,
                            int(dependency_index.remaining[event_id]),
                        ))
                        if len(remaining_sample) >= 16:
                            break
                    raise CycleConfigurationError(
                        f"deadlock at cycle {cycle}, pending={blocked_rows}, "
                        f"remaining_events={remaining_events}, completed_events={len(completed)}, "
                        f"remaining_sample={remaining_sample}, "
                        f"ready_state={ready_state}, relation_state={window_state}, "
                        f"window_references={relation_windows.live_reference_snapshot() if relation_windows is not None else {}}, "
                        f"owner_gradient_state={owner_gradient_state}"
                    )
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
        if relation_windows is not None:
            if relation_windows.live or relation_windows.relation_records_live:
                raise CycleConfigurationError(
                    "cycle replay ended with live relation-window state"
                )
            module_counters["bidirectional_query"].update(
                relation_windows.snapshot()
            )
        if query_replay is not None:
            if query_replay.active_queries:
                raise CycleConfigurationError(
                    "cycle replay ended with live adjoint replay entries"
                )
            module_counters["bidirectional_query"].update(
                query_replay.snapshot()
            )
        if owner_gradients is not None:
            if owner_gradients.active_by_cluster:
                raise CycleConfigurationError(
                    "cycle replay ended with live owner-gradient epoch slots"
                )
            module_counters["compute_pod"].update(owner_gradients.snapshot())
        if self.issue_scheduler.enforce_exact_readiness:
            try:
                self.issue_scheduler.require_drained()
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        self.issue_scheduler.release_completed()
        f_count, c_count, a_count = self.issue_scheduler.live_counter_totals()
        module_counters["fusion_issue"].update({
            **self.issue_scheduler.state_table_snapshot(),
            **self.issue_scheduler.history_snapshot(),
            "live_forward_count": f_count,
            "live_consumer_count": c_count,
            "live_adjoint_count": a_count,
        })
        audit_records = getattr(self.config.memory, "audit_records", None)
        memory_request_records = tuple(audit_records()) if callable(audit_records) else ()
        total_cycles = max(completed.values(), default=0)
        return CycleResult(
            total_cycles=total_cycles,
            module_counters=module_counters,
            stalls=self._stall_records(),
            completion_cycles=completed,
            event_counts={kind.name: int((trace.events["primitive_kind"] == int(kind)).sum())
                          for kind in PrimitiveKind},
            policy=self.policy,
            oracle_status=(
                "future_visible_resource_constrained"
                if self.policy.endswith("_oracle") else "not_applicable"
            ),
            memory_requests=memory_request_records,
            compute_telemetry=(
                compute_telemetry.finish(total_cycles)
                if compute_telemetry is not None else None
            ),
        )

    def _run_oracle_portfolio(
        self,
        trace: Trace,
        *,
        validate_input: bool,
        progress: Callable[[CycleProgress], None] | None,
        progress_interval_events: int | None,
        progress_interval_seconds: float | None,
        collect_compute_telemetry: bool,
    ) -> CycleResult:
        oracle_policy = self.policy
        actual_policy = "query" if self.selection.query_oracle else "residency"
        clone = getattr(self.config.memory, "clone", None)
        if callable(getattr(self.config.memory, "submit_async", None)) and not callable(clone):
            raise CycleConfigurationError(
                "Oracle portfolio requires a cloneable asynchronous memory backend"
            )

        def member_config() -> CycleConfig:
            return replace(
                self.config,
                memory=clone() if callable(clone) else self.config.memory,
            )

        last_progress: dict[str, CycleProgress] = {}

        def member_progress(member: str) -> Callable[[CycleProgress], None] | None:
            if progress is None:
                return None

            def report(item: CycleProgress) -> None:
                last_progress[member] = item
                progress(replace(
                    item, phase=f"oracle_portfolio_{member}_{item.phase}"
                ))

            return report

        base: CycleResult | None
        base_error: CycleConfigurationError | None = None
        try:
            base = CycleEngine(member_config(), policy="base").run(
                trace,
                validate_input=validate_input,
                progress=member_progress("base"),
                progress_interval_events=progress_interval_events,
                progress_interval_seconds=progress_interval_seconds,
                collect_compute_telemetry=collect_compute_telemetry,
            )
        except CycleConfigurationError as error:
            base = None
            base_error = error
        actual: CycleResult | None
        actual_error: CycleConfigurationError | None = None
        try:
            actual = CycleEngine(member_config(), policy=actual_policy).run(
                trace,
                validate_input=validate_input,
                progress=member_progress("actual"),
                progress_interval_events=progress_interval_events,
                progress_interval_seconds=progress_interval_seconds,
                collect_compute_telemetry=collect_compute_telemetry,
            )
        except CycleConfigurationError as error:
            actual = None
            actual_error = error
        future: CycleResult | None
        future_error: CycleConfigurationError | None = None
        try:
            future = CycleEngine(
                member_config(),
                policy=oracle_policy,
                _oracle_portfolio_member=True,
            ).run(
                trace,
                validate_input=False,
                progress=member_progress("future"),
                progress_interval_events=progress_interval_events,
                progress_interval_seconds=progress_interval_seconds,
                collect_compute_telemetry=collect_compute_telemetry,
            )
        except CycleConfigurationError as error:
            future = None
            future_error = error
        candidates = [
            (name, result) for name, result in (
                ("base", base), ("actual", actual), ("future", future),
            )
            if result is not None
        ]
        if not candidates:
            raise CycleConfigurationError(
                "all Oracle portfolio members failed: "
                f"base={base_error}; actual={actual_error}; future={future_error}"
            )
        winner_member, winner = min(
            candidates, key=lambda candidate: candidate[1].total_cycles
        )
        portfolio = OraclePortfolio(
            winner=winner_member,
            members=(
                OraclePortfolioMember(
                    name="base",
                    policy="base",
                    status="passed" if base is not None else "failed_cycle",
                    total_cycles=base.total_cycles if base is not None else None,
                    failure_reason=str(base_error) if base_error is not None else None,
                ),
                OraclePortfolioMember(
                    name="actual",
                    policy=actual_policy,
                    status="passed" if actual is not None else "failed_cycle",
                    total_cycles=actual.total_cycles if actual is not None else None,
                    failure_reason=str(actual_error) if actual_error is not None else None,
                ),
                OraclePortfolioMember(
                    name="future",
                    policy=oracle_policy,
                    status="passed" if future is not None else "failed_cycle",
                    total_cycles=future.total_cycles if future is not None else None,
                    failure_reason=str(future_error) if future_error is not None else None,
                ),
            ),
        )
        if progress is not None:
            final_progress = last_progress.get(winner_member)
            if final_progress is None:
                raise CycleConfigurationError(
                    "Oracle portfolio member produced no final progress sample"
                )
            progress(replace(
                final_progress,
                phase="replay",
                simulated_cycles=winner.total_cycles,
            ))
        return replace(
            winner,
            policy=oracle_policy,
            oracle_status="portfolio_best_known_not_proven_upper_bound",
            oracle_portfolio=portfolio,
            oracle_member_results={name: result for name, result in candidates},
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
        collect_compute_telemetry: bool = False,
    ) -> "CycleReplaySession":
        """Create a persistent packet consumer without trace-column staging."""

        if self.selection.query_oracle or self.selection.residency_oracle:
            raise CycleConfigurationError(
                "future-visible Oracle replay requires a complete trace"
            )

        return CycleReplaySession(
            self,
            max_events=max_events,
            max_frontier_events=max_frontier_events,
            initial_gaussian_count=initial_gaussian_count,
            semantic_workset_totals=semantic_workset_totals,
            retain_completion_cycles=retain_completion_cycles,
            progress=progress,
            progress_interval_seconds=progress_interval_seconds,
            collect_compute_telemetry=collect_compute_telemetry,
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
        expander = VirtualQueryEventExpander(
            max_events=max_events,
            relation_query_lanes=self.config.relation_query_lanes,
        )
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
        collect_compute_telemetry: bool = False,
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
        self.engine.issue_scheduler.reset()
        self.engine._query_history_bases.clear()
        self.max_events = max_events
        self.max_frontier_events = max_frontier_events
        self.retain_completion_cycles = retain_completion_cycles
        self.semantic_workset_totals = dict(semantic_workset_totals or {})
        if any(key[0] < 0 or key[1] < 0 or value <= 0
               for key, value in self.semantic_workset_totals.items()):
            raise ValueError("semantic workset totals must use positive keyed counts")
        self.progress = progress
        self.progress_interval_seconds = progress_interval_seconds
        self._expander = VirtualQueryEventExpander(
            max_events=max_events,
            relation_query_lanes=engine.config.relation_query_lanes,
        )
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
        self._ready: dict[
            tuple[str, int], _ReadyCandidateQueue
        ] = defaultdict(_ReadyCandidateQueue)
        self._fusion_pending: deque[tuple[int, TaskKind]] = deque()
        self._fusion_inputs: dict[TaskKind, deque[int]] = {
            TaskKind.FORWARD: deque(),
            TaskKind.CONSUMER: deque(),
            TaskKind.ADJOINT: deque(),
        }
        self._in_flight: list[tuple[int, int, int, str]] = []
        self._lane_outputs: list[tuple[int, int, int | None]] = []
        self._separate_lane_completion: set[int] = set()
        self._packet_stage_by_event: dict[int, PhysicalPacketStage] = {}
        self._packet_ready_members: dict[int, set[int]] = defaultdict(set)
        self._packet_ready_stages: set[tuple[int, int]] = set()
        self._adjoint_relation_ids: set[int] = set()
        self._consumer_credit_owner: dict[int, int] = {}
        self._completed_reduction_owner: dict[int, tuple[int, int]] = {}
        self._consumer_reduction_remaining: dict[int, int] = {}
        self._consumer_last_reduction: dict[int, tuple[int, int, int]] = {}
        self._relation_windows = engine._new_relation_window_tracker()
        self._query_replay = engine._new_query_replay_tracker()
        self._owner_gradients = engine._new_owner_gradient_tracker()
        self._compute_telemetry = (
            engine._new_compute_telemetry() if collect_compute_telemetry else None
        )
        self._module_busy_until = {
            name: [0] * engine._module_issue_ports(name)
            for name in engine.modules
        }
        self._fusion_busy_until = {
            name: [0] * limit for name, limit in (
                ("forward", engine.issue_scheduler.forward_ports),
                ("consumer", engine.issue_scheduler.consumer_ports),
                ("adjoint", engine.issue_scheduler.adjoint_ports),
            )
        }
        self._fusion_borrow_cursor = 0
        self._fusion_pop_ranks: dict[int, tuple[TaskKind, int]] = {}
        self._module_inflight = {
            (name, partition): 0
            for name in engine.modules
            for partition in range(engine._module_partition_count(name))
        }
        self._relation_seed_inflight = 0
        self._bank_busy: dict[tuple[object, ...], int] = {}
        self._shared_sram_address_busy: dict[tuple[int, int], set[str]] = {}
        self._cache_states = (
            engine._residency_states()
            if engine.selection.semantic_residency else {}
        )
        self._cache_fill_done: dict[tuple[int, tuple[int, int]], int] = {}
        self._cache_fill_request: dict[tuple[int, tuple[int, int]], int] = {}
        self._memory_waiters: dict[int, list[tuple[int, int, str, int]]] = {}
        self._cache_event_state: dict[int, tuple[SemanticCacheState, tuple[int, int], CacheLookup]] = {}
        self._cache_multicast_followers: dict[
            int, tuple[SemanticCacheState, tuple[int, int]]
        ] = {}
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
        self._last_completion_cycle = 0
        self._peak_frontier_events = 0
        self._cycle = 0
        self._source_packets = 0
        self._query_packets = 0
        self._closed_iterations = 0
        self._last_lifecycle_event: int | None = None
        self._open_lifecycle_begin: int | None = None
        self._open_lifecycle_events: list[int] = []
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
        if self._lane_outputs:
            violations.append("lane_outputs")
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
        if self._cache_multicast_followers:
            violations.append("cache_multicast")
        if self._workset_by_request:
            violations.append("workset_requests")
        if self._relation_seed_inflight:
            violations.append("relation_seed_fifo")
        if self._relation_windows is not None and (
            self._relation_windows.live
            or self._relation_windows.relation_records_live
        ):
            violations.append("relation_windows")
        if self._query_replay is not None and self._query_replay.active_queries:
            violations.append("replay_queue")
        if self._owner_gradients is not None and self._owner_gradients.active_by_cluster:
            violations.append("owner_gradient_slots")
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
            if (
                kind is PrimitiveKind.CONSUMER
                and self.engine.issue_scheduler.strict_lifecycle
            ):
                reduction_dependencies = tuple(
                    dependency for dependency in dependencies
                    if self._kinds.get(dependency) is PrimitiveKind.QUERY_REDUCTION
                    or dependency in self._completed_reduction_owner
                )
                if not reduction_dependencies:
                    raise CycleConfigurationError(
                        f"consumer {event_id} has no query reduction dependency"
                    )
                self._consumer_reduction_remaining[event_id] = len(
                    reduction_dependencies
                )
                for dependency in reduction_dependencies:
                    completed = self._completed_reduction_owner.get(dependency)
                    if completed is not None:
                        self._record_reduction_for_consumer(
                            event_id, dependency, completed[0], completed[1],
                        )
            physical_stage = self._packet_stage_by_event.get(event_id)
            physical_reader = (
                physical_stage is None
                or physical_stage.head_event_id == event_id
            )
            if (
                kind is PrimitiveKind.CACHE_REQUEST
                and physical_reader
                and self.semantic_workset_totals
            ):
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
        packets = tuple(packets)
        for packet in packets:
            self.engine._register_query_history_base(
                packet.iteration_id, packet.template_id, packet.query_base,
            )
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
        event_packets = tuple(self._expander.expand(
            packet, external_dependencies=external_dependencies
        ))
        expanded_rows = np.concatenate([item.events for item in event_packets])
        expanded_kinds = set(
            np.unique(expanded_rows["primitive_kind"]).astype(int).tolist()
        )
        if all(
            int(kind) in expanded_kinds
            for kind in (
                PrimitiveKind.RELATION,
                PrimitiveKind.QUERY_CLOSE,
                PrimitiveKind.FORWARD,
                PrimitiveKind.QUERY_REDUCTION,
            )
        ):
            self.engine.issue_scheduler.set_strict_lifecycle()
            self.engine.issue_scheduler.enable_exact_readiness()
        self._adjoint_relation_ids.update(
            int(row["relation_id"])
            for row in expanded_rows[
                expanded_rows["primitive_kind"] == int(PrimitiveKind.ADJOINT)
            ]
            if int(row["relation_id"]) >= 0
        )
        packet_plan = RelationPacketPlan.from_event_packets(
            event_packets,
            query_lanes=self.engine.config.relation_query_lanes,
        )
        if self._query_replay is not None:
            try:
                self._query_replay.register_rows(
                    expanded_rows
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if self._owner_gradients is not None:
            try:
                self._owner_gradients.register_rows(
                    expanded_rows
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if self._relation_windows is not None:
            window_plan = RelationWindowPlan.from_event_packets(
                event_packets,
                packet_plan,
                window_id=self._query_packets - 1,
            )
            try:
                self._relation_windows.register(window_plan.descriptors[0])
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        for physical_stage in packet_plan.stages:
            for event_id in physical_stage.event_ids:
                if event_id in self._packet_stage_by_event:
                    raise CycleConfigurationError(
                        f"online event {event_id} has duplicate packet metadata"
                    )
                self._packet_stage_by_event[event_id] = physical_stage
        for event_packet in event_packets:
            if (
                self.max_frontier_events is not None
                and self.pending_event_count + event_packet.event_count
                > self.max_frontier_events
            ):
                self._drain()
                self._compact_completed_prefix()
                if (
                    self.pending_event_count + event_packet.event_count
                    > self.max_frontier_events
                ):
                    raise CycleConfigurationError(
                        "max_frontier_events cannot hold the open physical packet frontier"
                    )
            terminal_ids.extend(
                int(event_id) for event_id in event_packet.events["event_id"][
                    event_packet.events["primitive_kind"]
                    == int(PrimitiveKind.GRADIENT_REDUCTION)
                ]
            )
            # The complete plan is known before registration.  A drain may
            # occur between expanded chunks, but incomplete physical stages
            # have no ready head and therefore cannot issue early.
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
        if totals and not self.engine.selection.semantic_residency:
            raise CycleConfigurationError(
                "semantic workset totals require semantic residency"
            )
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

        A transaction begin is ordered after the producer's backward/state
        frontier.  Events within one transaction share that begin dependency;
        they are not serialized through the preceding commit or modification.
        The end event joins all events generated by the transaction.
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
        if record.kind is VirtualLifecycleKind.UPDATE_BEGIN:
            base_dependencies = (
                tuple(record.dependency_ids)
                if record.dependency_ids else (
                    self._backward_frontier or tuple(self._events)
                )
            )
        elif record.kind is VirtualLifecycleKind.UPDATE_END:
            base_dependencies = (
                tuple(record.dependency_ids)
                if record.dependency_ids else tuple(
                    self._open_lifecycle_events
                    or ((self._open_lifecycle_begin,) if self._open_lifecycle_begin is not None else ())
                )
            )
        else:
            base_dependencies = (
                tuple(record.dependency_ids)
                if record.dependency_ids else (
                    (self._open_lifecycle_begin,)
                    if self._open_lifecycle_begin is not None
                    else (self._backward_frontier or tuple(self._events))
                )
            )
        if (
            record.kind is VirtualLifecycleKind.UPDATE_COMMIT
            and record.all_active
            and record.active_ids
        ):
            gaussian_ids = tuple(record.active_ids)
        elif record.kind is VirtualLifecycleKind.CLONE and record.child_ids:
            gaussian_ids = (record.parent_id, *record.child_ids)
        elif record.kind is VirtualLifecycleKind.SPLIT and record.child_ids:
            # Raw capture emits split children first and the parent update
            # after all child records have completed.
            gaussian_ids = (*record.child_ids, record.parent_id)
        else:
            gaussian_ids = (record.gaussian_id,)
        lineage_event_ids: list[int] = []
        for gaussian_id in gaussian_ids:
            event_id = self._stream_validator.next_event_id
            dependency_ids = base_dependencies
            if record.kind is VirtualLifecycleKind.CLONE and lineage_event_ids:
                dependency_ids = (*dependency_ids, lineage_event_ids[0])
            elif record.kind is VirtualLifecycleKind.SPLIT and gaussian_id == record.parent_id:
                dependency_ids = (*dependency_ids, *lineage_event_ids)
            if record.kind is VirtualLifecycleKind.UPDATE_BEGIN:
                self._open_lifecycle_events = []
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
            if record.kind is VirtualLifecycleKind.UPDATE_BEGIN:
                self._open_lifecycle_begin = event_id
            elif record.kind is not VirtualLifecycleKind.UPDATE_END:
                self._open_lifecycle_events.append(event_id)
            if record.kind in {
                VirtualLifecycleKind.CLONE, VirtualLifecycleKind.SPLIT,
            }:
                lineage_event_ids.append(event_id)
            self.accept_event_packet(event_packet)
        if record.kind is VirtualLifecycleKind.UPDATE_END:
            self._state_barrier_event = self._last_lifecycle_event
            self._backward_frontier = ()
            self._open_lifecycle_begin = None
            self._open_lifecycle_events = []

    def close_iteration(self, iteration_id: int) -> None:
        self._ensure_open()
        self._lifecycle.close_iteration(iteration_id)
        self._closed_iterations += 1
        self._drain()
        self._completed_reduction_owner.clear()

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
        if self.engine.issue_scheduler.enforce_exact_readiness:
            try:
                self.engine.issue_scheduler.require_drained()
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        self.engine.issue_scheduler.release_completed()
        f_count, c_count, a_count = (
            self.engine.issue_scheduler.live_counter_totals()
        )
        counters["fusion_issue"].update({
            **self.engine.issue_scheduler.state_table_snapshot(),
            **self.engine.issue_scheduler.history_snapshot(),
            "live_forward_count": f_count,
            "live_consumer_count": c_count,
            "live_adjoint_count": a_count,
        })
        if self._relation_windows is not None:
            counters["bidirectional_query"].update(
                self._relation_windows.snapshot()
            )
        if self._query_replay is not None:
            counters["bidirectional_query"].update(self._query_replay.snapshot())
        if self._owner_gradients is not None:
            counters["compute_pod"].update(self._owner_gradients.snapshot())
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
            total_cycles=self._last_completion_cycle,
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
            compute_telemetry=(
                self._compute_telemetry.finish(self._last_completion_cycle)
                if self._compute_telemetry is not None else None
            ),
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

    def _assign_consumer_credit(self, event_id: int, owner: int) -> None:
        existing = self._consumer_credit_owner.get(event_id)
        if existing is not None:
            if existing != owner:
                raise CycleConfigurationError(
                    f"consumer {event_id} changed credit owner"
                )
            return
        try:
            if not self.engine.issue_scheduler.successor_credit(owner):
                raise CycleConfigurationError(
                    f"consumer {event_id} credit owner {owner} is not live"
                )
        except ValueError as error:
            raise CycleConfigurationError(str(error)) from error
        self._consumer_credit_owner[event_id] = owner

    def _record_reduction_for_consumer(
        self, consumer_id: int, reduction_id: int, finish: int, owner: int,
    ) -> None:
        """Credit a consumer after its final query reduction completes."""

        if consumer_id in self._consumer_credit_owner:
            return
        remaining = self._consumer_reduction_remaining.get(consumer_id)
        if remaining is None:
            raise CycleConfigurationError(
                f"consumer {consumer_id} has no reduction dependency ledger"
            )
        remaining -= 1
        if remaining < 0:
            raise CycleConfigurationError(
                f"consumer {consumer_id} query dependency count underflow"
            )
        self._consumer_reduction_remaining[consumer_id] = remaining
        completed = (int(finish), int(reduction_id), int(owner))
        previous = self._consumer_last_reduction.get(
            consumer_id, (-1, -1, -1),
        )
        if completed > previous:
            self._consumer_last_reduction[consumer_id] = completed
        if remaining == 0:
            self._assign_consumer_credit(
                consumer_id, self._consumer_last_reduction[consumer_id][2],
            )

    def _push_ready(self, event_id: int, stage: int) -> None:
        physical_stage = self._packet_stage_by_event.get(event_id)
        kind = self._kinds[event_id]
        if self._compute_telemetry is not None and stage == 0:
            self._compute_telemetry.mark_dependency_ready(
                event_id, kind, self._cycle,
            )
        lane_granular = self.engine._is_lane_granular_stage(kind, stage)
        if physical_stage is not None and not lane_granular:
            packet_key = physical_stage.head_event_id
            if stage == 0:
                members = self._packet_ready_members[packet_key]
                members.add(event_id)
                if len(members) < len(physical_stage.event_ids):
                    return
                if len(members) > len(physical_stage.event_ids):
                    raise CycleConfigurationError(
                        f"online physical packet {packet_key} became ready twice"
                    )
            elif event_id != physical_stage.head_event_id:
                raise CycleConfigurationError(
                    "only an online physical packet head may advance stages"
                )
            ready_key = (packet_key, stage)
            if ready_key in self._packet_ready_stages:
                return
            self._packet_ready_stages.add(ready_key)
            event_id = physical_stage.head_event_id
        task_kind = self._fusion_kind(event_id, stage)
        if task_kind is None:
            module_name = self.engine._stages_for(kind)[stage]
            partition = self.engine._module_partition(
                module_name, self._events[event_id]
            )
            self._ready[(module_name, partition)].push((event_id, stage))
        else:
            self._fusion_pending.append((event_id, task_kind))

    def _requeue(self, event_id: int, stage: int) -> None:
        physical_stage = self._packet_stage_by_event.get(event_id)
        kind = self._kinds[event_id]
        if physical_stage is not None and not self.engine._is_lane_granular_stage(
            kind, stage
        ):
            event_id = physical_stage.head_event_id
        task_kind = self._fusion_kind(event_id, stage)
        if task_kind is None:
            module_name = self.engine._stages_for(kind)[stage]
            partition = self.engine._module_partition(
                module_name, self._events[event_id]
            )
            self._ready[(module_name, partition)].push((event_id, stage))
        else:
            popped = self._fusion_pop_ranks.get(event_id)
            if popped is None:
                self._fusion_inputs[task_kind].appendleft(event_id)
            else:
                _kind, rank = popped
                queue = self._fusion_inputs[task_kind]
                position = next((
                    index for index, queued_id in enumerate(queue)
                    if (
                        (queued := self._fusion_pop_ranks.get(queued_id)) is None
                        or queued[1] > rank
                    )
                ), len(queue))
                queue.insert(position, event_id)

    def _fusion_kind(self, event_id: int, stage: int) -> TaskKind | None:
        if not self.engine.selection.query_scheduler_enabled or stage != 0:
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
        physical_stage = self._packet_stage_by_event.get(event_id)
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
            target_resource=(
                (int(row["resource_class"]) << 56)
                | int(row["address_token"])
            ),
            consumer_owner_ids=(
                ((self._consumer_credit_owner[event_id],)
                 if event_id in self._consumer_credit_owner else ())
                if kind is PrimitiveKind.CONSUMER else ()
            ),
            workset_hit=self.engine._workset_hit(self._cache_states, row),
            conflict_query_ids=(
                tuple(
                    int(self._events[member]["query_id"])
                    for member in physical_stage.event_ids
                )
                if task_kind in {TaskKind.FORWARD, TaskKind.ADJOINT}
                and physical_stage is not None
                else ()
            ),
        )

    def _ordered(self, candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return candidates

    def _drain(self) -> None:
        async_memory = callable(getattr(self.engine.config.memory, "submit_async", None))
        while (
            self._ready
            or self._in_flight
            or self._memory_waiters
            or self._lane_outputs
            or self._fusion_pending
            or any(self._fusion_inputs.values())
        ):
            # Bank reservations are scoped to one cycle.  Drop old entries so
            # long packet streams do not retain one dictionary item per cycle.
            self._bank_busy = {
                key: value for key, value in self._bank_busy.items()
                if key[1] >= self._cycle
            }
            self._shared_sram_address_busy = {
                key: value
                for key, value in self._shared_sram_address_busy.items()
                if key[0] >= self._cycle
            }
            self._report_progress()
            self._cycle_fusion_issued = 0
            self._cycle_fusion_ports: dict[str, int] = {}
            self._cycle_module_issued: dict[tuple[str, int], int] = {}
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
            while self._lane_outputs and self._lane_outputs[0][0] <= self._cycle:
                finish, event_id, next_stage = heapq.heappop(self._lane_outputs)
                if next_stage is None:
                    physical_stage = self._packet_stage_by_event.get(event_id)
                    retain = bool(
                        physical_stage is not None
                        and physical_stage.head_event_id == event_id
                        and event_id in self._separate_lane_completion
                    )
                    self._complete_logical_event(event_id, finish, retain=retain)
                else:
                    self._push_ready(event_id, next_stage)
                progressed = True
            if self.engine.selection.query_scheduler_enabled:
                self.engine.issue_scheduler.set_clock(self._cycle)
                capacity = (
                    self.engine.config.candidate_fifo_entries
                    or self.engine.modules["fusion_issue"].timing.queue_capacity
                )
                blocked_pending: list[tuple[int, TaskKind]] = []
                while self._fusion_pending:
                    event_id, task_kind = self._fusion_pending.popleft()
                    if len(self._fusion_inputs[task_kind]) >= capacity:
                        blocked_pending.append((event_id, task_kind))
                        continue
                    self._fusion_inputs[task_kind].append(event_id)
                    self.engine.issue_scheduler.observe_arrival(
                        (self._task_packet(event_id),), arrival_cycle=self._cycle,
                    )
                for pending_entry in blocked_pending:
                    self._fusion_pending.append(pending_entry)
            candidates: list[tuple[int, int]] = []
            self._fusion_pop_ranks = {}
            for ready_key in tuple(self._ready):
                queue = self._ready[ready_key]
                module_name, _partition = ready_key
                candidates.extend(self.engine._pop_ready_candidates(
                    queue, module_name,
                    row_for=self._events.__getitem__,
                    physical_stage_for=self._packet_stage_by_event.get,
                    owner_gradients=self._owner_gradients,
                    query_replay=self._query_replay,
                ))
                if not queue:
                    del self._ready[ready_key]
            if self.engine.selection.query_scheduler_enabled:
                (
                    fusion_candidate_ids,
                    self._fusion_pop_ranks,
                    self._fusion_borrow_cursor,
                ) = self.engine._pop_fusion_input_candidates(
                    self._fusion_inputs,
                    borrow_cursor=self._fusion_borrow_cursor,
                )
                candidates.extend(
                    (event_id, 0) for event_id in fusion_candidate_ids
                )
            if candidates:
                ordered = self._ordered(candidates)
                fusion_packets = {
                    event_id: self._task_packet(event_id)
                    for event_id, stage in ordered
                    if self.engine.selection.query_scheduler_enabled and stage == 0
                    and PrimitiveKind(int(self._events[event_id]["primitive_kind"]))
                    in {PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT}
                }
                selected: set[int] = set()
                if fusion_packets:
                    decision = self.engine.issue_scheduler.select(
                        fusion_packets.values(),
                        use_load_rules=self.engine.selection.query_load_rules,
                    )
                    selected = {task.event_id for task in decision.accepted}
                    order = iter(task.event_id for task in (*decision.accepted, *decision.rejected))
                    ordered = [
                        (next(order), stage) if event_id in fusion_packets else (event_id, stage)
                        for event_id, stage in ordered
                    ]
                for ordered_position, (event_id, stage) in enumerate(ordered):
                    multicast_followers: tuple[int, ...] = ()
                    if (
                        stage == 0
                        and self._kinds[event_id] is PrimitiveKind.CACHE_REQUEST
                    ):
                        max_destinations = (
                            self.engine.config.cache_multicast_destinations
                        )
                        if max_destinations is not None and max_destinations > 1:
                            row = self._events[event_id]
                            key = (
                                int(row["gaussian_id"]),
                                int(row["state_version"]),
                            )
                            followers: list[int] = []
                            for candidate_id, candidate_stage in ordered[ordered_position + 1:]:
                                if (
                                    candidate_stage != 0
                                    or self._kinds[candidate_id]
                                    is not PrimitiveKind.CACHE_REQUEST
                                ):
                                    continue
                                candidate_row = self._events[candidate_id]
                                if (
                                    int(candidate_row["gaussian_id"]),
                                    int(candidate_row["state_version"]),
                                ) != key:
                                    continue
                                followers.append(candidate_id)
                                if len(followers) + 1 >= max_destinations:
                                    break
                            multicast_followers = tuple(followers)
                    if self._try_issue(
                        event_id, stage, selected, async_memory,
                        multicast_followers=multicast_followers,
                    ):
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
                    min((value for lanes in self._fusion_busy_until.values()
                         for value in lanes if value > self._cycle), default=None),
                    self._lane_outputs[0][0] if self._lane_outputs else None,
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

    def _try_issue(
        self,
        event_id: int,
        stage: int,
        selected: set[int],
        async_memory: bool,
        *,
        multicast_followers: tuple[int, ...] = (),
    ) -> bool:
        row = self._events[event_id]
        kind = self._kinds[event_id]
        physical_stage = self._packet_stage_by_event.get(event_id)
        stages = self.engine._stages_for(kind)
        module_name = stages[stage]
        module = self.engine.modules[module_name]
        timing = module.timing
        follower_state = (
            self._cache_multicast_followers.get(event_id)
            if kind is PrimitiveKind.CACHE_REQUEST and stage == 0 else None
        )
        if follower_state is not None:
            state, key = follower_state
            self._cache_multicast_followers.pop(event_id, None)
            self._cache_event_state[event_id] = (state, key, CacheLookup.HIT)
            self._cache_keys_by_version.setdefault(key[1], []).append((state, key))
            completion = self._cycle + self.engine._physical_service_cycles(
                module_name, row, kind, physical_stage,
            )
            heapq.heappush(
                self._in_flight, (completion, event_id, stage, module_name)
            )
            inflight_key = (
                module_name, self.engine._module_partition(module_name, row)
            )
            self._module_inflight[inflight_key] += 1
            module.counters.accepted += 1
            return True
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
        module_partition = self.engine._module_partition(module_name, row)
        module_issue_key = (module_name, module_partition)
        if module_name == "fusion_issue" and stage == 0 and self.engine.selection.overlap_guided_issue:
            port_name, limit = self.engine._fusion_port_limit(kind)
            used = getattr(self, "_cycle_fusion_ports", {}).get(port_name, 0)
            if used >= limit:
                module.counters.port_stalls += 1
                self.engine._record_stall(self._cycle, module_name, "port", event_id)
                self._requeue(event_id, stage)
                return False
        elif (
            getattr(self, "_cycle_module_issued", {}).get(module_issue_key, 0)
            >= self.engine._module_partition_issue_limit(
                module_name, module_partition,
            )
        ):
            module.counters.port_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "port", event_id)
            self._requeue(event_id, stage)
            return False
        module_lane: int | None = None
        fusion_port_lane: int | None = None
        query_allocation: tuple[int, ...] = ()
        if port_name:
            fusion_port_lane = next((
                index for index, point in enumerate(
                    self._fusion_busy_until[port_name]
                )
                if point <= self._cycle
            ), None)
            busy_until = (
                self._cycle if fusion_port_lane is not None
                else min(self._fusion_busy_until[port_name])
            )
        elif module_name == "bidirectional_query" and self.engine._has_query_resources():
            module_lanes = self._module_busy_until[module_name]
            allocation = self.engine._query_resource_allocation(
                module_lanes, row, kind, physical_stage, self._cycle,
            )
            if allocation is None:
                reason = (
                    "reduction_bank"
                    if kind in {
                        PrimitiveKind.FORWARD,
                        PrimitiveKind.QUERY_REDUCTION,
                    }
                    else "query_datapath"
                )
                if reason == "reduction_bank":
                    module.counters.bank_conflicts += 1
                else:
                    module.counters.port_stalls += 1
                self.engine._record_stall(
                    self._cycle, module_name, reason, event_id
                )
                self._requeue(event_id, stage)
                return False
            query_allocation = allocation
            busy_until = self._cycle
        else:
            module_lanes = self._module_busy_until[module_name]
            module_lane = next(
                (index for index in self.engine._module_lane_indices(module_name, row)
                 if module_lanes[index] <= self._cycle),
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
        if (self.engine._uses_generic_module_limits(module_name)
                and self._module_inflight[module_issue_key] >= timing.queue_capacity):
            module.counters.queue_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "queue_capacity", event_id)
            self._requeue(event_id, stage)
            return False
        if (
            self._owner_gradients is not None
            and kind is PrimitiveKind.ADJOINT
            and module_name == "compute_pod"
        ):
            owner_event_ids = (
                physical_stage.event_ids
                if physical_stage is not None else (event_id,)
            )
            if self._owner_gradients.blocks_adjoint(owner_event_ids):
                module.counters.queue_stalls += 1
                self.engine._record_stall(
                    self._cycle, module_name,
                    "owner_gradient_slot_capacity", event_id,
                )
                self._requeue(event_id, stage)
                return False
        if (
            self._query_replay is not None
            and module_name == "bidirectional_query"
            and kind is PrimitiveKind.CONSUMER
            and self._query_replay.blocks_consumer(event_id)
        ):
            module.counters.queue_stalls += 1
            self.engine._record_stall(
                self._cycle, module_name, "replay_queue_capacity", event_id
            )
            self._requeue(event_id, stage)
            return False
        if (
            self._query_replay is not None
            and module_name == "bidirectional_query"
            and kind is PrimitiveKind.ADJOINT
        ):
            replay_event_ids = (
                physical_stage.event_ids
                if physical_stage is not None else (event_id,)
            )
            try:
                replay_blocked = self._query_replay.blocks_adjoint(
                    replay_event_ids
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
            if replay_blocked:
                module.counters.queue_stalls += 1
                self.engine._record_stall(
                    self._cycle, module_name, "replay_queue_capacity", event_id
                )
                self._requeue(event_id, stage)
                return False
        if kind is PrimitiveKind.RELATION_CANDIDATE and self._relation_seed_inflight >= self.engine.config.relation_seed_fifo_entries:
            module.counters.queue_stalls += 1
            self.engine._record_stall(self._cycle, module_name, "seed_fifo", event_id)
            self._requeue(event_id, stage)
            return False
        if self._relation_windows is not None and stage == 0:
            physical_head = (
                physical_stage is None
                or physical_stage.head_event_id == event_id
            )
            window_reason = self._relation_windows.blocking_reason(
                event_id, kind,
                physical_stage_head=physical_head,
                cycle=self._cycle,
            )
            if window_reason is not None:
                query_module = self.engine.modules["bidirectional_query"]
                query_module.counters.queue_stalls += 1
                self.engine._record_stall(
                    self._cycle, "bidirectional_query", window_reason, event_id
                )
                self._requeue(event_id, stage)
                return False
            if (
                not self.engine.selection.query_load_rules
                and not self.engine.selection.query_oracle
            ):
                try:
                    stage_reason = (
                        self._relation_windows.full_stage_blocking_reason(
                            event_id, kind,
                        )
                    )
                except ValueError as error:
                    raise CycleConfigurationError(str(error)) from error
                if stage_reason is not None:
                    query_module = self.engine.modules["bidirectional_query"]
                    query_module.counters.queue_stalls += 1
                    self.engine._record_stall(
                        self._cycle, "bidirectional_query", stage_reason, event_id,
                    )
                    self._requeue(event_id, stage)
                    return False
        bank_keys = self.engine._module_bank_reservation_keys(
            module_name, row, self._cycle,
        )
        shared_sram_direction: str | None = None
        shared_sram_address_key: tuple[int, int] | None = None
        if module_name == "shared_sram":
            shared_sram_direction = self.engine._shared_sram_direction(kind)
            shared_sram_address_key = self.engine._shared_sram_address_key(
                row, self._cycle,
            )
            existing_directions = self._shared_sram_address_busy.get(
                shared_sram_address_key, set(),
            )
            if any(
                direction != shared_sram_direction
                for direction in existing_directions
            ):
                module.counters.bank_conflicts += 1
                self.engine._record_stall(
                    self._cycle, module_name, "same_address_replay", event_id,
                )
                self._requeue(event_id, stage)
                return False
        if (self.engine._uses_generic_module_bank_limits(module_name)
                and any(key in self._bank_busy for key in bank_keys)):
            module.counters.bank_conflicts += 1
            self.engine._record_stall(self._cycle, module_name, "bank", event_id)
            self._requeue(event_id, stage)
            return False
        compute_plan: tuple[tuple[str, int, int], ...] = ()
        if module_name == "compute_pod":
            try:
                compute_pod, compute_cluster = self.engine._compute_route(row, kind)
                compute_plan = module.reservation_plan(  # type: ignore[attr-defined]
                    int(row["template_id"]), kind, self._cycle,
                    pod=compute_pod, cluster_hint=compute_cluster,
                )
            except KeyError as error:
                raise CycleConfigurationError(str(error)) from error
            if not module.can_reserve(compute_plan, self._cycle):  # type: ignore[attr-defined]
                module.counters.queue_stalls += 1
                assert isinstance(module, ComputePod)
                self.engine._record_compute_resource_stall(
                    cycle=self._cycle, event_id=event_id, pod=compute_pod,
                    compute=module, plan=compute_plan,
                )
                self._requeue(event_id, stage)
                return False
        service_cycles = self.engine._physical_service_cycles(
            module_name, row, kind, physical_stage,
        )
        try:
            completion: int | None = self._issue_memory_if_needed(
                event_id, stage, kind, module_name, timing, service_cycles,
                async_memory, multicast_followers=multicast_followers,
            )
        except CacheBackpressure:
            module.counters.queue_stalls += 1
            self.engine._record_stall(
                self._cycle, module_name, "cache_capacity", event_id
            )
            self._requeue(event_id, stage)
            return False
        if completion is not None:
            heapq.heappush(self._in_flight, (completion, event_id, stage, module_name))
        if (
            physical_stage is not None
            and module_name == "compute_pod"
            and kind is PrimitiveKind.FORWARD
            and self.engine.config.compute_templates is not None
        ):
            compute = self.engine.modules[module_name]
            assert isinstance(compute, ComputePod)
            path = compute.path_for(int(row["template_id"]), kind)
            for logical_event, lane in zip(
                physical_stage.event_ids, physical_stage.lanes, strict=True,
            ):
                heapq.heappush(
                    self._lane_outputs,
                    (
                        self._cycle + path.packet_completion_offset(lane),
                        logical_event,
                        stage + 1 if stage + 1 < len(stages) else None,
                    ),
                )
            self._separate_lane_completion.add(event_id)
        if port_name:
            assert fusion_port_lane is not None
            self._fusion_busy_until[port_name][fusion_port_lane] = (
                self._cycle + timing.initiation_interval
            )
        elif query_allocation:
            for lane in query_allocation:
                self._module_busy_until[module_name][lane] = (
                    self._cycle + timing.initiation_interval
                )
        else:
            assert module_lane is not None
            self._module_busy_until[module_name][module_lane] = (
                self._cycle + timing.initiation_interval
            )
        if self.engine._uses_generic_module_bank_limits(module_name):
            for bank_key in bank_keys:
                self._bank_busy[bank_key] = self._cycle
        if (
            shared_sram_address_key is not None
            and shared_sram_direction is not None
        ):
            self._shared_sram_address_busy.setdefault(
                shared_sram_address_key, set(),
            ).add(shared_sram_direction)
        self._module_inflight[module_issue_key] += 1
        if kind is PrimitiveKind.RELATION_CANDIDATE:
            self._relation_seed_inflight += 1
        if self._relation_windows is not None and stage == 0:
            try:
                self._relation_windows.issue(
                    event_id, kind,
                    physical_stage_head=(
                        physical_stage is None
                        or physical_stage.head_event_id == event_id
                    ),
                    cycle=self._cycle,
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if (
            self._query_replay is not None
            and module_name == "bidirectional_query"
            and kind is PrimitiveKind.CONSUMER
        ):
            try:
                self._query_replay.reserve_consumer(event_id)
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if (
            self._query_replay is not None
            and module_name == "bidirectional_query"
            and kind is PrimitiveKind.ADJOINT
        ):
            try:
                self._query_replay.reserve_adjoint(
                    physical_stage.event_ids
                    if physical_stage is not None else (event_id,)
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if (
            self._owner_gradients is not None
            and kind is PrimitiveKind.ADJOINT
            and module_name == "compute_pod"
        ):
            try:
                self._owner_gradients.reserve_adjoint(
                    physical_stage.event_ids
                    if physical_stage is not None else (event_id,)
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        module.counters.accepted += 1
        module.counters.busy_cycles += service_cycles
        if compute_plan:
            module.reserve(compute_plan)  # type: ignore[attr-defined]
        if self._compute_telemetry is not None:
            self._compute_telemetry.mark_issue(
                self.engine._stage_event_ids(
                    event_id, kind, stage, physical_stage,
                ),
                kind, module_name, self._cycle, compute_plan=compute_plan,
            )
        self._cycle_module_issued[module_issue_key] = (
            self._cycle_module_issued.get(module_issue_key, 0) + 1
        )
        if module_name == "fusion_issue" and stage == 0:
            self._cycle_fusion_issued = getattr(self, "_cycle_fusion_issued", 0) + 1
            if kind is PrimitiveKind.CONSUMER and self.engine.issue_scheduler.strict_lifecycle:
                try:
                    owner = self._consumer_credit_owner.get(event_id)
                    dispatched = (
                        owner is not None
                        and self.engine.issue_scheduler.successor_dispatch(owner)
                    )
                    if not dispatched:
                        raise CycleConfigurationError(
                            f"consumer {event_id} has no query-state owner"
                        )
                except ValueError as error:
                    raise CycleConfigurationError(str(error)) from error
            if self.engine.selection.query_scheduler_enabled:
                self.engine.issue_scheduler.commit_issued((self._task_packet(event_id),))
            if port_name:
                self._cycle_fusion_ports[port_name] = self._cycle_fusion_ports.get(port_name, 0) + 1
        return True

    def _issue_memory_if_needed(self, event_id: int, stage: int, kind: PrimitiveKind,
                                module_name: str, timing: ModuleTiming,
                                service_cycles: int,
                                async_memory: bool, *,
                                multicast_followers: tuple[int, ...] = ()) -> int | None:
        if kind is not PrimitiveKind.CACHE_REQUEST or stage != 0:
            return self._cycle + service_cycles
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
                                                     self._cycle + service_cycles)]
                return None
            memory_done = self.engine.config.memory.submit(
                address=int(row["address_token"]), size_bytes=data_bytes,
                is_write=False, arrival_cycle=self._cycle,
            )
            return max(self._cycle + service_cycles, memory_done)
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
            if multicast_followers and key in state.active:
                available_followers = tuple(
                    follower_id for follower_id in multicast_followers
                    if follower_id not in self._cache_multicast_followers
                )
                if available_followers:
                    state.begin_multicast(
                        key, destinations=len(available_followers)
                    )
                    for follower_id in available_followers:
                        self._cache_multicast_followers[follower_id] = (state, key)
            return self._cycle + service_cycles
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
                (event_id, stage, module_name, self._cycle + service_cycles)
            )
            return None
        return max(self._cycle + service_cycles, request_id)

    def _complete_stage(self, finish: int, event_id: int, stage: int, module_name: str) -> None:
        module = self.engine.modules[module_name]
        module.complete(event_id, finish)
        row = self._events[event_id]
        inflight_key = (
            module_name, self.engine._module_partition(module_name, row)
        )
        self._module_inflight[inflight_key] -= 1
        kind = self._kinds[event_id]
        physical_stage = self._packet_stage_by_event.get(event_id)
        stages = self.engine._stages_for(kind)
        if (
            self._query_replay is not None
            and module_name == "bidirectional_query"
            and kind is PrimitiveKind.ADJOINT
        ):
            try:
                self._query_replay.dispatch_adjoint(
                    physical_stage.event_ids
                    if physical_stage is not None else (event_id,)
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        if (
            self._owner_gradients is not None
            and module_name == "compute_pod"
            and kind is PrimitiveKind.GRADIENT_REDUCTION
        ):
            try:
                self._owner_gradients.complete_gradient(
                    physical_stage.event_ids
                    if physical_stage is not None else (event_id,)
                )
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
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
                waiter_count = int(pending["waiters"])
                state.fill_complete(
                    key, remaining_uses=int(pending["remaining_uses"])
                )
                max_destinations = self.engine.config.cache_multicast_destinations
                if max_destinations is not None and max_destinations > 1:
                    remaining_waiters = waiter_count
                    while remaining_waiters > 1:
                        destinations = min(remaining_waiters, max_destinations)
                        state.begin_multicast(
                            key,
                            destinations=destinations,
                            readers_already_active=True,
                        )
                        remaining_waiters -= destinations
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
            if workset is not None and workset[3]:
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
        skip_shared_sram = (
            kind is PrimitiveKind.CACHE_REQUEST
            and stage == 0
            and event_id in self._cache_event_state
            and self._cache_event_state[event_id][2] is not CacheLookup.MISS
        )
        if skip_shared_sram:
            # Hits and merged requests use the leader's active/pending record;
            # only a miss leader performs the Active SRAM fill write.  Retire
            # all logical lanes represented by this physical cache packet.
            logical_events = (
                physical_stage.event_ids
                if physical_stage is not None else (event_id,)
            )
            for logical_event in logical_events:
                self._complete_logical_event(logical_event, finish)
            return
        if stage + 1 < len(stages):
            if event_id in self._separate_lane_completion:
                self._separate_lane_completion.remove(event_id)
                if event_id in self._completed_ids:
                    self._drop_event(event_id)
            else:
                self._push_ready(event_id, stage + 1)
            return
        if (
            event_id in self._separate_lane_completion
            and module_name == "compute_pod"
            and kind is PrimitiveKind.FORWARD
        ):
            self._separate_lane_completion.remove(event_id)
            self._drop_event(event_id)
            return
        logical_events = (
            physical_stage.event_ids
            if physical_stage is not None
            and not self.engine._is_lane_granular_stage(kind, stage)
            else (event_id,)
        )
        for logical_event in logical_events:
            retain = bool(
                physical_stage is not None
                and logical_event == physical_stage.head_event_id
                and logical_event in self._separate_lane_completion
            )
            self._complete_logical_event(logical_event, finish, retain=retain)

    def _complete_logical_event(
        self, event_id: int, finish: int, *, retain: bool = False,
    ) -> None:
        if event_id in self._completed_ids or event_id <= self._completed_through:
            raise CycleConfigurationError(f"online event {event_id} completed twice")
        self._completed_ids.add(event_id)
        row = self._events[event_id]
        kind = self._kinds[event_id]
        self.engine._update_query_state_for_completion(
            row, kind, self._adjoint_relation_ids,
        )
        if kind is PrimitiveKind.QUERY_REDUCTION:
            self._completed_reduction_owner[event_id] = (
                finish, int(row["query_id"]),
            )
        if self._compute_telemetry is not None:
            self._compute_telemetry.mark_finish(event_id, finish)
        if (
            self._relation_windows is not None
            and event_id in self._relation_windows.event_to_window
        ):
            try:
                self._relation_windows.complete_event(event_id)
            except ValueError as error:
                raise CycleConfigurationError(str(error)) from error
        self._completed_events += 1
        self._last_completion_cycle = max(self._last_completion_cycle, finish)
        if self.retain_completion_cycles:
            self._completion_cycles[event_id] = finish
        for dependent in self._dependents.pop(event_id, []):
            remaining = self._remaining[dependent] - 1
            if remaining < 0:
                raise CycleConfigurationError(f"dependency count underflow at event {dependent}")
            self._remaining[dependent] = remaining
            if (
                kind is PrimitiveKind.QUERY_REDUCTION
                and self._kinds[dependent] is PrimitiveKind.CONSUMER
                and self.engine.issue_scheduler.strict_lifecycle
            ):
                self._record_reduction_for_consumer(
                    dependent, event_id, finish, int(row["query_id"]),
                )
            if remaining == 0:
                self._push_ready(dependent, 0)
        if not retain:
            self._drop_event(event_id)

    def _drop_event(self, event_id: int) -> None:
        self._consumer_credit_owner.pop(event_id, None)
        self._consumer_reduction_remaining.pop(event_id, None)
        self._consumer_last_reduction.pop(event_id, None)
        self._events.pop(event_id, None)
        self._kinds.pop(event_id, None)
        self._dependencies.pop(event_id, None)
        self._remaining.pop(event_id, None)
        self._packet_stage_by_event.pop(event_id, None)

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
        if not self.session.engine.selection.semantic_residency:
            packets = tuple(self._packets)
            self._packets.clear()
            self.session.accept_query_packets(packets)
            return
        totals: dict[tuple[int, int], int] = defaultdict(int)
        for packet in self._packets:
            counts = packet.relation_packet_counts_by_candidate(
                query_lanes=self.session.engine.config.relation_query_lanes,
                max_relations=self.session.max_events,
            )
            for gaussian_id, count in zip(packet.point_ids, counts, strict=True):
                if int(count):
                    key = (int(gaussian_id), int(packet.state_version))
                    totals[key] += int(count)
        self._iteration_workset_keys.update(totals)
        self.session.register_semantic_workset_totals(totals)
        packets = tuple(self._packets)
        self._packets.clear()
        self.session.accept_query_packets(packets)
