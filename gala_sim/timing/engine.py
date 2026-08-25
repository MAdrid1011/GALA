"""Dependency-aware, event-jumping cycle executor."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

from gala_sim.clamp import FusionIssueScheduler, TaskKind, TaskPacket
from gala_sim.clamp.events import PrimitiveKind
from gala_sim.trace.model import Trace
from gala_sim.trace.validator import validate_trace

from .config import CycleConfig
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
class CycleResult:
    total_cycles: int
    module_counters: dict[str, dict[str, int]]
    stalls: tuple[StallRecord, ...]
    completion_cycles: dict[int, int]
    event_counts: dict[str, int]
    policy: str
    oracle_status: str
    memory_requests: tuple[MemoryRequestRecord, ...]


@dataclass
class _InFlight:
    completion_cycle: int
    event_id: int
    module: str


class CycleEngine:
    """Run the same trace for Base, Oracle, or mechanism-specific policies."""

    def __init__(self, config: CycleConfig, *, policy: str = "base") -> None:
        valid_variants = {f"variant:{bits:04b}" for bits in range(16)}
        if policy not in {"base", "query_oracle", "residency_oracle", "query", "residency", "full"} | valid_variants:
            raise ValueError(f"unknown cycle policy: {policy}")
        self.config = config
        self.policy = policy
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
        self.stalls: list[StallRecord] = []
        self._stall_index: dict[tuple[int, str, str], int] = {}
        self._stall_cycle: int | None = None

    def _record_stall(self, cycle: int, module: str, reason: str, event_id: int) -> None:
        if self._stall_cycle != cycle:
            self._stall_index.clear()
            self._stall_cycle = cycle
        key = (cycle, module, reason)
        index = self._stall_index.get(key)
        if index is None:
            self._stall_index[key] = len(self.stalls)
            self.stalls.append(StallRecord(cycle, module, reason, (event_id,)))
            return
        previous = self.stalls[index]
        event_ids = previous.event_ids
        if len(event_ids) < self.config.candidate_lanes:
            event_ids = (*event_ids, event_id)
        self.stalls[index] = StallRecord(
            cycle, module, reason, event_ids, previous.count + 1
        )

    def _stages_for(self, kind: PrimitiveKind) -> tuple[str, ...]:
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
        query_enabled, residency_enabled = self._mechanism_flags()
        if not query_enabled and not residency_enabled:
            return candidates
        if residency_enabled:
            if query_enabled:
                base_order = sorted(candidates, key=lambda event_id: (
                    0 if PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                    in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN} else 1,
                    int(trace.events[event_id]["query_id"]),
                    int(trace.events[event_id]["gaussian_id"]), event_id,
                ))
                return self._schedule_fusion(trace, base_order)
            return sorted(candidates, key=lambda event_id: (
                0 if PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN} else 1,
                int(trace.events[event_id]["gaussian_id"]), event_id,
            ))
        if query_enabled:
            base_order = sorted(candidates, key=lambda event_id: (
                int(trace.events[event_id]["query_id"]),
                int(trace.events[event_id]["relation_id"]), event_id,
            ))
            return self._schedule_fusion(trace, base_order)
        return candidates

    def _schedule_fusion(self, trace: Trace, base_order: list[int]) -> list[int]:
        fusion_ids = [event_id for event_id in base_order if PrimitiveKind(
            int(trace.events[event_id]["primitive_kind"])
        ) in {
            PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT,
        }]
        packets = [self._task_packet(trace, event_id) for event_id in fusion_ids]
        scheduled = [packet.event_id for packet in self.issue_scheduler.forecast(packets)]
        scheduled_iter = iter(scheduled)
        scheduled_set = set(fusion_ids)
        return [
            next(scheduled_iter) if event_id in scheduled_set else event_id
            for event_id in base_order
        ]

    @staticmethod
    def _task_packet(trace: Trace, event_id: int) -> TaskPacket:
        row = trace.events[event_id]
        kind = PrimitiveKind(int(row["primitive_kind"]))
        task_kind = {
            PrimitiveKind.FORWARD: TaskKind.FORWARD,
            PrimitiveKind.CONSUMER: TaskKind.CONSUMER,
            PrimitiveKind.ADJOINT: TaskKind.ADJOINT,
        }[kind]
        relation_id = int(row["relation_id"])
        query_id = int(row["query_id"])
        gaussian_id = int(row["gaussian_id"])
        reduction_key = int(row["reduction_key"])
        return TaskPacket(
            event_id=event_id,
            query_id=max(query_id, 0),
            gaussian_id=max(gaussian_id, 0),
            reduction_key=max(reduction_key, relation_id, event_id),
            resource=int(row["resource_class"]),
            state_version=int(row["state_version"]),
            template_id=int(row["template_id"]),
            address_token=int(row["address_token"]),
            task_kind=task_kind,
        )

    def _mechanism_flags(self) -> tuple[bool, bool]:
        """Return query-order and semantic-residency flags for this run."""
        if self.policy in {"base", "variant:0000"}:
            return False, False
        if self.policy in {"query", "query_oracle"}:
            return True, False
        if self.policy in {"residency", "residency_oracle"}:
            return False, True
        if self.policy == "full":
            return True, True
        bits = self.policy.removeprefix("variant:")
        return bits[0] == "1" or bits[2] == "1", bits[3] == "1"

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

    def run(self, trace: Trace) -> CycleResult:
        validate_trace(trace)
        if not self.config.modules:
            raise CycleConfigurationError("cycle modules are not configured")
        completed: dict[int, int] = {}
        remaining_dependencies: dict[int, int] = {}
        dependents: dict[int, list[int]] = {}
        ready: list[tuple[int, int]] = []
        for row in trace.events:
            event_id = int(row["event_id"])
            dependencies = trace.dependency_ids(row)
            remaining_dependencies[event_id] = int(dependencies.size)
            for dependency in dependencies:
                dependents.setdefault(int(dependency), []).append(event_id)
            if dependencies.size == 0:
                ready.append((event_id, 0))
        heapq.heapify(ready)
        remaining_events = len(trace.events)
        in_flight: list[tuple[int, int, int, str]] = []
        module_busy_until = {name: 0 for name in self.modules}
        fusion_busy_until = {name: 0 for name in ("forward", "consumer", "adjoint")}
        module_inflight = {name: 0 for name in self.modules}
        relation_seed_inflight = 0
        bank_busy: dict[tuple[str, int, int], int] = {}
        residency_enabled = self._mechanism_flags()[1]
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
        closed_versions: set[int] = set()
        memory_requests = 0
        cycle = 0
        ready_scan_window = max(
            self.config.candidate_lanes,
            sum(module.timing.ports for module in self.modules.values()),
        )
        while remaining_events or in_flight:
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
                        state.fill_complete(key, remaining_uses=1)
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
                    if int(trace.events[event_id]["state_version"]) in closed_versions:
                        state.close(key)
                if kind is PrimitiveKind.UPDATE_END and stage == len(stages) - 1:
                    state_version = int(trace.events[event_id]["state_version"])
                    if int(trace.events[event_id]["field_mask"]) != 0:
                        closed_versions.add(state_version)
                        for state, key in cache_keys_by_version.get(state_version, []):
                            state.close(key)
                if stage + 1 < len(stages):
                    heapq.heappush(ready, (event_id, stage + 1))
                else:
                    completed[event_id] = finish
                    remaining_events -= 1
                    for dependent in dependents.get(event_id, ()):
                        remaining_dependencies[dependent] -= 1
                        if remaining_dependencies[dependent] == 0:
                            heapq.heappush(ready, (dependent, 0))
                if (PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                        is PrimitiveKind.RELATION_CANDIDATE):
                    relation_seed_inflight -= 1
                    if relation_seed_inflight < 0:
                        raise CycleConfigurationError("negative relation seed FIFO occupancy")
                progressed = True
            candidates: list[tuple[int, int]] = []
            for _ in range(ready_scan_window):
                if not ready:
                    break
                candidates.append(heapq.heappop(ready))
            fusion_issued = 0
            fusion_port_issued: dict[str, int] = {}
            issued_modules: dict[str, int] = {}
            ordered_ids = self._ordered_candidates(trace, [item[0] for item in candidates])
            ordered = [(event_id, dict(candidates)[event_id]) for event_id in ordered_ids]
            for event_id, stage in ordered:
                row = trace.events[event_id]
                kind = PrimitiveKind(int(row["primitive_kind"]))
                stages = self._stages_for(kind)
                module_name = stages[stage]
                module = self.modules[module_name]
                timing = module.timing
                if module_name == "fusion_issue" and stage == 0:
                    if fusion_issued >= self.config.candidate_lanes:
                        module.counters.port_stalls += 1
                        self._record_stall(cycle, module_name, "candidate_width", event_id)
                        heapq.heappush(ready, (event_id, stage))
                        continue
                if not module.accepts_kind(kind):
                    raise CycleConfigurationError(f"{module_name} does not accept {kind.name}")
                fusion_port_name: str | None = None
                if module_name == "fusion_issue" and stage == 0:
                    fusion_port_name, fusion_port_limit = self._fusion_port_limit(kind)
                    if fusion_port_issued.get(fusion_port_name, 0) >= fusion_port_limit:
                        module.counters.port_stalls += 1
                        self._record_stall(cycle, module_name, "port", event_id)
                        heapq.heappush(ready, (event_id, stage))
                        continue
                elif issued_modules.get(module_name, 0) >= timing.ports:
                    module.counters.port_stalls += 1
                    self._record_stall(cycle, module_name, "port", event_id)
                    heapq.heappush(ready, (event_id, stage))
                    continue
                busy_until = (
                    fusion_busy_until[fusion_port_name]
                    if fusion_port_name is not None
                    else module_busy_until[module_name]
                )
                if busy_until > cycle:
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "initiation_interval", event_id)
                    heapq.heappush(ready, (event_id, stage))
                    continue
                if module_inflight[module_name] >= timing.queue_capacity:
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "queue_capacity", event_id)
                    heapq.heappush(ready, (event_id, stage))
                    continue
                if (kind is PrimitiveKind.RELATION_CANDIDATE
                        and relation_seed_inflight >= self.config.relation_seed_fifo_entries):
                    module.counters.queue_stalls += 1
                    self._record_stall(cycle, module_name, "seed_fifo", event_id)
                    heapq.heappush(ready, (event_id, stage))
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
                            lookup = state.request(key, remaining_uses=1)
                        except CacheBackpressure:
                            module.counters.queue_stalls += 1
                            self._record_stall(cycle, module_name, "cache_capacity", event_id)
                            heapq.heappush(ready, (event_id, stage))
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
                    module_busy_until[module_name] = cycle + timing.initiation_interval
                bank_busy[bank_key] = cycle
                module_inflight[module_name] += 1
                if kind is PrimitiveKind.RELATION_CANDIDATE:
                    relation_seed_inflight += 1
                module.counters.accepted += 1
                module.counters.busy_cycles += timing.latency
                issued_modules[module_name] = issued_modules.get(module_name, 0) + 1
                if module_name == "fusion_issue" and stage == 0:
                    fusion_issued += 1
                    assert fusion_port_name is not None
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
                                                   min((module_busy_until[name] for name in self.modules
                                                        if module_busy_until[name] > cycle), default=None),
                                                   min((point for point in fusion_busy_until.values()
                                                        if point > cycle), default=None),
                                                   memory_wakeup)
                               if point is not None and point > cycle]
                if not next_points:
                    blocked = tuple(event_id for event_id, _ in ready[:8])
                    raise CycleConfigurationError(f"deadlock at cycle {cycle}, pending={blocked}")
                cycle = min(next_points)
            else:
                cycle += 1
        module_counters = {name: module.counters.as_dict() for name, module in self.modules.items()}
        if cache_states:
            cache_totals: dict[str, int] = {key: 0 for key in next(iter(cache_states.values())).counters}
            for state in cache_states.values():
                for key, value in state.counters.items():
                    cache_totals[key] += value
            module_counters["semantic_cache"].update(cache_totals)
        module_counters["semantic_cache"]["memory_requests"] = memory_requests
        audit_records = getattr(self.config.memory, "audit_records", None)
        memory_request_records = tuple(audit_records()) if callable(audit_records) else ()
        return CycleResult(
            total_cycles=max(completed.values(), default=0),
            module_counters=module_counters,
            stalls=tuple(self.stalls),
            completion_cycles=completed,
            event_counts={kind.name: int((trace.events["primitive_kind"] == int(kind)).sum())
                          for kind in PrimitiveKind},
            policy=self.policy,
            oracle_status=("heuristic_unproven" if self.policy.endswith("_oracle") else "not_applicable"),
            memory_requests=memory_request_records,
        )
