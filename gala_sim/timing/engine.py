"""Dependency-aware, event-jumping cycle executor."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.trace.model import Trace
from gala_sim.trace.validator import validate_trace

from .config import CycleConfig
from .modules import (
    BidirectionalQueryUnit,
    ComputePod,
    CounterBlock,
    FusionIssueUnit,
    GaussianSemanticCache,
    RelationConstructor,
    ReconstructionUpdateUnit,
    SharedSram,
    StallRecord,
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


@dataclass
class _InFlight:
    completion_cycle: int
    event_id: int
    module: str


class CycleEngine:
    """Run the same trace for Base, Oracle, or mechanism-specific policies."""

    def __init__(self, config: CycleConfig, *, policy: str = "base") -> None:
        if policy not in {"base", "query_oracle", "residency_oracle", "query", "residency", "full"}:
            raise ValueError(f"unknown cycle policy: {policy}")
        self.config = config
        self.policy = policy
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

    def _stages_for(self, kind: PrimitiveKind) -> tuple[str, ...]:
        if kind in {PrimitiveKind.RELATION_CANDIDATE, PrimitiveKind.RELATION, PrimitiveKind.QUERY_CLOSE}:
            return ("relation_constructor",)
        if kind in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN}:
            return ("semantic_cache", "shared_sram")
        if kind in {PrimitiveKind.UPDATE_COMMIT, PrimitiveKind.SET_MODIFICATION}:
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
        if self.policy == "base":
            return candidates
        if self.policy in {"residency", "residency_oracle"}:
            return sorted(candidates, key=lambda event_id: (
                0 if PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN} else 1,
                int(trace.events[event_id]["gaussian_id"]), event_id,
            ))
        if self.policy in {"query", "query_oracle"}:
            return sorted(candidates, key=lambda event_id: (
                int(trace.events[event_id]["query_id"]),
                int(trace.events[event_id]["relation_id"]), event_id,
            ))
        if self.policy == "full":
            return sorted(candidates, key=lambda event_id: (
                0 if PrimitiveKind(int(trace.events[event_id]["primitive_kind"]))
                in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN} else 1,
                int(trace.events[event_id]["query_id"]),
                int(trace.events[event_id]["gaussian_id"]), event_id,
            ))
        return candidates

    def run(self, trace: Trace) -> CycleResult:
        validate_trace(trace)
        if not self.config.modules:
            raise CycleConfigurationError("cycle modules are not configured")
        completed: dict[int, int] = {}
        pending = set(int(item) for item in trace.events["event_id"])
        next_stage: dict[int, int | None] = {
            int(item): 0 for item in trace.events["event_id"]
        }
        in_flight: list[tuple[int, int, int, str]] = []
        module_busy_until = {name: 0 for name in self.modules}
        bank_busy: dict[tuple[str, int, int], int] = {}
        cycle = 0
        while pending or in_flight:
            progressed = False
            while in_flight and in_flight[0][0] <= cycle:
                finish, event_id, stage, module_name = heapq.heappop(in_flight)
                self.modules[module_name].complete(event_id, finish)
                stages = self._stages_for(PrimitiveKind(int(trace.events[event_id]["primitive_kind"])))
                if stage + 1 < len(stages):
                    next_stage[event_id] = stage + 1
                else:
                    completed[event_id] = finish
                    next_stage[event_id] = None
                    pending.remove(event_id)
                progressed = True
            candidates: list[tuple[int, int]] = []
            for event_id in sorted(pending):
                stage = next_stage[event_id]
                if stage is None:
                    continue
                row = trace.events[event_id]
                deps = trace.dependency_ids(row)
                if stage == 0 and not all(int(dep) in completed for dep in deps):
                    continue
                candidates.append((event_id, stage))
            issued = 0
            issued_modules: dict[str, int] = {}
            ordered_ids = self._ordered_candidates(trace, [item[0] for item in candidates])
            ordered = [(event_id, dict(candidates)[event_id]) for event_id in ordered_ids]
            for event_id, stage in ordered:
                if issued >= self.config.candidate_lanes:
                    break
                row = trace.events[event_id]
                kind = PrimitiveKind(int(row["primitive_kind"]))
                stages = self._stages_for(kind)
                module_name = stages[stage]
                module = self.modules[module_name]
                timing = module.timing
                if not module.accepts_kind(kind):
                    raise CycleConfigurationError(f"{module_name} does not accept {kind.name}")
                if issued_modules.get(module_name, 0) >= timing.ports:
                    module.counters.port_stalls += 1
                    self.stalls.append(StallRecord(cycle, module_name, "port", (event_id,)))
                    continue
                if module_busy_until[module_name] > cycle:
                    module.counters.queue_stalls += 1
                    self.stalls.append(StallRecord(cycle, module_name, "initiation_interval", (event_id,)))
                    continue
                bank_key = (module_name, cycle, module.bank(int(row["address_token"])))
                if bank_key in bank_busy:
                    module.counters.bank_conflicts += 1
                    self.stalls.append(StallRecord(cycle, module_name, "bank", (event_id,)))
                    continue
                if int(row["dependency_count"]) > timing.queue_capacity:
                    raise CycleConfigurationError(f"event dependency footprint exceeds {module_name} queue")
                if kind is PrimitiveKind.CACHE_REQUEST and stage == 0:
                    data_bytes = int(row["data_bytes"])
                    if data_bytes <= 0:
                        raise CycleConfigurationError(
                            f"{kind.name} event {event_id} has no explicit transfer size"
                        )
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
                heapq.heappush(in_flight, (completion, event_id, stage, module_name))
                module_busy_until[module_name] = cycle + timing.initiation_interval
                bank_busy[bank_key] = cycle
                module.counters.accepted += 1
                module.counters.busy_cycles += timing.latency
                issued_modules[module_name] = issued_modules.get(module_name, 0) + 1
                next_stage[event_id] = None
                issued += 1
                progressed = True
            if not progressed:
                next_points = [point for point in (in_flight[0][0] if in_flight else None,
                                                   min((module_busy_until[name] for name in self.modules
                                                        if module_busy_until[name] > cycle), default=None))
                               if point is not None and point > cycle]
                if not next_points:
                    blocked = tuple(sorted(pending))[:8]
                    raise CycleConfigurationError(f"deadlock at cycle {cycle}, pending={blocked}")
                cycle = min(next_points)
            else:
                cycle += 1
        return CycleResult(
            total_cycles=max(completed.values(), default=0),
            module_counters={name: module.counters.as_dict() for name, module in self.modules.items()},
            stalls=tuple(self.stalls),
            completion_cycles=completed,
            event_counts={kind.name: int((trace.events["primitive_kind"] == int(kind)).sum())
                          for kind in PrimitiveKind},
            policy=self.policy,
            oracle_status=("heuristic_unproven" if self.policy.endswith("_oracle") else "not_applicable"),
        )
