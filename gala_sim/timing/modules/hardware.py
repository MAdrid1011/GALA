"""Contract-level module implementations used by the discrete-event engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.timing.config import ComputeStage, ComputeTemplateProfile, ModuleTiming

from .base import CounterBlock, ModuleOutput
from .protocol import AcceptResult, EventBatch, ModuleOutputs


@dataclass
class HardwareModule:
    name: str
    timing: ModuleTiming
    counters: CounterBlock
    _wakeup: int | None = field(default=None, init=False, repr=False)

    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return True

    def service_cycles(self) -> int:
        return self.timing.latency

    def bank(self, address_token: int) -> int:
        # Trace addresses for state records are byte addresses aligned to the
        # 64-byte SRAM/DRAM sector.  Preserve the raw-token behavior for the
        # small synthetic logical tokens used by unit fixtures.
        line = address_token // 64 if address_token >= 64 and address_token % 64 == 0 else address_token
        return line % self.timing.banks

    def complete(self, event_id: int, cycle: int) -> ModuleOutput:
        self.counters.completed += 1
        return ModuleOutput(event_id, cycle)

    def next_wakeup(self) -> int | None:
        return self._wakeup

    def accept(self, batch: EventBatch, cycle: int) -> AcceptResult:
        accepted = batch.event_ids[: self.timing.ports]
        rejected = batch.event_ids[self.timing.ports:]
        self.counters.accepted += len(accepted)
        self.counters.port_stalls += len(rejected)
        self._wakeup = cycle + self.timing.initiation_interval if accepted else self._wakeup
        return AcceptResult(accepted, rejected, "port" if rejected else None)

    def advance(self, cycle: int) -> ModuleOutputs:
        if self._wakeup is not None and self._wakeup <= cycle:
            self._wakeup = None
        return ModuleOutputs(())

    def snapshot_counters(self) -> CounterBlock:
        return self.counters


class RelationConstructor(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.RELATION_CANDIDATE, PrimitiveKind.RELATION, PrimitiveKind.QUERY_CLOSE}


class FusionIssueUnit(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT}


class GaussianSemanticCache(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN}


class CacheBackpressure(RuntimeError):
    pass


class CacheLookup(Enum):
    HIT = "hit"
    MISS = "miss"
    MERGED = "merged"


@dataclass
class SemanticCacheState:
    """Directory, active-slot, waiter and release state for one Pod cache."""

    capacity: int
    directory_banks: int
    sector_bytes: int
    active: dict[tuple[int, int], dict[str, int | bool]]
    pending: dict[tuple[int, int], dict[str, int | bool]]
    closing_pending: set[tuple[int, int]]
    counters: dict[str, int]

    @classmethod
    def create(cls, *, capacity: int, directory_banks: int, sector_bytes: int) -> "SemanticCacheState":
        if min(capacity, directory_banks, sector_bytes) <= 0:
            raise ValueError("semantic cache resources must be positive")
        return cls(capacity, directory_banks, sector_bytes, {}, {}, set(), {
            "directory_hits": 0, "directory_misses": 0, "miss_merges": 0,
            "multicast_reads": 0, "fills": 0, "releases": 0,
            "oracle_evictions": 0,
        })

    def request(
        self,
        key: tuple[int, int],
        *,
        remaining_uses: int,
        workset_total_uses: int | None = None,
        resident_remaining_uses: int | None = None,
    ) -> CacheLookup:
        if remaining_uses <= 0:
            raise ValueError("remaining use count must be positive")
        if workset_total_uses is not None and workset_total_uses < remaining_uses:
            raise ValueError("workset total uses cannot be below remaining uses")
        if resident_remaining_uses is not None:
            if workset_total_uses is None:
                raise ValueError("resident remaining uses require a complete workset")
            if not 0 < resident_remaining_uses <= workset_total_uses:
                raise ValueError("resident remaining uses are outside the workset")
        effective_remaining = (
            resident_remaining_uses
            if resident_remaining_uses is not None else remaining_uses
        )
        if key in self.active:
            record = self.active[key]
            record["active_reads"] = int(record["active_reads"]) + 1
            if workset_total_uses is None:
                record["remaining_uses"] = int(record["remaining_uses"]) + 1
            elif (
                resident_remaining_uses is not None
                and int(record["remaining_uses"]) != effective_remaining
            ):
                raise ValueError("semantic workset remaining-use count diverged")
            self.counters["directory_hits"] += 1
            return CacheLookup.HIT
        if key in self.pending:
            pending = self.pending[key]
            pending["waiters"] = int(pending["waiters"]) + 1
            if workset_total_uses is None:
                pending["remaining_uses"] = int(pending["remaining_uses"]) + 1
            self.counters["miss_merges"] += 1
            return CacheLookup.MERGED
        if len(self.active) + len(self.pending) >= self.capacity:
            raise CacheBackpressure("semantic cache has no free slot or miss-merge entry")
        self.pending[key] = {
            "waiters": 1,
            "remaining_uses": (
                effective_remaining
                if resident_remaining_uses is not None
                else (workset_total_uses or 1)
            ),
            "predeclared": workset_total_uses is not None,
        }
        self.counters["directory_misses"] += 1
        return CacheLookup.MISS

    def fill_complete(
        self,
        key: tuple[int, int],
        *,
        remaining_uses: int,
        active_reads: int | None = None,
    ) -> None:
        pending = self.pending.pop(key, None)
        if pending is None:
            raise ValueError("cache fill has no pending miss")
        pending_uses = int(pending["remaining_uses"])
        if remaining_uses != pending_uses:
            raise ValueError("cache fill remaining uses diverged from pending workset")
        self.active[key] = {
            "remaining_uses": pending_uses,
            "active_reads": (
                active_reads if active_reads is not None else int(pending["waiters"])
            ),
            "closing": key in self.closing_pending,
            "predeclared": bool(pending["predeclared"]),
        }
        self.closing_pending.discard(key)
        self.counters["fills"] += 1

    def evict_for_oracle(self, key: tuple[int, int]) -> None:
        """Evict an idle active record without changing its future requests."""

        record = self.active.get(key)
        if record is None:
            raise KeyError(f"semantic cache key {key} is not active")
        if int(record["active_reads"]) != 0:
            raise CacheBackpressure("semantic cache record still has active readers")
        del self.active[key]
        self.counters["oracle_evictions"] += 1

    def begin_multicast(
        self,
        key: tuple[int, int],
        destinations: int,
        *,
        readers_already_active: bool = False,
    ) -> None:
        if destinations <= 0 or key not in self.active:
            raise ValueError("multicast requires an active key and destinations")
        if not readers_already_active:
            self.active[key]["active_reads"] = (
                int(self.active[key]["active_reads"]) + destinations
            )
            if not bool(self.active[key].get("predeclared", False)):
                self.active[key]["remaining_uses"] = (
                    int(self.active[key]["remaining_uses"]) + destinations
                )
        self.counters["multicast_reads"] += 1

    def complete_read(self, key: tuple[int, int]) -> bool:
        if key not in self.active or int(self.active[key]["active_reads"]) <= 0:
            raise ValueError("cache read completion has no active read")
        record = self.active[key]
        record["active_reads"] = int(record["active_reads"]) - 1
        record["remaining_uses"] = int(record["remaining_uses"]) - 1
        if int(record["remaining_uses"]) < 0:
            raise ValueError("cache remaining-use count became negative")
        if (
            bool(record["closing"])
            and int(record["active_reads"]) == 0
            and int(record["remaining_uses"]) == 0
        ):
            del self.active[key]
            self.counters["releases"] += 1
            return True
        return False

    def close(self, key: tuple[int, int]) -> bool:
        if key in self.pending:
            self.closing_pending.add(key)
            return False
        if key not in self.active:
            return False
        self.active[key]["closing"] = True
        if (int(self.active[key]["active_reads"]) == 0
                and int(self.active[key]["remaining_uses"]) == 0):
            del self.active[key]
            self.counters["releases"] += 1
            return True
        return False


@dataclass
class ComputePod(HardwareModule):
    """Reconfigurable Pod front-end with explicit template resource plans.

    The discrete-event engine calls :meth:`can_reserve` before accepting a
    compute event and :meth:`reserve` once it is accepted.  Reservations are
    cycle-granular, so overlapping template stages contend for the same
    cluster resources instead of collapsing into one average module latency.
    """

    template_profiles: dict[int, ComputeTemplateProfile] | None = None
    resource_capacities: dict[str, int] | None = None
    _resource_use: dict[tuple[str, int], int] = field(default_factory=dict, init=False, repr=False)

    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT,
                        PrimitiveKind.GRADIENT_REDUCTION}

    @staticmethod
    def _path_for(kind: PrimitiveKind) -> str:
        return {
            PrimitiveKind.FORWARD: "forward",
            PrimitiveKind.CONSUMER: "consumer",
            PrimitiveKind.ADJOINT: "adjoint",
            PrimitiveKind.GRADIENT_REDUCTION: "gradient_reduction",
        }[kind]

    def profile_for(self, template_id: int, kind: PrimitiveKind) -> ComputeTemplateProfile:
        if self.template_profiles is None:
            raise KeyError("ComputePod template profiles are not configured")
        try:
            profile = self.template_profiles[template_id]
        except KeyError as error:
            raise KeyError(f"no ComputePod profile for template {template_id}") from error
        # Validate the path eagerly at issue time, before any state is changed.
        profile.stages_for(self._path_for(kind))
        return profile

    def stages_for(self, template_id: int, kind: PrimitiveKind) -> tuple[ComputeStage, ...] | None:
        if self.template_profiles is None:
            return None
        return self.profile_for(template_id, kind).stages_for(self._path_for(kind))

    def path_for(self, template_id: int, kind: PrimitiveKind):
        return self.profile_for(template_id, kind).path_for(self._path_for(kind))

    def service_cycles_for(self, template_id: int, kind: PrimitiveKind) -> int:
        stages = self.stages_for(template_id, kind)
        if stages is None:
            # Compatibility is intentionally limited to hand-built fixture
            # CycleConfig objects; production CycleConfig always supplies
            # registered profiles.
            return self.timing.latency
        return sum(stage.latency for stage in stages)

    @staticmethod
    def _demands(stage: ComputeStage) -> tuple[tuple[str, int], ...]:
        return (
            ("fma_groups", stage.fma_groups),
            ("transcendental_lanes", stage.transcendental_lanes),
            ("reduction_trees", stage.reduction_trees),
            ("register_reads", stage.register_reads),
            ("register_writes", stage.register_writes),
            ("feedback_lanes", stage.feedback_lanes),
        )

    def reservation_plan(
        self, template_id: int, kind: PrimitiveKind, start_cycle: int,
    ) -> tuple[tuple[str, int, int], ...]:
        """Return ``(resource, cycle, demand)`` entries for one accepted task."""

        if self.template_profiles is None:
            return ()
        path = self.path_for(template_id, kind)
        stages = path.stages
        clusters = int((self.resource_capacities or {}).get("clusters", 1))
        self._discard_retired(start_cycle)
        fallback: tuple[tuple[str, int, int], ...] = ()
        for cluster in range(clusters):
            plan: list[tuple[str, int, int]] = []
            for point in range(start_cycle, start_cycle + path.cluster_issue_cycles):
                plan.append((
                    f"cluster_issue:{cluster}", point, path.cluster_issue_slots,
                ))
            offset = 0
            for stage in stages:
                point = start_cycle + offset
                for resource, demand in self._demands(stage):
                    if demand:
                        plan.append((f"{resource}:{cluster}", point, demand))
                offset += stage.latency
            # One physical RelationPacket occupies one microcontext.  Its
            # logical lane events share that context and never reserve eight
            # independent slots.
            context_cycles = path.packet_last_result_offset or offset
            for point in range(start_cycle, start_cycle + context_cycles):
                plan.append((f"microcontext_slots:{cluster}", point, 1))
            candidate = tuple(plan)
            fallback = fallback or candidate
            if self._fits(candidate):
                return candidate
        return fallback

    def _discard_retired(self, cycle: int) -> None:
        self._resource_use = {
            key: value for key, value in self._resource_use.items() if key[1] >= cycle
        }

    def _capacity_for(self, resource: str) -> int | None:
        if self.resource_capacities is None:
            return None
        base = resource.partition(":")[0]
        capacity = self.resource_capacities.get(base)
        if capacity is None:
            return None
        clusters = int(self.resource_capacities.get("clusters", 1))
        if ":" in resource:
            if capacity % clusters:
                raise ValueError(f"{base} capacity is not divisible by clusters")
            return capacity // clusters
        return capacity

    def _fits(self, plan: tuple[tuple[str, int, int], ...]) -> bool:
        for resource, point, demand in plan:
            capacity = self._capacity_for(resource)
            if capacity is not None and self._resource_use.get((resource, point), 0) + demand > capacity:
                return False
        return True

    def can_reserve(self, plan: tuple[tuple[str, int, int], ...], cycle: int) -> bool:
        if not plan or self.resource_capacities is None:
            return True
        self._discard_retired(cycle)
        return self._fits(plan)

    def reserve(self, plan: tuple[tuple[str, int, int], ...]) -> None:
        for resource, point, demand in plan:
            key = (resource, point)
            self._resource_use[key] = self._resource_use.get(key, 0) + demand


class BidirectionalQueryUnit(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.QUERY_REDUCTION, PrimitiveKind.CONSUMER,
                        PrimitiveKind.ADJOINT, PrimitiveKind.GRADIENT_REDUCTION}


class ReconstructionUpdateUnit(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {
            PrimitiveKind.UPDATE_BEGIN,
            PrimitiveKind.UPDATE_END,
            PrimitiveKind.UPDATE_COMMIT,
            PrimitiveKind.SET_MODIFICATION,
        }


class SharedSram(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN}
