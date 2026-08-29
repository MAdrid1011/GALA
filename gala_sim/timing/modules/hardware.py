"""Contract-level module implementations used by the discrete-event engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.timing.config import ComputeStage, ComputeTemplateProfile, ModuleTiming
from gala_sim.timing.packets import RelationWindowDescriptor

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
            "replacement_evictions": 0, "oracle_evictions": 0,
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
        if (
            workset_total_uses is None
            and len(self.active) + len(self.pending) >= self.capacity
        ):
            self.evict_oldest_idle()
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

    def evict_oldest_idle(self) -> tuple[int, int] | None:
        """Evict the oldest completed entry when no semantic workset is available."""

        for key, record in self.active.items():
            if (
                int(record["active_reads"]) == 0
                and int(record["remaining_uses"]) == 0
            ):
                del self.active[key]
                self.counters["replacement_evictions"] += 1
                return key
        return None

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
    _retirement_buckets: dict[int, set[tuple[str, int]]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _retirement_cycles: list[int] = field(
        default_factory=list, init=False, repr=False,
    )
    _next_cluster_by_pod: dict[int, int] = field(default_factory=dict, init=False, repr=False)

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
        self,
        template_id: int,
        kind: PrimitiveKind,
        start_cycle: int,
        *,
        pod: int | None = None,
        cluster_hint: int | None = None,
    ) -> tuple[tuple[str, int, int], ...]:
        """Return ``(resource, cycle, demand)`` entries for one accepted task."""

        if self.template_profiles is None:
            return ()
        path = self.path_for(template_id, kind)
        stages = path.stages
        clusters = int((self.resource_capacities or {}).get("clusters", 1))
        pods = int((self.resource_capacities or {}).get("pods", 1))
        clusters_per_pod = int(
            (self.resource_capacities or {}).get("clusters_per_pod", clusters)
        )
        if pods * clusters_per_pod != clusters:
            raise ValueError("ComputePod topology does not close its cluster count")
        if pod is not None and not 0 <= pod < pods:
            raise ValueError("ComputePod route names an invalid Pod")
        if cluster_hint is not None and not 0 <= cluster_hint < clusters:
            raise ValueError("ComputePod route names an invalid cluster")
        if cluster_hint is not None:
            if pod is not None and cluster_hint // clusters_per_pod != pod:
                raise ValueError("ComputePod owner cluster is outside its Pod")
            candidate_clusters = (cluster_hint,)
        elif pod is not None:
            begin = pod * clusters_per_pod
            next_local = self._next_cluster_by_pod.get(pod, 0)
            candidate_clusters = tuple(
                begin + (next_local + offset) % clusters_per_pod
                for offset in range(clusters_per_pod)
            )
        else:
            candidate_clusters = tuple(range(clusters))
        self._discard_retired(start_cycle)
        fallback: tuple[tuple[str, int, int], ...] = ()
        for cluster in candidate_clusters:
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
            # One physical RelationPacket occupies one microcontext while its
            # input lanes are admitted.  The downstream pipeline tokens retain
            # work after admission and do not keep the input context live.
            for point in range(
                start_cycle, start_cycle + path.cluster_issue_cycles,
            ):
                plan.append((f"microcontext_slots:{cluster}", point, 1))
            candidate = tuple(plan)
            fallback = fallback or candidate
            if self._fits(candidate):
                return candidate
        return fallback

    def _discard_retired(self, cycle: int) -> None:
        # Reservations are issued in nondecreasing cycle order.  Retire by
        # point buckets so each resource entry is removed once, instead of
        # scanning the complete in-flight table on every admission attempt.
        while self._retirement_cycles and self._retirement_cycles[0] <= cycle:
            retirement_cycle = heapq.heappop(self._retirement_cycles)
            keys = self._retirement_buckets.pop(retirement_cycle, ())
            for key in keys:
                self._resource_use.pop(key, None)

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

    def first_blocking_resource(
        self, plan: tuple[tuple[str, int, int], ...], cycle: int,
    ) -> tuple[str, int, int, int, int] | None:
        """Return resource, point, in-use, demand, and capacity for first conflict."""

        self._discard_retired(cycle)
        for resource, point, demand in plan:
            capacity = self._capacity_for(resource)
            in_use = self._resource_use.get((resource, point), 0)
            if capacity is not None and in_use + demand > capacity:
                return resource, point, in_use, demand, capacity
        return None

    def reserve(self, plan: tuple[tuple[str, int, int], ...]) -> None:
        reserved_cluster: int | None = None
        for resource, point, demand in plan:
            key = (resource, point)
            if key not in self._resource_use:
                retirement_cycle = point + 1
                bucket = self._retirement_buckets.setdefault(retirement_cycle, set())
                if not bucket:
                    heapq.heappush(self._retirement_cycles, retirement_cycle)
                bucket.add(key)
            self._resource_use[key] = self._resource_use.get(key, 0) + demand
            if reserved_cluster is None and resource.startswith("cluster_issue:"):
                reserved_cluster = int(resource.partition(":")[2])
        if reserved_cluster is not None and self.resource_capacities is not None:
            clusters_per_pod = int(
                self.resource_capacities.get(
                    "clusters_per_pod",
                    self.resource_capacities.get("clusters", 1),
                )
            )
            pod = reserved_cluster // clusters_per_pod
            self._next_cluster_by_pod[pod] = (
                reserved_cluster % clusters_per_pod + 1
            ) % clusters_per_pod


class BidirectionalQueryUnit(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {
            PrimitiveKind.FORWARD,
            PrimitiveKind.QUERY_REDUCTION,
            PrimitiveKind.CONSUMER,
            PrimitiveKind.ADJOINT,
        }


@dataclass
class _LiveRelationWindow:
    producer_events: set[int]
    forward_events: set[int]
    consumer_events: set[int]
    adjoint_events: set[int]
    relation_stage_heads: set[int]
    sealed: bool = True
    appended_relation_stages: set[int] = field(default_factory=set)
    released_relation_stages: set[int] = field(default_factory=set)
    relation_release_events: dict[int, set[int]] = field(default_factory=dict)
    release_stage_by_event: dict[int, int] = field(default_factory=dict)


@dataclass
class RelationWindowTracker:
    """Shared relation-window table, record store, and append-bank state."""

    window_capacity: int
    relation_capacity: int
    relation_banks: int
    descriptors: dict[int, RelationWindowDescriptor] = field(default_factory=dict)
    event_to_window: dict[int, int] = field(default_factory=dict)
    live: dict[int, _LiveRelationWindow] = field(default_factory=dict)
    relation_records_live: int = 0
    peak_windows: int = 0
    peak_relation_records: int = 0
    allocated_windows: int = 0
    released_windows: int = 0
    appended_relation_records: int = 0
    _append_bank_cycle: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if min(self.window_capacity, self.relation_capacity, self.relation_banks) <= 0:
            raise ValueError("relation-window resources must be positive")

    def _older_window_reservation_blocks(
        self, window_id: int, state: _LiveRelationWindow | None,
    ) -> bool:
        if state is None:
            return False
        older_states = tuple(
            older_state for older_id, older_state in self.live.items()
            if older_id < window_id
        )
        if not older_states:
            return False
        reserved_records = sum(
            len(older_state.relation_stage_heads)
            - len(older_state.appended_relation_stages)
            for older_state in older_states
        )
        return self.relation_records_live + reserved_records >= self.relation_capacity

    def register(self, descriptor: RelationWindowDescriptor) -> None:
        self.register_fragment(descriptor)

    def register_fragment(self, descriptor: RelationWindowDescriptor) -> None:
        """Register one future-reference fragment of a streamed window."""

        if descriptor.relation_records > self.relation_capacity:
            raise ValueError(
                "relation window exceeds the physical relation-store capacity; "
                "capture must provide capacity-continuation windows"
            )
        if not descriptor.event_ids:
            raise ValueError("relation window has no reference events")
        release_map = descriptor.relation_record_release_events
        if release_map:
            relation_heads = set(descriptor.relation_stage_heads)
            if set(release_map) != relation_heads:
                raise ValueError(
                    "relation record release map must cover every physical relation stage"
                )
            release_events = [
                event_id for event_ids in release_map.values() for event_id in event_ids
            ]
            if len(set(release_events)) != len(release_events):
                raise ValueError("relation record release event belongs to multiple stages")
            if not set(release_events) <= set(descriptor.adjoint_event_ids):
                raise ValueError("relation record release map contains a non-adjoint event")
        previous = self.descriptors.get(descriptor.window_id)
        if previous is not None and previous.sealed:
            raise ValueError("sealed relation window receives another fragment")
        for event_id in descriptor.event_ids:
            if event_id in self.event_to_window:
                raise ValueError("event belongs to multiple relation windows")
            self.event_to_window[event_id] = descriptor.window_id
        if previous is None:
            merged = descriptor
        else:
            merged_release = dict(previous.relation_record_release_events)
            if set(merged_release).intersection(release_map):
                raise ValueError("relation window fragment repeats a relation stage")
            merged_release.update(release_map)
            merged = RelationWindowDescriptor(
                window_id=descriptor.window_id,
                relation_stage_heads=(
                    previous.relation_stage_heads | descriptor.relation_stage_heads
                ),
                producer_event_ids=(
                    previous.producer_event_ids | descriptor.producer_event_ids
                ),
                forward_stage_heads=(
                    previous.forward_stage_heads | descriptor.forward_stage_heads
                ),
                consumer_event_ids=(
                    previous.consumer_event_ids | descriptor.consumer_event_ids
                ),
                adjoint_event_ids=(
                    previous.adjoint_event_ids | descriptor.adjoint_event_ids
                ),
                relation_record_release_events=merged_release,
                sealed=descriptor.sealed,
            )
            if merged.relation_records > self.relation_capacity:
                raise ValueError(
                    "relation window exceeds the physical relation-store capacity; "
                    "capture must provide capacity-continuation windows"
                )
        self.descriptors[descriptor.window_id] = merged
        state = self.live.get(descriptor.window_id)
        if state is not None:
            state.producer_events.update(descriptor.producer_event_ids)
            state.forward_events.update(descriptor.forward_stage_heads)
            state.consumer_events.update(descriptor.consumer_event_ids)
            state.adjoint_events.update(descriptor.adjoint_event_ids)
            state.relation_stage_heads.update(descriptor.relation_stage_heads)
            state.relation_release_events.update({
                int(stage_head): set(event_ids)
                for stage_head, event_ids in release_map.items()
            })
            state.release_stage_by_event.update({
                int(event_id): int(stage_head)
                for stage_head, event_ids in release_map.items()
                for event_id in event_ids
            })
            state.sealed = descriptor.sealed

    def blocking_reason(
        self,
        event_id: int,
        kind: PrimitiveKind,
        *,
        physical_stage_head: bool,
        cycle: int,
    ) -> str | None:
        window_id = self.event_to_window.get(event_id)
        if window_id is None:
            return None
        descriptor = self.descriptors[window_id]
        if window_id not in self.live:
            if event_id not in descriptor.producer_event_ids:
                raise ValueError("relation-window consumer became ready without a live producer")
            if len(self.live) >= self.window_capacity:
                return "relation_window_capacity"
        if kind is not PrimitiveKind.RELATION or not physical_stage_head:
            return None
        state = self.live.get(window_id)
        already_appended = (
            state is not None and event_id in state.appended_relation_stages
        )
        if already_appended:
            return None
        if self.relation_records_live >= self.relation_capacity:
            return "relation_store_capacity"
        if self._older_window_reservation_blocks(window_id, state):
            return "older_window_record_reservation"
        ordinal = tuple(sorted(descriptor.relation_stage_heads)).index(event_id)
        bank = ordinal % self.relation_banks
        if self._append_bank_cycle.get(bank) == cycle:
            return "relation_store_bank"
        return None

    def full_stage_blocking_reason(
        self, event_id: int, kind: PrimitiveKind,
    ) -> str | None:
        """Apply the A-disabled window barriers without changing dependencies."""

        window_id = self.event_to_window.get(event_id)
        if window_id is None:
            return None
        if kind not in {PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT}:
            return None
        state = self.live.get(window_id)
        if state is None:
            raise ValueError("relation-window task became ready without a live window")
        if not state.sealed:
            return "base_window_unsealed"
        if kind is PrimitiveKind.CONSUMER and state.forward_events:
            return "base_forward_stage"
        if kind is PrimitiveKind.ADJOINT and state.forward_events:
            return "base_forward_stage"
        if kind is PrimitiveKind.ADJOINT and state.consumer_events:
            return "base_consumer_stage"
        return None

    def issue(
        self,
        event_id: int,
        kind: PrimitiveKind,
        *,
        physical_stage_head: bool,
        cycle: int,
    ) -> None:
        window_id = self.event_to_window.get(event_id)
        if window_id is None:
            return
        descriptor = self.descriptors[window_id]
        state = self.live.get(window_id)
        if state is None:
            if event_id not in descriptor.producer_event_ids:
                raise ValueError("relation-window reference issues before allocation")
            if len(self.live) >= self.window_capacity:
                raise ValueError("relation window committed without a free table entry")
            state = _LiveRelationWindow(
                set(descriptor.producer_event_ids),
                set(descriptor.forward_stage_heads),
                set(descriptor.consumer_event_ids),
                set(descriptor.adjoint_event_ids),
                set(descriptor.relation_stage_heads),
                descriptor.sealed,
                relation_release_events={
                    int(stage_head): set(event_ids)
                    for stage_head, event_ids in
                    descriptor.relation_record_release_events.items()
                },
                release_stage_by_event={
                    int(event_id): int(stage_head)
                    for stage_head, event_ids in
                    descriptor.relation_record_release_events.items()
                    for event_id in event_ids
                },
            )
            self.live[window_id] = state
            self.allocated_windows += 1
            self.peak_windows = max(self.peak_windows, len(self.live))
        if kind is PrimitiveKind.RELATION and physical_stage_head:
            if event_id in state.appended_relation_stages:
                raise ValueError("relation record is appended twice")
            if self.relation_records_live >= self.relation_capacity:
                raise ValueError("relation record committed without store capacity")
            ordinal = tuple(sorted(descriptor.relation_stage_heads)).index(event_id)
            bank = ordinal % self.relation_banks
            if self._append_bank_cycle.get(bank) == cycle:
                raise ValueError("relation record committed through a busy append bank")
            state.appended_relation_stages.add(event_id)
            self._append_bank_cycle[bank] = cycle
            self.relation_records_live += 1
            self.appended_relation_records += 1
            self.peak_relation_records = max(
                self.peak_relation_records, self.relation_records_live
            )

    def complete_event(self, event_id: int) -> None:
        window_id = self.event_to_window.get(event_id)
        if window_id is None:
            return
        state = self.live.get(window_id)
        if state is None:
            raise ValueError("relation-window reference retires without a live window")
        reference_sets = (
            state.producer_events,
            state.forward_events,
            state.consumer_events,
            state.adjoint_events,
        )
        matches = [references for references in reference_sets if event_id in references]
        if len(matches) != 1:
            raise ValueError("relation-window event has no unique live reference")
        matches[0].remove(event_id)
        # Physical relation records have an independent lifetime from the
        # window table entry.  Reclaim each packet as soon as all of its
        # adjoint lanes retire; descriptors without this map retain the
        # historical conservative whole-window lifetime.
        if matches[0] is state.adjoint_events and state.relation_release_events:
            stage_head = state.release_stage_by_event.get(event_id)
            if stage_head is not None:
                remaining = state.relation_release_events[stage_head]
                remaining.remove(event_id)
                if (
                    not remaining
                    and stage_head in state.appended_relation_stages
                    and stage_head not in state.released_relation_stages
                ):
                    state.released_relation_stages.add(stage_head)
                    self.relation_records_live -= 1
                    if self.relation_records_live < 0:
                        raise ValueError("relation-store occupancy underflows")
        if any(reference_sets) or not state.sealed:
            return
        if state.appended_relation_stages != set(state.relation_stage_heads):
            raise ValueError("relation window releases before all records were appended")
        if state.relation_release_events:
            if state.released_relation_stages != state.appended_relation_stages:
                raise ValueError("relation window releases before all records retired")
        else:
            self.relation_records_live -= len(state.appended_relation_stages)
            if self.relation_records_live < 0:
                raise ValueError("relation-store occupancy underflows")
        del self.live[window_id]
        descriptor = self.descriptors.pop(window_id)
        for reference_event in descriptor.event_ids:
            self.event_to_window.pop(reference_event, None)
        self.released_windows += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "window_allocations": self.allocated_windows,
            "window_releases": self.released_windows,
            "window_peak_occupancy": self.peak_windows,
            "relation_records_appended": self.appended_relation_records,
            "relation_store_peak_records": self.peak_relation_records,
            "relation_windows_live": len(self.live),
            "relation_records_live": self.relation_records_live,
        }

    def live_reference_snapshot(self) -> dict[int, dict[str, int]]:
        """Expose per-window live-reference counts for deadlock diagnostics."""

        return {
            window_id: {
                "producer": len(state.producer_events),
                "forward": len(state.forward_events),
                "consumer": len(state.consumer_events),
                "adjoint": len(state.adjoint_events),
                "records": len(state.appended_relation_stages),
            }
            for window_id, state in sorted(self.live.items())
        }

    def has_append_bank_reservation(self, cycle: int) -> bool:
        return cycle in self._append_bank_cycle.values()


@dataclass
class QueryReplayTracker:
    """Consumer-input and adjoint-replay occupancy shared by both replay paths."""

    capacity: int
    stage_gated: bool = False
    remaining_adjoint_by_query: dict[tuple[int, int], int] = field(default_factory=dict)
    adjoint_query_by_event: dict[int, tuple[int, int]] = field(default_factory=dict)
    consumer_query_by_event: dict[int, tuple[int, int]] = field(default_factory=dict)
    ready_queries: set[tuple[int, int]] = field(default_factory=set)
    active_queries: set[tuple[int, int]] = field(default_factory=set)
    peak_entries: int = 0
    reservations: int = 0
    releases: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("replay queue capacity must be positive")

    def register_rows(self, rows) -> None:
        adjoint_counts: dict[tuple[int, int], int] = {}
        for row in rows:
            kind = PrimitiveKind(int(row["primitive_kind"]))
            event_id = int(row["event_id"])
            query_id = int(row["query_id"])
            query_key = (int(row["iteration_id"]), query_id)
            if kind is PrimitiveKind.ADJOINT:
                if query_id < 0 or event_id in self.adjoint_query_by_event:
                    raise ValueError("adjoint replay identity is invalid or duplicated")
                self.adjoint_query_by_event[event_id] = query_key
                adjoint_counts[query_key] = adjoint_counts.get(query_key, 0) + 1
            elif kind is PrimitiveKind.CONSUMER:
                if query_id < 0 or event_id in self.consumer_query_by_event:
                    raise ValueError("consumer replay identity is invalid or duplicated")
                if query_key in self.consumer_query_by_event.values():
                    raise ValueError("query has more than one consumer replay input")
                self.consumer_query_by_event[event_id] = query_key
        for query_key, count in adjoint_counts.items():
            self.remaining_adjoint_by_query[query_key] = (
                self.remaining_adjoint_by_query.get(query_key, 0) + count
            )

    def blocks_consumer(self, event_id: int) -> bool:
        query_key = self.consumer_query_by_event.get(event_id)
        return bool(
            not self.stage_gated
            and
            query_key is not None
            and self.remaining_adjoint_by_query.get(query_key, 0) > 0
            and query_key not in self.active_queries
            and len(self.active_queries) >= self.capacity
        )

    def requires_consumer_slot(self, event_id: int) -> bool:
        query_key = self.consumer_query_by_event.get(event_id)
        return bool(
            query_key is not None
            and self.remaining_adjoint_by_query.get(query_key, 0) > 0
            and query_key not in self.active_queries
        )

    def reserve_consumer(self, event_id: int) -> None:
        query_key = self.consumer_query_by_event.pop(event_id, None)
        if query_key is None or self.remaining_adjoint_by_query.get(query_key, 0) == 0:
            return
        if self.stage_gated:
            if query_key in self.ready_queries:
                raise ValueError("consumer replay input is reserved twice")
            self.ready_queries.add(query_key)
            return
        if query_key in self.active_queries:
            raise ValueError("consumer replay input is reserved twice")
        if len(self.active_queries) >= self.capacity:
            raise ValueError("consumer replay input commits without queue capacity")
        self.active_queries.add(query_key)
        self.reservations += 1
        self.peak_entries = max(self.peak_entries, len(self.active_queries))

    def _adjoint_query_keys(
        self, event_ids: tuple[int, ...],
    ) -> tuple[tuple[int, int], ...]:
        keys: list[tuple[int, int]] = []
        for event_id in event_ids:
            query_key = self.adjoint_query_by_event.get(event_id)
            if query_key is None:
                raise ValueError("adjoint dispatch has no registered replay entry")
            if query_key not in keys:
                keys.append(query_key)
        return tuple(keys)

    def blocks_adjoint(self, event_ids: tuple[int, ...]) -> bool:
        if not self.stage_gated:
            return False
        query_keys = self._adjoint_query_keys(event_ids)
        new_keys = tuple(
            query_key for query_key in query_keys
            if query_key not in self.active_queries
        )
        # A stage-gated replay may expose an adjoint before its consumer has
        # reserved the query-volume entry.  Keep that physical packet in the
        # ready queue until the consumer is actually ready; reserve_adjoint()
        # treats this as an invariant rather than a recoverable stall.
        if any(query_key not in self.ready_queries for query_key in new_keys):
            return True
        needed = len(new_keys)
        return len(self.active_queries) + needed > self.capacity

    def reserve_adjoint(self, event_ids: tuple[int, ...]) -> None:
        query_keys = self._adjoint_query_keys(event_ids)
        if not self.stage_gated:
            if any(query_key not in self.active_queries for query_key in query_keys):
                raise ValueError("adjoint dispatch has no active replay input")
            return
        new_keys = tuple(
            query_key for query_key in query_keys
            if query_key not in self.active_queries
        )
        if any(query_key not in self.ready_queries for query_key in new_keys):
            raise ValueError("adjoint replay input is not consumer-ready")
        if len(self.active_queries) + len(new_keys) > self.capacity:
            raise ValueError("adjoint replay input commits without queue capacity")
        self.active_queries.update(new_keys)
        self.reservations += len(new_keys)
        self.peak_entries = max(self.peak_entries, len(self.active_queries))

    def dispatch_adjoint(self, event_ids: tuple[int, ...]) -> None:
        completed_queries: set[tuple[int, int]] = set()
        for event_id in event_ids:
            query_key = self.adjoint_query_by_event.pop(event_id, None)
            if query_key is None:
                raise ValueError("adjoint dispatch has no registered replay entry")
            if query_key not in self.active_queries:
                raise ValueError("adjoint dispatch has no active replay input")
            remaining = self.remaining_adjoint_by_query[query_key] - 1
            if remaining < 0:
                raise ValueError("adjoint replay count underflows")
            self.remaining_adjoint_by_query[query_key] = remaining
            if remaining == 0:
                completed_queries.add(query_key)
        for query_key in completed_queries:
            self.active_queries.remove(query_key)
            self.ready_queries.discard(query_key)
            del self.remaining_adjoint_by_query[query_key]
            self.releases += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "replay_queue_reservations": self.reservations,
            "replay_queue_releases": self.releases,
            "replay_queue_peak_entries": self.peak_entries,
            "replay_queue_live_entries": len(self.active_queries),
        }


@dataclass
class OwnerGradientTracker:
    """Two-entry gradient-epoch reservation table in every owner cluster."""

    pods: int
    clusters_per_pod: int
    slots_per_cluster: int
    remaining_gradient_by_key: dict[tuple[int, int, int], int] = field(default_factory=dict)
    active_relations_by_key: dict[tuple[int, int, int], set[int]] = field(
        default_factory=dict
    )
    key_by_adjoint_event: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    key_by_gradient_event: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    relation_by_adjoint_event: dict[int, int] = field(default_factory=dict)
    relation_by_gradient_event: dict[int, int] = field(default_factory=dict)
    adjoint_event_by_relation: dict[int, int] = field(default_factory=dict)
    gradient_event_by_relation: dict[int, int] = field(default_factory=dict)
    key_by_relation: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    active_by_cluster: dict[int, set[tuple[int, int, int]]] = field(
        default_factory=dict
    )
    reservations: int = 0
    releases: int = 0
    peak_slots_per_cluster: int = 0

    def __post_init__(self) -> None:
        if min(self.pods, self.clusters_per_pod, self.slots_per_cluster) <= 0:
            raise ValueError("owner-gradient resources must be positive")

    @staticmethod
    def _key(row) -> tuple[int, int, int]:
        return (
            int(row["iteration_id"]),
            int(row["gaussian_id"]),
            int(row["state_version"]),
        )

    def _cluster(self, key: tuple[int, int, int]) -> int:
        gaussian_id = key[1]
        if gaussian_id < 0:
            raise ValueError("owner-gradient event has no Gaussian identity")
        pod = gaussian_id % self.pods
        owner = gaussian_id % self.clusters_per_pod
        return pod * self.clusters_per_pod + owner

    def register_rows(self, rows) -> None:
        for row in rows:
            kind = PrimitiveKind(int(row["primitive_kind"]))
            if kind not in {
                PrimitiveKind.ADJOINT, PrimitiveKind.GRADIENT_REDUCTION,
            }:
                continue
            event_id = int(row["event_id"])
            key = self._key(row)
            relation_id = int(row["relation_id"])
            if relation_id < 0:
                raise ValueError("owner-gradient event has no relation identity")
            existing_key = self.key_by_relation.setdefault(relation_id, key)
            if existing_key != key:
                raise ValueError("owner-gradient relation changes Gaussian epoch")
            if kind is PrimitiveKind.ADJOINT:
                if event_id in self.key_by_adjoint_event:
                    raise ValueError("owner-gradient adjoint event is registered twice")
                self.key_by_adjoint_event[event_id] = key
                self.relation_by_adjoint_event[event_id] = relation_id
                if relation_id in self.adjoint_event_by_relation:
                    raise ValueError("owner-gradient relation has duplicate adjoints")
                self.adjoint_event_by_relation[relation_id] = event_id
            else:
                if event_id in self.key_by_gradient_event:
                    raise ValueError("owner-gradient reduction event is registered twice")
                if relation_id in self.gradient_event_by_relation:
                    raise ValueError("owner-gradient relation has duplicate reductions")
                self.key_by_gradient_event[event_id] = key
                self.relation_by_gradient_event[event_id] = relation_id
                self.gradient_event_by_relation[relation_id] = event_id

    def key_for_adjoint(self, event_id: int) -> tuple[int, int, int]:
        try:
            return self.key_by_adjoint_event[event_id]
        except KeyError as error:
            raise ValueError("adjoint event lacks an owner-gradient identity") from error

    def cluster_for_key(self, key: tuple[int, int, int]) -> int:
        return self._cluster(key)

    def deadlock_snapshot(
        self,
        *,
        pending_adjoint_event_ids: tuple[int, ...],
        remaining_dependencies,
    ) -> dict[str, object]:
        """Describe live epochs and the reductions that can release them."""

        active = {
            cluster: tuple(sorted(keys))
            for cluster, keys in sorted(self.active_by_cluster.items())
        }
        pending: list[dict[str, object]] = []
        for event_id in pending_adjoint_event_ids[:16]:
            key = self.key_for_adjoint(event_id)
            cluster = self._cluster(key)
            pending.append({
                "event_id": event_id,
                "key": key,
                "cluster": cluster,
                "reuses_active_epoch": key in self.active_by_cluster.get(cluster, set()),
            })
        reductions: list[dict[str, object]] = []
        for cluster, keys in active.items():
            for key in keys:
                events = tuple(
                    self.gradient_event_by_relation[relation_id]
                    for relation_id in sorted(self.active_relations_by_key[key])
                    if relation_id in self.gradient_event_by_relation
                )
                ready_count = sum(
                    int(remaining_dependencies[event_id]) == 0
                    for event_id in events
                )
                reductions.append({
                    "key": key,
                    "cluster": cluster,
                    "event_count": len(events),
                    "ready_count": ready_count,
                    "event_sample": tuple(
                        (event_id, int(remaining_dependencies[event_id]))
                        for event_id in events[:16]
                    ),
                })
        return {
            "active_by_cluster": active,
            "pending_adjoint_count": len(pending_adjoint_event_ids),
            "pending_adjoint_sample": tuple(pending),
            "active_gradient_reductions": tuple(reductions),
        }

    def blocks_adjoint(self, event_ids: tuple[int, ...]) -> bool:
        additions: dict[int, set[tuple[int, int, int]]] = {}
        for event_id in event_ids:
            key = self.key_by_adjoint_event.get(event_id)
            if key is None:
                raise ValueError("adjoint event lacks an owner-gradient identity")
            cluster = self._cluster(key)
            if key not in self.active_by_cluster.get(cluster, set()):
                additions.setdefault(cluster, set()).add(key)
        return any(
            len(self.active_by_cluster.get(cluster, set())) + len(keys)
            > self.slots_per_cluster
            for cluster, keys in additions.items()
        )

    def reserve_adjoint(self, event_ids: tuple[int, ...]) -> None:
        if self.blocks_adjoint(event_ids):
            raise ValueError("adjoint commits without an owner-gradient epoch slot")
        for event_id in event_ids:
            key = self.key_by_adjoint_event[event_id]
            relation_id = self.relation_by_adjoint_event[event_id]
            cluster = self._cluster(key)
            active = self.active_by_cluster.setdefault(cluster, set())
            if key not in active:
                active.add(key)
                self.reservations += 1
                self.peak_slots_per_cluster = max(
                    self.peak_slots_per_cluster, len(active)
                )
            relations = self.active_relations_by_key.setdefault(key, set())
            relations.add(relation_id)
            self.remaining_gradient_by_key[key] = len(relations)

    def complete_gradient(self, event_ids: tuple[int, ...]) -> None:
        completed: set[tuple[int, tuple[int, int, int]]] = set()
        for event_id in event_ids:
            key = self.key_by_gradient_event.pop(event_id, None)
            if key is None:
                raise ValueError("gradient reduction lacks an owner-gradient identity")
            cluster = self._cluster(key)
            if key not in self.active_by_cluster.get(cluster, set()):
                raise ValueError("gradient reduction has no reserved owner-gradient slot")
            relation_id = self.relation_by_gradient_event.pop(event_id)
            adjoint_event = self.adjoint_event_by_relation.pop(relation_id, None)
            if adjoint_event is not None:
                self.key_by_adjoint_event.pop(adjoint_event, None)
                self.relation_by_adjoint_event.pop(adjoint_event, None)
            self.gradient_event_by_relation.pop(relation_id, None)
            self.key_by_relation.pop(relation_id, None)
            relations = self.active_relations_by_key[key]
            if relation_id not in relations:
                raise ValueError("gradient reduction has no in-flight adjoint relation")
            relations.remove(relation_id)
            self.remaining_gradient_by_key[key] = len(relations)
            if not relations:
                completed.add((cluster, key))
        for cluster, key in completed:
            del self.active_relations_by_key[key]
            del self.remaining_gradient_by_key[key]
            self.active_by_cluster[cluster].remove(key)
            if not self.active_by_cluster[cluster]:
                del self.active_by_cluster[cluster]
            self.releases += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "owner_gradient_slot_reservations": self.reservations,
            "owner_gradient_slot_releases": self.releases,
            "owner_gradient_peak_slots_per_cluster": self.peak_slots_per_cluster,
            "owner_gradient_live_slots": sum(
                len(active) for active in self.active_by_cluster.values()
            ),
        }


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
