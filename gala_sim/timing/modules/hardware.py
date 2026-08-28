"""Contract-level module implementations used by the discrete-event engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.timing.config import ModuleTiming

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
        })

    def request(
        self,
        key: tuple[int, int],
        *,
        remaining_uses: int,
        workset_total_uses: int | None = None,
    ) -> CacheLookup:
        if remaining_uses <= 0:
            raise ValueError("remaining use count must be positive")
        if workset_total_uses is not None and workset_total_uses < remaining_uses:
            raise ValueError("workset total uses cannot be below remaining uses")
        if key in self.active:
            record = self.active[key]
            record["active_reads"] = int(record["active_reads"]) + 1
            if workset_total_uses is None:
                record["remaining_uses"] = int(record["remaining_uses"]) + 1
            elif int(record["remaining_uses"]) != remaining_uses:
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
            "remaining_uses": workset_total_uses or 1,
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

    def begin_multicast(self, key: tuple[int, int], destinations: int) -> None:
        if destinations <= 0 or key not in self.active:
            raise ValueError("multicast requires an active key and destinations")
        self.active[key]["active_reads"] = int(self.active[key]["active_reads"]) + destinations
        if not bool(self.active[key].get("predeclared", False)):
            self.active[key]["remaining_uses"] = (
                int(self.active[key]["remaining_uses"]) + destinations
            )
        self.counters["multicast_reads"] += 1

    def complete_read(self, key: tuple[int, int]) -> None:
        if key not in self.active or int(self.active[key]["active_reads"]) <= 0:
            raise ValueError("cache read completion has no active read")
        record = self.active[key]
        record["active_reads"] = int(record["active_reads"]) - 1
        record["remaining_uses"] = int(record["remaining_uses"]) - 1
        if int(record["remaining_uses"]) < 0:
            raise ValueError("cache remaining-use count became negative")

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


class ComputePod(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT,
                        PrimitiveKind.GRADIENT_REDUCTION}


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
