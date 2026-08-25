"""Contract-level module implementations used by the discrete-event engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.timing.config import ModuleTiming

from .base import CounterBlock, ModuleOutput


@dataclass
class HardwareModule:
    name: str
    timing: ModuleTiming
    counters: CounterBlock

    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return True

    def service_cycles(self) -> int:
        return self.timing.latency

    def bank(self, address_token: int) -> int:
        return address_token % self.timing.banks

    def complete(self, event_id: int, cycle: int) -> ModuleOutput:
        self.counters.completed += 1
        return ModuleOutput(event_id, cycle)


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
    pending: dict[tuple[int, int], int]
    counters: dict[str, int]

    @classmethod
    def create(cls, *, capacity: int, directory_banks: int, sector_bytes: int) -> "SemanticCacheState":
        if min(capacity, directory_banks, sector_bytes) <= 0:
            raise ValueError("semantic cache resources must be positive")
        return cls(capacity, directory_banks, sector_bytes, {}, {}, {
            "directory_hits": 0, "directory_misses": 0, "miss_merges": 0,
            "multicast_reads": 0, "fills": 0, "releases": 0,
        })

    def request(self, key: tuple[int, int], *, remaining_uses: int) -> CacheLookup:
        if remaining_uses <= 0:
            raise ValueError("remaining use count must be positive")
        if key in self.active:
            record = self.active[key]
            record["active_reads"] = int(record["active_reads"]) + 1
            self.counters["directory_hits"] += 1
            return CacheLookup.HIT
        if key in self.pending:
            self.pending[key] += 1
            self.counters["miss_merges"] += 1
            return CacheLookup.MERGED
        if len(self.active) + len(self.pending) >= self.capacity:
            raise CacheBackpressure("semantic cache has no free slot or miss-merge entry")
        self.pending[key] = 1
        self.counters["directory_misses"] += 1
        return CacheLookup.MISS

    def fill_complete(self, key: tuple[int, int], *, remaining_uses: int) -> None:
        waiters = self.pending.pop(key, None)
        if waiters is None:
            raise ValueError("cache fill has no pending miss")
        self.active[key] = {
            "remaining_uses": remaining_uses,
            "active_reads": waiters,
            "closing": False,
        }
        self.counters["fills"] += 1

    def begin_multicast(self, key: tuple[int, int], destinations: int) -> None:
        if destinations <= 0 or key not in self.active:
            raise ValueError("multicast requires an active key and destinations")
        self.active[key]["active_reads"] = int(self.active[key]["active_reads"]) + destinations
        self.active[key]["remaining_uses"] = int(self.active[key]["remaining_uses"]) + destinations
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
        if key not in self.active:
            raise ValueError("cannot close a non-resident cache key")
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
        return kind in {PrimitiveKind.UPDATE_COMMIT, PrimitiveKind.SET_MODIFICATION}


class SharedSram(HardwareModule):
    def accepts_kind(self, kind: PrimitiveKind) -> bool:
        return kind in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN}
