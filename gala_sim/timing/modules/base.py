"""Shared module bookkeeping, kept separate from hardware responsibilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CounterBlock:
    accepted: int = 0
    completed: int = 0
    busy_cycles: int = 0
    bank_conflicts: int = 0
    port_stalls: int = 0
    queue_stalls: int = 0
    dependency_stalls: int = 0
    memory_wait_cycles: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "completed": self.completed,
            "busy_cycles": self.busy_cycles,
            "bank_conflicts": self.bank_conflicts,
            "port_stalls": self.port_stalls,
            "queue_stalls": self.queue_stalls,
            "dependency_stalls": self.dependency_stalls,
            "memory_wait_cycles": self.memory_wait_cycles,
        }


@dataclass(frozen=True)
class StallRecord:
    cycle: int
    module: str
    reason: str
    event_ids: tuple[int, ...]


@dataclass(frozen=True)
class ModuleOutput:
    event_id: int
    completion_cycle: int

