"""Common module API mandated by the cycle-model contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .base import CounterBlock, ModuleOutput


@dataclass(frozen=True)
class EventBatch:
    event_ids: tuple[int, ...]


@dataclass(frozen=True)
class AcceptResult:
    accepted: tuple[int, ...]
    rejected: tuple[int, ...]
    reason: str | None


@dataclass(frozen=True)
class ModuleOutputs:
    outputs: tuple[ModuleOutput, ...]


class CycleModule(Protocol):
    def next_wakeup(self) -> int | None: ...

    def accept(self, batch: EventBatch, cycle: int) -> AcceptResult: ...

    def advance(self, cycle: int) -> ModuleOutputs: ...

    def snapshot_counters(self) -> CounterBlock: ...
