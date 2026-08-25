"""Replay backend for a previously recorded Ramulator request/completion table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordedMemoryBackend:
    completions: dict[tuple[int, int, bool, int], int]

    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        key = (address, size_bytes, bool(is_write), arrival_cycle)
        if key not in self.completions:
            raise RuntimeError("memory request has no recorded Ramulator completion")
        completion = self.completions.pop(key)
        if completion < arrival_cycle:
            raise ValueError("recorded memory completion precedes request arrival")
        return completion
