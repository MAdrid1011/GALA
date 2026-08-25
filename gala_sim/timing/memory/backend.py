"""Memory bridge protocols and explicit Ramulator command adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class MissingMemoryBackend(RuntimeError):
    pass


@dataclass
class CallableMemoryBackend:
    submit_request: Callable[..., int]

    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        completion = self.submit_request(address=address, size_bytes=size_bytes,
                                         is_write=is_write, arrival_cycle=arrival_cycle)
        if not isinstance(completion, int) or completion < arrival_cycle:
            raise ValueError("memory backend returned an invalid completion cycle")
        return completion


@dataclass
class Ramulator2Backend:
    binding: Any

    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        try:
            completion = self.binding.submit(address, size_bytes, bool(is_write), arrival_cycle)
        except AttributeError as error:
            raise MissingMemoryBackend("Ramulator 2 binding lacks submit()") from error
        if not isinstance(completion, int) or completion < arrival_cycle:
            raise ValueError("Ramulator 2 returned an invalid completion cycle")
        return completion


@dataclass
class RecordedMemoryBackend:
    completions: dict[tuple[int, int, bool, int], int]

    def clone(self) -> "RecordedMemoryBackend":
        """Return a fresh replay cursor over the same external timing table."""

        return type(self)(dict(self.completions))

    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        key = (address, size_bytes, bool(is_write), arrival_cycle)
        if key not in self.completions:
            raise RuntimeError("memory request has no recorded Ramulator completion")
        completion = self.completions.pop(key)
        if completion < arrival_cycle:
            raise ValueError("recorded memory completion precedes request arrival")
        return completion
