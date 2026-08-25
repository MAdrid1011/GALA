"""Memory bridge protocols and explicit Ramulator command adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


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


@dataclass(frozen=True)
class MemoryRequestRecord:
    request_id: int
    address: int
    size_bytes: int
    is_write: bool
    arrival_cycle: int
    completion_cycle: int | None = None


@dataclass
class _RequestGroup:
    record: MemoryRequestRecord
    beat_ids: set[int]
    pending_beats: list[tuple[int, int]]


@dataclass
class Ramulator2Backend:
    """Asynchronous bridge over Ramulator 2's External frontend.

    The binding must expose ``metadata()``, ``try_issue(address, is_write,
    request_id)``, ``tick()``, ``drain_completions()``, and ``clone()``.  One
    GALA request is split into the binding's transaction-sized beats.  The
    cycle engine advances this backend alongside all other modules, so future
    arrivals can overlap earlier outstanding requests.
    """

    binding: Any
    current_cycle: int = field(default=0, init=False)
    _next_group_id: int = field(default=0, init=False, repr=False)
    _next_beat_id: int = field(default=0, init=False, repr=False)
    _groups: dict[int, _RequestGroup] = field(default_factory=dict, init=False, repr=False)
    _beat_to_group: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _completed: list[MemoryRequestRecord] = field(default_factory=list, init=False, repr=False)
    _audit: list[MemoryRequestRecord] = field(default_factory=list, init=False, repr=False)

    def metadata(self) -> Mapping[str, Any]:
        try:
            value = self.binding.metadata()
        except AttributeError as error:
            raise MissingMemoryBackend("Ramulator 2 binding lacks metadata()") from error
        if not isinstance(value, Mapping):
            raise MissingMemoryBackend("Ramulator 2 binding metadata is not a mapping")
        return value

    def submit_async(self, *, address: int, size_bytes: int, is_write: bool,
                     arrival_cycle: int) -> int:
        if arrival_cycle != self.current_cycle:
            raise ValueError("Ramulator request arrival does not match backend cycle")
        metadata = self.metadata()
        try:
            transaction_bytes = int(metadata["transaction_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise MissingMemoryBackend(
                "Ramulator 2 metadata lacks transaction_bytes"
            ) from error
        if transaction_bytes <= 0 or address % transaction_bytes != 0:
            raise ValueError("Ramulator request address is not transaction aligned")
        if size_bytes <= 0 or size_bytes % transaction_bytes != 0:
            raise ValueError("Ramulator request size is not transaction aligned")
        group_id = self._next_group_id
        self._next_group_id += 1
        record = MemoryRequestRecord(
            group_id, address, size_bytes, bool(is_write), arrival_cycle,
        )
        pending_beats: list[tuple[int, int]] = []
        beat_ids: set[int] = set()
        for beat_address in range(address, address + size_bytes, transaction_bytes):
            beat_id = self._next_beat_id
            self._next_beat_id += 1
            pending_beats.append((beat_id, beat_address))
            beat_ids.add(beat_id)
            self._beat_to_group[beat_id] = group_id
        self._groups[group_id] = _RequestGroup(record, beat_ids, pending_beats)
        self._audit.append(record)
        self._issue_waiting()
        return group_id

    def advance(self, cycle: int) -> None:
        if cycle < self.current_cycle:
            raise ValueError("Ramulator backend cannot move backwards")
        while self.current_cycle < cycle:
            self._issue_waiting()
            try:
                self.binding.tick()
            except AttributeError as error:
                raise MissingMemoryBackend("Ramulator 2 binding lacks tick()") from error
            self.current_cycle += 1
            self._drain_binding_completions()

    def next_wakeup(self) -> int | None:
        return self.current_cycle + 1 if self._groups else None

    def pop_completions(self) -> tuple[MemoryRequestRecord, ...]:
        completed = tuple(self._completed)
        self._completed.clear()
        return completed

    def audit_records(self) -> tuple[MemoryRequestRecord, ...]:
        return tuple(self._audit)

    def clone(self) -> "Ramulator2Backend":
        try:
            binding = self.binding.clone()
        except AttributeError as error:
            raise MissingMemoryBackend("Ramulator 2 binding lacks clone()") from error
        return type(self)(binding)

    def _issue_waiting(self) -> None:
        for group_id in sorted(self._groups):
            group = self._groups[group_id]
            while group.pending_beats:
                beat_id, address = group.pending_beats[0]
                try:
                    accepted = self.binding.try_issue(
                        address, group.record.is_write, beat_id
                    )
                except AttributeError as error:
                    raise MissingMemoryBackend(
                        "Ramulator 2 binding lacks try_issue()"
                    ) from error
                if not isinstance(accepted, bool):
                    raise ValueError("Ramulator 2 try_issue() must return bool")
                if not accepted:
                    break
                group.pending_beats.pop(0)

    def _drain_binding_completions(self) -> None:
        try:
            completions = tuple(self.binding.drain_completions())
        except AttributeError as error:
            raise MissingMemoryBackend(
                "Ramulator 2 binding lacks drain_completions()"
            ) from error
        for value in completions:
            beat_id = int(value)
            group_id = self._beat_to_group.pop(beat_id, None)
            if group_id is None or group_id not in self._groups:
                raise ValueError("Ramulator 2 completed an unknown request beat")
            group = self._groups[group_id]
            if beat_id not in group.beat_ids:
                raise ValueError("Ramulator 2 completed a request beat twice")
            group.beat_ids.remove(beat_id)
            if not group.beat_ids and not group.pending_beats:
                completed = MemoryRequestRecord(
                    group.record.request_id,
                    group.record.address,
                    group.record.size_bytes,
                    group.record.is_write,
                    group.record.arrival_cycle,
                    self.current_cycle,
                )
                self._completed.append(completed)
                self._audit[group_id] = completed
                del self._groups[group_id]


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
