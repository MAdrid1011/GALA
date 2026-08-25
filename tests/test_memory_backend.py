from __future__ import annotations

from gala_sim.timing.memory import Ramulator2Backend


class _SingleEntryBinding:
    def __init__(self) -> None:
        self.pending: int | None = None
        self.completed: list[int] = []

    def metadata(self) -> dict[str, object]:
        return {
            "implementation": "Ramulator 2", "version": "2.1.0",
            "config_sha256": "c" * 64,
            "channels": 8, "transaction_bytes": 64,
        }

    def try_issue(self, address: int, is_write: bool, request_id: int) -> bool:
        if self.pending is not None:
            return False
        self.pending = request_id
        return True

    def tick(self) -> None:
        if self.pending is not None:
            self.completed.append(self.pending)
            self.pending = None

    def drain_completions(self) -> tuple[int, ...]:
        result = tuple(self.completed)
        self.completed.clear()
        return result

    def clone(self) -> "_SingleEntryBinding":
        return type(self)()


def test_ramulator_backend_preserves_concurrent_arrivals_and_frontend_backpressure() -> None:
    backend = Ramulator2Backend(_SingleEntryBinding())
    first = backend.submit_async(
        address=0, size_bytes=64, is_write=False, arrival_cycle=0,
    )
    second = backend.submit_async(
        address=64, size_bytes=64, is_write=True, arrival_cycle=0,
    )
    backend.advance(1)
    first_completion = backend.pop_completions()
    assert [item.request_id for item in first_completion] == [first]
    backend.advance(2)
    second_completion = backend.pop_completions()
    assert [item.request_id for item in second_completion] == [second]
    records = backend.audit_records()
    assert [item.arrival_cycle for item in records] == [0, 0]
    assert [item.completion_cycle for item in records] == [1, 2]


def test_ramulator_backend_splits_multisector_request_and_waits_for_all_beats() -> None:
    backend = Ramulator2Backend(_SingleEntryBinding())
    request = backend.submit_async(
        address=128, size_bytes=128, is_write=False, arrival_cycle=0,
    )
    backend.advance(1)
    assert backend.pop_completions() == ()
    backend.advance(2)
    completion = backend.pop_completions()
    assert len(completion) == 1
    assert completion[0].request_id == request
    assert completion[0].size_bytes == 128
    assert completion[0].completion_cycle == 2


def test_ramulator_backend_clone_starts_with_independent_state() -> None:
    backend = Ramulator2Backend(_SingleEntryBinding())
    backend.submit_async(address=0, size_bytes=64, is_write=False, arrival_cycle=0)
    clone = backend.clone()
    assert clone.current_cycle == 0
    assert clone.audit_records() == ()
