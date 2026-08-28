"""Consumer for bounded virtual packets produced by the official capture hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

from gala_sim.clamp import PrimitiveKind
from gala_sim.trace import (
    VirtualEventStreamValidator,
    VirtualQueryEventExpander,
    VirtualTraceLifecycleValidator,
    VirtualTracePacket,
)
from gala_sim.trace.virtual import VirtualLifecycleRecord


RELATION_CHAIN_KINDS = (
    PrimitiveKind.RELATION,
    PrimitiveKind.CACHE_REQUEST,
    PrimitiveKind.CACHE_RETURN,
    PrimitiveKind.FORWARD,
    PrimitiveKind.ADJOINT,
    PrimitiveKind.GRADIENT_REDUCTION,
)
QUERY_CHAIN_KINDS = (
    PrimitiveKind.QUERY_CLOSE,
    PrimitiveKind.QUERY_REDUCTION,
    PrimitiveKind.CONSUMER,
)


@dataclass
class VirtualCaptureConsumer:
    """Consume query and lifecycle packets without materializing a full trace."""

    output_root: Path
    max_events: int = 65536
    state_record_bytes: int = 128
    relation_candidate_bytes: int = 0
    inactivity_timeout_seconds: float = 300.0
    progress_interval_seconds: float = 30.0
    expand_for_validation: bool = False
    packet_consumer: Callable[[VirtualTracePacket], None] | None = None
    _expander: VirtualQueryEventExpander = field(init=False)
    _event_validator: VirtualEventStreamValidator = field(
        default_factory=VirtualEventStreamValidator, init=False
    )
    _lifecycle: VirtualTraceLifecycleValidator | None = field(default=None, init=False)
    _current_iteration: int | None = field(default=None, init=False)
    _packet_count: int = field(default=0, init=False)
    _relation_count: int = field(default=0, init=False)
    _candidate_count: int = field(default=0, init=False)
    _query_count: int = field(default=0, init=False)
    _physical_bytes: int = field(default=0, init=False)
    _peak_packet_bytes: int = field(default=0, init=False)
    _logical_expanded_events: int = field(default=0, init=False)
    _last_progress: float = field(default_factory=time.monotonic, init=False)
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _last_report: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        if self.max_events <= 0 or self.state_record_bytes <= 0:
            raise ValueError("virtual capture consumer limits must be positive")
        if self.inactivity_timeout_seconds <= 0:
            raise ValueError("virtual capture inactivity timeout must be positive")
        if self.progress_interval_seconds <= 0:
            raise ValueError("virtual capture progress interval must be positive")
        self.output_root = Path(self.output_root)
        self._expander = VirtualQueryEventExpander(
            max_events=self.max_events,
            state_record_bytes=self.state_record_bytes,
            relation_candidate_bytes=self.relation_candidate_bytes,
        )

    def initialize_gaussians(self, count: int) -> None:
        if count < 0:
            raise ValueError("initial Gaussian count must be non-negative")
        if self._lifecycle is None:
            self._lifecycle = VirtualTraceLifecycleValidator(count)
        elif self._lifecycle.initial_gaussian_count != count:
            raise ValueError("virtual capture Gaussian count changed during initialization")

    @property
    def current_iteration(self) -> int | None:
        return self._current_iteration

    @property
    def initialized(self) -> bool:
        return self._lifecycle is not None

    def accept_query(self, packet: VirtualTracePacket) -> None:
        self._ensure_lifecycle()
        self._check_progress()
        if self._current_iteration is None:
            self._current_iteration = packet.iteration_id
        if packet.iteration_id != self._current_iteration:
            raise ValueError("virtual query packet changed iteration without a close")
        self._lifecycle.accept_packet(packet)  # type: ignore[union-attr]
        self._packet_count += 1
        self._query_count += packet.query_count
        self._candidate_count += packet.candidate_count
        self._relation_count += packet.logical_relation_count
        self._physical_bytes += packet.physical_bytes
        self._peak_packet_bytes = max(self._peak_packet_bytes, packet.physical_bytes)
        self._logical_expanded_events += (
            packet.candidate_count
            + len(RELATION_CHAIN_KINDS) * packet.logical_relation_count
            + len(QUERY_CHAIN_KINDS) * packet.query_count
        )
        if self.expand_for_validation:
            for event_packet in self._expander.expand(packet):
                self._check_progress()
                self._event_validator.accept(event_packet)
                self._last_progress = time.monotonic()
                self._report_progress()
        if self.packet_consumer is not None:
            self.packet_consumer(packet)
        self._last_progress = time.monotonic()
        self._report_progress()

    def accept_lifecycle(self, record: VirtualLifecycleRecord) -> None:
        self._ensure_lifecycle()
        self._check_progress()
        if self._current_iteration is None:
            self._current_iteration = record.iteration_id
        if record.iteration_id != self._current_iteration:
            raise ValueError("virtual lifecycle record changed iteration without a close")
        self._lifecycle.accept_lifecycle(record)  # type: ignore[union-attr]
        self._last_progress = time.monotonic()

    def close_iteration(self, iteration_id: int) -> None:
        self._ensure_lifecycle()
        if self._current_iteration != iteration_id:
            raise ValueError("virtual iteration close does not match the active iteration")
        self._lifecycle.close_iteration(iteration_id)  # type: ignore[union-attr]
        self._current_iteration = None
        self._last_progress = time.monotonic()
        self._report_progress(force=True)

    def finish(self, *, capture_audit: dict[str, int] | None = None) -> dict[str, Any]:
        self._ensure_lifecycle()
        if self._current_iteration is not None:
            self.close_iteration(self._current_iteration)
        if self.expand_for_validation:
            # Close the global packet frontier explicitly.  The empty terminal
            # packet distinguishes a clean end from an interrupted expansion.
            from gala_sim.trace import VirtualEventPacket
            from gala_sim.clamp.events import dependency_dtype, event_dtype

            terminal = VirtualEventPacket(
                packet_id=self._event_validator.next_packet_id,
                global_event_start=self._event_validator.next_event_id,
                events=np.empty(0, dtype=event_dtype()),
                dependencies=np.empty(0, dtype=dependency_dtype()),
                final_packet=True,
                frontier_complete=True,
            )
            self._event_validator.accept(terminal)
            self._event_validator.finalize()
        ledgers = self._lifecycle.finalize()  # type: ignore[union-attr]
        elapsed = time.monotonic() - self._started_at
        result: dict[str, Any] = {
            "schema_version": "gala-virtual-trace-capture-v1",
            "status": "passed",
            "formal_performance_eligible": False,
            "packet_count": self._packet_count,
            "query_count": self._query_count,
            "candidate_count": self._candidate_count,
            "relation_count": self._relation_count,
            "logical_expanded_event_count": self._logical_expanded_events,
            "validated_expanded_event_count": self._event_validator.accepted_events,
            "event_stream_validated": self.expand_for_validation,
            "packet_consumer_attached": self.packet_consumer is not None,
            "physical_stream_bytes": self._physical_bytes,
            "peak_resident_packet_bytes": self._peak_packet_bytes,
            "elapsed_seconds": elapsed,
            "event_id_mode": (
                "global_dense" if self.expand_for_validation
                else "deferred_to_packet_consumer"
            ),
            "dependency_id_mode": (
                "global_dense" if self.expand_for_validation
                else "deferred_to_packet_consumer"
            ),
            "iterations": [ledger.__dict__ for ledger in ledgers],
            "capture_audit": dict(sorted((capture_audit or {}).items())),
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "virtual_trace_manifest.json").write_text(
            json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _ensure_lifecycle(self) -> None:
        if self._lifecycle is None:
            raise RuntimeError("virtual capture consumer has no Gaussian initialization")

    def _check_progress(self) -> None:
        now = time.monotonic()
        if now - self._last_progress > self.inactivity_timeout_seconds:
            raise TimeoutError(
                "virtual trace capture made no packet progress for "
                f"{self.inactivity_timeout_seconds:g} seconds"
            )

    def _report_progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_report < self.progress_interval_seconds:
            return
        elapsed = max(now - self._started_at, np.finfo(np.float64).eps)
        report = {
            "phase": "virtual_trace_capture",
            "elapsed_seconds": elapsed,
            "query_packets": self._packet_count,
            "logical_relations": self._relation_count,
            "logical_expanded_events": self._logical_expanded_events,
            "logical_expanded_events_per_second": self._logical_expanded_events / elapsed,
        }
        print(json.dumps(report, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)
        self._last_report = now
