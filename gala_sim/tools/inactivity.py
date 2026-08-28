"""Shared inactivity watchdog for long-running workflow commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Hashable


class InactivityTimeoutError(RuntimeError):
    """Raised when a monitored command has no measurable forward progress."""


@dataclass
class InactivityWatchdog:
    timeout_seconds: float = 300.0
    monotonic_fn: Callable[[], float] = time.monotonic
    minimum_cpu_progress_seconds: float = 0.05
    _started_at: float = field(init=False)
    _last_progress_at: float = field(init=False)
    _last_progress_key: Hashable | None = field(default=None, init=False)
    _last_byte_count: int | None = field(default=None, init=False)
    _last_cpu_seconds: float | None = field(default=None, init=False)
    _maximum_inactivity_seconds: float = field(default=0.0, init=False)
    _last_evidence: str = field(default="startup", init=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("inactivity timeout must be positive")
        if self.minimum_cpu_progress_seconds < 0:
            raise ValueError("minimum CPU progress must be non-negative")
        self._started_at = self.monotonic_fn()
        self._last_progress_at = self._started_at

    def observe(
        self,
        *,
        progress_key: Hashable | None = None,
        gpu_active: bool = False,
        byte_count: int | None = None,
        cpu_seconds: float | None = None,
        allow_cpu_progress: bool = True,
    ) -> None:
        """Record independent evidence; calling this method is not progress itself."""

        now = self.monotonic_fn()
        evidence: str | None = None
        if progress_key is not None:
            if self._last_progress_key is not None and progress_key != self._last_progress_key:
                evidence = "progress_counter"
            self._last_progress_key = progress_key
        if byte_count is not None:
            if self._last_byte_count is not None and byte_count > self._last_byte_count:
                evidence = "output_bytes"
            self._last_byte_count = byte_count
        if cpu_seconds is not None:
            if (
                allow_cpu_progress
                and self._last_cpu_seconds is not None
                and cpu_seconds - self._last_cpu_seconds
                >= self.minimum_cpu_progress_seconds
            ):
                evidence = "cpu_time"
            self._last_cpu_seconds = cpu_seconds
        if gpu_active:
            evidence = "gpu_utilization"
        if evidence is not None:
            self._maximum_inactivity_seconds = max(
                self._maximum_inactivity_seconds, now - self._last_progress_at,
            )
            self._last_progress_at = now
            self._last_evidence = evidence
            return
        inactivity = now - self._last_progress_at
        self._maximum_inactivity_seconds = max(
            self._maximum_inactivity_seconds, inactivity,
        )
        if inactivity >= self.timeout_seconds:
            raise InactivityTimeoutError(
                f"no measurable progress for {inactivity:.3f} seconds"
            )

    def report(self, *, status: str) -> dict[str, object]:
        now = self.monotonic_fn()
        return {
            "status": status,
            "timeout_seconds": self.timeout_seconds,
            "elapsed_seconds": now - self._started_at,
            "current_inactivity_seconds": now - self._last_progress_at,
            "maximum_observed_inactivity_seconds": max(
                self._maximum_inactivity_seconds, now - self._last_progress_at,
            ),
            "last_progress_evidence": self._last_evidence,
        }


def observe_cycle_progress(
    watchdog: InactivityWatchdog,
    *,
    phase: str,
    completed_events: int,
    completed_iterations: int,
    cpu_seconds: float,
) -> None:
    """Apply cycle-specific evidence rules without counting callback heartbeats."""

    is_replay = phase == "replay" or phase.endswith("_replay")
    watchdog.observe(
        progress_key=(phase, completed_events, completed_iterations),
        cpu_seconds=cpu_seconds,
        # Preparation passes can legitimately be CPU-bound. During replay,
        # only retired events or iterations demonstrate forward progress.
        allow_cpu_progress=not is_replay,
    )
