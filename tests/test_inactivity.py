from __future__ import annotations

import pytest

from gala_sim.tools.inactivity import (
    InactivityTimeoutError,
    InactivityWatchdog,
    observe_cycle_progress,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_heartbeat_does_not_reset_inactivity_timeout() -> None:
    clock = _Clock()
    watchdog = InactivityWatchdog(timeout_seconds=5, monotonic_fn=clock)
    watchdog.observe(progress_key=("replay", 10, 1))
    for second in range(1, 5):
        clock.value = float(second)
        watchdog.observe(progress_key=("replay", 10, 1))
    clock.value = 5.0
    with pytest.raises(InactivityTimeoutError):
        watchdog.observe(progress_key=("replay", 10, 1))


def test_output_and_gpu_activity_are_real_progress() -> None:
    clock = _Clock()
    watchdog = InactivityWatchdog(timeout_seconds=5, monotonic_fn=clock)
    watchdog.observe(byte_count=0)
    clock.value = 4.0
    watchdog.observe(byte_count=1)
    clock.value = 8.0
    watchdog.observe(gpu_active=True, byte_count=1)
    clock.value = 12.0
    watchdog.observe(byte_count=1)


def test_cycle_replay_ignores_cpu_only_heartbeats() -> None:
    clock = _Clock()
    watchdog = InactivityWatchdog(timeout_seconds=5, monotonic_fn=clock)
    observe_cycle_progress(
        watchdog, phase="replay", completed_events=10,
        completed_iterations=1, cpu_seconds=0.0,
    )
    clock.value = 5.0
    with pytest.raises(InactivityTimeoutError):
        observe_cycle_progress(
            watchdog, phase="replay", completed_events=10,
            completed_iterations=1, cpu_seconds=100.0,
        )


@pytest.mark.parametrize("phase", [
    "online_replay",
    "oracle_portfolio_actual_replay",
    "oracle_portfolio_future_replay",
])
def test_nested_cycle_replay_ignores_cpu_only_heartbeats(phase: str) -> None:
    clock = _Clock()
    watchdog = InactivityWatchdog(timeout_seconds=5, monotonic_fn=clock)
    observe_cycle_progress(
        watchdog, phase=phase, completed_events=10,
        completed_iterations=1, cpu_seconds=0.0,
    )
    clock.value = 5.0
    with pytest.raises(InactivityTimeoutError):
        observe_cycle_progress(
            watchdog, phase=phase, completed_events=10,
            completed_iterations=1, cpu_seconds=100.0,
        )


def test_cycle_preparation_accepts_measured_cpu_progress() -> None:
    clock = _Clock()
    watchdog = InactivityWatchdog(timeout_seconds=5, monotonic_fn=clock)
    observe_cycle_progress(
        watchdog, phase="validation", completed_events=0,
        completed_iterations=0, cpu_seconds=0.0,
    )
    clock.value = 4.0
    observe_cycle_progress(
        watchdog, phase="validation", completed_events=0,
        completed_iterations=0, cpu_seconds=1.0,
    )
    clock.value = 8.0
    observe_cycle_progress(
        watchdog, phase="validation", completed_events=0,
        completed_iterations=0, cpu_seconds=2.0,
    )
