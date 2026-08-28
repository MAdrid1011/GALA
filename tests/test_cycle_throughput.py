from __future__ import annotations

import pytest

from gala_sim.timing import CycleProgress
from gala_sim.tools.cycle_throughput import (
    ThroughputConverged, ThroughputDiagnosticConfig, ThroughputMonitor,
    require_empty_diagnostic_output,
)


def _config() -> ThroughputDiagnosticConfig:
    return ThroughputDiagnosticConfig(
        report_interval_events=10,
        report_interval_seconds=1,
        warmup_samples=1,
        stability_window_samples=3,
        required_consecutive_stable_windows=2,
        stability_relative_span=0.01,
        minimum_completion_fraction=0.2,
    )


def _progress(iteration: int, *, cycles_per_iteration: int = 100) -> CycleProgress:
    return CycleProgress(
        phase="replay",
        completed_events=iteration * 10,
        total_events=100,
        completed_iterations=iteration,
        total_iterations=10,
        last_completed_iteration=iteration,
        simulated_cycles=iteration * cycles_per_iteration,
        elapsed_seconds=float(iteration),
    )


def test_monitor_stops_only_after_consecutive_iteration_windows() -> None:
    monitor = ThroughputMonitor(
        _config(), stop_when_stable=True,
        clock_frequency_hz=100, orin_anchor_seconds=20,
    )
    for iteration in range(1, 5):
        report = monitor.observe(_progress(iteration))
        assert report["stability"]["status"] == "collecting"
    with pytest.raises(ThroughputConverged) as stopped:
        monitor.observe(_progress(5))
    report = stopped.value.report
    assert report["stability"]["status"] == "stable"
    assert report["samples"][-1]["projected_total_cycles"] == pytest.approx(1000)
    assert report["samples"][-1]["projected_asic_seconds"] == pytest.approx(10)
    assert report["samples"][-1]["static_anchor_speedup_vs_orin"] == pytest.approx(2)
    assert report["formal_performance_eligible"] is False


def test_monitor_ignores_heartbeats_inside_one_iteration() -> None:
    monitor = ThroughputMonitor(_config())
    monitor.observe(_progress(1))
    duplicate = CycleProgress(
        phase="replay", completed_events=15, total_events=100,
        completed_iterations=1, total_iterations=10, last_completed_iteration=1,
        simulated_cycles=150, elapsed_seconds=1.5,
        last_completed_iteration_events=10,
        last_completed_iteration_cycles=100,
        last_completed_iteration_elapsed_seconds=1.0,
    )
    monitor.observe(duplicate)
    assert len(monitor.samples) == 1
    assert len(monitor.runtime_samples) == 2
    assert monitor.runtime_samples[-1].interval_events_per_second == pytest.approx(10)


def test_monitor_rejects_phase_change_that_breaks_cycle_rate_stability() -> None:
    monitor = ThroughputMonitor(_config())
    for iteration in range(1, 5):
        monitor.observe(_progress(iteration))
    report = monitor.observe(_progress(5, cycles_per_iteration=200))
    assert report["stability"]["status"] == "collecting"
    assert report["stability"]["cycles_per_event_relative_span"] > 0.01


def test_monitor_keeps_exact_result_when_stability_arrives_at_completion() -> None:
    monitor = ThroughputMonitor(_config(), stop_when_stable=True)
    report = {}
    for iteration in range(1, 6):
        report = monitor.observe(CycleProgress(
            phase="replay", completed_events=iteration * 10, total_events=50,
            completed_iterations=iteration, total_iterations=5,
            last_completed_iteration=iteration,
            simulated_cycles=iteration * 100, elapsed_seconds=float(iteration),
        ))
    assert report["complete_trace_replay"] is True
    assert report["stability"]["status"] == "stable"


def test_monitor_reports_orin_speedup_interval() -> None:
    monitor = ThroughputMonitor(
        _config(), clock_frequency_hz=100, orin_anchor_seconds=20,
        orin_anchor_interval_seconds=(15, 25),
    )
    sample = monitor.observe(_progress(1))["samples"][-1]
    assert sample["static_anchor_speedup_vs_orin"] == pytest.approx(2)
    assert sample["static_anchor_speedup_vs_orin_interval"] == pytest.approx({
        "low": 1.5, "high": 2.5,
    })


@pytest.mark.parametrize("field,value", [
    ("completed_events", 101),
    ("completed_iterations", 11),
])
def test_monitor_rejects_progress_outside_totals(field: str, value: int) -> None:
    monitor = ThroughputMonitor(_config())
    values = _progress(1).__dict__ | {field: value}
    with pytest.raises(ValueError):
        monitor.observe(CycleProgress(**values))


def test_early_stop_diagnostic_requires_empty_output(tmp_path) -> None:
    require_empty_diagnostic_output(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    require_empty_diagnostic_output(empty)
    (empty / "cycles.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        require_empty_diagnostic_output(empty)
