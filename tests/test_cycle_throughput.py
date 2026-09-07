from __future__ import annotations

import pytest

from gala_sim.timing import CycleProgress
from gala_sim.tools.cycle_throughput import (
    ArchiveSpeedupMonitor, ThroughputConverged, ThroughputDiagnosticConfig,
    ThroughputMonitor,
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
        clock_frequency_hz=100,
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


def test_adaptive_stop_uses_short_explicit_convergence_window() -> None:
    adaptive = _config().for_adaptive_stop()
    assert adaptive.warmup_samples == 1
    assert adaptive.stability_window_samples == 2
    assert adaptive.required_consecutive_stable_windows == 2
    assert adaptive.minimum_completion_fraction == pytest.approx(0.10)


def test_archive_speedup_monitor_requires_common_stable_intervals() -> None:
    monitor = ArchiveSpeedupMonitor(
        _config(), variants=("0000", "1010", "1111"),
    )
    report = {}
    for iteration in range(1, 6):
        report = monitor.observe(
            iteration_id=iteration,
            completed_iterations=iteration,
            total_iterations=10,
            completed_events=iteration * 100,
            cycles_by_variant={
                "0000": iteration * 100,
                "1010": iteration * 80,
                "1111": iteration * 50,
            },
        )
    assert report["stability"]["status"] == "stable"
    assert report["samples"][-1]["cumulative_cycle_ratio_vs_0000"] == {
        "0000": 1.0, "1010": 1.25, "1111": 2.0,
    }
    assert report["samples"][-1]["interval_cycle_ratio_vs_0000"] == {
        "0000": 1.0, "1010": 1.25, "1111": 2.0,
    }
    assert report["formal_performance_eligible"] is False


def test_archive_speedup_monitor_emits_adaptive_full_run_certificate() -> None:
    monitor = ArchiveSpeedupMonitor(
        _config(),
        variants=("0000", "1000", "1010", "0100", "0101", "1100", "1111"),
        target_speedups={"1010": 1.2, "0101": 1.2, "1111": 1.8},
        source_complete=True,
        source_validated=True,
        source_contract_passed=True,
    )
    report = {}
    for iteration in range(1, 6):
        report = monitor.observe(
            iteration_id=iteration,
            completed_iterations=iteration,
            total_iterations=10,
            completed_events=iteration * 100,
            cycles_by_variant={
                "0000": iteration * 100,
                "1000": iteration * 90,
                "1010": iteration * 80,
                "0100": iteration * 85,
                "0101": iteration * 75,
                "1100": iteration * 70,
                "1111": iteration * 50,
            },
        )
    assert report["result_scope"] == "adaptive_end_to_end_estimate"
    assert report["execution_path"] == "archive_replay_with_adaptive_stop"
    assert report["stability_certificate"]["ready"] is True
    assert report["adaptive_performance_eligible"] is True
    assert report["stability_certificate"]["composition"]["non_regressive"] is True
    assert report["stability_certificate"]["targets"]["all_targets_reached"] is True


def test_archive_speedup_monitor_rejects_recent_phase_change() -> None:
    monitor = ArchiveSpeedupMonitor(
        _config(), variants=("0000", "1111"),
    )
    for iteration in range(1, 5):
        monitor.observe(
            iteration_id=iteration,
            completed_iterations=iteration,
            total_iterations=10,
            completed_events=iteration * 100,
            cycles_by_variant={"0000": iteration * 100, "1111": iteration * 50},
        )
    report = monitor.observe(
        iteration_id=5,
        completed_iterations=5,
        total_iterations=10,
        completed_events=500,
        cycles_by_variant={"0000": 500, "1111": 400},
    )
    assert report["stability"]["status"] == "collecting"
    assert report["stability"]["interval_cycle_ratio_relative_span"]["1111"] > 0.01


def test_archive_speedup_monitor_rejects_variant_set_drift() -> None:
    monitor = ArchiveSpeedupMonitor(
        _config(), variants=("0000", "1111"),
    )
    with pytest.raises(ValueError, match="variant set changed"):
        monitor.observe(
            iteration_id=1,
            completed_iterations=1,
            total_iterations=10,
            completed_events=100,
            cycles_by_variant={"0000": 100},
        )
