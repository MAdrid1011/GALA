from __future__ import annotations

import json
import importlib
from pathlib import Path
from types import SimpleNamespace

from gala_sim.cli import main
from gala_sim.tools.inactivity import InactivityWatchdog
from gala_sim.timing import CycleProgress
from tests.test_trace_cycle import _trace
from gala_sim.trace import Trace, TraceWriter
from gala_sim.trace import VirtualPacketArchiveWriter
from tests.test_virtual_capture import _packet


def test_cli_validates_trace_and_reports_frozen_config(tmp_path: Path, capsys) -> None:
    trace_root = tmp_path / "trace"
    TraceWriter().write(_trace(), trace_root)
    assert main(["trace-validate", "--trace", str(trace_root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert main(["config-check", "--config", "configs/architecture/gala.yaml"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is True
    assert result["pending"] == []


def test_cli_validates_compact_packet_archive_without_false_formal_promotion(
    tmp_path: Path, capsys,
) -> None:
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=1024)
    writer.initialize_gaussians(1)
    writer.append_packet(_packet())
    writer.close_iteration(1)
    writer.finish()
    output = tmp_path / "validation.json"

    assert main([
        "trace-archive-validate", "--archive", str(archive_root),
        "--output", str(output), "--parallel-workers", "2",
    ]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "passed"
    assert report["validation_passed"] is True
    assert report["formal_performance_eligible"] is False
    assert report["parallel_validation_workers"] == 2
    progress = [json.loads(line) for line in captured.err.splitlines()]
    assert progress == [{
        "completed_chunks": 1,
        "percent": 100,
        "phase": "archive_validation",
        "total_chunks": 1,
    }]
    assert json.loads(output.read_text()) == report


def test_cli_snapshots_a_closed_archive_prefix(tmp_path: Path, capsys) -> None:
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=1)
    writer.initialize_gaussians(1)
    writer.append_packet(_packet())
    writer.close_iteration(1)
    writer.finish()
    output = tmp_path / "snapshot"

    assert main([
        "trace-archive-snapshot", "--archive", str(archive_root),
        "--output", str(output), "--initial-gaussian-count", "1",
        "--through-iteration", "1",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["iteration_count"] == 1
    assert (output / "manifest.json").is_file()


def test_formal_cycle_cli_writes_failed_preflight_without_timing_bypass(
    tmp_path: Path, capsys,
) -> None:
    trace_root = tmp_path / "trace"
    TraceWriter().write(_trace(), trace_root)
    usage = tmp_path / "resource_usage.json"
    usage.write_text(json.dumps({
        "shared_sram_bytes": 1,
        "pods": 4,
        "clusters": 20,
        "fma_lanes": 320,
        "transcendental_lanes": 40,
        "external_channels": 8,
        "regions": {"fixture": 1},
    }), encoding="utf-8")
    output = tmp_path / "run"
    assert main([
        "cycle-replay", "--trace", str(trace_root),
        "--config", "configs/architecture/gala.yaml",
        "--resource-usage", str(usage), "--output", str(output),
    ]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed_preflight"
    assert (output / "status.json").is_file()
    assert not (output / "cycles.json").exists()


def test_cycle_cli_watchdog_rejects_repeated_heartbeats(
    monkeypatch, tmp_path: Path,
) -> None:
    cli = importlib.import_module("gala_sim.cli.main")

    class Clock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()

    class FakeCycleConfig:
        clock_frequency_hz = 500_000_000

        @classmethod
        def from_gala(cls, *args, **kwargs):
            return cls()

    class FakeCycleEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, trace, *, progress, **kwargs):
            heartbeat = CycleProgress(
                phase="replay", completed_events=10, total_events=100,
                completed_iterations=1, total_iterations=10,
                last_completed_iteration=1, simulated_cycles=100,
                elapsed_seconds=1.0,
            )
            progress(heartbeat)
            clock.value = 300.0
            progress(heartbeat)
            raise AssertionError("watchdog did not stop the replay")

    monkeypatch.setattr(cli.TraceReader, "read", lambda *args, **kwargs: _trace())
    monkeypatch.setattr(cli, "_load_binding", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_load_resource_usage", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli, "run_cycle_preflight",
        lambda *args, **kwargs: SimpleNamespace(status="passed"),
    )
    monkeypatch.setattr(cli, "CycleConfig", FakeCycleConfig)
    monkeypatch.setattr(cli, "CycleEngine", FakeCycleEngine)
    monkeypatch.setattr(
        cli, "InactivityWatchdog",
        lambda **kwargs: InactivityWatchdog(
            timeout_seconds=300, monotonic_fn=clock,
        ),
    )
    output = tmp_path / "cycle"
    result = cli.main([
        "cycle-replay", "--trace", str(tmp_path / "trace"),
        "--config", "configs/architecture/gala.yaml",
        "--ramulator-binding", "fixture:binding",
        "--resource-usage", str(tmp_path / "usage.json"),
        "--output", str(output),
    ])
    assert result == 2
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_cycle"
    assert status["reason"] == "watchdog_inactivity_timeout"
    assert not (output / "cycles.json").exists()


def test_cycle_cli_rejects_windowed_trace_without_quick_scope(
    tmp_path: Path, capsys,
) -> None:
    source = _trace()
    windowed = Trace(
        source.events, source.dependencies, source.payload,
        {**source.metadata, "trace_window": {
            "schema_version": "gala-iteration-window-v1",
            "result_scope": "quick_trace_validation",
            "formal_performance_eligible": False,
            "quality_eligible": False,
            "selection": "inclusive_training_iteration_range",
            "iteration_start": 600,
            "iteration_end": 601,
        }},
    )
    trace_root = tmp_path / "trace"
    TraceWriter().write(windowed, trace_root)
    usage = tmp_path / "resource_usage.json"
    usage.write_text(json.dumps({
        "shared_sram_bytes": 1,
        "pods": 4,
        "clusters": 20,
        "fma_lanes": 320,
        "transcendental_lanes": 40,
        "external_channels": 8,
        "regions": {"fixture": 1},
    }), encoding="utf-8")

    assert main([
        "cycle-replay", "--trace", str(trace_root),
        "--config", "configs/architecture/gala.yaml",
        "--resource-usage", str(usage), "--output", str(tmp_path / "run"),
    ]) == 2
    assert "--quick-validation" in capsys.readouterr().err
