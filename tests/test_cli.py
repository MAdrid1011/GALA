from __future__ import annotations

import json
from pathlib import Path

from gala_sim.cli import main
from tests.test_trace_cycle import _trace
from gala_sim.trace import Trace, TraceWriter


def test_cli_validates_trace_and_reports_frozen_config(tmp_path: Path, capsys) -> None:
    trace_root = tmp_path / "trace"
    TraceWriter().write(_trace(), trace_root)
    assert main(["trace-validate", "--trace", str(trace_root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert main(["config-check", "--config", "configs/architecture/gala.yaml"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is True
    assert result["pending"] == []


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
