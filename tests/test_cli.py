from __future__ import annotations

import json
from pathlib import Path

from gala_sim.cli import main
from tests.test_trace_cycle import _trace
from gala_sim.trace import TraceWriter


def test_cli_validates_trace_and_reports_pending_config(tmp_path: Path, capsys) -> None:
    trace_root = tmp_path / "trace"
    TraceWriter().write(_trace(), trace_root)
    assert main(["trace-validate", "--trace", str(trace_root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert main(["config-check", "--config", "configs/architecture/gala.yaml"]) == 2
    assert json.loads(capsys.readouterr().out)["ready"] is False
