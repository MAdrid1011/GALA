from __future__ import annotations

from pathlib import Path

import pytest

from gala_sim.clamp import PrimitiveKind, ResourceClass, TraceBuilder, TraceEvent
from gala_sim.trace import (
    QueryRange, TraceSampleConfig, TraceWriter, dependency_closed_query_sample,
    validate_trace,
)
from gala_sim.cli import main


def _two_query_trace():
    builder = TraceBuilder()
    for query_id in range(2):
        candidate = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
            gaussian_id=query_id, state_version=0,
            resource_class=int(ResourceClass.RELATION),
        ))
        relation = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, state_version=0,
            resource_class=int(ResourceClass.RELATION),
        ), dependencies=[candidate])
        close = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=query_id,
            state_version=0, resource_class=int(ResourceClass.RELATION),
        ), dependencies=[relation])
        request = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, state_version=0,
            data_bytes=64, address_token=query_id * 64,
            resource_class=int(ResourceClass.CACHE),
        ), dependencies=[relation])
        returned = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, state_version=0,
            resource_class=int(ResourceClass.CACHE),
        ), dependencies=[request])
        forward = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, state_version=0,
            resource_class=int(ResourceClass.ISSUE),
        ), dependencies=[relation, returned])
        reduction = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.QUERY_REDUCTION), query_id=query_id,
            state_version=0, resource_class=int(ResourceClass.QUERY),
        ), dependencies=[close, forward])
        consumer = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CONSUMER), query_id=query_id,
            consumer_id=query_id, state_version=0,
            resource_class=int(ResourceClass.QUERY),
        ), dependencies=[reduction], payload=[float(query_id)])
        adjoint = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.ADJOINT), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, state_version=0,
            resource_class=int(ResourceClass.ISSUE),
        ), dependencies=[consumer])
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.GRADIENT_REDUCTION), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, state_version=0,
            resource_class=int(ResourceClass.QUERY),
        ), dependencies=[adjoint])
    return builder.finish(metadata={
        "initial_gaussian_count": 2,
        "trace_chunk_events": 4,
        "capture_audit_schema_version": "gala-r2-capture-audit-v4",
        "capture_audit": {},
    })


def test_query_sample_keeps_full_dependency_chain_and_marks_quick_scope() -> None:
    trace = _two_query_trace()
    sample = dependency_closed_query_sample(
        trace, TraceSampleConfig(
            (QueryRange(1, 1),), max_events=10, max_dependencies=20,
            scan_events=4, scan_backend="cpu",
        ), source_identity="fixture",
    )

    report = validate_trace(sample)
    assert report.event_count == 10
    assert set(sample.events["query_id"]) == {-1, 1}
    assert sample.events["event_id"].tolist() == list(range(10))
    assert sample.payload.tolist() == [1.0]
    assert sample.metadata["trace_sample"]["formal_performance_eligible"] is False
    assert "capture_audit" not in sample.metadata


def test_query_sample_enforces_explicit_event_limit() -> None:
    with pytest.raises(ValueError, match="exceeds max_events"):
        dependency_closed_query_sample(
            _two_query_trace(), TraceSampleConfig(
                (QueryRange(1, 1),), max_events=9, max_dependencies=20,
                scan_events=4, scan_backend="cpu",
            ),
        )


def test_query_sample_enforces_dependency_and_id_limits() -> None:
    with pytest.raises(ValueError, match="max_dependencies"):
        dependency_closed_query_sample(
            _two_query_trace(), TraceSampleConfig(
                (QueryRange(1, 1),), max_events=10, max_dependencies=1,
                scan_events=4, scan_backend="cpu",
            ),
        )
    with pytest.raises(ValueError, match="query ranges"):
        QueryRange(2**63, 1)

    with pytest.raises(ValueError, match="max_dependencies"):
        dependency_closed_query_sample(
            _two_query_trace(), TraceSampleConfig(
                (QueryRange(0, 2),), max_events=20, max_dependencies=11,
                scan_events=10, scan_backend="cpu",
            ),
        )


def test_cli_writes_sample_and_cycle_cli_rejects_implicit_formal_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sample"
    TraceWriter().write(_two_query_trace(), source, validate=False)
    assert main([
        "trace-sample", "--trace", str(source), "--output", str(output),
        "--query-range", "1:1", "--max-events", "10", "--scan-events", "4",
        "--max-dependencies", "20",
        "--scan-backend", "cpu",
    ]) == 0
    capsys.readouterr()

    assert main([
        "cycle-replay", "--trace", str(output),
        "--config", "configs/architecture/gala.yaml",
        "--resource-usage", str(tmp_path / "missing.json"),
        "--output", str(tmp_path / "run"),
    ]) == 2
    assert "--quick-validation" in capsys.readouterr().err
