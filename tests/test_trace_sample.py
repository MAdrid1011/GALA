from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import PrimitiveKind, ResourceClass, TraceBuilder, TraceEvent
from gala_sim.trace import (
    QueryPacketSampleConfig, QueryRange, TraceReader, TraceSampleConfig, TraceWriter,
    dependency_closed_query_sample, real_query_packet_sample, validate_trace,
)
from gala_sim.trace.sample import _gather_value_batches
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


def _two_iteration_relation_source():
    builder = TraceBuilder()
    gaussian_id = 0
    for iteration_id, query_base, state_version in ((1, 10, 4), (2, 20, 5)):
        for query_id in range(query_base, query_base + 2):
            candidate = builder.emit(TraceEvent(
                iteration_id=iteration_id,
                primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
                gaussian_id=gaussian_id,
                state_version=state_version,
                resource_class=int(ResourceClass.RELATION),
                address_token=gaussian_id + 1,
                template_id=1,
                field_mask=3,
            ))
            builder.emit(TraceEvent(
                iteration_id=iteration_id,
                primitive_kind=int(PrimitiveKind.RELATION),
                query_id=query_id,
                gaussian_id=gaussian_id,
                relation_id=gaussian_id,
                state_version=state_version,
                resource_class=int(ResourceClass.RELATION),
                template_id=1,
                field_mask=3,
            ), dependencies=[candidate])
            builder.emit(TraceEvent(
                iteration_id=iteration_id,
                primitive_kind=int(PrimitiveKind.CONSUMER),
                query_id=query_id,
                consumer_id=query_id,
                state_version=state_version,
                resource_class=int(ResourceClass.QUERY),
                template_id=1,
                field_mask=3,
                flags=1,
            ))
            gaussian_id += 1
    return builder.finish(metadata={
        "model": "R2-Gaussian",
        "dataset": "Chest",
        "initial_gaussian_count": 2,
        "state_record_bytes": 128,
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


def test_dependency_value_batches_split_duplicate_frontier_edges() -> None:
    source = np.asarray([0, 1, 0, 1], dtype=np.uint64)
    begins = np.asarray([0, 2], dtype=np.uint64)
    counts = np.asarray([2, 2], dtype=np.uint64)

    batches = list(_gather_value_batches(source, begins, counts, max_values=3))

    assert [batch.tolist() for batch in batches] == [[0, 1], [0, 1]]


def test_real_query_packet_sample_preserves_supports_and_rebases_history() -> None:
    sample = real_query_packet_sample(
        _two_iteration_relation_source(),
        QueryPacketSampleConfig(
            (QueryRange(10, 2), QueryRange(20, 2)),
            scan_events=4,
            scan_backend="cpu",
            query_lanes=2,
        ),
        source_identity="fixture",
    )

    report = validate_trace(sample)
    relation_rows = sample.events[
        sample.events["primitive_kind"] == int(PrimitiveKind.RELATION)
    ]
    metadata = sample.metadata["trace_sample"]
    assert report.event_count == 40
    assert relation_rows["query_id"].tolist() == [10, 11, 20, 21]
    assert relation_rows["gaussian_id"].tolist() == [0, 1, 2, 3]
    assert relation_rows["state_version"].tolist() == [4, 4, 5, 5]
    assert metadata["source_state_versions"] == [4, 5]
    assert metadata["state_versions_preserved"] is True
    assert [packet["source_relation_count"] for packet in metadata["packets"]] == [2, 2]
    assert [packet["query_base"] for packet in metadata["packets"]] == [10, 20]
    assert [packet["query_shape"] for packet in metadata["packets"]] == [[1, 2], [1, 2]]
    assert metadata["eligible_policies"][:6] == [
        "base", "query", "residency", "full",
        "query_oracle", "residency_oracle",
    ]
    assert metadata["eligible_policies"][6:] == [
        "variant:0000", "variant:1000", "variant:1010",
        "variant:0100", "variant:0101", "variant:1100", "variant:1111",
    ]


def test_real_query_packet_sample_accepts_matching_rows_per_iteration() -> None:
    sample = real_query_packet_sample(
        _two_iteration_relation_source(),
        QueryPacketSampleConfig(
            (
                QueryRange(10, 1), QueryRange(11, 1),
                QueryRange(20, 1), QueryRange(21, 1),
            ),
            scan_events=4,
            scan_backend="cpu",
            query_lanes=2,
        ),
        source_identity="fixture",
    )

    packets = sample.metadata["trace_sample"]["packets"]
    assert [(item["iteration_id"], item["query_base"]) for item in packets] == [
        (1, 10), (1, 11), (2, 20), (2, 21),
    ]
    assert sample.event_count == 40


def test_cli_writes_real_query_packet_sample(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    output = tmp_path / "query-packets"
    TraceWriter().write(_two_iteration_relation_source(), source, validate=False)

    assert main([
        "trace-query-packets", "--trace", str(source), "--output", str(output),
        "--query-range", "10:2", "--query-range", "20:2",
        "--scan-events", "4", "--scan-backend", "cpu",
        "--query-lanes", "2",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["logical_relation_events"] == 4
    assert result["formal_performance_eligible"] is False
    assert TraceReader().read(output).metadata["trace_sample"][
        "state_versions_preserved"
    ] is True


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
