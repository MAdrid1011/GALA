from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.cli import main
from gala_sim.timing import RelationPacketPlan
from gala_sim.trace import (
    CapturedPacketSpec,
    captured_virtual_packet,
    complete_captured_packet_sample,
)


def _write_records(root: Path) -> tuple[Path, Path]:
    candidates = np.asarray([
        [0, 0, 4, 0 << 32],
        [0, 1, 5, 0 << 32],
        [0, 2, 6, 1 << 32],
    ], dtype=np.int64)
    relations = np.asarray([
        [1, 0, 0, 0],
        [1, 0, 7, 0],
        [1, 1, 1, 0],
        [1, 2, 3, 0],
    ], dtype=np.int64)
    candidate_path = root / "candidates.raw"
    relation_path = root / "relations.raw"
    candidates.tofile(candidate_path)
    relations.tofile(relation_path)
    return candidate_path, relation_path


def test_captured_packet_extracts_exact_candidate_major_tile(tmp_path: Path) -> None:
    candidates, relations = _write_records(tmp_path)
    spec = CapturedPacketSpec(
        candidates, relations,
        iteration_id=1,
        template_id=1,
        query_base=20,
        query_shape=(16, 16),
        tile_id=0,
        loss_flags=1,
    )

    packet = captured_virtual_packet(spec)

    assert packet.point_ids.tolist() == [4, 5]
    assert packet.candidate_count == 2
    assert packet.logical_relation_count == 3
    assert packet.masks[:, 0].tolist() == [0b10000001, 0b00000010]
    assert not np.any(packet.masks[:, 1:])


def test_captured_packet_can_rebase_a_nonzero_tile(tmp_path: Path) -> None:
    candidates, relations = _write_records(tmp_path)
    packet = captured_virtual_packet(CapturedPacketSpec(
        candidates, relations,
        iteration_id=1,
        template_id=1,
        query_base=1000,
        query_shape=(16, 16),
        tile_id=1,
        loss_flags=1,
    ))

    assert packet.point_ids.tolist() == [6]
    assert packet.point_keys.tolist() == [0]
    assert packet.query_base == 1000
    assert packet.logical_relation_count == 1
    assert packet.masks[0, 0] == np.uint32(1 << 3)


def test_captured_packet_sample_expands_complete_forward_and_backward(
    tmp_path: Path,
) -> None:
    candidates, relations = _write_records(tmp_path)
    spec = CapturedPacketSpec(
        candidates, relations,
        iteration_id=1,
        template_id=1,
        query_base=20,
        query_shape=(16, 16),
        tile_id=0,
        loss_flags=1,
    )

    trace = complete_captured_packet_sample(
        (spec,), max_events=2, query_lanes=8, initial_gaussian_count=7,
    )

    counts = {
        kind: int(np.count_nonzero(trace.events["primitive_kind"] == int(kind)))
        for kind in PrimitiveKind
    }
    assert counts[PrimitiveKind.RELATION_CANDIDATE] == 2
    for kind in (
        PrimitiveKind.RELATION,
        PrimitiveKind.CACHE_REQUEST,
        PrimitiveKind.CACHE_RETURN,
        PrimitiveKind.FORWARD,
        PrimitiveKind.ADJOINT,
        PrimitiveKind.GRADIENT_REDUCTION,
    ):
        assert counts[kind] == 3
    for kind in (
        PrimitiveKind.QUERY_CLOSE,
        PrimitiveKind.QUERY_REDUCTION,
        PrimitiveKind.CONSUMER,
    ):
        assert counts[kind] == 256
    plan = RelationPacketPlan.from_trace(trace, query_lanes=8)
    assert plan.relation_packet_count == 2
    assert trace.metadata["trace_sample"]["selection"] == (
        "complete_captured_physical_packets"
    )
    assert trace.metadata["trace_sample"]["eligible_policies"][-7:] == [
        "variant:0000", "variant:1000", "variant:1010",
        "variant:0100", "variant:0101", "variant:1100", "variant:1111",
    ]


def test_cli_builds_captured_packet_sample(
    tmp_path: Path, capsys,
) -> None:
    candidates, relations = _write_records(tmp_path)
    manifest = tmp_path / "packets.json"
    manifest.write_text(json.dumps({
        "schema_version": "gala-captured-packet-sample-v1",
        "packets": [{
            "candidate_records": str(candidates),
            "relation_records": str(relations),
            "iteration_id": 1,
            "template_id": 1,
            "query_base": 20,
            "query_shape": [16, 16],
            "tile_id": 0,
            "loss_flags": 1,
        }],
    }), encoding="utf-8")

    assert main([
        "trace-captured-packets",
        "--manifest", str(manifest),
        "--output", str(tmp_path / "trace"),
        "--max-events", "64",
        "--query-lanes", "8",
        "--initial-gaussian-count", "7",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["events"] == 788
    assert report["physical_relation_packets"] == 2
