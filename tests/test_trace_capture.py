from __future__ import annotations

from pathlib import Path

import numpy as np

from gala_sim.adapters.trace_capture import (
    LOSS_L1,
    LOSS_SSIM,
    LOSS_TV,
    RASTER_TEMPLATE_ID,
    STATE_FIELD_MASK,
    VOXEL_TEMPLATE_ID,
    TraceSession,
    _QueryContext,
)
from gala_sim.clamp import PrimitiveKind
from gala_sim.trace import validate_trace


def _rows(trace, kind: PrimitiveKind) -> np.ndarray:
    return trace.events[trace.events["primitive_kind"] == int(kind)]


def test_capture_expands_each_valid_mask_bit_into_a_query_relation(tmp_path: Path) -> None:
    session = TraceSession(tmp_path / "trace", chunk_events=16)
    session._ensure_gaussians(2)
    tile_one_key = np.int64(np.uint64(1) << np.uint64(32))
    records = np.asarray([
        [0, 0, 0, 0],
        [0, 1, 1, tile_one_key],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 16, 0],
    ], dtype=np.int64)
    query_count = 2 * 17
    session._audit.update({
        "official_raster_kernel_calls": 1,
        "captured_raster_kernel_calls": 1,
        "captured_query_kernel_calls": 1,
        "cuda_relation_candidates": 2,
        "captured_logical_queries": query_count,
    })
    session._emit_query_records(
        records, rendered=2, query_base=0, query_shape=(2, 17),
        binning_pointer=10, output_pointer=20,
        template_id=RASTER_TEMPLATE_ID, field_mask=STATE_FIELD_MASK,
    )
    context = session._contexts[10]
    context.loss_flags = LOSS_L1 | LOSS_SSIM
    context.ssim_radius = 1
    session._capture_backward(10, voxel=False)
    trace = session.finish()

    report = validate_trace(trace)
    assert report.counts[PrimitiveKind.RELATION.name] == 3
    assert report.counts[PrimitiveKind.QUERY_CLOSE.name] == query_count
    assert report.counts[PrimitiveKind.CONSUMER.name] == query_count
    relations = _rows(trace, PrimitiveKind.RELATION)
    assert relations["query_id"].tolist() == [0, 16, 33]
    candidates = _rows(trace, PrimitiveKind.RELATION_CANDIDATE)
    assert candidates["flags"].tolist() == [1, 1]
    candidate_event_ids = candidates["event_id"].tolist()
    assert trace.dependency_ids(relations[0]).tolist() == [candidate_event_ids[0]]
    assert trace.dependency_ids(relations[1]).tolist() == [candidate_event_ids[1]]

    consumer = _rows(trace, PrimitiveKind.CONSUMER)[16]
    dependency_queries = trace.events[trace.dependency_ids(consumer)]["query_id"].tolist()
    assert dependency_queries == [15, 16, 32, 33]


def test_voxel_query_offsets_match_official_x_y_z_layout() -> None:
    keys = np.zeros(8, dtype=np.uint64)
    local_queries = np.asarray([0, 1, 8, 9, 64, 65, 72, 73], dtype=np.int64)
    offsets = TraceSession._decode_query_offsets(
        keys, local_queries, (2, 2, 2), VOXEL_TEMPLATE_ID
    )
    assert offsets.tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


def test_tv_consumer_uses_axis_adjacent_query_reductions() -> None:
    context = _QueryContext(
        query_base=0,
        query_shape=(2, 2, 2),
        relation_query_offsets=np.empty(0, dtype=np.int64),
        gaussian_ids=np.empty(0, dtype=np.int64),
        relation_ids=np.empty(0, dtype=np.int64),
        reduction_events=np.arange(8, dtype=np.int64),
        buffer_pointer=0,
        output_pointer=0,
        template_id=VOXEL_TEMPLATE_ID,
        field_mask=STATE_FIELD_MASK,
        loss_flags=LOSS_TV,
    )
    assert TraceSession._consumer_query_offsets(context, 0).tolist() == [0, 4, 2, 1]
