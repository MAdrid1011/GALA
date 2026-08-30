from __future__ import annotations

import numpy as np
import pytest

from gala_sim.timing.kernel import (
    iter_packed_relation_packet_batches, packed_tile_statistics,
)
from gala_sim.trace.virtual import (
    RASTER_TEMPLATE_ID,
    VOXEL_TEMPLATE_ID,
    VirtualInterleavedQueryEventExpander,
    VirtualTracePacket,
)


def _packet(
    *, template_id: int, query_shape: tuple[int, ...],
    tiles: tuple[int, ...], bits: tuple[tuple[int, ...], ...],
    loss_flags: int,
) -> VirtualTracePacket:
    words = 8 if template_id == RASTER_TEMPLATE_ID else 16
    masks = np.zeros((len(tiles), words), dtype=np.dtype("<u4"))
    for candidate, active_bits in enumerate(bits):
        for bit in active_bits:
            masks[candidate, bit // 32] |= np.uint32(1 << (bit % 32))
    point_ids = np.arange(100, 100 + len(tiles), dtype=np.int64)
    point_keys = (
        np.asarray(tiles, dtype=np.uint64) << np.uint64(32)
    ) | np.arange(len(tiles), dtype=np.uint64)
    return VirtualTracePacket(
        iteration_id=1,
        template_id=template_id,
        query_base=1000,
        query_shape=query_shape,
        point_ids=point_ids,
        point_keys=point_keys,
        masks=masks,
        state_version=2,
        field_mask=7,
        loss_flags=loss_flags,
        ssim_radius=1,
        backward_confirmed=True,
    )


def _legacy_records(source: VirtualTracePacket) -> list[tuple[int, ...]]:
    expander = VirtualInterleavedQueryEventExpander(
        max_events=1024,
        relation_capacity=1024,
        window_capacity=16,
        continuation_query_packs=4,
        relation_query_lanes=8,
    )
    result: list[tuple[int, ...]] = []
    packet_ordinal = 0
    for query_pack, columns in enumerate(
        source.iter_query_pack_relation_arrays(query_lanes=8)
    ):
        candidates, query_ids, _gaussian_ids, _point_keys = columns
        stages, _positions = expander._relation_groups(
            candidates, query_ids, source, 0, packet_ordinal,
        )
        for stage in stages:
            positions = np.asarray(stage.event_ids, dtype=np.int64)
            candidate = int(candidates[positions[0]])
            release = int(source._consumer_ready_pack_indices(
                query_ids[positions] - source.query_base,
                query_lanes=8,
            ).max())
            result.append((
                int(stage.relation_packet_id),
                query_pack,
                candidate,
                stage.query_base,
                int(source.point_ids[candidate]),
                int(source.point_keys[candidate]),
                release,
                stage.lane_mask,
                stage.lane_mask.bit_count(),
            ))
        packet_ordinal += len(stages)
    return result


@pytest.mark.parametrize("source", [
    _packet(
        template_id=RASTER_TEMPLATE_ID,
        query_shape=(17, 19),
        tiles=(0, 0, 1, 2, 3),
        bits=(
            (0, 3, 7, 8, 15, 17, 31, 255),
            (1, 4, 9, 18, 24),
            (0, 2, 17),
            (0, 7, 8, 15),
            (0, 2),
        ),
        loss_flags=2,
    ),
    _packet(
        template_id=VOXEL_TEMPLATE_ID,
        query_shape=(9, 10, 11),
        tiles=(0, 0, 1, 2, 4, 7),
        bits=(
            (0, 1, 7, 8, 15, 64, 71, 511),
            (2, 9, 65, 72),
            (0, 2, 8, 10),
            (0, 1, 7, 64),
            (0, 2, 8, 10),
            (0, 1, 2),
        ),
        loss_flags=4,
    ),
])
def test_packed_relation_packets_match_legacy_physical_groups(
    source: VirtualTracePacket,
) -> None:
    batches = tuple(iter_packed_relation_packet_batches(
        source, query_lanes=8, max_packets=2,
    ))
    records = np.concatenate([batch.records for batch in batches])
    observed = [tuple(int(row[name]) for name in records.dtype.names) for row in records]

    assert observed == _legacy_records(source)
    assert sum(batch.logical_relation_count for batch in batches) == (
        source.logical_relation_count
    )
    assert [batch.packet_count for batch in batches[:-1]] == [2] * (
        len(batches) - 1
    )


def test_packed_relation_packets_require_frozen_eight_lane_contract() -> None:
    source = _packet(
        template_id=RASTER_TEMPLATE_ID,
        query_shape=(1, 1),
        tiles=(0,),
        bits=((0,),),
        loss_flags=0,
    )
    with pytest.raises(ValueError, match="eight lanes"):
        tuple(iter_packed_relation_packet_batches(
            source, query_lanes=4, max_packets=1,
        ))


def test_packed_tile_statistics_close_exact_packet_totals() -> None:
    source = _packet(
        template_id=RASTER_TEMPLATE_ID,
        query_shape=(17, 19),
        tiles=(0, 0, 1, 2, 3),
        bits=(
            (0, 3, 7, 8, 15, 17, 31, 255),
            (1, 4, 9, 18, 24),
            (0, 2, 17),
            (0, 7, 8, 15),
            (0, 2),
        ),
        loss_flags=2,
    )
    statistics = packed_tile_statistics(source)
    records = np.concatenate([
        batch.records for batch in iter_packed_relation_packet_batches(
            source, query_lanes=8, max_packets=3,
        )
    ])
    record_tiles = np.right_shift(
        records["point_key"], np.uint64(32),
    ).astype(np.int64, copy=False)

    assert statistics.candidate_counts.tolist() == [2, 1, 1, 1]
    assert int(statistics.logical_relation_counts.sum()) == source.logical_relation_count
    assert int(statistics.physical_packet_counts.sum()) == records.size
    assert np.array_equal(
        statistics.physical_packet_counts,
        np.bincount(record_tiles, minlength=4),
    )
    assert int(statistics.lane_histograms[:, 1:].sum()) == records.size
    for lane_count in range(1, 9):
        assert np.array_equal(
            statistics.lane_histograms[:, lane_count],
            np.bincount(
                record_tiles[records["active_lanes"] == lane_count],
                minlength=4,
            ),
        )
