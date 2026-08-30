from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from gala_sim.trace import (
    VirtualEventStreamValidator,
    VirtualQueryEventExpander,
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualTraceLifecycleValidator,
    Trace,
    VirtualRelationEventExpander,
    VirtualTracePacket,
    VirtualTraceStream,
    compare_virtual_packet_records,
)
from gala_sim.trace.virtual import (
    LOSS_SSIM,
    LOSS_TV,
    VirtualInterleavedQueryEventExpander,
    _consumer_dependency_batch,
    _consumer_offsets,
)
from gala_sim.clamp.events import (
    EVENT_SCHEMA_VERSION,
    PrimitiveKind,
    decode_relation_packet_flags,
    dependency_dtype,
    event_dtype,
)
from gala_sim.trace import validate_trace
from gala_sim.timing.packets import RelationPacketPlan


def _mask(candidate_count: int, words: int) -> np.ndarray:
    return np.zeros((candidate_count, words), dtype=np.dtype("<u4"))


def test_raster_packet_matches_decoder_query_major_order() -> None:
    masks = _mask(3, 8)
    masks[0, 0] = np.uint32(1 << 0)
    masks[1, 0] = np.uint32((1 << 0) | (1 << 16))
    masks[2, 0] = np.uint32(1 << 0)
    keys = np.asarray([
        np.uint64(0) << np.uint64(32),
        np.uint64(1) << np.uint64(32),
        np.uint64(1) << np.uint64(32),
    ], dtype=np.uint64)
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=100,
        query_shape=(2, 17),
        point_ids=np.asarray([4, 5, 6], dtype=np.int64),
        point_keys=keys,
        masks=masks,
    )

    assert packet.logical_relation_count == 4
    assert packet.materialize_relations().tolist() == [
        [0, 100, 4, 0],
        [1, 116, 5, 1 << 32],
        [2, 116, 6, 1 << 32],
        [1, 133, 5, 1 << 32],
    ]
    assert list(packet.iter_relations_candidate_order()) == [
        (0, 100, 4, 0),
        (1, 116, 5, 1 << 32),
        (1, 133, 5, 1 << 32),
        (2, 116, 6, 1 << 32),
    ]


def test_packet_comparator_matches_legacy_candidate_and_relation_columns() -> None:
    masks = _mask(3, 8)
    masks[0, 0] = np.uint32(1 << 0)
    masks[1, 0] = np.uint32((1 << 0) | (1 << 16))
    masks[2, 0] = np.uint32(1 << 0)
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=100,
        query_shape=(2, 17),
        point_ids=np.asarray([40, 50, 60], dtype=np.int64),
        point_keys=np.asarray([0, 1 << 32, 1 << 32], dtype=np.uint64),
        masks=masks,
    )
    candidates = np.asarray([
        [0, 0, 0, 0],
        [0, 1, 1, 1 << 32],
        [0, 2, 2, 1 << 32],
    ], dtype=np.int64)
    relations = np.asarray([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 16, 0],
        [1, 2, 0, 0],
    ], dtype=np.int64)
    result = compare_virtual_packet_records(
        packet, candidates, relations,
        gaussian_ids=np.asarray([40, 50, 60], dtype=np.int64),
        relation_batch_size=2,
    )
    assert result.ok
    assert result.candidate_mismatches == 0
    assert result.relation_mismatches == 0


def test_packet_comparator_reports_relation_field_mismatch() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32(1)
    packet = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0]), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
    )
    result = compare_virtual_packet_records(
        packet,
        np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        np.asarray([[1, 0, 1, 0]], dtype=np.int64),
    )
    assert not result.ok
    assert result.relation_mismatches == 1
    assert any(
        "local_query" in message or "mask-bit" in message
        for message in result.first_mismatches
    )


def test_packet_comparator_supports_global_candidate_chunk_indexes() -> None:
    masks = _mask(1, 16)
    masks[0, 0] = np.uint32(1)
    packet = VirtualTracePacket(
        iteration_id=600, template_id=2, query_base=0, query_shape=(9, 8, 8),
        point_ids=np.asarray([7]),
        point_keys=np.asarray([np.uint64(1) << np.uint64(32)], dtype=np.uint64),
        masks=masks,
    )
    result = compare_virtual_packet_records(
        packet,
        np.asarray([[0, 50, 7, 1 << 32]], dtype=np.int64),
        np.asarray([[1, 50, 0, 0]], dtype=np.int64),
        candidate_index_base=50,
    )
    assert result.ok


def test_packet_comparator_rejects_duplicate_relation_mask_bit() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32(0b11)
    packet = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 2),
        point_ids=np.asarray([0]), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
    )
    result = compare_virtual_packet_records(
        packet,
        np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int64),
        relation_batch_size=1,
    )
    assert not result.ok
    assert any(
        "order/duplicate" in value or "local_query" in value
        for value in result.first_mismatches
    )


def test_packet_comparator_detects_missing_mask_bit_replaced_by_another() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32(0b11)
    packet = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 2),
        point_ids=np.asarray([0]), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
    )
    result = compare_virtual_packet_records(
        packet,
        np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        np.asarray([[1, 0, 0, 0], [1, 0, 2, 0]], dtype=np.int64),
    )
    assert not result.ok
    assert result.relation_mismatches == 1
    assert any("local_query" in value for value in result.first_mismatches)


def test_voxel_packet_maps_tile_and_local_bit_to_global_query() -> None:
    masks = _mask(2, 16)
    masks[0, 0] = np.uint32(1 << 0)
    masks[1, 0] = np.uint32(1 << 0)  # tile x=1, local x=0, y=0, z=0
    packet = VirtualTracePacket(
        iteration_id=600,
        template_id=2,
        query_base=0,
        query_shape=(9, 8, 8),
        point_ids=np.asarray([7, 8], dtype=np.int64),
        point_keys=np.asarray([0, np.uint64(1) << np.uint64(32)], dtype=np.uint64),
        masks=masks,
    )

    assert packet.materialize_relations().tolist() == [
        [0, 0, 7, 0],
        [1, 512, 8, 1 << 32],
    ]


def test_packet_rejects_edge_tile_bit_outside_query_shape() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32(1 << 1)
    with pytest.raises(ValueError, match="outside"):
        VirtualTracePacket(
            iteration_id=1,
            template_id=1,
            query_base=0,
            query_shape=(1, 17),
            point_ids=np.asarray([0], dtype=np.int64),
            point_keys=np.asarray([np.uint64(1) << np.uint64(32)], dtype=np.uint64),
            masks=masks,
        )


def test_relation_batches_are_bounded_and_exact() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32((1 << 0) | (1 << 1) | (1 << 2))
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 3),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
    )
    assert [len(batch) for batch in packet.iter_relation_batches(2)] == [2, 1]
    with pytest.raises(ValueError, match="exceeds"):
        packet.materialize_relations(max_relations=2)


def test_relation_store_wavefront_accounts_for_ssim_consumer_frontier() -> None:
    masks = _mask(2, 8)
    for y in range(3):
        for x in range(8):
            local_query = y * 16 + x
            masks[:, local_query // 32] |= np.uint32(1 << (local_query % 32))
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(3, 8),
        point_ids=np.asarray([4, 5], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=LOSS_SSIM,
        ssim_radius=1,
        backward_confirmed=True,
    )

    infeasible = packet.relation_store_wavefront(
        query_lanes=8, relation_capacity=3,
    )
    feasible = packet.relation_store_wavefront(
        query_lanes=8, relation_capacity=4,
    )

    assert infeasible.query_pack_count == 3
    assert infeasible.physical_relation_records == 6
    assert infeasible.peak_live_records == 4
    assert infeasible.peak_query_pack == 1
    assert not infeasible.feasible
    assert feasible.feasible


def test_query_stream_schedule_closes_relation_and_window_frontiers() -> None:
    masks = _mask(2, 8)
    for y in range(3):
        for x in range(8):
            local_query = y * 16 + x
            masks[:, local_query // 32] |= np.uint32(1 << (local_query % 32))
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(3, 8),
        point_ids=np.asarray([4, 5], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=LOSS_SSIM,
        ssim_radius=1,
        backward_confirmed=True,
    )

    schedule = packet.query_stream_schedule(
        query_lanes=8,
        relation_capacity=4,
        window_capacity=2,
        continuation_query_packs=2,
    )

    assert [info.relation_count for info in schedule.infos] == [16, 16, 16]
    assert [info.physical_relation_records for info in schedule.infos] == [2, 2, 2]
    assert schedule.operations == (
        ("producer", 0),
        ("producer", 1),
        ("backward", 0),
        ("producer", 2),
        ("backward", 1),
        ("backward", 2),
    )
    assert schedule.peak_relation_records == 4
    assert schedule.peak_windows == 2

    with pytest.raises(ValueError, match="relation_store_capacity_infeasible"):
        packet.query_stream_schedule(
            query_lanes=8,
            relation_capacity=3,
            window_capacity=2,
            continuation_query_packs=2,
        )


@pytest.mark.parametrize(
    ("template_id", "query_shape", "tile_ids", "candidate_bits"),
    [
        (
            1,
            (3, 18),
            (0, 0, 1, 1),
            (
                (0, 1, 7, 8, 15, 16, 31, 32, 47),
                (0, 2, 8, 33),
                (0, 1, 16, 17, 32, 33),
                (1, 17, 32),
            ),
        ),
        (
            2,
            (9, 3, 10),
            (0, 1, 2, 3),
            (
                (0, 1, 7, 8, 15, 64, 72, 136, 471),
                (0, 1, 7, 8, 15, 16, 23),
                (0, 1, 8, 9, 64, 65, 465),
                (0, 1, 8, 9, 16, 17),
            ),
        ),
    ],
    ids=("raster", "voxel"),
)
def test_relation_pack_counts_match_materialized_pack_units(
    template_id: int,
    query_shape: tuple[int, ...],
    tile_ids: tuple[int, ...],
    candidate_bits: tuple[tuple[int, ...], ...],
) -> None:
    mask_words = 8 if template_id == 1 else 16
    masks = _mask(len(tile_ids), mask_words)
    for candidate, bits in enumerate(candidate_bits):
        for bit in bits:
            masks[candidate, bit // 32] |= (
                np.uint32(1) << np.uint32(bit % 32)
            )
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=template_id,
        query_base=0,
        query_shape=query_shape,
        point_ids=np.arange(len(tile_ids), dtype=np.int64),
        point_keys=np.left_shift(
            np.asarray(tile_ids, dtype=np.uint64), np.uint64(32),
        ),
        masks=masks,
    )
    query_lanes = 4
    tiles = np.right_shift(packet.point_keys, np.uint64(32))
    order = np.argsort(tiles, kind="stable")
    sorted_tiles = tiles[order]
    tile_domain = np.arange(packet._tile_count())
    starts = np.searchsorted(sorted_tiles, tile_domain, side="left")
    ends = np.searchsorted(sorted_tiles, tile_domain, side="right")

    units = tuple(packet._iter_relation_pack_units(
        order, starts, ends, query_lanes=query_lanes,
    ))
    counts = tuple(packet._iter_relation_pack_counts(
        order, starts, ends, query_lanes=query_lanes,
    ))

    assert len(counts) == len(units)
    for (pack_index, relation_count, record_count), (
        candidates, query_offsets,
    ) in zip(counts, units, strict=True):
        pack_indices = packet._query_pack_indices(
            query_offsets, query_lanes=query_lanes,
        )
        assert np.all(pack_indices == pack_index)
        assert relation_count == candidates.size
        assert record_count == np.unique(candidates).size


@pytest.mark.parametrize(
    ("template_id", "query_shape", "loss_flags", "ssim_radius", "mask_bits"),
    [
        (1, (3, 6), LOSS_SSIM, 1, tuple(
            y * 16 + x for y in range(3) for x in range(6)
        )),
        (2, (2, 2, 8), LOSS_TV, 0, tuple(
            x * 64 + y * 8 + z
            for x in range(2) for y in range(2) for z in range(8)
        )),
    ],
)
def test_interleaved_query_expander_preserves_full_event_graph(
    template_id: int,
    query_shape: tuple[int, ...],
    loss_flags: int,
    ssim_radius: int,
    mask_bits: tuple[int, ...],
) -> None:
    mask_words = 8 if template_id == 1 else 16
    masks = _mask(2, mask_words)
    for bit in mask_bits:
        masks[:, bit // 32] |= np.uint32(1 << (bit % 32))
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=template_id,
        query_base=20,
        query_shape=query_shape,
        point_ids=np.asarray([4, 5], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=loss_flags,
        ssim_radius=ssim_radius,
        backward_confirmed=True,
    )
    reference_packets = tuple(VirtualQueryEventExpander(
        max_events=7, relation_query_lanes=4,
    ).expand(source))
    continuations = tuple(VirtualInterleavedQueryEventExpander(
        max_events=7,
        relation_capacity=100,
        window_capacity=100,
        continuation_query_packs=2,
        relation_query_lanes=4,
    ).expand(source))
    streamed_packets = tuple(
        packet for continuation in continuations
        for packet in continuation.event_packets
    )

    validator = VirtualEventStreamValidator()
    for packet in streamed_packets:
        validator.accept(packet)
    assert validator.accepted_events == source.logical_expanded_event_count

    reference_rows = np.concatenate([
        packet.events for packet in reference_packets
    ])
    streamed_rows = np.concatenate([
        packet.events for packet in streamed_packets
    ])

    def identities(rows: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]]]:
        by_identity: dict[tuple[int, int], int] = {}
        candidate_ordinal = 0
        for row in rows:
            kind = PrimitiveKind(int(row["primitive_kind"]))
            if kind is PrimitiveKind.RELATION_CANDIDATE:
                identity = (int(kind), candidate_ordinal)
                candidate_ordinal += 1
            elif kind in {
                PrimitiveKind.QUERY_CLOSE,
                PrimitiveKind.QUERY_REDUCTION,
                PrimitiveKind.CONSUMER,
            }:
                identity = (int(kind), int(row["query_id"]))
            else:
                identity = (int(kind), int(row["relation_id"]))
            assert identity not in by_identity
            by_identity[identity] = int(row["event_id"])
        return by_identity, {
            event_id: identity for identity, event_id in by_identity.items()
        }

    reference_ids, reference_identity = identities(reference_rows)
    streamed_ids, streamed_identity = identities(streamed_rows)
    assert reference_ids.keys() == streamed_ids.keys()
    ignored_fields = {"event_id", "dependency_begin", "dependency_count"}
    reference_by_identity = {
        reference_identity[int(row["event_id"])]: row for row in reference_rows
    }
    streamed_by_identity = {
        streamed_identity[int(row["event_id"])]: row for row in streamed_rows
    }
    for identity in reference_ids:
        reference = reference_by_identity[identity]
        streamed = streamed_by_identity[identity]
        for field in event_dtype().names:
            if field not in ignored_fields:
                assert int(streamed[field]) == int(reference[field]), (
                    identity, field,
                )

    def dependency_graph(
        packets: tuple,
        identity_by_event: dict[int, tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
        return {
            identity_by_event[int(row["event_id"])]: tuple(
                identity_by_event[int(dependency)]
                for dependency in packet.dependency_ids(index)
            )
            for packet in packets
            for index, row in enumerate(packet.events)
        }

    assert dependency_graph(
        streamed_packets, streamed_identity,
    ) == dependency_graph(reference_packets, reference_identity)

    inferred = RelationPacketPlan.from_event_packets(
        streamed_packets, query_lanes=4,
    )
    streamed_stages = tuple(
        stage for continuation in continuations
        for stage in continuation.physical_stages
    )
    assert [
        (stage.kind, stage.event_ids, stage.lanes, stage.lane_mask, stage.query_base)
        for stage in streamed_stages
    ] == [
        (stage.kind, stage.event_ids, stage.lanes, stage.lane_mask, stage.query_base)
        for stage in inferred.stages
    ]


def test_relation_event_expander_preserves_global_ids_and_external_dependencies() -> None:
    masks = _mask(2, 8)
    masks[0, 0] = np.uint32(1)
    masks[1, 0] = np.uint32((1 << 0) | (1 << 1))
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 2),
        point_ids=np.asarray([0, 1], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
    )
    expander = VirtualRelationEventExpander(max_events=2)
    output = list(expander.expand(source))

    assert [packet.global_event_start for packet in output] == [0, 2, 4]
    assert [packet.event_count for packet in output] == [2, 2, 1]
    assert output[1].external_dependencies.tolist() == [0, 1]
    assert output[2].external_dependencies.tolist() == [1]
    assert output[1].events["query_id"].tolist() == [0, 0]
    assert output[2].events["query_id"].tolist() == [1]
    assert expander.next_event_id == 5
    assert expander.next_relation_id == 3


def test_relation_packet_metadata_preserves_sparse_lane_positions() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
    )
    packets = tuple(
        VirtualRelationEventExpander(
            max_events=1, relation_query_lanes=8,
        ).expand(source)
    )
    relation_rows = np.concatenate([
        packet.events[
            packet.events["primitive_kind"] == int(PrimitiveKind.RELATION)
        ]
        for packet in packets
    ])

    assert relation_rows["query_id"].tolist() == [20, 27]
    assert [
        decode_relation_packet_flags(int(flags))
        for flags in relation_rows["flags"]
    ] == [(0, 0b10000001), (7, 0b10000001)]


def test_raster_relation_packet_metadata_does_not_cross_row_boundary() -> None:
    masks = _mask(1, 8)
    for local_query in (8, 9, 16):
        masks[0, local_query // 32] |= np.uint32(1 << (local_query % 32))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=100,
        query_shape=(2, 10), point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
    )
    packets = tuple(
        VirtualRelationEventExpander(
            max_events=2, relation_query_lanes=8,
        ).expand(source)
    )
    relation_rows = np.concatenate([
        packet.events[
            packet.events["primitive_kind"] == int(PrimitiveKind.RELATION)
        ]
        for packet in packets
    ])

    assert relation_rows["query_id"].tolist() == [108, 109, 110]
    assert [
        decode_relation_packet_flags(int(flags))
        for flags in relation_rows["flags"]
    ] == [(0, 0b00000011), (1, 0b00000011), (0, 0b00000001)]


def test_voxel_relation_packet_metadata_does_not_cross_brick_x_boundary() -> None:
    masks = _mask(1, 16)
    for local_query in (6, 7, 8):
        masks[0, local_query // 32] |= np.uint32(1 << (local_query % 32))
    source = VirtualTracePacket(
        iteration_id=1, template_id=2, query_base=100,
        query_shape=(2, 2, 8), point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
    )
    packets = tuple(
        VirtualRelationEventExpander(
            max_events=2, relation_query_lanes=8,
        ).expand(source)
    )
    relation_rows = np.concatenate([
        packet.events[
            packet.events["primitive_kind"] == int(PrimitiveKind.RELATION)
        ]
        for packet in packets
    ])

    assert relation_rows["query_id"].tolist() == [106, 107, 108]
    assert [
        decode_relation_packet_flags(int(flags))
        for flags in relation_rows["flags"]
    ] == [(6, 0b11000000), (7, 0b11000000), (0, 0b00000001)]


def test_full_relation_chain_inherits_packet_metadata_without_reduction_key_alias() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32((1 << 0) | (1 << 7))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=20, query_shape=(1, 8),
        point_ids=np.asarray([4], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    rows = np.concatenate([
        packet.events
        for packet in VirtualQueryEventExpander(
            max_events=1, relation_query_lanes=8,
        ).expand(source)
    ])
    chain_kinds = {
        PrimitiveKind.RELATION,
        PrimitiveKind.CACHE_REQUEST,
        PrimitiveKind.CACHE_RETURN,
        PrimitiveKind.FORWARD,
        PrimitiveKind.ADJOINT,
        PrimitiveKind.GRADIENT_REDUCTION,
    }
    chain_rows = rows[np.isin(
        rows["primitive_kind"], [int(kind) for kind in chain_kinds]
    )]

    for relation_id in (0, 1):
        relation_chain = chain_rows[chain_rows["relation_id"] == relation_id]
        expected_lane = 0 if relation_id == 0 else 7
        assert {
            decode_relation_packet_flags(int(flags))
            for flags in relation_chain["flags"]
        } == {(expected_lane, 0b10000001)}
    gradients = rows[
        rows["primitive_kind"] == int(PrimitiveKind.GRADIENT_REDUCTION)
    ]
    assert gradients["reduction_key"].tolist() == [4, 4]


def test_event_stream_validator_rejects_rebased_or_unclosed_packets() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = np.uint32(1)
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
    )
    packets = list(VirtualRelationEventExpander(max_events=1).expand(source))
    validator = VirtualEventStreamValidator()
    validator.accept(packets[0])
    rebased_events = packets[1].events.copy()
    rebased_events["event_id"] = 99
    with pytest.raises(ValueError, match="rebased"):
        validator.accept(type(packets[1])(
            packet_id=1, global_event_start=99,
            events=rebased_events, dependencies=packets[1].dependencies,
        ))
    with pytest.raises(ValueError, match="final"):
        validator.finalize()


def test_query_expander_builds_a_valid_full_query_chain() -> None:
    masks = _mask(2, 8)
    masks[0, 0] = np.uint32(1)
    masks[1, 0] = np.uint32((1 << 0) | (1 << 1))
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=10, query_shape=(1, 2),
        point_ids=np.asarray([0, 1], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    expander = VirtualQueryEventExpander(max_events=2)
    packets = list(expander.expand(source))
    validator = VirtualEventStreamValidator()
    for packet in packets:
        validator.accept(packet)
    assert validator.accepted_events == 26
    rows = []
    dependencies = []
    dependency_offset = 0
    for packet in packets:
        part = packet.events.copy()
        part["dependency_begin"] += dependency_offset
        rows.append(part)
        dependencies.append(packet.dependencies)
        dependency_offset += packet.dependencies.size
    trace = Trace(
        np.concatenate(rows),
        np.concatenate(dependencies or [np.empty(0, dtype=dependency_dtype())]),
        np.empty(0, dtype=np.dtype("<f4")),
        {"schema_version": EVENT_SCHEMA_VERSION},
    )
    report = validate_trace(trace)
    assert report.event_count == 26


@pytest.mark.parametrize(
    ("template_id", "query_shape", "loss_flags", "ssim_radius"),
    [
        (1, (7, 9), LOSS_SSIM, 2),
        (2, (3, 4, 5), LOSS_TV, 0),
    ],
)
def test_vectorized_consumer_dependencies_match_scalar_order(
    template_id: int,
    query_shape: tuple[int, ...],
    loss_flags: int,
    ssim_radius: int,
) -> None:
    mask_words = 8 if template_id == 1 else 16
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=template_id,
        query_base=0,
        query_shape=query_shape,
        point_ids=np.empty(0, dtype=np.int64),
        point_keys=np.empty(0, dtype=np.uint64),
        masks=np.empty((0, mask_words), dtype=np.dtype("<u4")),
        loss_flags=loss_flags,
        ssim_radius=ssim_radius,
    )
    reduction_start = 123
    dependencies, counts = _consumer_dependency_batch(
        source, 0, source.query_count, reduction_start,
    )
    expected_parts = tuple(
        reduction_start + _consumer_offsets(source, query)
        for query in range(source.query_count)
    )
    expected = np.concatenate(expected_parts)

    assert np.array_equal(dependencies, expected)
    assert counts.tolist() == [part.size for part in expected_parts]


def test_query_expander_carries_external_state_barrier_on_every_candidate() -> None:
    masks = _mask(3, 8)
    masks[:, 0] = 1
    source = VirtualTracePacket(
        iteration_id=2, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0, 1, 2]),
        point_keys=np.asarray([0, 0, 0], dtype=np.uint64),
        masks=masks, loss_flags=1, backward_confirmed=True,
    )
    expander = VirtualQueryEventExpander(max_events=2, next_event_id=4)
    packets = tuple(expander.expand(source, external_dependencies=(1, 3)))
    candidate_packets = packets[:2]
    assert [
        packet.dependency_ids(index).tolist()
        for packet in candidate_packets
        for index in range(packet.event_count)
    ] == [[1, 3], [1, 3], [1, 3]]
    close = next(
        (packet, index)
        for packet in packets
        for index, row in enumerate(packet.events)
        if int(row["primitive_kind"]) == int(PrimitiveKind.QUERY_CLOSE)
    )
    assert close[0].dependency_ids(close[1]).tolist()[-2:] == [1, 3]


def test_lifecycle_validator_tracks_updates_lineage_and_iteration_ledger() -> None:
    validator = VirtualTraceLifecycleValidator(initial_gaussian_count=2)
    masks = _mask(2, 8)
    masks[:, 0] = np.uint32(1)
    packet = VirtualTracePacket(
        iteration_id=600, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0, 1], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    validator.accept_packet(packet)
    validator.accept_lifecycle(VirtualLifecycleRecord(
        600, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=15, transaction_kind=2,
    ))
    for gaussian_id in (0, 1):
        validator.accept_lifecycle(VirtualLifecycleRecord(
            600, VirtualLifecycleKind.UPDATE_COMMIT, 0,
            field_mask=15, gaussian_id=gaussian_id, transaction_kind=2,
        ))
    validator.accept_lifecycle(VirtualLifecycleRecord(
        600, VirtualLifecycleKind.UPDATE_END, 0,
        field_mask=15, transaction_kind=2,
    ))
    validator.accept_lifecycle(VirtualLifecycleRecord(
        600, VirtualLifecycleKind.UPDATE_BEGIN, 1,
        field_mask=15, transaction_kind=1,
    ))
    validator.accept_lifecycle(VirtualLifecycleRecord(
        600, VirtualLifecycleKind.CLONE, 1,
        parent_id=0, child_ids=(2,), transaction_kind=1,
    ))
    validator.accept_lifecycle(VirtualLifecycleRecord(
        600, VirtualLifecycleKind.PRUNE, 1,
        gaussian_id=1, transaction_kind=1,
    ))
    validator.accept_lifecycle(VirtualLifecycleRecord(
        600, VirtualLifecycleKind.UPDATE_END, 1,
        field_mask=15, transaction_kind=1,
    ))
    ledger = validator.close_iteration(600)

    assert ledger.relation_count == ledger.backward_relation_count == 2
    assert ledger.optimizer_commits == 2
    assert ledger.collection_transactions == 1
    assert ledger.state_version_start == 0
    assert ledger.state_version_end == 2
    assert ledger.active_gaussian_count_start == ledger.active_gaussian_count_end == 2
    assert validator.finalize() == (ledger,)


def test_lifecycle_validator_rejects_cross_iteration_open_transaction() -> None:
    validator = VirtualTraceLifecycleValidator(initial_gaussian_count=1)
    validator.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=15, transaction_kind=2,
    ))
    with pytest.raises(ValueError, match="open update"):
        validator.close_iteration(1)


def test_lifecycle_validator_rejects_incomplete_optimizer_commit_set() -> None:
    validator = VirtualTraceLifecycleValidator(initial_gaussian_count=2)
    validator.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=1, transaction_kind=2,
    ))
    validator.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_COMMIT, 0,
        field_mask=1, gaussian_id=0, transaction_kind=2,
    ))
    with pytest.raises(ValueError, match="incomplete commits"):
        validator.accept_lifecycle(VirtualLifecycleRecord(
            1, VirtualLifecycleKind.UPDATE_END, 0,
            field_mask=1, transaction_kind=2,
        ))


def test_lifecycle_validator_rejects_noncontiguous_query_base() -> None:
    validator = VirtualTraceLifecycleValidator(initial_gaussian_count=1)
    masks = _mask(1, 8)
    masks[0, 0] = 1
    first = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0]), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks, backward_confirmed=True,
    )
    second = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=2, query_shape=(1, 1),
        point_ids=np.asarray([0]), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks, backward_confirmed=True,
    )
    validator.accept_packet(first)
    with pytest.raises(ValueError, match="query base is not contiguous"):
        validator.accept_packet(second)


def test_stream_bounds_inflight_packets_and_accounts_physical_bytes() -> None:
    packets = []
    for iteration in range(3):
        masks = _mask(1, 8)
        masks[0, 0] = np.uint32(1)
        packets.append(VirtualTracePacket(
            iteration_id=iteration,
            template_id=1,
            query_base=iteration,
            query_shape=(1, 1),
            point_ids=np.asarray([iteration], dtype=np.int64),
            point_keys=np.asarray([0], dtype=np.uint64),
            masks=masks,
        ))
    consumed = []
    result = VirtualTraceStream(
        packets, max_inflight_packets=1, inactivity_timeout_seconds=1.0,
    ).run(lambda packet: consumed.append(packet.iteration_id))

    assert consumed == [0, 1, 2]
    assert result.produced_packets == result.consumed_packets == 3
    assert result.logical_relation_count == 3
    assert result.physical_stream_bytes == sum(packet.physical_bytes for packet in packets)
    assert result.peak_resident_packet_bytes >= packets[0].physical_bytes


def test_stream_peak_resident_bytes_includes_queue_and_consumer() -> None:
    packets = []
    for iteration in range(3):
        masks = _mask(1, 8)
        masks[0, 0] = np.uint32(1)
        packets.append(VirtualTracePacket(
            iteration_id=iteration, template_id=1, query_base=iteration,
            query_shape=(1, 1), point_ids=np.asarray([0]),
            point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        ))
    result = VirtualTraceStream(
        packets, max_inflight_packets=2, inactivity_timeout_seconds=1.0,
    ).run(lambda _packet: time.sleep(0.02))
    assert result.peak_resident_packet_bytes >= 2 * packets[0].physical_bytes


def test_stream_stops_when_producer_makes_no_progress() -> None:
    release = threading.Event()

    def packets():
        release.wait(1.0)
        return iter(())

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no progress"):
        VirtualTraceStream(
            packets, max_inflight_packets=1, inactivity_timeout_seconds=0.05,
        ).run(lambda _packet: None)
    release.set()
    assert time.monotonic() - started < 0.5
