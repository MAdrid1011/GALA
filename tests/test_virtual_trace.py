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
from gala_sim.clamp.events import (
    EVENT_SCHEMA_VERSION,
    PrimitiveKind,
    decode_relation_packet_flags,
    dependency_dtype,
    event_dtype,
)
from gala_sim.trace import validate_trace


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


def test_query_expander_carries_external_state_barrier_on_first_candidate() -> None:
    masks = _mask(1, 8)
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=2, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0]), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks, loss_flags=1, backward_confirmed=True,
    )
    expander = VirtualQueryEventExpander(max_events=2, next_event_id=4)
    packets = tuple(expander.expand(source, external_dependencies=(1, 3)))
    assert packets[0].dependency_ids(0).tolist() == [1, 3]


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
