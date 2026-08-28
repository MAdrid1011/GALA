from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from gala_sim.trace import (
    VirtualEventStreamValidator,
    VirtualQueryEventExpander,
    Trace,
    VirtualRelationEventExpander,
    VirtualTracePacket,
    VirtualTraceStream,
)
from gala_sim.clamp.events import event_dtype, dependency_dtype, EVENT_SCHEMA_VERSION
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
        loss_flags=1,
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
    assert result.peak_resident_packet_bytes == packets[0].physical_bytes


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
