from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from gala_sim.adapters.trace_capture import LOSS_L1, STATE_FIELD_MASK, TraceSession
from gala_sim.adapters.virtual_capture import VirtualCaptureConsumer
from gala_sim.trace import (
    VirtualPacketArchiveReader,
    VirtualPacketArchiveWriter,
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualTracePacket,
)
from gala_sim.tools.relation_capacity import run_relation_capacity_preflight


class _ArrayRef:
    def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
        self.shape = shape
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer


def _packet(*, iteration: int = 1, point_id: int = 0) -> VirtualTracePacket:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    return VirtualTracePacket(
        iteration_id=iteration,
        template_id=1,
        query_base=0,
        query_shape=(1, 1),
        point_ids=np.asarray([point_id], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        field_mask=STATE_FIELD_MASK,
        loss_flags=LOSS_L1,
        backward_confirmed=True,
    )


def test_virtual_packet_archive_roundtrip_preserves_payload_and_order(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=1024)
    writer.initialize_gaussians(2)
    packet = _packet(point_id=0)
    begin = VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2,
    )
    end = VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_END, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2,
    )
    writer.append_packet(packet)
    writer.append_lifecycle(begin)
    writer.append_lifecycle(end)
    writer.close_iteration(1)
    manifest = writer.finish()

    assert manifest["formal_performance_eligible"] is False
    assert manifest["chunk_count"] == 1
    assert not (archive_root / "events.raw").exists()
    records = list(VirtualPacketArchiveReader(archive_root).records())
    assert [kind for kind, _ in records] == [
        "packet", "lifecycle", "lifecycle", "close_iteration",
    ]
    restored = records[0][1]
    assert np.array_equal(restored.point_ids, packet.point_ids)
    assert np.array_equal(restored.point_keys, packet.point_keys)
    assert np.array_equal(restored.masks, packet.masks)
    assert restored.backward_confirmed is True
    assert records[1][1] == begin
    assert records[2][1] == end
    assert records[3][1] == 1


def test_virtual_packet_archive_readers_have_independent_packet_objects(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=1024)
    writer.initialize_gaussians(1)
    writer.append_packet(_packet())
    writer.close_iteration(1)
    writer.finish()

    first = list(VirtualPacketArchiveReader(archive_root).records())
    second = list(VirtualPacketArchiveReader(archive_root).records())
    first_packet = first[0][1]
    second_packet = second[0][1]
    first_packet.point_ids[0] = 99
    assert second_packet.point_ids[0] == 0


def test_virtual_packet_archive_rejects_cross_chunk_iteration_regression(tmp_path: Path) -> None:
    writer = VirtualPacketArchiveWriter(tmp_path / "archive", max_chunk_bytes=1)
    writer.initialize_gaussians(1)
    writer.append_packet(_packet(iteration=2))
    with pytest.raises(ValueError, match="iteration order"):
        writer.append_packet(_packet(iteration=1))


def test_virtual_packet_archive_reader_retains_only_current_chunk(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=1)
    writer.initialize_gaussians(2)
    writer.append_packet(_packet(point_id=0))
    writer.append_packet(_packet(point_id=1))
    writer.close_iteration(1)
    manifest = writer.finish()
    assert manifest["chunk_count"] == 2

    reader = VirtualPacketArchiveReader(archive_root)
    records = reader.records()
    assert next(records)[1].point_ids.tolist() == [0]
    first_chunk = reader._chunk
    assert reader._chunk_index == 0
    assert next(records)[1].point_ids.tolist() == [1]
    assert reader._chunk_index == 1
    assert reader._chunk is not first_chunk
    assert set(vars(reader)).intersection({"_chunks", "_chunk_cache"}) == set()


def test_virtual_packet_archive_failed_formal_gate_keeps_writer_open(tmp_path: Path) -> None:
    writer = VirtualPacketArchiveWriter(tmp_path / "archive", max_chunk_bytes=1024)
    writer.initialize_gaussians(1)
    writer.append_packet(_packet())
    writer.close_iteration(1)
    with pytest.raises(ValueError, match="iterations 1 through 30000"):
        writer.finish(complete_30k=True, validation_passed=True)

    manifest = writer.finish()
    assert manifest["complete_30k"] is False
    assert manifest["formal_performance_eligible"] is False


def test_relation_capacity_preflight_reports_exact_topology_failure(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=1024)
    writer.initialize_gaussians(2)
    masks = np.zeros((2, 8), dtype=np.dtype("<u4"))
    for y in range(3):
        for x in range(8):
            local_query = y * 16 + x
            masks[:, local_query // 32] |= np.uint32(1 << (local_query % 32))
    writer.append_packet(VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(3, 8),
        point_ids=np.asarray([0, 1], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=2,
        ssim_radius=1,
        backward_confirmed=True,
    ))
    writer.close_iteration(1)
    writer.finish()

    class _Config:
        sha256 = "recorded-not-gated"

        @staticmethod
        def require_ready() -> None:
            return None

        @staticmethod
        def value(path: str) -> int:
            return {
                "compute.relations_per_microcontext": 8,
                "query.relation_store_records": 3,
                "query.relation_candidate_ordinal_bits": 16,
            }[path]

    report = run_relation_capacity_preflight(archive_root, _Config())

    assert report["status"] == "failed_preflight"
    assert report["reason"] == "relation_store_capacity_infeasible"
    assert report["configuration_sha256_recorded_only"] == "recorded-not-gated"
    assert report["packets"][0]["peak_live_records"] == 4
    assert report["packets"][0]["capacity_deficit_records"] == 1
    assert report["topology_scope"] == (
        "exact_for_declared_topology_not_global_lower_bound"
    )


def test_virtual_consumer_closes_event_and_lifecycle_frontiers(tmp_path: Path) -> None:
    consumer = VirtualCaptureConsumer(tmp_path, max_events=2, expand_for_validation=True)
    consumer.initialize_gaussians(2)
    consumer.accept_query(_packet())
    consumer.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2,
    ))
    consumer.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_COMMIT, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2, all_active=True,
    ))
    consumer.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_END, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2,
    ))
    consumer.close_iteration(1)

    result = consumer.finish(capture_audit={"captured_query_kernel_calls": 1})

    assert result["logical_expanded_event_count"] == 10
    assert result["validated_expanded_event_count"] == 10
    assert result["event_stream_validated"] is True
    assert result["iterations"][0]["optimizer_commits"] == 2
    assert result["iterations"][0]["state_version_end"] == 1
    assert result["capture_audit"] == {"captured_query_kernel_calls": 1}
    assert json.loads((tmp_path / "virtual_trace_manifest.json").read_text()) == result


class _PacketLifecycleSink:
    def __init__(self) -> None:
        self.queries = 0
        self.lifecycle = []
        self.closed = []
        self.finished = False

    def accept_query_packet(self, packet: VirtualTracePacket) -> None:
        self.queries += 1

    def accept_lifecycle(self, record: VirtualLifecycleRecord) -> None:
        self.lifecycle.append(record.kind)

    def close_iteration(self, iteration_id: int) -> None:
        self.closed.append(iteration_id)

    def finish(self) -> None:
        self.finished = True


def test_virtual_consumer_dispatches_query_and_lifecycle_to_online_sink(tmp_path: Path) -> None:
    sink = _PacketLifecycleSink()
    consumer = VirtualCaptureConsumer(tmp_path, packet_consumer=sink)
    consumer.initialize_gaussians(1)
    consumer.accept_query(_packet())
    consumer.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2,
    ))
    consumer.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_COMMIT, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2, all_active=True,
    ))
    consumer.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_END, 0,
        field_mask=STATE_FIELD_MASK, transaction_kind=2,
    ))
    consumer.close_iteration(1)
    consumer.finish()
    assert sink.queries == 1
    assert len(sink.lifecycle) == 3
    assert sink.closed == [1]
    assert sink.finished is True


def test_trace_session_creates_online_consumer_after_gaussian_initialization(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    sink = _PacketLifecycleSink()

    def factory(initial_count: int) -> _PacketLifecycleSink:
        calls.append(initial_count)
        return sink

    session = TraceSession(
        tmp_path,
        virtual_capture=True,
        virtual_packet_consumer_factory=factory,
    )
    session._ensure_gaussians(3)
    assert calls == []
    session._ensure_virtual_consumer()
    assert calls == [3]
    session._ensure_virtual_consumer()
    assert calls == [3]
    session.finish()
    assert sink.finished is True


def test_trace_session_virtual_path_skips_relation_record_decoder(tmp_path: Path) -> None:
    session = TraceSession(tmp_path, chunk_events=2, virtual_capture=True)
    session._iteration = 1
    session._ensure_gaussians(2)
    session._ensure_virtual_consumer()
    output = _ArrayRef((1, 1), 20)
    decoder_called = False

    def forbidden_decoder():
        nonlocal decoder_called
        decoder_called = True
        raise AssertionError("full relation decoder must not run in virtual mode")

    session._capture_query(
        _ArrayRef((2, 3), 0), _ArrayRef((1,), 10), output, 1, (1, 1),
        forbidden_decoder, template_id=1, field_mask=STATE_FIELD_MASK,
        virtual_packet_fn=lambda _query_base: _packet(point_id=1),
    )
    session._pending_query_by_output[20].loss_flags = LOSS_L1
    session._capture_backward(10, voxel=False)
    result = session.finish()

    assert decoder_called is False
    assert result["packet_count"] == 1
    assert result["relation_count"] == 1
    assert result["logical_expanded_event_count"] == 10
    assert result["validated_expanded_event_count"] == 0
    assert result["iterations"][0]["active_gaussian_count_end"] == 2
    assert result["capture_audit"]["captured_backward_calls"] == 1


def test_virtual_collection_and_optimizer_keep_exact_counts(tmp_path: Path) -> None:
    class Model:
        get_xyz = np.empty((2, 3), dtype=np.float32)

    session = TraceSession(tmp_path, virtual_capture=True)
    session._iteration = 1
    session._ensure_gaussians(2)
    session._ensure_virtual_consumer()
    session._capture_update(Model(), field_mask=STATE_FIELD_MASK)
    session._start_collection_transaction()
    session._ensure_virtual_collection_begin()
    session._accept_virtual_clone_records((0,), [2])
    session._finish_collection_transaction()
    result = session.finish()

    ledger = result["iterations"][0]
    assert ledger["optimizer_commits"] == 2
    assert ledger["collection_transactions"] == 1
    assert ledger["state_version_end"] == 2
    assert ledger["active_gaussian_count_end"] == 3
