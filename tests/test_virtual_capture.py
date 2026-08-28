from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gala_sim.adapters.trace_capture import LOSS_L1, STATE_FIELD_MASK, TraceSession
from gala_sim.adapters.virtual_capture import VirtualCaptureConsumer
from gala_sim.trace import (
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualTracePacket,
)


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
