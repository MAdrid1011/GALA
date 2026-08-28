from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import (
    ChunkedTraceBuilder,
    PrimitiveKind,
    ReductionDomain,
    ResourceClass,
    TaskKind,
    TraceBuilder,
    TraceEvent,
    UpdateBeginKind,
)
from gala_sim.adapters.trace_capture import FIELD_DENSITY
from gala_sim.ablation import run_matrix
from gala_sim.timing import (
    BufferedVirtualCycleConsumer,
    CycleConfig,
    CycleConfigurationError,
    CycleEngine,
    CycleProgress,
    ModuleTiming,
)
from gala_sim.timing.engine import _DependencyIndex
from gala_sim.timing.memory import Ramulator2Backend, RecordedMemoryBackend
from gala_sim.config import load_config
from gala_sim.trace import NumpyChunkSink, TraceReader, TraceWriter, TraceValidationError, validate_trace
from gala_sim.trace import (
    VirtualEventPacket,
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualQueryEventExpander,
    VirtualTracePacket,
)
from gala_sim.adapters.r2_gaussian import _push_trace_chunks
from gala_sim.clamp.events import dependency_dtype, event_dtype


class _Memory:
    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        return arrival_cycle + size_bytes // 64 + int(is_write)


def _trace():
    builder = TraceBuilder()
    relation = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=2, gaussian_id=4,
        relation_id=7, state_version=0, resource_class=int(ResourceClass.RELATION),
    ))
    forward = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=2, gaussian_id=4,
        relation_id=7, state_version=0, resource_class=int(ResourceClass.ISSUE),
    ), dependencies=[relation])
    consumer = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=2, gaussian_id=4,
        relation_id=7, consumer_id=1, state_version=0, resource_class=int(ResourceClass.QUERY),
    ), dependencies=[forward])
    close = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=2,
        resource_class=int(ResourceClass.RELATION),
    ), dependencies=[relation])
    return builder.finish(metadata={"model_commit": "fixture"})


def _config() -> CycleConfig:
    timing = ModuleTiming(latency=2, initiation_interval=1, queue_capacity=8, ports=1, banks=2)
    return CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000, relation_seed_fifo_entries=8,
        candidate_lanes=3,
    )


def test_trace_round_trip_and_cycle_result(tmp_path: Path) -> None:
    trace = _trace()
    report = validate_trace(trace)
    assert report.event_count == 4
    root = tmp_path / "trace"
    TraceWriter().write(trace, root)
    loaded = TraceReader().read(root)
    result = CycleEngine(_config()).run(loaded)
    assert result.total_cycles > 0
    assert result.completion_cycles[3] <= result.total_cycles
    assert result.module_counters["relation_constructor"]["completed"] == 2


def test_cycle_progress_uses_capture_iteration_totals_metadata() -> None:
    trace = _trace()
    trace = replace(
        trace,
        metadata={**trace.metadata, "iteration_event_counts": {"0": trace.event_count}},
    )
    progress: list[CycleProgress] = []

    result = CycleEngine(_config()).run(
        trace,
        progress=progress.append,
        progress_interval_events=2,
        progress_interval_seconds=0.001,
    )

    assert result.total_cycles > 0
    replay = [item for item in progress if item.phase == "replay"]
    assert replay
    assert replay[-1].total_iterations == 1
    assert replay[-1].completed_iterations == 1
    assert any(item.phase == "dependency_index_dependency_fill" for item in progress)


def test_virtual_cycle_replay_preserves_global_packet_state(tmp_path: Path) -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    packets = [
        VirtualTracePacket(
            iteration_id=1,
            template_id=1,
            query_base=index,
            query_shape=(1, 1),
            point_ids=np.asarray([index], dtype=np.int64),
            point_keys=np.asarray([0], dtype=np.uint64),
            masks=masks,
            loss_flags=1,
        )
        for index in (0, 1)
    ]
    result = CycleEngine(_config()).run_virtual(
        packets,
        trace_root=tmp_path / "virtual_replay",
        max_events=2,
        max_total_events=64,
    )
    assert len(result.completion_cycles) == 20
    assert result.total_cycles > 0
    manifest = tmp_path / "virtual_replay" / "chunk_manifest.json"
    assert manifest.is_file()
    loaded = TraceReader().read(tmp_path / "virtual_replay", validate=True, mmap_mode="r")
    assert loaded.events["event_id"].tolist() == list(range(20))
    assert loaded.metadata["result_scope"] == "quick_cycle_validation"
    assert loaded.metadata["formal_performance_eligible"] is False
    assert loaded.metadata["virtual_source_packets"] == 2
    assert loaded.metadata["virtual_max_events"] == 2
    assert loaded.metadata["virtual_max_total_events"] == 64


def test_virtual_cycle_replay_rejects_unbounded_expansion(tmp_path: Path) -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    with pytest.raises(CycleConfigurationError, match="max_total_events"):
        CycleEngine(_config()).run_virtual(
            [source], trace_root=tmp_path / "rejected",
            max_events=2, max_total_events=9,
        )
    assert not (tmp_path / "rejected" / "chunk_manifest.json").exists()


def test_virtual_cycle_replay_rejects_reused_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "reused"
    root.mkdir()
    (root / "chunk_manifest.json").write_text("stale\n", encoding="utf-8")
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    with pytest.raises(CycleConfigurationError, match="new empty trace_root"):
        CycleEngine(_config()).run_virtual(
            [source], trace_root=root, max_events=2, max_total_events=64
        )
    assert (root / "chunk_manifest.json").read_text(encoding="utf-8") == "stale\n"


def test_virtual_forward_dependencies_preserve_relation_pairing() -> None:
    masks = np.zeros((3, 8), dtype=np.dtype("<u4"))
    masks[:, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 3),
        point_ids=np.asarray([0, 1, 2], dtype=np.int64),
        point_keys=np.asarray([0, 0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    packets = tuple(VirtualQueryEventExpander(max_events=8).expand(source))
    events = np.concatenate([packet.events for packet in packets])
    relation = events[events["primitive_kind"] == int(PrimitiveKind.RELATION)]
    returns = events[events["primitive_kind"] == int(PrimitiveKind.CACHE_RETURN)]
    forwards = events[events["primitive_kind"] == int(PrimitiveKind.FORWARD)]
    assert relation.size == returns.size == forwards.size == 3
    for packet in packets:
        for forward in packet.events[
            packet.events["primitive_kind"] == int(PrimitiveKind.FORWARD)
        ]:
            begin = int(forward["dependency_begin"])
            deps = packet.dependencies[
                begin:begin + int(forward["dependency_count"])
            ]
            relation_row = relation[relation["event_id"] == deps[0]][0]
            return_row = returns[returns["event_id"] == deps[1]][0]
            assert int(relation_row["relation_id"]) == int(forward["relation_id"])
            assert int(return_row["relation_id"]) == int(forward["relation_id"])
            assert int(relation_row["query_id"]) == int(forward["query_id"])
            assert int(return_row["query_id"]) == int(forward["query_id"])
            assert int(relation_row["gaussian_id"]) == int(forward["gaussian_id"])
            assert int(return_row["gaussian_id"]) == int(forward["gaussian_id"])


def test_online_cycle_replay_consumes_packets_without_trace_columns() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    engine = CycleEngine(_config())
    session = engine.online_session(
        max_events=4, initial_gaussian_count=1, retain_completion_cycles=True
    )
    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()
    assert result.total_cycles > 0
    assert len(result.completion_cycles) == 10
    assert result.event_counts["RELATION"] == 1
    assert session.pending_event_count == 0
    assert session.resident_completion_markers == 0
    assert session.quiescent
    assert session.accepted_event_count == session.completed_event_count
    assert session.source_packet_count > 0
    assert session.query_packet_count == 1
    assert session.closed_iteration_count == 1


def test_online_cycle_replay_reports_progress() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    progress = []
    session = CycleEngine(_config()).online_session(
        max_events=4, initial_gaussian_count=1,
        progress=progress.append, progress_interval_seconds=0.001,
    )
    session.accept_query_packet(source)
    session.close_iteration(1)
    session.finish()
    assert progress
    assert progress[-1].phase == "online_replay"
    assert progress[-1].completed_events > 0


def test_online_query_packet_batches_expanded_subpackets_before_drain(
    tmp_path: Path,
) -> None:
    masks = np.zeros((32, 8), dtype=np.dtype("<u4"))
    masks[:, 0] = 0b1111
    sources = tuple(
        VirtualTracePacket(
            iteration_id=1,
            template_id=1,
            query_base=index * 32,
            query_shape=(1, 32),
            point_ids=np.arange(index * 32, (index + 1) * 32, dtype=np.int64),
            point_keys=np.arange(32, dtype=np.uint64),
            masks=masks,
            loss_flags=1,
            backward_confirmed=True,
        )
        for index in range(2)
    )
    offline = CycleEngine(_config()).run_virtual(
        sources,
        trace_root=tmp_path / "offline",
        max_events=64,
        max_total_events=10000,
    )
    session = CycleEngine(_config()).online_session(
        max_events=64,
        initial_gaussian_count=64,
    )
    session.accept_query_packets(sources)
    session.close_iteration(1)
    online = session.finish()
    assert online.event_counts == offline.event_counts
    assert online.total_cycles - offline.total_cycles <= 1
    assert session.quiescent


def test_online_query_packet_batch_respects_frontier_bound() -> None:
    masks = np.zeros((32, 8), dtype=np.dtype("<u4"))
    masks[:, 0] = 0b1111
    sources = tuple(
        VirtualTracePacket(
            iteration_id=1,
            template_id=1,
            query_base=index * 32,
            query_shape=(1, 32),
            point_ids=np.arange(index * 32, (index + 1) * 32, dtype=np.int64),
            point_keys=np.arange(32, dtype=np.uint64),
            masks=masks,
            loss_flags=1,
            backward_confirmed=True,
        )
        for index in range(2)
    )
    session = CycleEngine(_config()).online_session(
        max_events=64,
        max_frontier_events=128,
        initial_gaussian_count=64,
    )
    session.accept_query_packets(sources)
    session.close_iteration(1)
    result = session.finish()
    assert result.event_counts["RELATION"] == 256
    assert session.peak_frontier_events <= 128
    assert session.quiescent


def test_online_cycle_replay_rejects_frontier_overflow_before_advancing_ids() -> None:
    rows = np.empty(2, dtype=event_dtype())
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = (0, 1)
    rows["primitive_kind"] = int(PrimitiveKind.RELATION_CANDIDATE)
    rows["resource_class"] = int(ResourceClass.RELATION)
    packet = VirtualEventPacket(
        packet_id=0,
        global_event_start=0,
        events=rows,
        dependencies=np.empty(0, dtype=dependency_dtype()),
    )
    session = CycleEngine(_config()).online_session(
        max_events=4, max_frontier_events=1
    )
    with pytest.raises(CycleConfigurationError, match="max_frontier_events"):
        session.accept_event_packet(packet)
    assert session.global_event_id == 0
    assert session.pending_event_count == 0
    assert session.peak_frontier_events == 0


def test_online_cycle_releases_relation_seed_fifo_slots() -> None:
    masks = np.zeros((2, 8), dtype=np.dtype("<u4"))
    masks[:, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0, 1], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    config = replace(_config(), relation_seed_fifo_entries=1)
    session = CycleEngine(config).online_session(
        max_events=4, initial_gaussian_count=2,
    )
    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()
    assert result.event_counts["RELATION_CANDIDATE"] == 2
    assert session.pending_event_count == 0


def test_online_cycle_replay_keeps_lifecycle_events_in_same_frontier() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    engine = CycleEngine(_config())
    session = engine.online_session(max_events=4, initial_gaussian_count=1)
    session.accept_query_packet(source)
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=1, transaction_kind=2,
    ))
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_COMMIT, 0,
        field_mask=1, transaction_kind=2, all_active=True,
    ))
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_END, 0,
        field_mask=1, transaction_kind=2,
    ))
    session.close_iteration(1)
    result = session.finish()
    assert result.event_counts["UPDATE_BEGIN"] == 1
    assert result.event_counts["UPDATE_COMMIT"] == 1
    assert result.event_counts["UPDATE_END"] == 1


def test_online_cycle_replay_consumes_exact_semantic_workset_sidecar() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 0b11
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 2),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks,
        loss_flags=1,
        backward_confirmed=True,
    )
    timing = ModuleTiming(latency=2, initiation_interval=1, queue_capacity=8, ports=1, banks=2)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=1, cache_capacity_per_instance=4,
        cache_directory_banks=1, cache_sector_bytes=64,
        cache_multicast_destinations=4,
    )
    session = CycleEngine(config, policy="variant:0001").online_session(
        max_events=4,
        initial_gaussian_count=1,
        semantic_workset_totals={(0, 0): 2},
    )
    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()
    assert result.module_counters["semantic_cache"]["workset_keys"] == 1
    assert result.module_counters["semantic_cache"]["workset_uses"] == 2
    assert result.module_counters["semantic_cache"]["workset_releases"] == 1
    assert result.module_counters["semantic_cache"]["miss_merges"] == 1
    assert result.module_counters["semantic_cache"]["multicast_reads"] == 1
    assert result.event_counts["CACHE_REQUEST"] == 2
    assert result.event_counts["CACHE_RETURN"] == 2
    assert session.completed_event_count == session.accepted_event_count
    assert session.quiescent


def test_buffered_virtual_consumer_derives_totals_before_lifecycle() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 0b11
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 2),
        point_ids=np.asarray([0], dtype=np.int64), point_keys=np.asarray([0], dtype=np.uint64),
        masks=masks, loss_flags=1, backward_confirmed=True,
    )
    timing = ModuleTiming(latency=2, initiation_interval=1, queue_capacity=8, ports=1, banks=2)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=1, cache_capacity_per_instance=4,
        cache_directory_banks=1, cache_sector_bytes=64,
    )
    sink = BufferedVirtualCycleConsumer(
        CycleEngine(config, policy="variant:0001").online_session(
            max_events=4, initial_gaussian_count=1,
        )
    )
    sink.accept_query_packet(source)
    sink.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0, field_mask=1, transaction_kind=2,
    ))
    sink.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_COMMIT, 0, field_mask=1,
        transaction_kind=2, all_active=True,
    ))
    sink.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_END, 0, field_mask=1, transaction_kind=2,
    ))
    sink.close_iteration(1)
    result = sink.finish()
    assert result.module_counters["semantic_cache"]["workset_uses"] == 2
    assert sink.result is result
    assert sink.session.semantic_workset_totals == {}
    assert sink.session._workset_seen == {}
    assert sink.session._workset_by_request == {}
    assert sink.session._cache_event_state == {}


def test_formal_cycle_records_but_does_not_gate_on_configuration_hash() -> None:
    trace = _trace()
    config = replace(_config(), config_sha256="a" * 64)
    assert CycleEngine(config).run(trace).total_cycles > 0

    matching = type(trace)(
        trace.events, trace.dependencies, trace.payload,
        {**trace.metadata, "config_sha256": "a" * 64},
    )
    assert CycleEngine(config).run(matching).total_cycles > 0


def test_chunked_trace_builder_preserves_global_offsets() -> None:
    builder = ChunkedTraceBuilder(chunk_events=2)
    first = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0, gaussian_id=0,
        relation_id=0,
    ))
    second = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0, gaussian_id=1,
        relation_id=1,
    ), dependencies=[first])
    builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0), dependencies=[second])
    trace = builder.finish()
    assert trace.event_count == 3
    assert trace.dependency_ids(trace.events[2]).tolist() == [1]
    assert validate_trace(trace).event_count == 3


@pytest.mark.parametrize("disk", [False, True])
def test_chunked_trace_builder_batch_matches_single_event_writer(
    tmp_path: Path, disk: bool,
) -> None:
    single = TraceBuilder()
    rows = []
    dependencies: list[int] = []
    dependency_counts: list[int] = []
    payload = np.asarray([1.25, 2.5, 3.75], dtype=np.dtype("<f4"))
    payload_counts = [1, 2, 0, 0, 0]
    payload_offset = 0
    for index in range(5):
        event = TraceEvent(
            primitive_kind=int(
                PrimitiveKind.RELATION if index < 4 else PrimitiveKind.QUERY_CLOSE
            ),
            query_id=index,
            gaussian_id=index,
            relation_id=index,
            resource_class=int(ResourceClass.RELATION),
        )
        deps = () if index == 0 else (index - 1,)
        rows.append(event.as_tuple())
        dependencies.extend(deps)
        dependency_counts.append(len(deps))
        next_offset = payload_offset + payload_counts[index]
        single.emit(event, dependencies=deps, payload=payload[payload_offset:next_offset])
        payload_offset = next_offset
    kwargs = {"chunk_root": tmp_path / "chunks"} if disk else {}
    batched = ChunkedTraceBuilder(chunk_events=2, **kwargs)
    ids = batched.emit_batch(
        np.asarray(rows, dtype=event_dtype()),
        dependencies=np.asarray(dependencies, dtype=dependency_dtype()),
        dependency_counts=np.asarray(dependency_counts, dtype=np.int64),
        payload=payload,
        payload_counts=np.asarray(payload_counts, dtype=np.int64),
    )
    result = batched.finish(metadata={"fixture": "batch"})
    expected = single.finish(metadata={"fixture": "single"})
    assert ids.tolist() == list(range(5))
    assert np.array_equal(result.events, expected.events)
    assert np.array_equal(result.dependencies, expected.dependencies)
    assert np.array_equal(result.payload, expected.payload)
    assert validate_trace(result).event_count == 5


def test_stream_only_builder_writes_raw_columns_readable_as_mmap(tmp_path: Path) -> None:
    root = tmp_path / "stream"
    builder = ChunkedTraceBuilder(
        chunk_events=2, chunk_root=root / ".capture_chunks", stream_only=True
    )
    relation = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0, gaussian_id=0,
        relation_id=0,
    ))
    builder.emit(
        TraceEvent(primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0),
        dependencies=[relation],
    )
    manifest = builder.finish(
        metadata={"model": "fixture"}, materialize=False
    )
    assert manifest.storage_format == "raw_columns"
    assert (root / "events.raw").is_file()
    loaded = TraceReader().read(root, mmap_mode="r")
    assert isinstance(loaded.events, np.memmap)
    assert loaded.metadata["trace_storage_format"] == "raw_columns"
    assert loaded.event_count == 2
    assert validate_trace(loaded).event_count == 2


def test_dependency_index_uses_compressed_stable_reverse_edges() -> None:
    builder = TraceBuilder()
    root = builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION)))
    left = builder.emit(
        TraceEvent(primitive_kind=int(PrimitiveKind.RELATION)), dependencies=[root]
    )
    right = builder.emit(
        TraceEvent(primitive_kind=int(PrimitiveKind.RELATION)), dependencies=[root]
    )
    builder.emit(
        TraceEvent(primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0),
        dependencies=[left, right],
    )
    index = _DependencyIndex.from_trace(builder.finish())
    assert index.remaining.tolist() == [0, 1, 1, 2]
    assert index.offsets.tolist() == [0, 2, 3, 4, 4]
    assert index.for_event(root).tolist() == [left, right]
    assert index.for_event(left).tolist() == [3]
    assert index.for_event(right).tolist() == [3]


def test_disk_chunked_builder_streams_all_columns(tmp_path: Path) -> None:
    builder = ChunkedTraceBuilder(chunk_events=2, chunk_root=tmp_path / "chunks")
    first = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0, gaussian_id=0,
        relation_id=0,
    ), payload=[1.5])
    second = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0, gaussian_id=1,
        relation_id=1,
    ), dependencies=[first], payload=[2.5, 3.5])
    builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0), dependencies=[second])
    trace = builder.finish()
    assert trace.event_count == 3
    assert trace.dependency_ids(trace.events[2]).tolist() == [1]
    assert trace.payload_values(trace.events[1]).tolist() == [2.5, 3.5]
    assert validate_trace(trace).event_count == 3
    assert (tmp_path / "chunks").exists() is False
    TraceWriter().write(trace, tmp_path)
    loaded = TraceReader().read(tmp_path, mmap_mode="r")
    assert loaded.event_count == 3
    assert isinstance(loaded.events, np.memmap)


def test_sink_handoff_rebases_chunk_offsets() -> None:
    sink = NumpyChunkSink(chunk_events=2, max_inflight_chunks=2)
    _push_trace_chunks(_trace(), sink)
    events, dependencies, payload = sink.collect()
    from gala_sim.trace.model import Trace

    handed_off = Trace(events, dependencies, payload, _trace().metadata)
    assert validate_trace(handed_off).event_count == 4


def test_trace_rejects_forward_dependency() -> None:
    trace = _trace()
    broken = trace.events.copy()
    broken[1]["dependency_begin"] = 0
    broken[1]["dependency_count"] = 1
    dependencies = trace.dependencies.copy()
    dependencies[0] = 2
    from gala_sim.trace.model import Trace
    with pytest.raises(TraceValidationError, match="prior event"):
        validate_trace(Trace(broken, dependencies, trace.payload, trace.metadata))


def test_trace_schema_rejects_object_arrays() -> None:
    trace = _trace()
    from gala_sim.trace.model import Trace
    with pytest.raises(ValueError, match="frozen schema"):
        Trace(trace.events.astype(object), trace.dependencies, trace.payload, trace.metadata)


def test_cache_transfer_requires_memory_size_and_pairing() -> None:
    builder = TraceBuilder()
    request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), gaussian_id=4,
        state_version=0, address_token=128, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), gaussian_id=4,
        state_version=0, address_token=128, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[request])
    result = CycleEngine(_config()).run(builder.finish())
    assert result.module_counters["semantic_cache"]["memory_wait_cycles"] >= 0


class _RamulatorBinding:
    def __init__(self) -> None:
        self.pending: list[int] = []
        self.completed: list[int] = []

    def metadata(self) -> dict[str, object]:
        return {
            "implementation": "Ramulator 2", "version": "2.1.0",
            "config_sha256": "c" * 64,
            "channels": 8, "transaction_bytes": 64,
        }

    def try_issue(self, address: int, is_write: bool, request_id: int) -> bool:
        self.pending.append(request_id)
        return True

    def tick(self) -> None:
        self.completed.extend(self.pending)
        self.pending.clear()

    def drain_completions(self) -> tuple[int, ...]:
        result = tuple(self.completed)
        self.completed.clear()
        return result

    def clone(self) -> "_RamulatorBinding":
        return type(self)()


def test_cycle_engine_wakes_cache_event_from_async_ramulator_completion() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0,
        gaussian_id=0, state_version=0, address_token=128, data_bytes=128,
        resource_class=int(ResourceClass.CACHE),
    ))
    trace = builder.finish()
    timing = ModuleTiming(latency=1, initiation_interval=1, queue_capacity=2, ports=1, banks=2)
    memory = Ramulator2Backend(_RamulatorBinding())
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=memory, clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=2, candidate_lanes=3,
    )
    result = CycleEngine(config).run(trace)
    assert result.total_cycles == 2
    assert result.module_counters["semantic_cache"]["memory_requests"] == 1
    assert memory.audit_records()[0].completion_cycle == 1
    assert result.memory_requests == memory.audit_records()


def test_semantic_residency_variant_merges_repeated_state_reads() -> None:
    builder = TraceBuilder()
    first_request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    first_return = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[first_request])
    second_request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=1,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[first_return])
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=1,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[second_request])
    trace = builder.finish()
    config = CycleConfig(
        modules={name: ModuleTiming(latency=1, initiation_interval=1,
                                    queue_capacity=8, ports=1, banks=2)
                 for name in (
                     "relation_constructor", "fusion_issue", "semantic_cache",
                     "compute_pod", "bidirectional_query", "reconstruction_update",
                     "shared_sram",
                 )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=1, cache_capacity_per_instance=2,
        cache_directory_banks=2, cache_sector_bytes=64,
        cache_multicast_destinations=1,
    )
    base = CycleEngine(config, policy="variant:0000").run(trace)
    worksets = CycleEngine(config, policy="variant:0100").run(trace)
    residency = CycleEngine(config, policy="variant:0001").run(trace)
    combined = CycleEngine(config, policy="variant:0101").run(trace)
    assert base.module_counters["semantic_cache"]["memory_requests"] == 2
    assert worksets.module_counters["semantic_cache"]["memory_requests"] == 2
    assert worksets.module_counters["semantic_cache"]["workset_keys"] == 1
    assert worksets.module_counters["semantic_cache"]["workset_uses"] == 2
    assert residency.module_counters["semantic_cache"]["memory_requests"] == 1
    assert residency.module_counters["semantic_cache"]["directory_misses"] == 1
    assert residency.module_counters["semantic_cache"]["directory_hits"] == 1
    assert residency.module_counters["semantic_cache"]["workset_uses"] == 0
    assert combined.module_counters["semantic_cache"]["memory_requests"] == 1
    assert combined.module_counters["semantic_cache"]["workset_releases"] == 1
    assert combined.module_counters["semantic_cache"]["releases"] == 1


def test_semantic_residency_multicasts_real_ready_requests_without_dropping_dependencies() -> None:
    builder = TraceBuilder()
    first_request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=128,
        resource_class=int(ResourceClass.CACHE),
    ))
    first_return = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=128,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[first_request])
    return_ids: list[int] = []
    for query_id in range(1, 5):
        request_id = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=query_id,
            gaussian_id=7, state_version=0, address_token=448, data_bytes=128,
            resource_class=int(ResourceClass.CACHE),
        ), dependencies=[first_return])
        return_ids.append(builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=query_id,
            gaussian_id=7, state_version=0, address_token=448, data_bytes=128,
            resource_class=int(ResourceClass.CACHE),
        ), dependencies=[request_id]))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_END), state_version=0,
        resource_class=int(ResourceClass.UPDATE), field_mask=FIELD_DENSITY,
    ), dependencies=return_ids)
    trace = builder.finish()
    dependencies_before = trace.dependencies.copy()
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=16, ports=1, banks=2,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=1, cache_capacity_per_instance=2,
        cache_directory_banks=2, cache_sector_bytes=64,
        cache_multicast_destinations=4,
    )
    result = CycleEngine(config, policy="variant:0001").run(
        trace, validate_input=False,
    )
    single_destination = CycleEngine(
        replace(config, cache_multicast_destinations=1), policy="variant:0001",
    ).run(trace, validate_input=False)
    cache = result.module_counters["semantic_cache"]
    assert cache["directory_misses"] == 1
    assert cache["directory_hits"] == 1
    assert cache["multicast_reads"] == 1
    assert cache["memory_requests"] == 1
    assert cache["releases"] == 1
    assert single_destination.module_counters["semantic_cache"]["multicast_reads"] == 0
    assert cache["busy_cycles"] < single_destination.module_counters["semantic_cache"]["busy_cycles"]
    assert result.event_counts["CACHE_REQUEST"] == 5
    assert result.event_counts["CACHE_RETURN"] == 5
    assert set(result.completion_cycles) == set(range(trace.event_count))
    assert np.array_equal(trace.dependencies, dependencies_before)


def test_semantic_residency_ordering_is_scoped_to_cache_candidates() -> None:
    builder = TraceBuilder()
    event_ids = [
        builder.emit(TraceEvent(
            primitive_kind=int(kind), gaussian_id=gaussian_id,
            query_id=event_id, resource_class=int(resource),
            data_bytes=64 if kind is PrimitiveKind.CACHE_REQUEST else 0,
        ))
        for event_id, (kind, gaussian_id, resource) in enumerate((
            (PrimitiveKind.RELATION, 9, ResourceClass.RELATION),
            (PrimitiveKind.CACHE_REQUEST, 8, ResourceClass.CACHE),
            (PrimitiveKind.FORWARD, 1, ResourceClass.COMPUTE),
            (PrimitiveKind.CACHE_REQUEST, 2, ResourceClass.CACHE),
            (PrimitiveKind.RELATION, 0, ResourceClass.RELATION),
        ))
    ]
    trace = builder.finish()
    engine = CycleEngine(_config(), policy="residency")

    ordered = engine._ordered_candidates(trace, event_ids)

    assert ordered == [event_ids[0], event_ids[3], event_ids[2], event_ids[1], event_ids[4]]


def test_query_close_keeps_state_resident_until_update_end() -> None:
    builder = TraceBuilder()
    first_request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    first_return = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[first_request])
    close = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0,
        state_version=0, resource_class=int(ResourceClass.RELATION),
    ))
    second_request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=1,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[first_return, close])
    second_return = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=1,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[second_request])
    begin = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_BEGIN), state_version=0,
        resource_class=int(ResourceClass.UPDATE),
        flags=int(UpdateBeginKind.OPTIMIZER),
    ), dependencies=[second_return])
    commit = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_COMMIT), gaussian_id=7,
        state_version=0, resource_class=int(ResourceClass.UPDATE),
        field_mask=FIELD_DENSITY,
    ), dependencies=[begin])
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_END), state_version=0,
        reduction_key=begin, resource_class=int(ResourceClass.UPDATE),
        flags=int(UpdateBeginKind.OPTIMIZER), field_mask=FIELD_DENSITY,
    ), dependencies=[commit])
    trace = builder.finish()
    config = CycleConfig(
        modules={name: ModuleTiming(latency=1, initiation_interval=1,
                                    queue_capacity=8, ports=1, banks=2)
                 for name in (
                     "relation_constructor", "fusion_issue", "semantic_cache",
                     "compute_pod", "bidirectional_query", "reconstruction_update",
                     "shared_sram",
                 )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=1, cache_capacity_per_instance=2,
        cache_directory_banks=2, cache_sector_bytes=64,
        cache_multicast_destinations=1,
    )
    result = CycleEngine(config, policy="variant:0001").run(trace)
    cache = result.module_counters["semantic_cache"]
    assert cache["directory_misses"] == 1
    assert cache["directory_hits"] == 1
    assert cache["releases"] == 1


def test_noop_update_keeps_same_state_version_resident() -> None:
    builder = TraceBuilder()
    request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    returned = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=0,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[request])
    begin = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_BEGIN), state_version=0,
        resource_class=int(ResourceClass.UPDATE), flags=int(UpdateBeginKind.OPTIMIZER),
    ), dependencies=[returned])
    end = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_END), state_version=0,
        reduction_key=begin, resource_class=int(ResourceClass.UPDATE),
        flags=int(UpdateBeginKind.OPTIMIZER), field_mask=0,
    ), dependencies=[begin])
    second_request = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=1,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[end])
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=1,
        gaussian_id=7, state_version=0, address_token=448, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[second_request])
    config = CycleConfig(
        modules={name: ModuleTiming(latency=1, initiation_interval=1,
                                    queue_capacity=8, ports=1, banks=2)
                 for name in (
                     "relation_constructor", "fusion_issue", "semantic_cache",
                     "compute_pod", "bidirectional_query", "reconstruction_update",
                     "shared_sram",
                 )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=1, cache_capacity_per_instance=2,
        cache_directory_banks=2, cache_sector_bytes=64,
        cache_multicast_destinations=1,
    )
    result = CycleEngine(config, policy="variant:0001").run(builder.finish())
    cache = result.module_counters["semantic_cache"]
    assert cache["directory_misses"] == 1
    assert cache["directory_hits"] == 1
    assert cache["releases"] == 0


def test_cycle_engine_applies_module_queue_and_seed_fifo_backpressure() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE), query_id=1,
        gaussian_id=2, state_version=0, resource_class=int(ResourceClass.RELATION),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE), query_id=1,
        gaussian_id=3, state_version=0, resource_class=int(ResourceClass.RELATION),
    ))
    trace = builder.finish()
    timing = ModuleTiming(latency=3, initiation_interval=1, queue_capacity=1, ports=1, banks=2)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000, relation_seed_fifo_entries=1,
        candidate_lanes=3,
    )
    result = CycleEngine(config).run(trace)
    reasons = {stall.reason for stall in result.stalls}
    assert "queue_capacity" in reasons or "seed_fifo" in reasons
    assert result.module_counters["relation_constructor"]["completed"] == 2


def _fusion_port_trace():
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, relation_id=0, reduction_key=0, address_token=0,
        state_version=0, resource_class=int(ResourceClass.ISSUE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=1,
        consumer_id=1, reduction_key=1, address_token=1,
        state_version=0, resource_class=int(ResourceClass.ISSUE),
    ))
    return builder.finish()


def test_fusion_issue_uses_independent_forward_consumer_and_adjoint_ports() -> None:
    timing = ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8, ports=1, banks=4)
    modules = {name: timing for name in (
        "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
        "bidirectional_query", "reconstruction_update", "shared_sram",
    )}
    independent_port_config = CycleConfig(
        modules=modules, memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        fusion_forward_ports=1, fusion_consumer_ports=1, fusion_adjoint_ports=1,
    )
    independent = CycleEngine(
        independent_port_config, policy="variant:0010"
    ).run(_fusion_port_trace())
    independent_fusion_stalls = [
        stall for stall in independent.stalls
        if stall.module == "fusion_issue" and stall.reason == "port"
    ]
    assert not independent_fusion_stalls

    base = CycleEngine(
        independent_port_config, policy="variant:0000"
    ).run(_fusion_port_trace())
    assert any(
        stall.module == "fusion_issue" and stall.reason == "base_single_issue"
        for stall in base.stalls
    )


def test_fusion_issue_applies_each_task_class_port_limit() -> None:
    builder = TraceBuilder()
    for event_id in range(2):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=event_id,
            gaussian_id=event_id, relation_id=event_id, reduction_key=event_id,
            address_token=event_id, state_version=0,
            resource_class=int(ResourceClass.ISSUE),
        ))
    timing = ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8, ports=3, banks=4)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        fusion_forward_ports=1, fusion_consumer_ports=1, fusion_adjoint_ports=1,
    )
    result = CycleEngine(config, policy="variant:0010").run(builder.finish())
    assert result.module_counters["fusion_issue"]["accepted"] == 2
    assert result.completion_cycles[0] < result.completion_cycles[1]


@pytest.mark.parametrize("policy,expected", [
    ("variant:0000", (False, False, False, False)),
    ("variant:1000", (True, False, False, False)),
    ("variant:0100", (False, True, False, False)),
    ("variant:0010", (False, False, True, False)),
    ("variant:0001", (False, False, False, True)),
    ("variant:1111", (True, True, True, True)),
    ("query", (True, False, True, False)),
    ("residency", (False, True, False, True)),
])
def test_cycle_policy_preserves_independent_mechanism_bits(
    policy: str, expected: tuple[bool, bool, bool, bool],
) -> None:
    selection = CycleEngine(_config(), policy=policy).selection
    assert (
        selection.query_load_rules,
        selection.semantic_worksets,
        selection.overlap_guided_issue,
        selection.semantic_residency,
    ) == expected


def test_task_packet_uses_semantic_reduction_domain_and_not_event_id() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=7,
        gaussian_id=3, relation_id=11, reduction_key=-1,
        resource_class=int(ResourceClass.ISSUE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.ADJOINT), query_id=7,
        gaussian_id=3, relation_id=11, reduction_key=-1,
        resource_class=int(ResourceClass.ISSUE),
    ))
    trace = builder.finish()
    forward = CycleEngine._task_packet(trace, 0)
    adjoint = CycleEngine._task_packet(trace, 1)
    assert (forward.reduction_domain, forward.reduction_key) == (
        ReductionDomain.QUERY, 7,
    )
    assert (adjoint.reduction_domain, adjoint.reduction_key) == (
        ReductionDomain.GAUSSIAN, 3,
    )


def test_overlap_issue_rejects_same_query_reduction_in_one_cycle() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, relation_id=0, reduction_key=0, address_token=0,
        resource_class=int(ResourceClass.ISSUE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=0,
        gaussian_id=1, consumer_id=0, reduction_key=0, address_token=1,
        resource_class=int(ResourceClass.ISSUE),
    ))
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=8, ports=3, banks=4,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        fusion_forward_ports=1, fusion_consumer_ports=1, fusion_adjoint_ports=1,
    )
    result = CycleEngine(config, policy="variant:0010").run(builder.finish())
    assert any(
        stall.reason == "scheduler_conflict_or_port" for stall in result.stalls
    )


def test_overlap_scheduler_observes_three_real_input_heads() -> None:
    builder = TraceBuilder()
    for event_id in range(8):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=event_id,
            gaussian_id=event_id, relation_id=event_id, reduction_key=event_id,
            address_token=event_id, resource_class=int(ResourceClass.ISSUE),
        ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=20,
        gaussian_id=20, consumer_id=20, reduction_key=20, address_token=20,
        resource_class=int(ResourceClass.ISSUE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.ADJOINT), query_id=21,
        gaussian_id=21, relation_id=21, reduction_key=21, address_token=21,
        resource_class=int(ResourceClass.ISSUE),
    ))
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=16, ports=3, banks=32,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        fusion_forward_ports=1, fusion_consumer_ports=1, fusion_adjoint_ports=1,
    )
    engine = CycleEngine(config, policy="variant:0010")
    observed: list[tuple[TaskKind, ...]] = []
    original_select = engine.issue_scheduler.select

    def recording_select(candidates, **kwargs):
        packets = tuple(candidates)
        observed.append(tuple(packet.task_kind for packet in packets))
        return original_select(packets, **kwargs)

    engine.issue_scheduler.select = recording_select  # type: ignore[method-assign]
    engine.run(builder.finish(), validate_input=False)
    assert set(observed[0]) == {
        TaskKind.FORWARD, TaskKind.CONSUMER, TaskKind.ADJOINT,
    }


def test_event_driven_engine_replays_large_dependency_chain() -> None:
    builder = TraceBuilder()
    previous: int | None = None
    event_count = 4096
    for relation_id in range(event_count):
        dependencies = () if previous is None else (previous,)
        previous = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION), query_id=0,
            gaussian_id=relation_id, relation_id=relation_id,
            state_version=0, resource_class=int(ResourceClass.RELATION),
        ), dependencies=dependencies)
    trace = builder.finish()
    timing = ModuleTiming(latency=2, initiation_interval=1, queue_capacity=4, ports=1, banks=4)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=4, candidate_lanes=3,
    )
    result = CycleEngine(config).run(trace)
    # Each dependency becomes runnable after the two-cycle service completes.
    assert result.total_cycles == event_count * timing.latency
    assert len(result.completion_cycles) == event_count
    assert result.completion_cycles[event_count - 1] == result.total_cycles


def test_cycle_replay_is_deterministic_and_compresses_stall_counts() -> None:
    builder = TraceBuilder()
    for gaussian_id in range(4):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE), query_id=0,
            gaussian_id=gaussian_id, relation_id=gaussian_id,
            state_version=0, resource_class=int(ResourceClass.RELATION),
        ))
    trace = builder.finish()
    timing = ModuleTiming(latency=3, initiation_interval=1, queue_capacity=4, ports=1, banks=1)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=4, candidate_lanes=3,
    )
    first = CycleEngine(config).run(trace)
    second = CycleEngine(config).run(trace)
    assert first.total_cycles == second.total_cycles
    assert first.completion_cycles == second.completion_cycles
    assert first.stalls == second.stalls
    compressed = [stall for stall in first.stalls if stall.reason == "port"]
    assert compressed
    assert any(stall.count > 1 for stall in compressed)
    assert all(len(stall.event_ids) <= config.candidate_lanes for stall in compressed)


def test_production_cycle_config_uses_frozen_parameters() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    cycle_config = CycleConfig.from_gala(config, _Memory())
    assert cycle_config.relation_seed_fifo_entries == 256


def test_ablation_runner_uses_one_validation_for_all_variants(monkeypatch) -> None:
    import gala_sim.ablation.runner as runner

    validation_calls = 0
    original_validate = runner.validate_trace

    def counted_validate(trace):
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(trace)

    monkeypatch.setattr(runner, "validate_trace", counted_validate)
    runs = runner.run_matrix(_trace(), _config())
    assert len(runs) == 16
    assert validation_calls == 1
    assert runs[0].variant.bits == "0000"
    assert runs[-1].variant.bits == "1111"
    assert runs[-1].result.policy == "variant:1111"


def test_ablation_runner_reports_each_completed_variant() -> None:
    from gala_sim.ablation.runner import run_matrix

    completed: list[str] = []
    runs = run_matrix(_trace(), _config(), progress=lambda run: completed.append(run.variant.bits))
    assert len(runs) == 16
    assert completed == [run.variant.bits for run in runs]


def test_cycle_progress_reports_complete_iterations_without_duplicate_final_sample() -> None:
    from gala_sim.timing import CycleProgress

    samples: list[CycleProgress] = []
    result = CycleEngine(_config()).run(
        _trace(), progress=samples.append,
        progress_interval_events=2, progress_interval_seconds=3600,
    )
    assert result.total_cycles > 0
    assert samples[-1].completed_events == _trace().event_count
    assert samples[-1].completed_iterations == samples[-1].total_iterations == 1
    replay_samples = [sample for sample in samples if sample.phase == "replay"]
    assert len({sample.completed_events for sample in replay_samples}) == len(replay_samples)


def test_default_cycle_run_does_not_build_iteration_diagnostics(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("default replay must not scan iteration diagnostics")

    monkeypatch.setattr(np, "bincount", forbidden)
    assert CycleEngine(_config()).run(_trace()).total_cycles > 0


def test_progress_diagnostics_do_not_change_cycle_result() -> None:
    baseline = CycleEngine(_config()).run(_trace())
    samples = []
    monitored = CycleEngine(_config()).run(
        _trace(), progress=samples.append,
        progress_interval_events=2, progress_interval_seconds=3600,
    )
    assert monitored == baseline
    assert {sample.phase for sample in samples} >= {
        "validation", "dependency_index", "ready_queue", "replay",
    }


def test_cycle_progress_uses_contiguous_iteration_prefix() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        iteration_id=2, primitive_kind=int(PrimitiveKind.RELATION),
        query_id=2, gaussian_id=2, relation_id=2,
        resource_class=int(ResourceClass.RELATION),
    ))
    builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.RELATION),
        query_id=1, gaussian_id=1, relation_id=1,
        resource_class=int(ResourceClass.RELATION),
    ))
    samples = []
    CycleEngine(_config()).run(
        builder.finish(), progress=samples.append,
        progress_interval_events=1, progress_interval_seconds=3600,
    )
    replay = [sample for sample in samples if sample.phase == "replay"]
    assert replay[0].completed_events == 1
    assert replay[0].completed_iterations == 0
    assert replay[-1].completed_iterations == replay[-1].total_iterations == 2


def test_ablation_runner_can_parallelize_independent_variants() -> None:
    from gala_sim.ablation.runner import run_matrix

    runs = run_matrix(_trace(), _config(), parallel_workers=2)
    assert [run.variant.bits for run in runs] == [f"{value:04b}" for value in range(16)]


def test_ablation_replays_recorded_memory_for_each_variant() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), gaussian_id=4,
        state_version=0, address_token=128, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    trace = builder.finish()
    timing = ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8, ports=1, banks=2)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=RecordedMemoryBackend({(128, 64, False, 0): 1}),
        clock_frequency_hz=500_000_000, relation_seed_fifo_entries=8, candidate_lanes=3,
    )
    assert len(run_matrix(trace, config)) == 16
