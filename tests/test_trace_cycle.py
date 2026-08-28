from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import (
    ChunkedTraceBuilder,
    PrimitiveKind,
    ResourceClass,
    TraceBuilder,
    TraceEvent,
    UpdateBeginKind,
)
from gala_sim.adapters.trace_capture import FIELD_DENSITY
from gala_sim.ablation import run_matrix
from gala_sim.timing import CycleConfig, CycleConfigurationError, CycleEngine, ModuleTiming
from gala_sim.timing.engine import _DependencyIndex
from gala_sim.timing.memory import Ramulator2Backend, RecordedMemoryBackend
from gala_sim.config import load_config
from gala_sim.trace import NumpyChunkSink, TraceReader, TraceWriter, TraceValidationError, validate_trace
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
    residency = CycleEngine(config, policy="variant:0001").run(trace)
    assert base.module_counters["semantic_cache"]["memory_requests"] == 2
    assert residency.module_counters["semantic_cache"]["memory_requests"] == 1
    assert residency.module_counters["semantic_cache"]["directory_misses"] == 1
    assert residency.module_counters["semantic_cache"]["directory_hits"] == 1


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
    independent = CycleEngine(independent_port_config).run(_fusion_port_trace())
    independent_fusion_stalls = [
        stall for stall in independent.stalls
        if stall.module == "fusion_issue" and stall.reason == "port"
    ]
    assert not independent_fusion_stalls


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
    result = CycleEngine(config).run(builder.finish())
    assert any(
        stall.module == "fusion_issue" and stall.reason == "port"
        for stall in result.stalls
    )


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
