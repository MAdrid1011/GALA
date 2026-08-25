from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import ChunkedTraceBuilder, PrimitiveKind, ResourceClass, TraceBuilder, TraceEvent
from gala_sim.ablation import run_matrix
from gala_sim.timing import CycleConfig, CycleEngine, ModuleTiming
from gala_sim.timing.memory import RecordedMemoryBackend
from gala_sim.config import load_config
from gala_sim.trace import NumpyChunkSink, TraceReader, TraceWriter, TraceValidationError, validate_trace
from gala_sim.adapters.r2_gaussian import _push_trace_chunks


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


def test_production_cycle_config_cannot_bypass_unfrozen_parameters() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    with pytest.raises(ValueError, match="clock or seed FIFO"):
        CycleConfig.from_gala(config, _Memory())


def test_ablation_runner_uses_one_trace_for_all_variants() -> None:
    runs = run_matrix(_trace(), _config())
    assert len(runs) == 16
    assert runs[0].variant.bits == "0000"
    assert runs[-1].variant.bits == "1111"
    assert runs[-1].result.policy == "variant:1111"


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
