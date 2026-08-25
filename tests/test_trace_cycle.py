from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import PrimitiveKind, ResourceClass, TraceBuilder, TraceEvent
from gala_sim.ablation import run_matrix
from gala_sim.timing import CycleConfig, CycleEngine, ModuleTiming
from gala_sim.config import load_config
from gala_sim.trace import TraceReader, TraceWriter, TraceValidationError, validate_trace


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


def test_production_cycle_config_cannot_bypass_unfrozen_parameters() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    with pytest.raises(ValueError, match="clock or seed FIFO"):
        CycleConfig.from_gala(config, _Memory())


def test_ablation_runner_uses_one_trace_for_all_variants() -> None:
    runs = run_matrix(_trace(), _config())
    assert len(runs) == 16
    assert runs[0].variant.bits == "0000"
    assert runs[-1].variant.bits == "1111"
    assert runs[-1].result.policy == "full"
