from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gala_sim.clamp import (
    ChunkedTraceBuilder,
    FusionIssueScheduler,
    PrimitiveKind,
    ReductionDomain,
    ResourceClass,
    TaskKind,
    TaskPacket,
    TraceBuilder,
    TraceEvent,
    UpdateBeginKind,
)
from gala_sim.adapters.trace_capture import FIELD_DENSITY
from gala_sim.ablation import (
    run_archive_matrix, run_archive_speedup_diagnostic, run_matrix,
)
from gala_sim.timing import (
    BufferedVirtualCycleConsumer,
    CycleConfig,
    CycleConfigurationError,
    CycleEngine,
    CycleProgress,
    ModuleTiming,
)
from gala_sim.timing.engine import (
    _BankedFusionSourceQueue,
    _DependencyIndex,
    _ReadyCandidateQueue,
)
from gala_sim.timing.modules import OwnerGradientTracker
from gala_sim.timing.memory import Ramulator2Backend, RecordedMemoryBackend
from gala_sim.config import load_config
from gala_sim.tools.cycle_throughput import ThroughputDiagnosticConfig
from gala_sim.trace import (
    NumpyChunkSink, Trace, TraceReader, TraceWriter, TraceValidationError,
    validate_trace,
)
from gala_sim.trace import (
    VirtualPacketArchiveReader,
    VirtualPacketArchiveWriter,
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


@pytest.mark.parametrize("policy", ["base", "variant:0101", "full"])
def test_archived_packet_replay_matches_live_buffered_cycle(
    tmp_path: Path, policy: str,
) -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 0b11
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 2),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    writer = VirtualPacketArchiveWriter(tmp_path / "archive", max_chunk_bytes=64)
    writer.initialize_gaussians(1)
    writer.append_packet(source)
    writer.close_iteration(1)
    writer.finish()

    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    live_config = CycleConfig.from_gala(load_config(config_path), _Memory())
    archive_config = CycleConfig.from_gala(load_config(config_path), _Memory())
    live = BufferedVirtualCycleConsumer(
        CycleEngine(live_config, policy=policy).online_session(
            max_events=4, initial_gaussian_count=1,
        )
    )
    live.accept_query_packet(source)
    live.close_iteration(1)
    live_result = live.finish()
    archive_result = VirtualPacketArchiveReader(tmp_path / "archive").replay_session(
        CycleEngine(archive_config, policy=policy), max_events=4,
    )

    assert archive_result.total_cycles == live_result.total_cycles
    assert archive_result.event_counts == live_result.event_counts
    assert archive_result.module_counters == live_result.module_counters
    assert archive_result.stalls == live_result.stalls


def test_archive_ablation_runs_canonical_independent_variants(tmp_path: Path) -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    archive_root = tmp_path / "archive"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=64)
    writer.initialize_gaussians(1)
    writer.append_packet(source)
    writer.close_iteration(1)
    writer.finish()
    completed: list[str] = []
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    config = CycleConfig.from_gala(load_config(config_path), _Memory())

    runs = run_archive_matrix(
        archive_root, config, max_events=4,
        progress=lambda run: completed.append(run.variant.bits),
    )

    expected = ["0000", "1000", "1010", "0100", "0101", "1100", "1111"]
    assert [run.variant.bits for run in runs] == expected
    assert completed == expected
    assert len({tuple(sorted(run.result.event_counts.items())) for run in runs}) == 1
    assert len({id(run.result) for run in runs}) == len(runs)


def test_archive_speedup_diagnostic_stops_all_variants_at_common_boundary(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive-speedup"
    writer = VirtualPacketArchiveWriter(archive_root, max_chunk_bytes=64)
    writer.initialize_gaussians(1)
    for iteration in range(1, 11):
        masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
        masks[0, 0] = 1
        writer.append_packet(VirtualTracePacket(
            iteration_id=iteration, template_id=1, query_base=iteration - 1,
            query_shape=(1, 1), point_ids=np.asarray([0], dtype=np.int64),
            point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
            loss_flags=1, backward_confirmed=True,
        ))
        writer.close_iteration(iteration)
    writer.finish()
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    config = CycleConfig.from_gala(load_config(config_path), _Memory())
    diagnostic = ThroughputDiagnosticConfig(
        report_interval_events=1,
        report_interval_seconds=1,
        warmup_samples=1,
        stability_window_samples=3,
        required_consecutive_stable_windows=2,
        stability_relative_span=0.01,
        minimum_completion_fraction=0.2,
    )

    report = run_archive_speedup_diagnostic(
        archive_root, config, diagnostic, max_events=4,
    )

    assert report["termination"] == "stopped_on_stable_speedup"
    assert report["measured_iteration_count"] == 5
    assert report["measured_last_iteration"] == 5
    assert report["complete_trace_replay"] is False
    assert set(report["variant_results"]) == {
        "0000", "1000", "1010", "0100", "0101", "1100", "1111",
    }
    assert {
        item["completed_events"] for item in report["variant_results"].values()
    } == {report["samples"][-1]["completed_events"]}


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
    assert progress[-1].completed_iterations == 1
    assert progress[-1].total_iterations == 1
    assert progress[-1].last_completed_iteration == 1


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
    config = replace(
        _config(),
        candidate_fifo_entries=32,
        fusion_query_state_banks=8,
        fusion_bank_head_lookahead=True,
        fusion_bank_head_index_bytes=224,
        fusion_forward_ports=2,
        fusion_consumer_ports=1,
        fusion_adjoint_ports=2,
    )
    offline = CycleEngine(config).run_virtual(
        sources,
        trace_root=tmp_path / "offline",
        max_events=64,
        max_total_events=10000,
    )
    session = CycleEngine(config).online_session(
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


def test_online_streaming_waits_for_window_seal_before_declaring_deadlock() -> None:
    masks = np.full((9, 8), np.uint32(0xFFFF_FFFF), dtype=np.dtype("<u4"))
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(16, 16),
        point_ids=np.arange(9, dtype=np.int64),
        point_keys=np.zeros(9, dtype=np.uint64),
        masks=masks,
        loss_flags=2,
        ssim_radius=1,
        backward_confirmed=True,
    )
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    config = CycleConfig.from_gala(load_config(config_path), _Memory())
    session = CycleEngine(config).online_session(
        max_events=128,
        max_frontier_events=1024,
        initial_gaussian_count=9,
    )

    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()

    assert source.query_pack_count(query_lanes=config.relation_query_lanes) == 32
    assert session.accepted_event_count == source.logical_expanded_event_count
    assert session.completed_event_count == source.logical_expanded_event_count
    assert result.event_counts["RELATION"] == 9 * 256
    assert session.peak_frontier_events <= 1024
    assert session.quiescent


def test_online_streaming_reuses_schedule_for_capacity_preflight(
    monkeypatch,
) -> None:
    masks = np.zeros((2, 8), dtype=np.dtype("<u4"))
    for y in range(3):
        for x in range(8):
            local_query = y * 16 + x
            masks[:, local_query // 32] |= np.uint32(1 << (local_query % 32))
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(3, 8),
        point_ids=np.asarray([4, 5], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=2,
        ssim_radius=1,
        backward_confirmed=True,
    )

    def unexpected_wavefront(*_args, **_kwargs):
        raise AssertionError("streaming replay repeated the capacity scan")

    monkeypatch.setattr(
        VirtualTracePacket, "relation_store_wavefront", unexpected_wavefront,
    )
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    config = CycleConfig.from_gala(load_config(config_path), _Memory())
    session = CycleEngine(config).online_session(
        max_events=64,
        max_frontier_events=256,
        initial_gaussian_count=6,
    )

    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()

    assert result.event_counts["RELATION"] == 48
    assert session.quiescent


def test_online_streaming_cycles_are_transport_chunk_invariant() -> None:
    candidate_count = 44
    masks = np.full(
        (candidate_count, 16),
        np.uint32(0xFFFF_FFFF),
        dtype=np.dtype("<u4"),
    )
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=2,
        query_base=0,
        query_shape=(8, 8, 8),
        point_ids=np.arange(candidate_count, dtype=np.int64),
        point_keys=np.zeros(candidate_count, dtype=np.uint64),
        masks=masks,
        loss_flags=1,
        backward_confirmed=True,
    )
    assert source.logical_expanded_event_count > 131_072
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    gala_config = load_config(config_path)
    results = []
    max_frontier_events = (
        int(gala_config.value("trace.chunk_events"))
        * int(gala_config.value("trace.max_inflight_chunks"))
    )
    for max_events in (65_536, 131_072, 2_000_000):
        config = CycleConfig.from_gala(gala_config, _Memory())
        session = CycleEngine(config).online_session(
            max_events=max_events,
            max_frontier_events=max_frontier_events,
            initial_gaussian_count=candidate_count,
            retain_completion_cycles=True,
        )
        session.accept_query_packet(source)
        session.close_iteration(1)
        result = session.finish()
        assert session.peak_frontier_events <= max_frontier_events
        assert session.accepted_event_count == source.logical_expanded_event_count
        assert session.completed_event_count == source.logical_expanded_event_count
        assert session.quiescent
        results.append(result)

    reference = results[0]
    for result in results[1:]:
        assert result.total_cycles == reference.total_cycles
        assert result.event_counts == reference.event_counts
        assert result.module_counters == reference.module_counters
        assert result.stalls == reference.stalls
        assert result.completion_cycles == reference.completion_cycles


def test_online_query_packet_rejects_atomic_planning_overflow_before_expansion() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    session = CycleEngine(_config()).online_session(
        max_events=4,
        max_frontier_events=64,
        max_atomic_packet_events=9,
        initial_gaussian_count=1,
    )

    with pytest.raises(CycleConfigurationError, match="capacity-continuation"):
        session.accept_query_packet(source)

    assert session.accepted_event_count == 0
    assert session.pending_event_count == 0
    assert session.global_event_id == 0


def test_online_query_packet_rejects_infeasible_relation_store_wavefront() -> None:
    masks = np.zeros((2, 8), dtype=np.dtype("<u4"))
    for y in range(3):
        for x in range(8):
            local_query = y * 16 + x
            masks[:, local_query // 32] |= np.uint32(1 << (local_query % 32))
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(3, 8),
        point_ids=np.asarray([4, 5], dtype=np.int64),
        point_keys=np.asarray([0, 0], dtype=np.uint64),
        masks=masks,
        loss_flags=2,
        ssim_radius=1,
        backward_confirmed=True,
    )
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    config = replace(
        CycleConfig.from_gala(load_config(config_path), _Memory()),
        query_relation_store_records=3,
        resource_envelope=None,
        resource_usage=None,
    )
    session = CycleEngine(config).online_session(
        max_events=64,
        max_frontier_events=256,
        max_atomic_packet_events=1_000,
        initial_gaussian_count=6,
    )

    with pytest.raises(
        CycleConfigurationError, match="relation_store_capacity_infeasible",
    ):
        session.accept_query_packet(source)

    assert session.accepted_event_count == 0
    assert session.global_event_id == 0


def test_query_reduction_banks_serialize_same_bank_and_parallelize_distinct_banks() -> None:
    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    engine = CycleEngine(CycleConfig.from_gala(
        load_config(config_path), _Memory(),
    ))
    rows = np.empty(3, dtype=event_dtype())
    rows[:] = TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
    ).as_tuple()
    rows[1]["query_id"] = 64
    rows[2]["query_id"] = 1
    lanes = [0] * engine._module_issue_ports("bidirectional_query")

    first = engine._query_resource_allocation(
        lanes, rows[0], PrimitiveKind.FORWARD, None, 0,
    )
    second = engine._query_resource_allocation(
        lanes, rows[1], PrimitiveKind.FORWARD, None, 0,
    )
    repeated = engine._query_resource_allocation(
        lanes, rows[2], PrimitiveKind.FORWARD, None, 0,
    )

    assert first == (0,)
    assert second == (0,)
    lanes[first[0]] = engine.modules["bidirectional_query"].timing.latency
    assert engine._query_resource_allocation(
        lanes, rows[1], PrimitiveKind.FORWARD, None, 0,
    ) is None
    assert repeated == (1,)
    ready_cycle = engine.modules["bidirectional_query"].timing.latency
    assert engine._query_resource_allocation(
        lanes, rows[1], PrimitiveKind.FORWARD, None, ready_cycle,
    ) == (0,)
    assert engine._query_resource_allocation(
        lanes, rows[2], PrimitiveKind.FORWARD, None, ready_cycle,
    ) == (1,)


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


def test_online_lifecycle_commits_share_transaction_begin_dependency() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    session = CycleEngine(_config()).online_session(
        max_events=16, initial_gaussian_count=3,
    )
    # Keep the frontier resident while inspecting the dependency graph.  The
    # normal online path drains after each packet and retires completed rows.
    accept_event_packet = session.accept_event_packet
    session.accept_event_packet = lambda packet, **kwargs: accept_event_packet(
        packet, _drain_after=False
    )
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
    begin = next(event_id for event_id, kind in session._kinds.items()
                 if kind is PrimitiveKind.UPDATE_BEGIN)
    commits = [event_id for event_id, kind in session._kinds.items()
               if kind is PrimitiveKind.UPDATE_COMMIT]
    end = next(event_id for event_id, kind in session._kinds.items()
               if kind is PrimitiveKind.UPDATE_END)
    assert len(commits) == 3
    assert all(session._dependencies[event_id] == (begin,) for event_id in commits)
    assert set(session._dependencies[end]) == set(commits)
    session.close_iteration(1)
    session.finish()


def test_online_lifecycle_lineage_preserves_clone_and_split_order() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    session = CycleEngine(_config()).online_session(
        max_events=16, initial_gaussian_count=2,
    )
    accept_event_packet = session.accept_event_packet
    session.accept_event_packet = lambda packet, **kwargs: accept_event_packet(
        packet, _drain_after=False
    )
    session.accept_query_packet(source)
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_BEGIN, 0,
        field_mask=1, transaction_kind=1, active_ids=(0, 1),
    ))
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.CLONE, 0, parent_id=0, child_ids=(2, 3),
        transaction_kind=1,
    ))
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.SPLIT, 0, parent_id=1, child_ids=(4, 5),
        transaction_kind=1,
    ))
    session.accept_lifecycle(VirtualLifecycleRecord(
        1, VirtualLifecycleKind.UPDATE_END, 0,
        field_mask=1, transaction_kind=1,
    ))

    modifications = [
        event_id for event_id, kind in session._kinds.items()
        if kind is PrimitiveKind.SET_MODIFICATION
    ]
    assert len(modifications) == 6
    clone_parent, clone_child_a, clone_child_b, split_child_a, split_child_b, split_parent = modifications
    assert session._dependencies[clone_child_a][-1] == clone_parent
    assert session._dependencies[clone_child_b][-1] == clone_parent
    assert session._dependencies[split_parent][-2:] == (split_child_a, split_child_b)
    session.close_iteration(1)
    session.finish()


def test_online_cycle_replay_consumes_exact_semantic_workset_sidecar() -> None:
    masks = np.zeros((5, 8), dtype=np.dtype("<u4"))
    masks[:, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=(1, 1),
        point_ids=np.zeros(5, dtype=np.int64),
        point_keys=np.arange(5, dtype=np.uint64),
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
    session = CycleEngine(config, policy="variant:0101").online_session(
        max_events=4,
        initial_gaussian_count=1,
        semantic_workset_totals={(0, 0): 5},
    )
    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()
    assert result.module_counters["semantic_cache"]["workset_keys"] == 1
    assert result.module_counters["semantic_cache"]["workset_uses"] == 5
    assert result.module_counters["semantic_cache"]["workset_releases"] == 1
    assert result.module_counters["semantic_cache"]["miss_merges"] == 1
    assert result.module_counters["semantic_cache"]["multicast_reads"] == 1
    assert result.module_counters["shared_sram"]["accepted"] == 3
    assert result.event_counts["CACHE_REQUEST"] == 5
    assert result.event_counts["CACHE_RETURN"] == 5
    assert session.completed_event_count == session.accepted_event_count
    assert session.quiescent


def test_online_residency_retains_cache_return_request_dependency() -> None:
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
        loss_flags=1,
        backward_confirmed=True,
    )
    config = replace(
        _config(),
        cache_instances=1,
        cache_capacity_per_instance=4,
        cache_directory_banks=1,
        cache_sector_bytes=64,
    )
    session = CycleEngine(config, policy="variant:0101").online_session(
        max_events=16,
        initial_gaussian_count=1,
        semantic_workset_totals={(0, 0): 1},
    )
    drain = session._drain
    session._drain = lambda *args, **kwargs: None

    session.accept_query_packet(source)

    request = next(
        event_id for event_id, kind in session._kinds.items()
        if kind is PrimitiveKind.CACHE_REQUEST
    )
    cache_return = next(
        event_id for event_id, kind in session._kinds.items()
        if kind is PrimitiveKind.CACHE_RETURN
    )
    assert session._dependencies[cache_return] == (request,)
    session._drain = drain
    session.close_iteration(1)
    session.finish()


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
            CycleEngine(config, policy="variant:0101").online_session(
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


def test_semantic_residency_with_worksets_merges_repeated_state_reads() -> None:
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
    combined = CycleEngine(config, policy="variant:0101").run(trace)
    assert base.module_counters["semantic_cache"]["memory_requests"] == 2
    assert worksets.module_counters["semantic_cache"]["memory_requests"] == 2
    assert worksets.module_counters["semantic_cache"]["workset_keys"] == 1
    assert worksets.module_counters["semantic_cache"]["workset_uses"] == 2
    assert combined.module_counters["semantic_cache"]["memory_requests"] == 1
    assert combined.module_counters["semantic_cache"]["directory_misses"] == 1
    assert combined.module_counters["semantic_cache"]["directory_hits"] == 1
    assert combined.module_counters["semantic_cache"]["workset_uses"] == 2
    assert combined.module_counters["semantic_cache"]["workset_releases"] == 1
    assert combined.module_counters["semantic_cache"]["releases"] == 1
    assert combined.module_counters["shared_sram"]["accepted"] == 3


def _semantic_fusion_bundle_fixture() -> tuple[Trace, CycleConfig]:
    builder = TraceBuilder()
    request_ids = [
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=query_id,
            gaussian_id=7, state_version=3, template_id=1,
            address_token=448, data_bytes=64,
            resource_class=int(ResourceClass.CACHE),
        ))
        for query_id in range(4)
    ]
    return_ids = [
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=query_id,
            gaussian_id=7, state_version=3, template_id=1,
            address_token=448, data_bytes=64,
            resource_class=int(ResourceClass.CACHE),
        ), dependencies=[request_id])
        for query_id, request_id in enumerate(request_ids)
    ]
    for query_id in range(4):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
            gaussian_id=7, state_version=3, template_id=1,
            relation_id=query_id, reduction_key=query_id,
            address_token=448, resource_class=int(ResourceClass.ISSUE),
        ), dependencies=return_ids)
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=16, ports=1, banks=2,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache",
            "compute_pod", "bidirectional_query", "reconstruction_update",
            "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        fusion_query_state_banks=8, fusion_bank_head_lookahead=True,
        fusion_bank_head_index_bytes=224,
        fusion_semantic_bundle_index_bytes=320,
        fusion_forward_ports=2, fusion_consumer_ports=1,
        fusion_adjoint_ports=2,
        cache_instances=1, cache_capacity_per_instance=8,
        cache_directory_banks=2, cache_sector_bytes=64,
        cache_multicast_destinations=4,
    )
    return builder.finish(), config


@pytest.mark.parametrize(("policy", "expected_issues", "expected_followers"), [
    ("variant:0000", 0, 0),
    ("variant:0100", 0, 0),
    ("variant:0101", 1, 2),
    ("variant:1010", 0, 0),
    ("variant:1111", 2, 2),
])
def test_semantic_fusion_bundle_is_isolated_to_combined_residency(
    policy: str, expected_issues: int, expected_followers: int,
) -> None:
    trace, config = _semantic_fusion_bundle_fixture()

    result = CycleEngine(config, policy=policy).run(
        trace, validate_input=False,
    )

    fusion = result.module_counters["fusion_issue"]
    assert fusion["semantic_bundle_issues"] == expected_issues
    assert fusion["semantic_bundle_followers"] == expected_followers
    assert set(result.completion_cycles) == set(range(trace.event_count))


@pytest.mark.parametrize(("policy", "expected_issues"), [
    ("variant:0101", 1),
    ("variant:1111", 2),
])
def test_semantic_fusion_bundle_matches_offline_and_online_replay(
    policy: str, expected_issues: int,
) -> None:
    trace, config = _semantic_fusion_bundle_fixture()
    offline = CycleEngine(config, policy=policy).run(
        trace, validate_input=False,
    )
    session = CycleEngine(
        config, policy=policy,
    ).online_session(
        max_events=16, semantic_workset_totals={(7, 3): 4},
    )

    session.accept_event_packet(VirtualEventPacket(
        packet_id=0, global_event_start=0,
        events=trace.events, dependencies=trace.dependencies,
    ))

    online_issues = sum(
        queue.semantic_bundle_issues for queue in session._fusion_inputs.values()
    )
    online_followers = sum(
        queue.semantic_bundle_followers for queue in session._fusion_inputs.values()
    )
    assert session._last_completion_cycle == offline.total_cycles
    assert online_issues == offline.module_counters["fusion_issue"][
        "semantic_bundle_issues"
    ] == expected_issues
    assert online_followers == offline.module_counters["fusion_issue"][
        "semantic_bundle_followers"
    ] == 2
    assert session.completed_event_count == session.accepted_event_count
    assert session.quiescent


def test_semantic_adjoint_bundle_matches_offline_and_online_replay() -> None:
    builder = TraceBuilder()
    for query_id in range(4):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.ADJOINT), query_id=query_id,
            gaussian_id=7, state_version=3, template_id=1,
            relation_id=query_id, reduction_key=query_id,
            address_token=448, resource_class=int(ResourceClass.ISSUE),
        ))
    trace = builder.finish()
    _forward_trace, config = _semantic_fusion_bundle_fixture()

    offline = CycleEngine(config, policy="variant:0101").run(
        trace, validate_input=False,
    )
    session = CycleEngine(config, policy="variant:0101").online_session(
        max_events=16,
        retain_completion_cycles=True,
    )
    session.accept_event_packet(VirtualEventPacket(
        packet_id=0, global_event_start=0,
        events=trace.events, dependencies=trace.dependencies,
    ))

    adjoint_queue = session._fusion_inputs[TaskKind.ADJOINT]
    forward_queue = session._fusion_inputs[TaskKind.FORWARD]
    assert session._last_completion_cycle == offline.total_cycles
    assert session._completion_cycles == offline.completion_cycles
    assert adjoint_queue.semantic_bundle_issues == 1
    assert adjoint_queue.semantic_bundle_followers == 2
    assert forward_queue.semantic_bundle_issues == 0
    assert session.quiescent


def test_shared_sram_replays_same_address_read_against_fill_write() -> None:
    class _ImmediateMemory:
        def submit(self, *, address: int, size_bytes: int, is_write: bool,
                   arrival_cycle: int) -> int:
            return arrival_cycle

    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0,
        gaussian_id=2, state_version=0, address_token=0, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=1,
        gaussian_id=3, state_version=0, address_token=0, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ))
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=8, ports=2, banks=2,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache",
            "compute_pod", "bidirectional_query", "reconstruction_update",
            "shared_sram",
        )},
        memory=_ImmediateMemory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        cache_instances=2,
        shared_sram_read_ports_per_bank=1,
        shared_sram_write_ports_per_bank=1,
    )
    result = CycleEngine(config, policy="variant:0000").run(
        builder.finish(), validate_input=False,
    )

    assert any(
        stall.module == "shared_sram" and stall.reason == "same_address_replay"
        for stall in result.stalls
    )
    assert result.module_counters["shared_sram"]["bank_conflicts"] >= 1


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
    result = CycleEngine(config, policy="variant:0101").run(
        trace, validate_input=False,
    )
    single_destination = CycleEngine(
        replace(config, cache_multicast_destinations=1), policy="variant:0101",
    ).run(trace, validate_input=False)
    cache = result.module_counters["semantic_cache"]
    assert cache["directory_misses"] == 1
    assert cache["directory_hits"] == 1
    assert cache["multicast_reads"] == 1
    assert cache["memory_requests"] == 1
    assert cache["releases"] == 1
    assert result.module_counters["shared_sram"]["accepted"] == 3
    assert single_destination.module_counters["shared_sram"]["accepted"] == 6
    assert single_destination.module_counters["semantic_cache"]["multicast_reads"] == 0
    assert cache["busy_cycles"] < single_destination.module_counters["semantic_cache"]["busy_cycles"]
    assert result.event_counts["CACHE_REQUEST"] == 5
    assert result.event_counts["CACHE_RETURN"] == 5
    assert set(result.completion_cycles) == set(range(trace.event_count))
    assert np.array_equal(trace.dependencies, dependencies_before)


def test_semantic_residency_preserves_ready_candidate_order() -> None:
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

    assert ordered == event_ids


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
    result = CycleEngine(config, policy="variant:0101").run(trace)
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
    result = CycleEngine(config, policy="variant:0101").run(builder.finish())
    cache = result.module_counters["semantic_cache"]
    assert cache["directory_misses"] == 1
    assert cache["directory_hits"] == 1
    assert cache["memory_requests"] == 1
    assert cache["workset_releases"] == 1


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
        independent_port_config, policy="variant:1010"
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
    assert base.total_cycles > independent.total_cycles


def test_idle_fusion_candidate_slot_borrows_real_fifo_heads_fairly() -> None:
    config = replace(
        _config(), candidate_lanes=3,
        fusion_forward_ports=2, fusion_consumer_ports=1,
        fusion_adjoint_ports=2,
    )
    engine = CycleEngine(config, policy="variant:1010")
    def packet(event_id: int, kind: TaskKind) -> TaskPacket:
        return TaskPacket(
            event_id=event_id, query_id=event_id, gaussian_id=event_id,
            reduction_key=event_id, resource=event_id, state_version=0,
            template_id=0, address_token=event_id, task_kind=kind,
        )

    inputs = {
        kind: _BankedFusionSourceQueue(capacity=32, banks=8)
        for kind in TaskKind
    }
    for event_id, bank in ((10, 0), (11, 1), (12, 2)):
        inputs[TaskKind.FORWARD].append(
            packet(event_id, TaskKind.FORWARD), bank=bank,
        )
    for event_id, bank in ((20, 0), (21, 1), (22, 2)):
        inputs[TaskKind.ADJOINT].append(
            packet(event_id, TaskKind.ADJOINT), bank=bank,
        )

    first, first_borrowed = engine._peek_fusion_input_candidates(
        inputs, borrow_cursor=0,
    )
    for selected in first:
        borrow_commit = first_borrowed.get(selected.event_id)
        inputs[selected.task_kind].commit(
            selected.event_id,
            bank_cursor=(borrow_commit[1] if borrow_commit is not None else None),
        )
    cursor = next(iter(first_borrowed.values()))[0]
    second, _second_borrowed = engine._peek_fusion_input_candidates(
        inputs, borrow_cursor=cursor,
    )

    assert [item.event_id for item in first] == [10, 20, 11]
    assert [item.event_id for item in second] == [12, 21, 22]


def test_banked_fusion_fifo_exposes_only_a_real_head_per_bank() -> None:
    queue = _BankedFusionSourceQueue(capacity=32, banks=8)

    def packet(event_id: int, query_id: int) -> TaskPacket:
        return TaskPacket(
            event_id=event_id, query_id=query_id, gaussian_id=event_id,
            reduction_key=query_id, resource=event_id, state_version=0,
            template_id=0, address_token=event_id,
            task_kind=TaskKind.FORWARD,
        )

    queue.append(packet(10, 0), bank=0)
    queue.append(packet(11, 1), bank=0)
    queue.append(packet(12, 8), bank=1)

    offer = queue.first_compatible_bank_head(
        (queue.head(),), compatible=CycleEngine._fusion_packets_compatible,
    )

    assert offer is not None
    borrowed, bank_cursor = offer
    assert borrowed.event_id == 12
    assert bank_cursor == 2
    assert len(queue) == 3
    assert queue.head().event_id == 10


def test_semantic_residency_prioritizes_fullest_ready_fusion_group() -> None:
    config = replace(
        _config(), candidate_lanes=3,
        fusion_forward_ports=2, fusion_consumer_ports=1,
        fusion_adjoint_ports=2, fusion_semantic_bundle_index_bytes=320,
        cache_multicast_destinations=4,
    )
    engine = CycleEngine(config, policy="variant:0101")
    inputs = {
        kind: _BankedFusionSourceQueue(capacity=32, banks=8)
        for kind in TaskKind
    }

    def packet(event_id: int, query_id: int, gaussian_id: int) -> TaskPacket:
        return TaskPacket(
            event_id=event_id, query_id=query_id, gaussian_id=gaussian_id,
            reduction_key=query_id, resource=event_id, state_version=0,
            template_id=1, address_token=gaussian_id,
            task_kind=TaskKind.FORWARD,
        )

    inputs[TaskKind.ADJOINT].append(TaskPacket(
        event_id=1, query_id=0, gaussian_id=9, reduction_key=9,
        resource=1, state_version=0, template_id=1, address_token=9,
        task_kind=TaskKind.ADJOINT,
    ), bank=0)
    inputs[TaskKind.FORWARD].append(packet(10, 0, 7), bank=0)
    inputs[TaskKind.FORWARD].append(packet(11, 8, 7), bank=1)
    inputs[TaskKind.FORWARD].append(packet(12, 16, 7), bank=2)
    inputs[TaskKind.FORWARD].append(packet(20, 24, 8), bank=3)
    inputs[TaskKind.FORWARD].append(packet(21, 32, 8), bank=4)

    selected, borrowed = engine._peek_fusion_input_candidates(
        inputs, borrow_cursor=0,
    )

    assert [item.event_id for item in selected] == [10]
    assert borrowed == {}
    bundle = engine._semantic_fusion_bundle(inputs, selected)
    assert bundle is not None
    assert bundle[0].event_id == 10
    assert [item.event_id for item in bundle[1]] == [11, 12]


def test_semantic_residency_groups_ready_adjoint_state() -> None:
    config = replace(
        _config(), candidate_lanes=3,
        fusion_forward_ports=2, fusion_consumer_ports=1,
        fusion_adjoint_ports=2, fusion_semantic_bundle_index_bytes=320,
        cache_multicast_destinations=4,
    )
    engine = CycleEngine(config, policy="variant:0101")
    inputs = {
        kind: _BankedFusionSourceQueue(capacity=32, banks=8)
        for kind in TaskKind
    }

    for event_id, query_id in ((10, 0), (11, 8), (12, 16)):
        inputs[TaskKind.ADJOINT].append(TaskPacket(
            event_id=event_id, query_id=query_id, gaussian_id=7,
            reduction_key=query_id, resource=event_id, state_version=3,
            template_id=1, address_token=7, task_kind=TaskKind.ADJOINT,
        ), bank=query_id // 8)

    selected, borrowed = engine._peek_fusion_input_candidates(
        inputs, borrow_cursor=0,
    )
    bundle = engine._semantic_fusion_bundle(inputs, selected)

    assert [item.event_id for item in selected] == [10]
    assert borrowed == {}
    assert bundle is not None
    assert bundle[0].event_id == 10
    assert [item.event_id for item in bundle[1]] == [11, 12]


def test_semantic_ready_group_priority_isolated_from_other_variants() -> None:
    config = replace(
        _config(), candidate_lanes=3,
        fusion_forward_ports=2, fusion_consumer_ports=1,
        fusion_adjoint_ports=2, fusion_semantic_bundle_index_bytes=320,
        cache_multicast_destinations=4,
    )

    def candidates(policy: str) -> list[int]:
        engine = CycleEngine(config, policy=policy)
        inputs = {
            kind: _BankedFusionSourceQueue(capacity=32, banks=8)
            for kind in TaskKind
        }
        inputs[TaskKind.ADJOINT].append(TaskPacket(
            event_id=1, query_id=0, gaussian_id=9, reduction_key=9,
            resource=1, state_version=0, template_id=1, address_token=9,
            task_kind=TaskKind.ADJOINT,
        ), bank=0)
        for event_id, query_id, bank in ((10, 0, 0), (11, 8, 1), (12, 16, 2)):
            inputs[TaskKind.FORWARD].append(TaskPacket(
                event_id=event_id, query_id=query_id, gaussian_id=7,
                reduction_key=query_id, resource=event_id, state_version=0,
                template_id=1, address_token=7,
                task_kind=TaskKind.FORWARD,
            ), bank=bank)
        selected, _borrowed = engine._peek_fusion_input_candidates(
            inputs, borrow_cursor=0,
        )
        return [item.event_id for item in selected]

    assert candidates("variant:0101") == [10]
    assert candidates("variant:0000") == [10, 1, 11]
    assert candidates("variant:1010") == [10, 1, 11]


def test_banked_fusion_fifo_advances_cursor_only_after_issue_commit() -> None:
    queue = _BankedFusionSourceQueue(capacity=32, banks=8)

    def packet(event_id: int, query_id: int) -> TaskPacket:
        return TaskPacket(
            event_id=event_id, query_id=query_id, gaussian_id=event_id,
            reduction_key=query_id, resource=event_id, state_version=0,
            template_id=0, address_token=event_id,
            task_kind=TaskKind.FORWARD,
        )

    queue.append(packet(10, 0), bank=0)
    queue.append(packet(11, 8), bank=1)
    queue.append(packet(12, 16), bank=2)

    first = queue.first_compatible_bank_head(
        (queue.head(),), compatible=CycleEngine._fusion_packets_compatible,
    )
    repeated = queue.first_compatible_bank_head(
        (queue.head(),), compatible=CycleEngine._fusion_packets_compatible,
    )
    assert first is not None and repeated is not None
    assert first[0].event_id == repeated[0].event_id == 11

    queue.commit(first[0].event_id, bank_cursor=first[1])
    advanced = queue.first_compatible_bank_head(
        (queue.head(),), compatible=CycleEngine._fusion_packets_compatible,
    )
    assert advanced is not None
    assert advanced[0].event_id == 12


def test_banked_fusion_fifo_capacity_is_shared_across_banks() -> None:
    queue = _BankedFusionSourceQueue(capacity=32, banks=8)
    for event_id in range(32):
        queue.append(TaskPacket(
            event_id=event_id, query_id=event_id, gaussian_id=event_id,
            reduction_key=event_id, resource=event_id, state_version=0,
            template_id=0, address_token=event_id,
            task_kind=TaskKind.FORWARD,
        ), bank=event_id % 8)

    with pytest.raises(OverflowError, match="full"):
        queue.append(TaskPacket(
            event_id=32, query_id=32, gaussian_id=32, reduction_key=32,
            resource=32, state_version=0, template_id=0, address_token=32,
            task_kind=TaskKind.FORWARD,
        ), bank=0)
    assert len(queue) == 32
    queue.commit(8)
    assert len(queue) == 31
    assert queue.head().event_id == 0


def test_owner_gradient_capacity_blocks_compute_not_query_replay() -> None:
    rows = np.empty(4, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(4)
    rows["iteration_id"] = [1, 1, 2, 2]
    rows["gaussian_id"] = 0
    rows["state_version"] = [0, 0, 1, 1]
    rows["relation_id"] = [10, 10, 20, 20]
    rows["primitive_kind"] = [
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.GRADIENT_REDUCTION),
        int(PrimitiveKind.ADJOINT),
        int(PrimitiveKind.GRADIENT_REDUCTION),
    ]
    tracker = OwnerGradientTracker(
        pods=1, clusters_per_pod=1, slots_per_cluster=1,
    )
    tracker.register_rows(rows)
    tracker.reserve_adjoint((0,))

    def pop(module_name: str) -> list[tuple[int, int]]:
        queue = _ReadyCandidateQueue()
        queue.push((2, 0))
        return queue.pop_acceptable(
            1, module_name, row_for=rows.__getitem__,
            physical_stage_for=lambda _event_id: None,
            owner_gradients=tracker, query_replay=None,
        )

    assert pop("bidirectional_query") == [(2, 0)]
    assert pop("compute_pod") == []


def test_semantic_ready_index_selects_noncontiguous_same_key_followers() -> None:
    rows = np.empty(6, dtype=TraceBuilder().finish().events.dtype)
    rows[:] = TraceEvent().as_tuple()
    rows["event_id"] = np.arange(6)
    rows["primitive_kind"] = int(PrimitiveKind.CACHE_RETURN)
    rows["gaussian_id"] = [7, 8, 7, 9, 7, 7]
    rows["state_version"] = [3, 3, 3, 3, 3, 3]
    queue = _ReadyCandidateQueue()
    for event_id in range(6):
        queue.push((event_id, 0))

    selected = queue.pop_acceptable(
        1, "semantic_cache", capacity=8, row_for=rows.__getitem__,
        physical_stage_for=lambda _event_id: None,
        owner_gradients=None, query_replay=None,
        semantic_multicast_destinations=4,
    )

    assert selected == [(0, 0), (2, 0), (4, 0), (5, 0)]
    assert list(queue) == [(1, 0), (3, 0)]


def test_fusion_issue_banks_by_physical_query_state_index() -> None:
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=8, ports=3, banks=1,
    )
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=8, candidate_lanes=3,
        relation_query_lanes=1, fusion_query_state_banks=8,
        fusion_forward_ports=1, fusion_consumer_ports=1, fusion_adjoint_ports=1,
    )

    grouped_engine = CycleEngine(
        replace(config, relation_query_lanes=8), policy="variant:1010",
    )
    grouped_rows = []
    for query_id in (0, 7, 8):
        builder = TraceBuilder()
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CONSUMER), query_id=query_id,
        ))
        grouped_rows.append(builder.finish().events[0])
    assert [
        grouped_engine._fusion_query_state_bank(row) for row in grouped_rows
    ] == [0, 0, 1]

    different_banks = TraceBuilder()
    different_banks.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, relation_id=0, reduction_key=0, address_token=0,
        resource_class=int(ResourceClass.ISSUE),
    ))
    different_banks.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=1,
        consumer_id=1, reduction_key=1, address_token=0,
        resource_class=int(ResourceClass.ISSUE),
    ))
    parallel = CycleEngine(
        config, policy="variant:1010",
    ).run(different_banks.finish())
    assert not any(
        stall.module == "fusion_issue" and stall.reason == "bank"
        for stall in parallel.stalls
    )

    same_bank = TraceBuilder()
    same_bank.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=0,
        gaussian_id=0, relation_id=0, reduction_key=0, address_token=0,
        resource_class=int(ResourceClass.ISSUE),
    ))
    same_bank.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=8,
        consumer_id=8, reduction_key=8, address_token=1,
        resource_class=int(ResourceClass.ISSUE),
    ))
    serialized = CycleEngine(
        config, policy="variant:1010",
    ).run(same_bank.finish())
    assert any(
        stall.module == "fusion_issue" and stall.reason == "bank"
        for stall in serialized.stalls
    )


def test_relation_constructor_separates_append_banks_and_dynamic_support_lanes() -> None:
    config = CycleConfig.from_gala(
        load_config(
            Path(__file__).parents[1] / "configs/architecture/gala.yaml"
        ),
        _Memory(),
    )
    engine = CycleEngine(config)
    dtype = TraceBuilder().finish().events.dtype

    def row(kind: PrimitiveKind, query_id: int, gaussian_id: int = 0):
        value = np.zeros((), dtype=dtype)
        value["primitive_kind"] = int(kind)
        value["query_id"] = query_id
        value["gaussian_id"] = gaussian_id
        return value

    assert engine._module_issue_ports("relation_constructor") == 24
    assert engine._module_partition_count("relation_constructor") == 17
    assert engine._module_partition_issue_limit("relation_constructor", 0) == 8
    assert engine._module_partition_issue_limit("relation_constructor", 1) == 1
    assert engine._module_partition(
        "relation_constructor", row(PrimitiveKind.RELATION, 8)
    ) == 2
    assert engine._module_partition(
        "relation_constructor", row(PrimitiveKind.RELATION_CANDIDATE, -1, 9)
    ) == 0
    candidate = row(PrimitiveKind.RELATION_CANDIDATE, -1, 9)
    assert list(engine._module_lane_indices(
        "relation_constructor", candidate,
    )) == list(range(8))

    builder = TraceBuilder()
    for gaussian_id in (9, 17):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
            gaussian_id=gaussian_id,
        ))
    result = engine.run(builder.finish())
    assert result.module_counters["relation_constructor"]["accepted"] == 2
    assert not any(
        stall.module == "relation_constructor" for stall in result.stalls
    )


def test_query_load_rules_order_three_fifo_heads_without_enabling_fusion() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.FORWARD), query_id=9,
        gaussian_id=9, relation_id=9, reduction_key=9, address_token=9,
        state_version=0, resource_class=int(ResourceClass.ISSUE),
    ))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=1,
        gaussian_id=1, consumer_id=1, reduction_key=1, address_token=1,
        state_version=0, resource_class=int(ResourceClass.ISSUE),
    ))
    timing = ModuleTiming(
        latency=1, initiation_interval=1, queue_capacity=8, ports=3, banks=16,
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
    engine = CycleEngine(config, policy="variant:1000")
    committed: list[int] = []
    original_commit = engine.issue_scheduler.commit_issued

    def record_commit(tasks) -> None:
        packets = tuple(tasks)
        committed.extend(packet.event_id for packet in packets)
        original_commit(packets)

    engine.issue_scheduler.commit_issued = record_commit  # type: ignore[method-assign]
    result = engine.run(builder.finish())

    assert committed == [1, 0]
    assert any(
        stall.module == "fusion_issue" and stall.reason == "base_single_issue"
        for stall in result.stalls
    )
    assert set(result.completion_cycles) == {0, 1}


def test_online_query_load_rules_drain_pending_fifo_heads() -> None:
    masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
    masks[0, 0] = 1
    source = VirtualTracePacket(
        iteration_id=1, template_id=1, query_base=0, query_shape=(1, 1),
        point_ids=np.asarray([0], dtype=np.int64),
        point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
        loss_flags=1, backward_confirmed=True,
    )
    engine = CycleEngine(_config(), policy="variant:1000")
    committed: list[int] = []
    original_commit = engine.issue_scheduler.commit_issued

    def record_commit(tasks) -> None:
        packets = tuple(tasks)
        committed.extend(item.event_id for item in packets)
        original_commit(packets)

    engine.issue_scheduler.commit_issued = record_commit  # type: ignore[method-assign]
    session = engine.online_session(max_events=4, initial_gaussian_count=1)
    session.accept_query_packet(source)
    session.close_iteration(1)
    result = session.finish()

    assert committed
    assert result.event_counts["FORWARD"] == 1
    assert result.event_counts["CONSUMER"] == 1
    assert result.event_counts["ADJOINT"] == 1
    assert session.quiescent


def test_query_load_rules_do_not_deadlock_consumers_behind_replay_capacity() -> None:
    builder = TraceBuilder()
    consumers = [
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.CONSUMER), query_id=query_id,
            consumer_id=query_id, reduction_key=query_id,
            resource_class=int(ResourceClass.ISSUE),
        ))
        for query_id in range(2)
    ]
    for query_id in range(2):
        builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.ADJOINT), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id,
            reduction_key=query_id, resource_class=int(ResourceClass.ISSUE),
        ), dependencies=consumers)

    result = CycleEngine(
        replace(
            _config(),
            query_reduction_banks=2,
            query_partial_sum_groups_per_bank=2,
            query_loss_fma_lanes=2,
            query_loss_queries_per_cycle=1,
            query_adjoint_replay_lanes=1,
            query_replay_queue_entries=1,
            query_relation_window_entries=1,
            query_relation_window_entry_bytes=32,
            query_relation_store_banks=1,
            query_relation_store_records=8,
            query_relation_store_record_bytes=3,
            query_relation_candidate_ordinal_bits=16,
            query_volume_banks=2,
            query_volume_word_bytes=16,
        ),
        policy="variant:1000",
    ).run(builder.finish())

    assert set(result.completion_cycles) == set(range(4))
    replay = result.module_counters["bidirectional_query"]
    assert replay["replay_queue_peak_entries"] == 1
    assert replay["replay_queue_live_entries"] == 0


def test_online_query_history_uses_template_local_identity_across_iterations() -> None:
    engine = CycleEngine(_config(), policy="variant:1000")
    session = engine.online_session(max_events=16, initial_gaussian_count=1)

    for iteration, iteration_query_base in ((1, 0), (2, 2)):
        packets = []
        for template_id, query_base, query_shape, mask_words in (
            (1, iteration_query_base, (1, 1), 8),
            (2, iteration_query_base + 1, (1, 1, 1), 16),
        ):
            masks = np.zeros((1, mask_words), dtype=np.dtype("<u4"))
            masks[0, 0] = 1
            packets.append(VirtualTracePacket(
                iteration_id=iteration, template_id=template_id,
                query_base=query_base, query_shape=query_shape,
                point_ids=np.asarray([0], dtype=np.int64),
                point_keys=np.asarray([0], dtype=np.uint64), masks=masks,
                loss_flags=1, backward_confirmed=True,
            ))
        session.accept_query_packets(packets)
        session.close_iteration(iteration)

    result = session.finish()

    assert result.module_counters["fusion_issue"]["query_history_restored"] == 2
    assert result.module_counters["fusion_issue"][
        "query_history_candidate_evaluations"
    ] == 0
    assert result.module_counters["fusion_issue"][
        "query_load_rule_evaluations"
    ] > 0
    assert session.quiescent


def test_offline_query_packet_boundary_releases_iterations_in_order() -> None:
    expander = VirtualQueryEventExpander(max_events=16)
    row_parts = []
    dependency_parts = []
    dependency_offset = 0
    packet_metadata = []
    for iteration, query_base in ((1, 10), (2, 20)):
        masks = np.zeros((1, 8), dtype=np.dtype("<u4"))
        masks[0, 0] = 1
        packet = VirtualTracePacket(
            iteration_id=iteration,
            template_id=1,
            query_base=query_base,
            query_shape=(1, 1),
            point_ids=np.asarray([0], dtype=np.int64),
            point_keys=np.asarray([0], dtype=np.uint64),
            masks=masks,
            loss_flags=1,
            backward_confirmed=True,
        )
        for event_packet in expander.expand(packet):
            rows = event_packet.events.copy()
            rows["dependency_begin"] += dependency_offset
            row_parts.append(rows)
            dependency_parts.append(event_packet.dependencies)
            dependency_offset += event_packet.dependencies.size
        packet_metadata.append({
            "query_base": query_base,
            "query_count": 1,
            "candidate_count": 1,
        })
    trace = Trace(
        np.concatenate(row_parts),
        np.concatenate(dependency_parts),
        np.empty(0, dtype=np.dtype("<f4")),
        {
            "schema_version": "gala-clamp-events-v2",
            "initial_gaussian_count": 1,
            "trace_sample": {
                "schema_version": "gala-query-packet-sample-v1",
                "result_scope": "quick_cycle_validation",
                "formal_performance_eligible": False,
                "quality_eligible": False,
                "boundary_condition": (
                    "prior_selected_packet_completes_before_next_iteration"
                ),
                "packets": packet_metadata,
            },
        },
    )

    config_path = Path(__file__).parents[1] / "configs/architecture/gala.yaml"
    config = CycleConfig.from_gala(load_config(config_path), _Memory())
    result = CycleEngine(config, policy="variant:1000").run(trace)

    assert result.event_counts["GRADIENT_REDUCTION"] == 2
    assert result.module_counters["fusion_issue"]["query_history_restored"] == 1


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
    result = CycleEngine(config, policy="variant:1010").run(builder.finish())
    assert result.module_counters["fusion_issue"]["accepted"] == 2
    assert result.completion_cycles[0] < result.completion_cycles[1]


@pytest.mark.parametrize("policy,expected", [
    ("variant:0000", (False, False, False, False)),
    ("variant:1000", (True, False, False, False)),
    ("variant:1010", (True, False, True, False)),
    ("variant:0100", (False, True, False, False)),
    ("variant:0101", (False, True, False, True)),
    ("variant:1100", (True, True, False, False)),
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


@pytest.mark.parametrize(
    "policy", ["variant:0010", "variant:0001", "variant:0111"],
)
def test_cycle_engine_rejects_unsupported_mechanism_combinations(
    policy: str,
) -> None:
    with pytest.raises(ValueError, match="unknown cycle policy"):
        CycleEngine(_config(), policy=policy)


def test_overlap_guided_issue_uses_history_without_compiler_load_rules() -> None:
    scheduler = FusionIssueScheduler(
        candidate_lanes=3, forward_ports=1, consumer_ports=1,
        adjoint_ports=1,
    )
    scheduler.set_strict_lifecycle()
    for _ in range(3):
        scheduler.relation_accept(
            (1,), iteration_id=0, history_keys=((1, 0),),
        )
    scheduler.producer_close((1,))
    scheduler.reduction_writeback((1,))
    scheduler.forward_retire((1, 1, 1))
    assert scheduler.release_completed() == 1
    scheduler.relation_accept(
        (101,), iteration_id=1, history_keys=((1, 0),),
    )
    scheduler.producer_close((101,))
    scheduler.reduction_writeback((101,))

    task = TaskPacket(
        0, 101, 1, 101, 1, 0, 1, 0, TaskKind.FORWARD,
    )
    scheduler.select(
        (task,), use_load_rules=False, use_history_prediction=True,
    )

    history = scheduler.history_snapshot()
    assert history["query_history_candidate_evaluations"] == 1
    assert history["query_load_rule_evaluations"] == 0


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
    result = CycleEngine(config, policy="variant:1010").run(builder.finish())
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
    engine = CycleEngine(config, policy="variant:1010")
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


def test_consumer_credit_owner_is_last_query_dependency_not_last_generic_dependency() -> None:
    builder = TraceBuilder()
    reductions = []
    for query_id in (0, 1):
        relation = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id,
            resource_class=int(ResourceClass.RELATION),
        ))
        close = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=query_id,
            resource_class=int(ResourceClass.RELATION),
        ), dependencies=[relation])
        forward = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
            gaussian_id=query_id, relation_id=query_id, reduction_key=query_id,
            resource_class=int(ResourceClass.ISSUE),
        ), dependencies=[relation])
        reductions.append(builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.QUERY_REDUCTION), query_id=query_id,
            reduction_key=query_id, resource_class=int(ResourceClass.QUERY),
        ), dependencies=[close, forward]))
    tail = None
    for index in range(8):
        tail = builder.emit(TraceEvent(
            primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
            gaussian_id=100 + index,
            resource_class=int(ResourceClass.RELATION),
        ), dependencies=(() if tail is None else (tail,)))
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CONSUMER), query_id=0,
        consumer_id=0, reduction_key=0,
        resource_class=int(ResourceClass.QUERY),
    ), dependencies=[*reductions, tail])
    engine = CycleEngine(_config(), policy="query")
    credited: list[int] = []
    original_credit = engine.issue_scheduler.successor_credit

    def record_credit(owner: int, count: int = 1) -> bool:
        credited.append(owner)
        return original_credit(owner, count)

    engine.issue_scheduler.successor_credit = record_credit  # type: ignore[method-assign]

    result = engine.run(builder.finish(), validate_input=False)

    assert result.event_counts["CONSUMER"] == 1
    assert len(credited) == 1
    assert credited[0] in {0, 1}


def test_loaded_architecture_uses_three_independent_144_entry_candidate_fifos() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    cycle_config = CycleConfig.from_gala(config, _Memory())

    assert cycle_config.candidate_fifo_entries == 144
    assert cycle_config.fusion_bank_head_lookahead is True
    assert cycle_config.fusion_bank_head_index_bytes == 224
    assert cycle_config.fusion_semantic_bundle_index_bytes == 640


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
    assert len(runs) == 7
    assert validation_calls == 1
    assert [run.variant.bits for run in runs] == [
        "0000", "1000", "1010", "0100", "0101", "1100", "1111",
    ]
    assert runs[-1].result.policy == "variant:1111"


def test_ablation_runner_reports_each_completed_variant() -> None:
    from gala_sim.ablation.runner import run_matrix

    completed: list[str] = []
    runs = run_matrix(_trace(), _config(), progress=lambda run: completed.append(run.variant.bits))
    assert len(runs) == 7
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
    assert [run.variant.bits for run in runs] == [
        "0000", "1000", "1010", "0100", "0101", "1100", "1111",
    ]


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
    assert len(run_matrix(trace, config)) == 7
