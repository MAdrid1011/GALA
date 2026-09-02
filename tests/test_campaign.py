from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gala_sim.campaign import (
    ALL_COMBINATIONS,
    _campaign_result_root,
    _expand_trace_iterations,
    _first_prepared_root,
    _trace_for_official_campaign,
    prepared_dataset,
    run_campaign,
)
from gala_sim.clamp import PrimitiveKind, TraceBuilder, TraceEvent, ResourceClass
from gala_sim.trace import VirtualPacketArchiveReader, validate_trace
from gala_sim.workspace import WorkspacePaths


def _prepared_dataset(root: Path, dataset_id: str) -> None:
    root.mkdir(parents=True)
    projection_root = root / "projections"
    projection_root.mkdir()
    np.save(projection_root / "000.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
    np.save(root / "init_prepared.npy", np.asarray([
        [-1.0, -1.0, -1.0, 0.2],
        [-0.2, 0.1, 0.0, 0.4],
        [0.3, -0.2, 0.5, 0.8],
        [1.0, 1.0, 1.0, 0.6],
    ], dtype=np.float32))
    (root / "metadata.json").write_text(json.dumps({
        "angles_radians": [0.0], "detector_shape": [4, 4],
        "volume_shape": [4, 4, 4], "DSO": 5.0, "DSD": 7.0,
        "initialization": "init_prepared.npy", "train_indices": [0],
        "test_indices": [],
    }), encoding="utf-8")


def test_all_campaign_identifiers_are_explicit_and_unique() -> None:
    assert len(ALL_COMBINATIONS) == 12
    assert len(set(ALL_COMBINATIONS)) == 12


def test_campaign_result_root_keeps_iteration_variants_separate(tmp_path: Path) -> None:
    paths = WorkspacePaths.discover(
        repository=Path(__file__).resolve().parents[1],
        workspace=tmp_path / "workspace",
    )
    assert _campaign_result_root(paths, "r2_gaussian", "walnut", 1).name == (
        "cpu-validation-v1"
    )
    assert _campaign_result_root(paths, "r2_gaussian", "walnut", 8).name == (
        "cpu-validation-8iter-v1"
    )
    assert _campaign_result_root(
        paths, "r2_gaussian", "walnut", 1, official_trace=True,
    ).name == "official-validation-1-1-v1"


def test_prepared_dataset_selects_newest_complete_cache(tmp_path: Path) -> None:
    paths = WorkspacePaths.discover(
        repository=Path(__file__).resolve().parents[1],
        workspace=tmp_path / "workspace",
    ).ensure()
    older = paths.cache / "prepared" / "walnut" / "aaa"
    newer = paths.cache / "prepared" / "walnut" / "zzz"
    _prepared_dataset(older, "walnut")
    _prepared_dataset(newer, "walnut")
    import os

    os.utime(older / "metadata.json", ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer / "metadata.json", ns=(2_000_000_000, 2_000_000_000))
    assert _first_prepared_root(paths, "walnut") == newer


def test_official_campaign_capture_is_identity_bound_and_reusable(
    tmp_path: Path, monkeypatch,
) -> None:
    import gala_sim.campaign as campaign

    repository = Path(__file__).resolve().parents[1]
    workspace = WorkspacePaths.discover(
        repository=repository, workspace=tmp_path / "workspace",
    ).ensure()
    _prepared_dataset(
        workspace.cache / "prepared" / "walnut" / "fixture", "walnut",
    )
    dataset = prepared_dataset(workspace, "walnut")
    config = campaign.load_config(repository / "configs/architecture/gala.yaml")
    builder = TraceBuilder()
    builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE)))
    source_trace = builder.finish(metadata={
        "model_id": "exact_gs", "dataset": "walnut",
        "config_sha256": config.sha256, "model_commit": "fixture-commit",
    })

    class Adapter:
        def prepare(self, prepared, prepared_config):
            assert prepared is dataset
            assert prepared_config is config
            return SimpleNamespace()

        def capture_trace(self, _run, sink):
            sink.push(source_trace.events, source_trace.dependencies, source_trace.payload)
            sink.close()
            return SimpleNamespace(
                trace=source_trace,
                reference=SimpleNamespace(gpu_reference={"status": "passed"}),
            )

    monkeypatch.setattr(campaign, "get_model_adapter", lambda *args, **kwargs: Adapter())
    trace_root, trace, reference = _trace_for_official_campaign(
        workspace, "exact_gs", "walnut", dataset, config, (1, 1),
    )

    assert trace_root == workspace.traces / "exact_gs/walnut/official-capture-1-1-v1/trace"
    assert trace.metadata["official_model_trace"] is True
    assert trace.metadata["config_sha256"] == config.sha256
    assert reference == {"status": "passed"}

    monkeypatch.setattr(
        campaign, "get_model_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid official trace should be reused")
        ),
    )
    _, reused, reused_reference = _trace_for_official_campaign(
        workspace, "exact_gs", "walnut", dataset, config, (1, 1),
    )
    assert reused.event_count == source_trace.event_count
    assert reused_reference == reference


def test_campaign_writes_identity_bound_trace_and_seven_variant_result(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    workspace = WorkspacePaths.discover(
        repository=repository, workspace=tmp_path / "workspace",
    ).ensure()
    _prepared_dataset(
        workspace.cache / "prepared" / "walnut" / "fixture", "walnut",
    )
    result = run_campaign("exact_gs", "walnut", workspace=workspace)
    assert result.base_cycles == result.ablation_cycles["0000"]
    assert set(result.ablation_cycles) == {
        "0000", "1000", "1010", "0100", "0101", "1100", "1111",
    }
    metadata = json.loads(
        (result.trace_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_id"] == "exact_gs"
    assert metadata["dataset_id"] == "walnut"
    assert metadata["formal_performance_eligible"] is False
    ablation = json.loads(result.ablation_path.read_text(encoding="utf-8"))
    assert ablation["gpu_reference_status"] == "not_measured_cpu_only"
    assert all(value is None for value in ablation["gpu_base_speedups"].values())
    assert ablation["variant_order"] == [
        "0000", "1000", "1010", "0100", "0101", "1100", "1111",
    ]
    assert set(ablation["static_anchor_targets"]) == set(ablation["cycles"])
    assert ablation["target_assessment"]["1000"]["status"] == (
        "not_measured_cpu_only"
    )
    assert ablation["target_assessment"]["1010"]["observed_speedup"] is not None
    oracle = ablation["configured_oracle_diagnostics"]
    assert set(oracle) == {"query", "residency"}
    assert all(item["status"] == "diagnostic_only" for item in oracle.values())
    assert set(result.oracle_cycles) == {"query", "residency"}


def test_representative_archive_campaign_uses_packet_archive_contract(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    workspace = WorkspacePaths.discover(
        repository=repository, workspace=tmp_path / "workspace",
    ).ensure()
    _prepared_dataset(
        workspace.cache / "prepared" / "walnut" / "fixture", "walnut",
    )
    result = run_campaign(
        "exact_gs", "walnut", workspace=workspace, representative_archive=True,
    )
    archive_root = (
        workspace.traces / "exact_gs" / "walnut" / "cpu-packet-archive-v1"
    )
    reader = VirtualPacketArchiveReader(archive_root)
    reader.validate()
    assert reader.manifest["metadata"]["model_id"] == "exact_gs"
    assert reader.manifest["metadata"]["dataset_id"] == "walnut"
    assert result.trace_root.name == "cpu-archive-validation-v1"
    metadata = json.loads(
        (result.trace_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["capture_backend"] == (
        "deterministic_cpu_packet_archive_contract"
    )
    assert metadata["trace_sample"]["selection"] == (
        "single_iteration_median_physical_tile"
    )
    ablation = json.loads(result.ablation_path.read_text(encoding="utf-8"))
    assert ablation["result_scope"] == "quick_cpu_packet_archive_validation"
    assert ablation["formal_performance_eligible"] is False


def test_gr_campaign_uses_optimized_independent_cpu_reference_trace(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    workspace = WorkspacePaths.discover(
        repository=repository, workspace=tmp_path / "workspace",
    ).ensure()
    _prepared_dataset(
        workspace.cache / "prepared" / "walnut" / "fixture", "walnut",
    )

    result = run_campaign(
        "gr_gaussian", "walnut", workspace=workspace, iterations=2,
    )

    assert result.trace_root.name == "trace"
    assert result.trace_root.parent.name == "cpu-reference-2iter-v2"
    metadata = json.loads(
        (result.trace_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["capture_backend"] == "independent_cpu_ray_bundle"
    assert metadata["radiative_state_source"] == "optimized_reference_state"
    assert metadata["campaign_capture"] == "optimized_independent_cpu_reference"
    assert metadata["expanded_iterations"] == 2
    assert metadata["trace_window"]["iterations"] == [1, 2]
    ablation = json.loads(result.ablation_path.read_text(encoding="utf-8"))
    assert ablation["result_scope"] == "quick_cpu_reference_trace_validation"


def test_iteration_expansion_preserves_state_versions_and_cross_iteration_barrier() -> None:
    builder = TraceBuilder()
    relation = builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.RELATION), query_id=0,
        relation_id=0, gaussian_id=0, resource_class=int(ResourceClass.RELATION),
    ))
    close = builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0,
        resource_class=int(ResourceClass.RELATION),
    ), dependencies=[relation])
    begin = builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.UPDATE_BEGIN),
        resource_class=int(ResourceClass.UPDATE), flags=1, field_mask=1,
    ))
    end = builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.UPDATE_END),
        resource_class=int(ResourceClass.UPDATE), flags=1, field_mask=1,
        reduction_key=begin,
    ), dependencies=[begin])
    trace = builder.finish(metadata={
        "schema_version": "gala-clamp-events-v2",
        "initial_gaussian_count": 1,
    })
    expanded = _expand_trace_iterations(trace, 2)
    report = validate_trace(expanded)
    assert report.event_count == 2 * trace.event_count
    assert set(expanded.events["state_version"]) == {0, 1}
    second_close = expanded.events[trace.event_count + 1]
    assert int(second_close["primitive_kind"]) == int(PrimitiveKind.QUERY_CLOSE)
    deps = expanded.dependency_ids(second_close)
    assert int(end) in deps
