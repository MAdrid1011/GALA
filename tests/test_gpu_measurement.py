from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gala_sim.campaign import _trace_for_campaign, prepared_dataset, run_campaign
from gala_sim.gpu_measurement import (
    GPU_COMPILER_VARIANTS,
    GpuMeasurementError,
    build_gpu_compiler_measurement,
    default_gpu_measurement_path,
    load_gpu_compiler_measurement,
)
from gala_sim.workspace import WorkspacePaths


def _document() -> dict:
    return build_gpu_compiler_measurement(
        model_id="exact_gs",
        dataset_id="walnut",
        iteration_range=(1, 2),
        included_training_operations=["forward", "backward", "optimizer_step"],
        median_gpu_ms={
            "gpu_base": 100.0,
            "1000": 75.0,
            "0100": 80.0,
            "1100": 60.0,
        },
        sample_counts={bits: 3 for bits in GPU_COMPILER_VARIANTS},
        source_probe={"schema_version": "fixture-probe-v1"},
        gpu_isolated=True,
        same_workload_across_variants=True,
        numerical_equivalence_passed=True,
        gpu_platform={"gpu_name": "NVIDIA Test GPU", "gpu_uuid": "GPU-test"},
    )


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _prepared_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    projection_root = root / "projections"
    projection_root.mkdir()
    np.save(projection_root / "000.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
    np.save(root / "init.npy", np.asarray([
        [-1.0, -1.0, -1.0, 0.2],
        [0.0, 0.1, 0.2, 0.4],
        [1.0, 1.0, 1.0, 0.6],
    ], dtype=np.float32))
    (root / "metadata.json").write_text(json.dumps({
        "angles_radians": [0.0],
        "detector_shape": [4, 4],
        "volume_shape": [4, 4, 4],
        "DSO": 5.0,
        "DSD": 7.0,
        "initialization": "init.npy",
        "train_indices": [0],
        "test_indices": [],
    }), encoding="utf-8")


def test_gpu_measurement_derives_compiler_speedups_from_gpu_base(tmp_path: Path) -> None:
    path = tmp_path / "measurement.json"
    _write(path, _document())

    measurement = load_gpu_compiler_measurement(
        path,
        model_id="exact_gs",
        dataset_id="walnut",
        iteration_range=(1, 2),
    )

    assert measurement.speedups_vs_gpu_base == {
        "1000": pytest.approx(4.0 / 3.0),
        "0100": pytest.approx(1.25),
        "1100": pytest.approx(5.0 / 3.0),
    }
    assert measurement.gpu_platform == {
        "gpu_name": "NVIDIA Test GPU", "gpu_uuid": "GPU-test",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(model_id="fact_gs"), "model identity"),
        (lambda value: value.update(dataset_id="chest"), "dataset identity"),
        (
            lambda value: value["workload"].update(iteration_range=[2, 3]),
            "iteration range",
        ),
        (
            lambda value: value.update(trace_instrumentation_enabled=True),
            "trace-instrumented",
        ),
        (
            lambda value: value["comparability"].update(gpu_isolated=False),
            "comparability",
        ),
        (
            lambda value: value["variants"]["1000"].update(
                speedup_vs_gpu_base=99.0,
            ),
            "inconsistent speedup",
        ),
        (lambda value: value.update(gpu_platform={"gpu_name": ""}), "platform name"),
    ],
)
def test_gpu_measurement_rejects_noncomparable_evidence(
    tmp_path: Path, mutation, message: str,
) -> None:
    document = _document()
    mutation(document)
    path = tmp_path / "measurement.json"
    _write(path, document)

    with pytest.raises(GpuMeasurementError, match=message):
        load_gpu_compiler_measurement(
            path,
            model_id="exact_gs",
            dataset_id="walnut",
            iteration_range=(1, 2),
        )


def test_official_campaign_ingests_only_uninstrumented_gpu_measurement(
    tmp_path: Path, monkeypatch,
) -> None:
    import gala_sim.campaign as campaign

    repository = Path(__file__).resolve().parents[1]
    paths = WorkspacePaths.discover(
        repository=repository,
        workspace=tmp_path / "workspace",
    ).ensure()
    _prepared_dataset(paths.cache / "prepared" / "walnut" / "fixture")
    dataset = prepared_dataset(paths, "walnut")
    trace_root, trace = _trace_for_campaign(
        paths, "exact_gs", "walnut", dataset, iterations=2,
    )
    trace.metadata.update({
        "official_model_trace": True,
        "capture_iteration_range": [1, 2],
    })
    monkeypatch.setattr(
        campaign,
        "_trace_for_official_campaign",
        lambda *_args, **_kwargs: (
            trace_root,
            trace,
            {"wall_seconds": 999.0, "trace_instrumentation_enabled": True},
        ),
    )
    measurement_path = default_gpu_measurement_path(
        paths.results / "gpu-measurements", "exact_gs", "walnut",
    )
    document = _document()
    _write(measurement_path, document)

    result = run_campaign(
        "exact_gs",
        "walnut",
        workspace=paths,
        official_trace=True,
        capture_iteration_range=(1, 2),
    )

    ablation = json.loads(result.ablation_path.read_text(encoding="utf-8"))
    assert result.gpu_base_speedups["1000"] == pytest.approx(4.0 / 3.0)
    assert ablation["gpu_reference_status"] == (
        "measured_uninstrumented_official_training"
    )
    assert ablation["target_assessment"]["1000"]["status"] == (
        "target_met_gpu_measurement"
    )
    assert ablation["target_assessment"]["0100"]["status"] == (
        "below_target_gpu_measurement"
    )
    assert ablation["gpu_compiler_measurement"]["trace_instrumentation_enabled"] is False
    bounds = json.loads(result.bounds_path.read_text(encoding="utf-8"))
    assert bounds["gpu_compiler_bounds"]["1000"]["observed_speedup"] == pytest.approx(
        4.0 / 3.0,
    )


def test_explicit_gpu_measurement_root_requires_every_selected_artifact(
    tmp_path: Path, monkeypatch,
) -> None:
    import gala_sim.campaign as campaign

    repository = Path(__file__).resolve().parents[1]
    paths = WorkspacePaths.discover(
        repository=repository,
        workspace=tmp_path / "workspace",
    ).ensure()
    _prepared_dataset(paths.cache / "prepared" / "walnut" / "fixture")
    dataset = prepared_dataset(paths, "walnut")
    trace_root, trace = _trace_for_campaign(
        paths, "exact_gs", "walnut", dataset,
    )
    monkeypatch.setattr(
        campaign,
        "_trace_for_official_campaign",
        lambda *_args, **_kwargs: (trace_root, trace, {"wall_seconds": 1.0}),
    )

    with pytest.raises(ValueError, match="measurement is missing"):
        run_campaign(
            "exact_gs",
            "walnut",
            workspace=paths,
            official_trace=True,
            gpu_measurement_root=tmp_path / "measurements",
        )
