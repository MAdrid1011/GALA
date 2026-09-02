from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gala_sim.gpu_measurement import load_gpu_compiler_measurement
from gala_sim.tools.exact_gs_training_probe import (
    COMPILER_VARIANTS,
    ExactTrainingProbeError,
    OVERLAY_TRANSFORMS,
    ProbeObservation,
    render_training_overlay,
    run_matrix,
    summarize_matrix,
)
from gala_sim.gpu_coverage import GpuCoverageError, analyze_gpu_compiler_coverage


def _training_source() -> str:
    return "".join(item[1] for item in OVERLAY_TRANSFORMS)


def _observation(variant: str, elapsed_ms: float, value: float = 1.0) -> ProbeObservation:
    gradients = {"xyz": np.asarray([value, 2.0], dtype=np.float32)}
    return ProbeObservation({
        "workload": {
            "model": "Exact-GS",
            "model_id": "exact_gs",
            "dataset": "walnut",
            "dataset_id": "walnut",
            "compiler_variant": variant,
            "requested_iterations": 2,
            "warmup_iterations": 0,
        },
        "measurement": {"cuda_elapsed_ms": elapsed_ms},
        "gpu": {
            "isolation": {"status": "isolated"},
            "external_compute_processes": [],
        },
    }, gradients, gradients)


def test_exact_training_overlay_removes_observer_synchronization() -> None:
    transformed, manifest = render_training_overlay(_training_source())

    assert "torch.cuda.synchronize()" not in transformed
    assert ".item()" not in transformed
    assert [item["id"] for item in manifest] == [item[0] for item in OVERLAY_TRANSFORMS]


def test_exact_training_overlay_rejects_source_drift() -> None:
    source = _training_source().replace("        iter_start.record()\n", "", 1)
    with pytest.raises(ExactTrainingProbeError, match="expected one source fragment"):
        render_training_overlay(source)


def test_exact_matrix_uses_gpu_base_and_checks_gradients() -> None:
    observations = {
        "gpu_base": [_observation("gpu_base", 100.0)],
        "1000": [_observation("1000", 75.0)],
        "0100": [_observation("0100", 70.0)],
        "1100": [_observation("1100", 60.0)],
    }

    result = summarize_matrix(
        observations,
        relative_tolerance=1.0e-4,
        absolute_tolerance=1.0e-6,
    )

    assert tuple(result["summary"]) == COMPILER_VARIANTS
    assert result["summary"]["1000"]["speedup_vs_gpu_base"] == pytest.approx(
        4.0 / 3.0,
    )
    assert result["summary"]["0100"]["speedup_vs_gpu_base"] == pytest.approx(
        10.0 / 7.0,
    )
    assert result["performance_comparison_eligible"] is True


def test_exact_matrix_rejects_changed_gradient() -> None:
    observations = {
        "gpu_base": [_observation("gpu_base", 100.0)],
        "1000": [_observation("1000", 75.0)],
        "0100": [_observation("0100", 70.0, 1.2)],
        "1100": [_observation("1100", 60.0)],
    }
    with pytest.raises(ExactTrainingProbeError, match="gradients beyond tolerance"):
        summarize_matrix(
            observations,
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-6,
        )


def test_exact_matrix_writes_campaign_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.exact_gs_training_probe as probe

    elapsed = {
        "gpu_base": 100.0,
        "1000": 75.0,
        "0100": 70.0,
        "1100": 60.0,
    }

    def fake_probe(**kwargs):
        assert kwargs["dataset_id"] == "walnut"
        return _observation(kwargs["compiler_variant"], elapsed[kwargs["compiler_variant"]])

    monkeypatch.setattr(probe, "run_probe", fake_probe)
    output = tmp_path / "exact_gs" / "walnut"
    run_matrix(
        source_root=tmp_path,
        extension_root=tmp_path,
        dataset_root=tmp_path,
        initial_state=tmp_path / "initial.ply",
        dataset_id="walnut",
        output=output,
        requested_iterations=2,
        warmup_iterations=0,
        repeats=1,
        inactivity_timeout_seconds=300.0,
        relative_tolerance=1.0e-4,
        absolute_tolerance=1.0e-6,
    )

    document = json.loads(
        (output / "gpu-compiler-measurement.json").read_text(encoding="utf-8")
    )
    assert document["model_id"] == "exact_gs"
    assert document["dataset_id"] == "walnut"
    loaded = load_gpu_compiler_measurement(
        output / "gpu-compiler-measurement.json",
        model_id="exact_gs",
        dataset_id="walnut",
        iteration_range=(1, 2),
    )
    assert loaded.speedups_vs_gpu_base["1100"] == pytest.approx(5.0 / 3.0)


def test_gpu_compiler_coverage_reports_necessary_amdahl_bounds() -> None:
    result = analyze_gpu_compiler_coverage(
        total_gpu_ms=100.0,
        stage_profile={"summaries": {"backward": {"total_ms": 40.0}}},
        coverable_stages={
            "1000": ("backward",),
            "0100": ("backward",),
            "1100": ("backward",),
        },
    )

    query = result["bounds"]["1000"]
    assert query["minimum_uncovered_ms"] == pytest.approx(60.0)
    assert query["maximum_possible_speedup_vs_gpu_base"] == pytest.approx(5.0 / 3.0)
    assert query["status"] == "not_ruled_out_by_necessary_stage_bound"
    assert result["bounds"]["1100"]["target_speedup_vs_gpu_base"] == pytest.approx(
        1.482
    )


def test_gpu_compiler_coverage_rejects_missing_or_overlapping_stage_time() -> None:
    stages = {"1000": ("backward",), "0100": ("backward",), "1100": ("backward",)}
    with pytest.raises(GpuCoverageError, match="missing"):
        analyze_gpu_compiler_coverage(
            total_gpu_ms=100.0,
            stage_profile={"summaries": {}},
            coverable_stages=stages,
        )
    with pytest.raises(GpuCoverageError, match="exceeds"):
        analyze_gpu_compiler_coverage(
            total_gpu_ms=100.0,
            stage_profile={"summaries": {"backward": {"total_ms": 101.0}}},
            coverable_stages=stages,
        )


def test_exact_matrix_keeps_stage_profile_out_of_four_way_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.exact_gs_training_probe as probe

    calls: list[tuple[str, bool]] = []

    def fake_probe(**kwargs):
        variant = kwargs["compiler_variant"]
        profiled = kwargs["profile_stages"]
        calls.append((variant, profiled))
        observation = _observation(variant, 100.0)
        if profiled:
            observation.record["measurement"]["stage_profile"] = {
                "summaries": {"backward": {"total_ms": 40.0}}
            }
        return observation

    monkeypatch.setattr(probe, "run_probe", fake_probe)
    output = tmp_path / "profiled-matrix"
    result = run_matrix(
        source_root=tmp_path,
        extension_root=tmp_path,
        dataset_root=tmp_path,
        initial_state=tmp_path / "initial.ply",
        dataset_id="walnut",
        output=output,
        requested_iterations=2,
        warmup_iterations=0,
        repeats=1,
        inactivity_timeout_seconds=300.0,
        relative_tolerance=1.0e-4,
        absolute_tolerance=1.0e-6,
        profile_stages=True,
    )

    assert calls == [
        ("gpu_base", False),
        ("1000", False),
        ("0100", False),
        ("1100", False),
        ("gpu_base", True),
    ]
    assert result["compiler_coverage_bounds"]["bounds"]["1000"][
        "maximum_possible_speedup_vs_gpu_base"
    ] == pytest.approx(5.0 / 3.0)
    written = json.loads((output / "matrix.json").read_text(encoding="utf-8"))
    assert written["compiler_coverage_bounds"] == result["compiler_coverage_bounds"]
