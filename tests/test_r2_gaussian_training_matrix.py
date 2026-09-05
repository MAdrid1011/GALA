from __future__ import annotations

import json
from pathlib import Path

import pytest

from gala_sim.tools.r2_gaussian_training_matrix import (
    COMPILER_VARIANTS,
    _resolve_extension_root,
    run_matrix,
    summarize_matrix,
)
from gala_sim.gpu_measurement import load_gpu_compiler_measurement
from gala_sim.tools.r2_gaussian_training_probe import TrainingProbeError


def _record(variant: str, elapsed_ms: float, *, state_sum: float = 3.0) -> dict:
    return {
        "workload": {
            "model": "R2-Gaussian",
            "model_id": "r2_gaussian",
            "dataset": "walnut",
            "dataset_id": "walnut",
            "compiler_variant": variant,
            "requested_iterations": 2,
            "warmup_iterations": 0,
        },
        "measurement": {
            "cuda_elapsed_ms": elapsed_ms,
            "final_losses": {"loss_total": 0.25},
            "final_state": {
                "fields": {
                    "xyz": {
                        "shape": [1, 3],
                        "dtype": "float32",
                        "sum": state_sum,
                        "absolute_sum": abs(state_sum),
                    },
                },
            },
        },
        "gpu": {
            "isolation": {"status": "isolated"},
            "external_compute_processes": [],
        },
    }


def test_r2_matrix_discovers_isolated_overlay_for_official_checkout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "workspace" / "upstream" / "r2_gaussian"
    source_root.mkdir(parents=True)
    (source_root / "train.py").write_text("# fixture\n", encoding="utf-8")
    overlay = (
        tmp_path / "workspace" / "build" / "r2-gaussian-compiler-overlay-v1"
        / "xray_gaussian_rasterization_voxelization"
    )
    overlay.mkdir(parents=True)
    (overlay / "_C.fixture.so").write_bytes(b"fixture")

    assert _resolve_extension_root(source_root, None) == overlay.parent


def test_r2_matrix_rejects_official_checkout_without_overlay(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "workspace" / "upstream" / "r2_gaussian"
    source_root.mkdir(parents=True)
    (source_root / "train.py").write_text("# fixture\n", encoding="utf-8")

    with pytest.raises(TrainingProbeError, match="isolated CUDA overlay"):
        _resolve_extension_root(source_root, None)


def test_r2_matrix_honors_explicit_overlay_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "workspace" / "upstream" / "r2_gaussian"
    source_root.mkdir(parents=True)
    explicit = tmp_path / "external-overlay"
    explicit.mkdir()
    monkeypatch.setenv("GALA_R2_EXTENSION_ROOT", str(explicit))

    assert _resolve_extension_root(source_root, None) == explicit


def test_r2_matrix_uses_gpu_base_and_checks_numerics() -> None:
    records = {
        "gpu_base": [_record("gpu_base", 100.0)],
        "1000": [_record("1000", 75.0)],
        "0100": [_record("0100", 70.0)],
        "1100": [_record("1100", 60.0)],
    }

    result = summarize_matrix(
        records,
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


def test_r2_matrix_rejects_changed_state() -> None:
    records = {
        variant: [_record(variant, 100.0, state_sum=4.0 if variant == "0100" else 3.0)]
        for variant in COMPILER_VARIANTS
    }

    with pytest.raises(TrainingProbeError, match="final state beyond tolerance"):
        summarize_matrix(
            records,
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-6,
        )


def test_r2_matrix_rejects_changed_workload() -> None:
    records = {
        variant: [_record(variant, 100.0)]
        for variant in COMPILER_VARIANTS
    }
    records["1100"][0]["workload"]["dataset_id"] = "chest"

    with pytest.raises(TrainingProbeError, match="comparable official workload"):
        summarize_matrix(
            records,
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-6,
        )


def test_r2_matrix_writes_campaign_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.r2_gaussian_training_matrix as matrix

    def fake_probe(**kwargs):
        variant = kwargs["compiler_variant"]
        assert kwargs["dataset_id"] == "walnut"
        record = _record(variant, {
            "gpu_base": 100.0,
            "1000": 75.0,
            "0100": 70.0,
            "1100": 60.0,
        }[variant])
        record["workload"].update({
            "warmup_iterations": 0,
            "measured_iterations": 1,
            "included_training_operations": ["forward", "backward", "optimizer_step"],
        })
        record["gpu"] = {
            "isolation": {"status": "isolated"},
            "external_compute_processes": [],
        }
        return record

    monkeypatch.setattr(matrix, "run_probe", fake_probe)
    output = tmp_path / "r2_gaussian" / "walnut"
    run_matrix(
        source_root=tmp_path,
        extension_root=None,
        dataset_root=tmp_path,
        initial_state=tmp_path / "init.npy",
        dataset_id="walnut",
        output=output,
        requested_iterations=1,
        warmup_iterations=0,
        repeats=1,
        progress_interval=1,
        inactivity_timeout_seconds=300.0,
        relative_tolerance=1.0e-4,
        absolute_tolerance=1.0e-6,
    )

    document = json.loads(
        (output / "gpu-compiler-measurement.json").read_text(encoding="utf-8")
    )
    assert document["model_id"] == "r2_gaussian"
    assert document["dataset_id"] == "walnut"
    loaded = load_gpu_compiler_measurement(
        output / "gpu-compiler-measurement.json",
        model_id="r2_gaussian",
        dataset_id="walnut",
        iteration_range=(1, 1),
    )
    assert loaded.speedups_vs_gpu_base["1000"] == pytest.approx(4.0 / 3.0)


def test_r2_matrix_keeps_stage_profile_out_of_four_way_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.r2_gaussian_training_matrix as matrix

    calls: list[tuple[str, bool]] = []

    def fake_probe(**kwargs):
        variant = kwargs["compiler_variant"]
        profiled = kwargs["profile_stages"]
        calls.append((variant, profiled))
        record = _record(variant, 100.0)
        record["workload"].update({
            "measured_iterations": 2,
            "included_training_operations": ["forward", "backward", "optimizer"],
        })
        if profiled:
            record["measurement"]["stage_profile"] = {
                "stage_summaries": {"backward": {"total_ms": 40.0}}
            }
        return record

    monkeypatch.setattr(matrix, "run_probe", fake_probe)
    output = tmp_path / "profiled-matrix"
    result = run_matrix(
        source_root=tmp_path,
        extension_root=None,
        dataset_root=tmp_path,
        initial_state=tmp_path / "init.npy",
        dataset_id="walnut",
        output=output,
        requested_iterations=2,
        warmup_iterations=0,
        repeats=1,
        progress_interval=1,
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
    assert result["compiler_coverage_bounds"]["bounds"]["1100"][
        "maximum_possible_speedup_vs_gpu_base"
    ] == pytest.approx(5.0 / 3.0)
