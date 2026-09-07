from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gala_sim.tools.fact_gs_training_probe import (
    BACKWARD_PATHS,
    COMPILER_VARIANTS,
    FactTrainingProbeError,
    ProbeObservation,
    _semantic_hot_fraction,
    compare_states,
    run_matrix,
    summarize_matrix,
)
from gala_sim.gpu_measurement import load_gpu_compiler_measurement


def _observation(variant: str, elapsed_ms: float, value: float = 1.0) -> ProbeObservation:
    state = {"xyz": np.asarray([value, 2.0], dtype=np.float32)}
    return ProbeObservation({
        "workload": {
            "model": "FaCT-GS",
            "dataset": "Walnut",
            "compiler_variant": variant,
            "requested_iterations": 2,
        },
        "measurement": {"cuda_elapsed_ms": elapsed_ms},
        "gpu": {
            "isolation": {"status": "isolated"},
            "external_compute_processes": [],
        },
    }, state, state)


def test_compiler_variants_independently_select_the_two_official_paths() -> None:
    assert COMPILER_VARIANTS == ("gpu_base", "1000", "0100", "1100")
    assert BACKWARD_PATHS["gpu_base"].projection_per_gaussian is False
    assert BACKWARD_PATHS["gpu_base"].volume_per_gaussian is False
    assert BACKWARD_PATHS["1000"].projection_per_gaussian is True
    assert BACKWARD_PATHS["1000"].volume_per_gaussian is False
    assert BACKWARD_PATHS["0100"].projection_per_gaussian is False
    assert BACKWARD_PATHS["0100"].volume_per_gaussian is True
    assert BACKWARD_PATHS["0100"].semantic_hot_relation_fraction == pytest.approx(0.40)
    assert BACKWARD_PATHS["1100"].projection_per_gaussian is True
    assert BACKWARD_PATHS["1100"].volume_per_gaussian is True


def test_semantic_hot_fraction_override_is_scoped_and_validated(monkeypatch) -> None:
    selection = BACKWARD_PATHS["0100"]
    assert _semantic_hot_fraction(selection) == pytest.approx(0.40)
    monkeypatch.setenv("GALA_FACT_SEMANTIC_HOT_FRACTION", "0.25")
    assert _semantic_hot_fraction(selection) == pytest.approx(0.25)
    monkeypatch.setenv("GALA_FACT_SEMANTIC_HOT_FRACTION", "1.5")
    with pytest.raises(FactTrainingProbeError, match=r"in \(0, 1\]"):
        _semantic_hot_fraction(selection)
    assert _semantic_hot_fraction(BACKWARD_PATHS["1000"]) is None


def test_state_comparison_reports_defined_numeric_tolerance() -> None:
    report = compare_states(
        {"state": np.asarray([1.0, 2.0], dtype=np.float32)},
        {"state": np.asarray([1.0001, 2.0], dtype=np.float32)},
        relative_tolerance=1.0e-4,
        absolute_tolerance=0.0,
    )
    assert report["passed"]
    assert report["fields"]["state"]["maximum_absolute_error"] > 0.0


def test_matrix_uses_gpu_base_for_all_compiler_speedups() -> None:
    observations = {
        "gpu_base": [_observation("gpu_base", 100.0)],
        "1000": [_observation("1000", 75.0)],
        "0100": [_observation("0100", 70.0)],
        "1100": [_observation("1100", 60.0)],
    }
    result = summarize_matrix(
        observations, relative_tolerance=1.0e-4, absolute_tolerance=1.0e-6,
    )
    assert result["summary"]["1000"]["speedup_vs_gpu_base"] == pytest.approx(4.0 / 3.0)
    assert result["summary"]["0100"]["speedup_vs_gpu_base"] == pytest.approx(10.0 / 7.0)
    assert result["summary"]["1100"]["target_met"] is True
    assert result["performance_comparison_eligible"] is True


def test_matrix_marks_measurements_without_gpu_isolation_as_ineligible() -> None:
    observations = {
        variant: [_observation(variant, 100.0)]
        for variant in COMPILER_VARIANTS
    }
    observations["0100"][0].record["gpu"] = {}
    result = summarize_matrix(
        observations, relative_tolerance=1.0e-4, absolute_tolerance=1.0e-6,
    )
    assert result["performance_comparison_eligible"] is False


def test_matrix_rejects_numerically_changed_variant() -> None:
    observations = {
        "gpu_base": [_observation("gpu_base", 100.0)],
        "1000": [_observation("1000", 75.0, 1.2)],
        "0100": [_observation("0100", 70.0)],
        "1100": [_observation("1100", 60.0)],
    }
    with pytest.raises(FactTrainingProbeError, match="pre-optimizer gradients"):
        summarize_matrix(
            observations, relative_tolerance=1.0e-4, absolute_tolerance=1.0e-6,
        )


def test_matrix_writes_campaign_measurement_with_dataset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.fact_gs_training_probe as probe

    def fake_probe(**kwargs):
        variant = kwargs["compiler_variant"]
        assert kwargs["dataset_id"] == "hdtomo_usb"
        observation = _observation(variant, {
            "gpu_base": 100.0,
            "1000": 75.0,
            "0100": 70.0,
            "1100": 60.0,
        }[variant])
        observation.record["workload"].update({
            "model_id": "fact_gs",
            "dataset": "hdtomo_usb",
            "dataset_id": "hdtomo_usb",
            "warmup_iterations": 0,
            "measured_iterations": 1,
            "included_training_operations": ["forward", "backward", "optimizer_step"],
        })
        return observation

    monkeypatch.setattr(probe, "run_probe", fake_probe)
    output = tmp_path / "fact_gs" / "hdtomo_usb"
    run_matrix(
        source_root=tmp_path,
        dataset_root=tmp_path,
        dataset_id="hdtomo_usb",
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
    assert document["model_id"] == "fact_gs"
    assert document["dataset_id"] == "hdtomo_usb"
    loaded = load_gpu_compiler_measurement(
        output / "gpu-compiler-measurement.json",
        model_id="fact_gs",
        dataset_id="hdtomo_usb",
        iteration_range=(1, 1),
    )
    assert loaded.speedups_vs_gpu_base["1100"] == pytest.approx(5.0 / 3.0)


def test_fact_matrix_keeps_stage_profile_out_of_four_way_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.fact_gs_training_probe as probe

    calls: list[tuple[str, bool]] = []

    def fake_probe(**kwargs):
        variant = kwargs["compiler_variant"]
        profiled = kwargs["profile_stages"]
        calls.append((variant, profiled))
        observation = _observation(variant, 100.0)
        observation.record["workload"].update({
            "model_id": "fact_gs",
            "dataset_id": "walnut",
            "warmup_iterations": 0,
            "measured_iterations": 2,
            "included_training_operations": ["forward", "backward", "optimizer"],
        })
        if profiled:
            observation.record["measurement"]["stage_profile"] = {
                "summaries": {"backward": {"total_ms": 40.0}}
            }
        return observation

    monkeypatch.setattr(probe, "run_probe", fake_probe)
    output = tmp_path / "profiled-matrix"
    result = run_matrix(
        source_root=tmp_path,
        dataset_root=tmp_path,
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
    assert result["compiler_coverage_bounds"]["bounds"]["0100"][
        "maximum_possible_speedup_vs_gpu_base"
    ] == pytest.approx(5.0 / 3.0)
