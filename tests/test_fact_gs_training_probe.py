from __future__ import annotations

import numpy as np
import pytest

from gala_sim.tools.fact_gs_training_probe import (
    BACKWARD_PATHS,
    COMPILER_VARIANTS,
    FactTrainingProbeError,
    ProbeObservation,
    compare_states,
    summarize_matrix,
)


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
