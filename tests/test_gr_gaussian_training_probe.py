from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gala_sim.gpu_measurement import load_gpu_compiler_measurement
from gala_sim.tools.gr_gaussian_training_probe import (
    COMPILER_VARIANTS,
    GRTrainingProbeError,
    ProbeObservation,
    VARIANT_PATHS,
    run_matrix,
    summarize_matrix,
)


def _observation(
    variant: str,
    elapsed_ms: float,
    densities: np.ndarray | None = None,
) -> ProbeObservation:
    return ProbeObservation({
        "workload": {
            "model": "GR-Gaussian",
            "model_id": "gr_gaussian",
            "dataset": "walnut",
            "dataset_id": "walnut",
            "compiler_variant": variant,
            "requested_iterations": 10,
            "warmup_iterations": 2,
            "included_training_operations": ["radiative_splat", "density_gradient"],
        },
        "measurement": {"cuda_elapsed_ms": elapsed_ms},
        "gpu": {
            "isolation": {"status": "isolated"},
            "external_compute_processes": [],
        },
    }, np.asarray(
        [0.1, 0.2, 0.3] if densities is None else densities,
        dtype=np.float64,
    ))


def test_gr_compiler_paths_are_orthogonal() -> None:
    assert VARIANT_PATHS == {
        "gpu_base": (False, False),
        "1000": (True, False),
        "0100": (False, True),
        "1100": (True, True),
    }


def test_gr_matrix_uses_gpu_base_and_accepts_equivalent_states() -> None:
    observations = {
        "gpu_base": [_observation("gpu_base", 100.0)],
        "1000": [_observation("1000", 75.0)],
        "0100": [_observation("0100", 70.0)],
        "1100": [_observation("1100", 60.0)],
    }

    result = summarize_matrix(
        observations,
        relative_tolerance=1.0e-5,
        absolute_tolerance=1.0e-8,
    )

    assert tuple(result["summary"]) == COMPILER_VARIANTS
    assert result["summary"]["1000"]["speedup_vs_gpu_base"] == pytest.approx(
        4.0 / 3.0,
    )
    assert result["summary"]["0100"]["speedup_vs_gpu_base"] == pytest.approx(
        10.0 / 7.0,
    )
    assert result["performance_comparison_eligible"] is True


def test_gr_matrix_rejects_numerically_changed_variant() -> None:
    observations = {
        variant: [_observation(
            variant,
            100.0,
            np.asarray([0.1, 0.2, 0.4] if variant == "0100" else [0.1, 0.2, 0.3]),
        )]
        for variant in COMPILER_VARIANTS
    }

    with pytest.raises(GRTrainingProbeError, match="final densities beyond tolerance"):
        summarize_matrix(
            observations,
            relative_tolerance=1.0e-5,
            absolute_tolerance=1.0e-8,
        )


def test_gr_matrix_rejects_changed_dataset_identity() -> None:
    observations = {
        variant: [_observation(variant, 100.0)]
        for variant in COMPILER_VARIANTS
    }
    observations["1100"][0].record["workload"]["dataset_id"] = "chest"

    with pytest.raises(GRTrainingProbeError, match="comparable workload"):
        summarize_matrix(
            observations,
            relative_tolerance=1.0e-5,
            absolute_tolerance=1.0e-8,
        )


def test_gr_matrix_writes_campaign_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gala_sim.tools.gr_gaussian_training_probe as probe

    elapsed = {
        "gpu_base": 100.0,
        "1000": 75.0,
        "0100": 70.0,
        "1100": 60.0,
    }

    def fake_probe(**kwargs):
        assert kwargs["dataset_id"] == "hdtomo_usb"
        result = _observation(
            kwargs["compiler_variant"], elapsed[kwargs["compiler_variant"]],
        )
        result.record["workload"]["dataset"] = "hdtomo_usb"
        result.record["workload"]["dataset_id"] = "hdtomo_usb"
        return result

    monkeypatch.setattr(probe, "run_probe", fake_probe)
    output = tmp_path / "gr_gaussian" / "hdtomo_usb"
    run_matrix(
        bundle=tmp_path / "bundle.npz",
        dataset_id="hdtomo_usb",
        output=output,
        requested_iterations=10,
        warmup_iterations=2,
        repeats=1,
        learning_rate=0.05,
        graph_weight=0.01,
        inactivity_timeout_seconds=300.0,
        relative_tolerance=1.0e-5,
        absolute_tolerance=1.0e-8,
    )

    document = json.loads(
        (output / "gpu-compiler-measurement.json").read_text(encoding="utf-8")
    )
    assert document["model_id"] == "gr_gaussian"
    assert document["dataset_id"] == "hdtomo_usb"
    loaded = load_gpu_compiler_measurement(
        output / "gpu-compiler-measurement.json",
        model_id="gr_gaussian",
        dataset_id="hdtomo_usb",
        iteration_range=(3, 10),
    )
    assert loaded.speedups_vs_gpu_base["1100"] == pytest.approx(5.0 / 3.0)
