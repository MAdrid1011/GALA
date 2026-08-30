from __future__ import annotations

import pytest

from gala_sim.tools.gpu_compiler_ablation import (
    GpuCompilerProbeError,
    RUNNER_VARIANTS,
    summarize_samples,
)


def _sample(total_ms: float) -> dict[str, float | int]:
    return {
        "relation_generation_ms": total_ms - 10.0,
        "primitive_ms": 10.0,
        "total_ms": total_ms,
        "query_count": 1024,
        "active_gaussians": 50000,
        "candidate_pixels": 20000,
        "accepted_relations": 6000,
        "consumer_dequeues": 1024,
        "mean_l1_loss": 0.1,
        "position_gradient_l1": 1.0,
        "covariance_gradient_l1": 2.0,
        "opacity_gradient_l1": 3.0,
        "scale_latent_gradient_l1": 4.0,
        "quaternion_latent_gradient_l1": 0.0,
        "density_latent_gradient_l1": 5.0,
    }


def test_summary_uses_gpu_base_and_repeated_medians() -> None:
    totals = {
        "A0B0": (100.0, 101.0, 99.0),
        "A1B0": (80.0, 79.0, 81.0),
        "A0B1": (75.0, 76.0, 74.0),
        "A1B1": (65.0, 66.0, 64.0),
    }
    summary = summarize_samples({
        variant: [_sample(value) for value in totals[variant]]
        for variant in RUNNER_VARIANTS
    })
    assert summary["A1B0"]["speedup_vs_gpu_base"] == pytest.approx(1.25)
    assert summary["A0B1"]["speedup_vs_gpu_base"] == pytest.approx(4.0 / 3.0)
    assert summary["A1B1"]["speedup_vs_gpu_base"] == pytest.approx(100.0 / 65.0)
    assert summary["A1B0"]["speedup_vs_base_asic"] is None


def test_summary_rejects_changed_workload() -> None:
    samples = {variant: [_sample(100.0)] for variant in RUNNER_VARIANTS}
    samples["A0B1"][0]["accepted_relations"] = 5999
    with pytest.raises(GpuCompilerProbeError, match="accepted_relations"):
        summarize_samples(samples)


def test_summary_rejects_changed_numerics() -> None:
    samples = {variant: [_sample(100.0)] for variant in RUNNER_VARIANTS}
    samples["A1B1"][0]["mean_l1_loss"] = 0.2
    with pytest.raises(GpuCompilerProbeError, match="mean_l1_loss"):
        summarize_samples(samples)
