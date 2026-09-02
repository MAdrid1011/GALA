from __future__ import annotations

import pytest

from gala_sim.ablation.anchors import (
    SUPPORTED_DATASETS, SUPPORTED_MODELS, assess_upper_bound,
    hardware_target_speedups, static_anchors,
)


def test_all_twelve_campaigns_use_the_seven_baseline_correct_targets() -> None:
    for model_id in SUPPORTED_MODELS:
        for dataset_id in SUPPORTED_DATASETS:
            anchors = static_anchors(model_id, dataset_id)
            assert tuple(anchors) == (
                "0000", "1000", "1010", "0100", "0101", "1100", "1111",
            )
            assert anchors["0000"].comparison_baseline == "agx_orin_gpu_base_estimate"
            for bits in ("1000", "0100", "1100"):
                assert anchors[bits].comparison_baseline == "gpu_base"
            for bits in ("1010", "0101", "1111"):
                assert anchors[bits].comparison_baseline == "base_asic"


def test_hardware_targets_are_derived_from_matching_static_endpoints() -> None:
    targets = hardware_target_speedups()
    assert targets["query"] == pytest.approx(3.513 / 2.430)
    assert targets["residency"] == pytest.approx(3.507 / 2.430)
    assert targets["full"] == pytest.approx(6.436 / 2.430)


def test_upper_bound_gate_identifies_uncovered_engineering_work() -> None:
    observed = {
        bits: anchor.target_speedup + 0.01
        for bits, anchor in static_anchors("exact_gs", "chest").items()
    }
    observed["0101"] = 1.0
    results = assess_upper_bound("exact_gs", "chest", observed)
    statuses = {item.bits: item.status for item in results}
    assert statuses["0101"] == "engineering_optimization_required"
    assert all(
        status == "target_reachable"
        for bits, status in statuses.items() if bits != "0101"
    )


def test_upper_bound_gate_rejects_incomplete_or_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="all seven"):
        assess_upper_bound("exact_gs", "chest", {"0000": 2.5})
    observed = {
        bits: anchor.target_speedup
        for bits, anchor in static_anchors("exact_gs", "chest").items()
    }
    observed["1111"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        assess_upper_bound("exact_gs", "chest", observed)
