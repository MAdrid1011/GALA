from __future__ import annotations

import pytest

from gala_sim.tools.gpu_orin_estimate import OrinEstimateError, estimate_normalized


def _reference() -> dict:
    categories = (
        "fp32_fma", "exp", "log", "rcp", "sqrt", "memory_bandwidth",
        "atomic", "kernel_launch", "synchronization",
    )
    return {
        "method": "test-proxy",
        "local": {
            "name": "local",
            "fp32_cores": 10,
            "sm_clock_hz": 10,
            "memory_bandwidth_bytes_per_second": 100,
            "source": "test",
        },
        "target": {
            "name": "target",
            "fp32_cores": 5,
            "sm_clock_hz": 10,
            "memory_bandwidth_bytes_per_second": 50,
            "source": "test",
        },
        "uncertainty": {category: 0.1 for category in categories},
    }


def test_proxy_estimate_preserves_stage_weights_and_nonformal_scope() -> None:
    result = estimate_normalized(
        {
            "local_sampling_performance_eligible": True,
            "stages": {
                "forward": {
                    "local_ms": 10,
                    "weights": {"fp32_fma": 0.5, "memory_bandwidth": 0.5},
                },
                "missing": {"local_ms": None, "weights": {}},
            },
        },
        _reference(),
    )

    assert result["status"] == "proxy_estimate"
    assert result["formal_performance_eligible"] is False
    assert result["local_sampling_performance_eligible"] is True
    assert result["stages"]["forward"]["estimated_orin_ms"] == pytest.approx(20.0)
    assert result["stages"]["missing"]["status"] == "unavailable"
    assert result["total"]["estimated_orin_ms"] == pytest.approx(20.0)


def test_proxy_rejects_non_unit_stage_weights() -> None:
    with pytest.raises(OrinEstimateError, match="weights do not sum to one"):
        estimate_normalized(
            {"stages": {"forward": {"local_ms": 1, "weights": {"fp32_fma": 0.9}}}},
            _reference(),
        )
