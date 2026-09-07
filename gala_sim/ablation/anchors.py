"""Static design targets and baseline-aware reachability gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .matrix import comparison_baseline


SUPPORTED_MODELS = frozenset({
    "r2_gaussian", "fact_gs", "exact_gs", "gr_gaussian",
})
SUPPORTED_DATASETS = frozenset({"chest", "walnut", "hdtomo_usb"})

# The endpoint estimates were frozen during the R2+Chest calibration.  Keep
# them in a workload-indexed table even when two workloads currently share the
# same estimate: this prevents a result from one trace being silently reused
# for another combination and gives later calibration a stable extension point.
STATIC_ANCHOR_VERSION = "legacy-static-endpoints-v1"
ANCHOR_ACCEPTABLE_MIN_FRACTION = 0.90
_STATIC_ENDPOINTS = {
    (model_id, dataset_id): {
        "software": {
            "1000": 1.254,
            "0100": 1.282,
            "1100": 1.482,
        },
        "hardware": {
            "0000": 2.430,
            "1010": 3.513,
            "0101": 3.507,
            "1111": 6.436,
        },
    }
    for model_id in SUPPORTED_MODELS
    for dataset_id in SUPPORTED_DATASETS
}


@dataclass(frozen=True)
class StaticAnchor:
    bits: str
    comparison_baseline: str
    target_speedup: float
    decomposition: str


@dataclass(frozen=True)
class AnchorGateResult:
    bits: str
    comparison_baseline: str
    target_speedup: float
    observed_upper_bound_speedup: float
    margin: float
    status: str


def static_anchor_matrix() -> Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]]:
    """Return the immutable per-workload endpoint estimate table."""

    return {
        key: {
            "software": dict(value["software"]),
            "hardware": dict(value["hardware"]),
        }
        for key, value in _STATIC_ENDPOINTS.items()
    }


def static_anchors(model_id: str, dataset_id: str) -> Mapping[str, StaticAnchor]:
    """Return the baseline-correct design targets for one supported campaign."""

    if model_id not in SUPPORTED_MODELS:
        raise KeyError(f"unsupported anchor model: {model_id}")
    if dataset_id not in SUPPORTED_DATASETS:
        raise KeyError(f"unsupported anchor dataset: {dataset_id}")
    try:
        endpoints = _STATIC_ENDPOINTS[(model_id, dataset_id)]
    except KeyError as error:
        raise KeyError(f"unsupported anchor workload: {model_id}/{dataset_id}") from error
    software_endpoints = endpoints["software"]
    hardware_endpoints = endpoints["hardware"]
    base_endpoint = hardware_endpoints["0000"]
    anchors = {
        "0000": StaticAnchor(
            "0000", comparison_baseline("0000"), base_endpoint,
            "ASIC_A0B0 / GPU_or_Orin_base",
        ),
    }
    anchors.update({
        bits: StaticAnchor(
            bits, comparison_baseline(bits), endpoint,
            f"CLAMP_CUDA_{bits} / GPU_base",
        )
        for bits, endpoint in software_endpoints.items()
    })
    anchors.update({
        bits: StaticAnchor(
            bits, comparison_baseline(bits), endpoint / base_endpoint,
            f"ASIC_{bits} / ASIC_A0B0",
        )
        for bits, endpoint in hardware_endpoints.items()
        if bits != "0000"
    })
    canonical = ("0000", "1000", "1010", "0100", "0101", "1100", "1111")
    return {bits: anchors[bits] for bits in canonical}


def compiler_target_speedups(
    model_id: str | None = None, dataset_id: str | None = None,
) -> Mapping[str, float]:
    """Return compiler targets, optionally scoped to one workload."""

    if (model_id is None) != (dataset_id is None):
        raise ValueError("model_id and dataset_id must be supplied together")
    if model_id is None:
        return dict(_STATIC_ENDPOINTS[("r2_gaussian", "chest")]["software"])
    anchors = static_anchors(model_id, dataset_id)
    return {
        bits: anchors[bits].target_speedup
        for bits in ("1000", "0100", "1100")
    }


def hardware_target_speedups(
    model_id: str = "r2_gaussian", dataset_id: str = "chest",
) -> Mapping[str, float]:
    anchors = static_anchors(model_id, dataset_id)
    return {
        "query": anchors["1010"].target_speedup,
        "residency": anchors["0101"].target_speedup,
        "full": anchors["1111"].target_speedup,
    }


def speedup_within_anchor_tolerance(observed: float | None, target: float) -> bool:
    """Return whether a measured speedup is no more than 10% below its anchor.

    Faster-than-anchor measurements remain valid; the tolerance is a lower
    reachability bound rather than an upper cap.
    """

    if observed is None:
        return False
    value = float(observed)
    return math.isfinite(value) and value >= target * ANCHOR_ACCEPTABLE_MIN_FRACTION


def assess_upper_bound(
    model_id: str,
    dataset_id: str,
    observed_upper_bounds: Mapping[str, float],
) -> tuple[AnchorGateResult, ...]:
    """Require every bound to use the target's declared comparison baseline."""

    anchors = static_anchors(model_id, dataset_id)
    if set(observed_upper_bounds) != set(anchors):
        raise ValueError("upper bounds must cover all seven canonical variants")
    results = []
    for bits, anchor in anchors.items():
        observed = float(observed_upper_bounds[bits])
        if not math.isfinite(observed) or observed <= 0:
            raise ValueError(f"upper bound for {bits} must be finite and positive")
        margin = observed - anchor.target_speedup
        results.append(AnchorGateResult(
            bits=bits,
            comparison_baseline=anchor.comparison_baseline,
            target_speedup=anchor.target_speedup,
            observed_upper_bound_speedup=observed,
            margin=margin,
            status=(
                "target_reachable"
                if margin >= 0 else "engineering_optimization_required"
            ),
        ))
    return tuple(results)
