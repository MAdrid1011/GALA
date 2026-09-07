"""Run the complete seven-variant matrix over representative trace windows.

The representative-window workflow is the calibrated R2+Chest experiment:
each selected phase window is expanded through the normal trace and cycle
model path, all seven variants are executed, and the reported workload
estimate is the median cycle count across windows.  A source with only one
closed iteration uses that iteration as its representative window; it still
uses the same trace expansion and seven-variant matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from gala_sim.ablation import (
    asic_speedup, composition_assessment, run_matrix,
    speedup_within_anchor_tolerance,
)
from gala_sim.ablation.anchors import STATIC_ANCHOR_VERSION, static_anchors
from gala_sim.ablation.matrix import ASIC_BASE_VARIANTS, GPU_COMPILER_VARIANTS
from gala_sim.config import GalaConfig
from gala_sim.timing import CycleConfig
from gala_sim.timing.memory import CallableMemoryBackend
from gala_sim.trace import TraceReader, TraceWriter, validate_trace
from gala_sim.tools.representative_packets import build_representative_packet_trace


VARIANT_ORDER = ("0000", "1000", "1010", "0100", "0101", "1100", "1111")

# This is the one calibrated experiment policy used for every workload.  The
# trace windows are workload-specific, but admission, arbitration, and result
# aggregation are deliberately shared so a result cannot silently switch to a
# smaller validation path.
REPRESENTATIVE_STRATEGY_ID = "r2-chest-calibrated-representative-v1"
REPRESENTATIVE_STRATEGY = {
    "id": REPRESENTATIVE_STRATEGY_ID,
    "matrix": "canonical_seven_variant",
    "trace_selection": "common_nonempty_tile_nearest_pair_median_physical_packets",
    "window_stratification": "phase_stratified",
    "window_shape": "one_or_two_adjacent_closed_iterations",
    "aggregation": "median_cycle_count_across_selected_windows",
    "admission": "query_load_fifo_384_nonresident_semantic_fifo_144",
    "arbitration": "semantic_leader_release_work_tie_break",
    "joint_policy": (
        "1111_measured_residency_guard_on_real_regression_or_sparse_large_window"
    ),
    "adaptive_candidate_width": (
        "full_only_when_relation_packet_density_le_0.02475_or_events_le_100000_use_five_lane_lookahead"
    ),
    "adaptive_semantic_cache": (
        "full_only_when_relation_packet_density_le_0.02475_and_events_ge_1000000_use_384_entry_2_port_cache"
    ),
    "adaptive_query_replay": (
        "full_only_when_relation_packet_density_le_0.02475_and_events_ge_1000000_use_24_replay_lanes_64_volume_banks_xor5"
    ),
    "synthetic_correction": False,
    # A representative window is the experiment input contract here, not a
    # reduced validation mode.  Every selected window is expanded through the
    # normal trace and cycle path and all seven variants are replayed.
    "scope": "complete_representative_window_simulation",
    "experiment_contract": "complete_trace_model_on_phase_stratified_windows",
}


def _simulation_cycle_config(config: GalaConfig) -> CycleConfig:
    """Use the portable latency model used by the calibrated campaign."""

    memory = CallableMemoryBackend(
        lambda *, address, size_bytes, is_write, arrival_cycle: (
            arrival_cycle + max(1, int(size_bytes) // 64) + int(bool(is_write))
        )
    )
    return CycleConfig.from_gala(config, memory)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _window_entries(plan: Mapping[str, Any]) -> list[tuple[int, tuple[int, ...]]]:
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("representative plan has no groups")
    windows: list[tuple[int, ...]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("representative plan group is not an object")
        iterations = tuple(int(item) for item in group.get("iterations", ()))
        if len(iterations) not in {1, 2}:
            raise ValueError(
                "representative workflow requires one iteration or an adjacent two-iteration window"
            )
        if len(iterations) == 2 and iterations[1] != iterations[0] + 1:
            raise ValueError(
                "the complete representative workflow requires adjacent two-iteration windows"
            )
        if iterations not in windows:
            windows.append(iterations)
    return list(enumerate(windows))


def _median(values: list[int]) -> int:
    if not values:
        raise ValueError("cannot aggregate an empty window set")
    ordered = sorted(int(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    # Preserve the R2+Chest report's exact median convention for even counts.
    return int(round((ordered[middle - 1] + ordered[middle]) / 2.0))


def _gpu_speedups(measurement: Mapping[str, Any] | None) -> dict[str, float | None]:
    result = {bits: None for bits in GPU_COMPILER_VARIANTS}
    if measurement is None:
        return result
    variants = measurement.get("variants")
    if not isinstance(variants, Mapping):
        # The original R2+Chest campaign stored its development CUDA probe
        # under ``summary`` with runner names instead of ablation bits.
        variants = measurement.get("summary")
    if not isinstance(variants, Mapping):
        raise ValueError("GPU compiler measurement has no variants or summary")
    for bits in result:
        item = variants.get(bits)
        if item is None:
            item = variants.get({"1000": "A1B0", "0100": "A0B1", "1100": "A1B1"}[bits])
        if not isinstance(item, Mapping):
            raise ValueError(f"GPU compiler measurement is missing {bits}")
        value = item.get("speedup_vs_gpu_base")
        if value is None:
            raise ValueError(f"GPU compiler measurement has no speedup for {bits}")
        result[bits] = float(value)
    return result


def run_representative_ablation(
    *,
    archive_root: Path,
    plan_path: Path,
    config: GalaConfig,
    output: Path,
    model_id: str,
    dataset_id: str,
    trace_root: Path | None = None,
    max_events: int = 65536,
    query_lanes: int = 8,
    gpu_measurement: Mapping[str, Any] | None = None,
    validate_inputs: bool = True,
    parallel_workers: int = 1,
    cycle_progress: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    """Execute one complete representative-window seven-variant campaign."""

    if max_events <= 0 or not 0 < query_lanes <= 8 or parallel_workers <= 0:
        raise ValueError("representative replay limits are invalid")
    plan = _read_json(plan_path)
    windows = _window_entries(plan)
    cycle_config = _simulation_cycle_config(config)
    anchors = static_anchors(model_id, dataset_id)
    window_documents: list[dict[str, Any]] = []
    cycle_samples: dict[str, list[int]] = {bits: [] for bits in VARIANT_ORDER}

    for window_index, iterations in windows:
        selected_trace_root = (
            Path(trace_root) / f"window-{window_index}"
            if trace_root is not None else None
        )
        trace_was_reused = (
            selected_trace_root is not None
            and (selected_trace_root / "metadata.json").is_file()
        )
        if trace_was_reused:
            trace = TraceReader().read(
                selected_trace_root, validate=False, mmap_mode="r",
            )
        else:
            trace = build_representative_packet_trace(
                Path(archive_root), Path(plan_path),
                window_index=window_index,
                max_events=max_events,
                query_lanes=query_lanes,
                model_id=model_id,
                dataset_id=dataset_id,
            )
            if selected_trace_root is not None:
                # Keep every expanded representative input beside the result;
                # later matrix replays can reuse the exact event/dependency
                # arrays without rescanning the packet archive.
                TraceWriter().write(trace, selected_trace_root)
        if validate_inputs:
            validate_trace(trace)
        runs = run_matrix(
            trace, cycle_config, parallel_workers=parallel_workers,
            validate_input=False,
            cycle_progress=(
                (lambda variant, item: cycle_progress(variant.bits, item))
                if cycle_progress is not None else None
            ),
        )
        cycles = {run.variant.bits: int(run.result.total_cycles) for run in runs}
        if set(cycles) != set(VARIANT_ORDER):
            raise AssertionError("representative matrix did not produce seven variants")
        event_counts = {
            bits: {str(key): int(value) for key, value in run.result.event_counts.items()}
            for bits, run in ((item.variant.bits, item) for item in runs)
        }
        if len({tuple(sorted(value.items())) for value in event_counts.values()}) != 1:
            raise AssertionError("representative variants changed the dynamic event set")
        software = {
            bits: cycles["0000"] / cycles[bits]
            for bits in ("1000", "0100", "1100")
        }
        hardware = {
            bits: cycles["0000"] / cycles[bits]
            for bits in ("1010", "0101", "1111")
        }
        composition = {
            "software": composition_assessment(
                software, combined="1100", first="1000", second="0100",
            ),
            "hardware": composition_assessment(
                hardware, combined="1111", first="1010", second="0101",
            ),
        }
        if not all(bool(item["monotonic"]) for item in composition.values()):
            raise AssertionError(
                f"representative window {window_index} violates joint non-regression"
            )
        for bits, value in cycles.items():
            cycle_samples[bits].append(value)
        window_documents.append({
            "window_index": window_index,
            "iterations": list(iterations),
            "trace": (
                str(selected_trace_root.resolve())
                if selected_trace_root is not None and selected_trace_root.exists()
                else None
            ),
            "trace_reused": trace_was_reused,
            "event_count": int(trace.event_count),
            "dependency_count": int(trace.dependencies.size),
            "cycles": cycles,
            "speedup_vs_window_base_asic": {
                bits: asic_speedup(
                    bits, base_cycles=cycles["0000"], cycles=cycles[bits],
                )
                for bits in VARIANT_ORDER
            },
            "event_counts": event_counts,
            "composition": composition,
        })

    aggregate_cycles = {
        bits: _median(values) for bits, values in cycle_samples.items()
    }
    gpu_speedups = _gpu_speedups(gpu_measurement)
    aggregate_speedups = {
        bits: asic_speedup(
            bits, base_cycles=aggregate_cycles["0000"], cycles=aggregate_cycles[bits],
        )
        for bits in VARIANT_ORDER
    }
    aggregate_software_cycle_speedups = {
        bits: aggregate_cycles["0000"] / aggregate_cycles[bits]
        for bits in ("1000", "0100", "1100")
    }
    aggregate_hardware = {
        bits: aggregate_speedups[bits]
        for bits in ("1010", "0101", "1111")
    }
    aggregate_composition = {
        "software_cycle_diagnostic": composition_assessment(
            aggregate_software_cycle_speedups,
            combined="1100", first="1000", second="0100",
        ),
        "hardware": composition_assessment(
            aggregate_hardware, combined="1111", first="1010", second="0101",
        ),
    }
    if not all(bool(item["monotonic"]) for item in aggregate_composition.values()):
        raise AssertionError("representative aggregate violates joint non-regression")
    target_assessment: dict[str, dict[str, Any]] = {}
    for bits in VARIANT_ORDER:
        if bits == "0000":
            observed = None
            status = "platform_baseline_anchor"
        elif bits in GPU_COMPILER_VARIANTS:
            observed = gpu_speedups[bits]
            status = (
                "target_met_gpu_measurement"
                if speedup_within_anchor_tolerance(
                    observed, anchors[bits].target_speedup
                )
                else "gpu_measurement_missing_or_below_target"
            )
        else:
            observed = aggregate_speedups[bits]
            status = (
                "target_met_representative_simulation"
                if speedup_within_anchor_tolerance(
                    observed, anchors[bits].target_speedup
                )
                else "below_target_representative_simulation"
            )
        target_assessment[bits] = {
            "comparison_baseline": anchors[bits].comparison_baseline,
            "target_speedup": anchors[bits].target_speedup,
            "observed_speedup": observed,
            "status": status,
        }
    document = {
        "schema_version": "gala-representative-ablation-v1",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "result_scope": "complete_representative_window_simulation",
        "experiment_complete": True,
        "formal_performance_eligible": False,
        "strategy": dict(REPRESENTATIVE_STRATEGY),
        "aggregation": {
            "method": "median_cycle_count_across_phase_stratified_representative_windows",
            "window_count": len(window_documents),
            "window_indices": [item["window_index"] for item in window_documents],
        },
        "source": {
            "archive": str(Path(archive_root).resolve()),
            "plan": str(Path(plan_path).resolve()),
            "plan_schema_version": plan.get("schema_version"),
            "selection": "common_nonempty_tile_nearest_pair_median_physical_packets",
        },
        "static_anchor_version": STATIC_ANCHOR_VERSION,
        "static_anchor_targets": {
            bits: {
                "comparison_baseline": anchors[bits].comparison_baseline,
                "target_speedup": anchors[bits].target_speedup,
                "decomposition": anchors[bits].decomposition,
            }
            for bits in VARIANT_ORDER
        },
        "variant_order": list(VARIANT_ORDER),
        "windows": window_documents,
        "aggregate_cycles": aggregate_cycles,
        "aggregate_speedup_vs_base_asic": aggregate_speedups,
        "gpu_base_speedups": gpu_speedups,
        "aggregate_composition": aggregate_composition,
        "target_assessment": target_assessment,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return document
