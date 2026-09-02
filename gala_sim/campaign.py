"""Portable model/dataset campaign orchestration for bounded validation runs.

The campaign runner keeps generated traces and measurements in the ignored
workspace.  It is deliberately explicit about which comparisons require a
GPU reference so a CPU trace cannot be mistaken for a GPU performance result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from gala_sim.ablation import asic_speedup, run_matrix
from gala_sim.ablation.anchors import hardware_target_speedups, static_anchors
from gala_sim.adapters import get_model_adapter
from gala_sim.adapters.datasets import DatasetManifest, get_dataset_adapter
from gala_sim.adapters.gr_gaussian import (
    _RAY_BUNDLE_SCHEMA_VERSION,
    _TRACE_FIELD_MASK,
    _active_relation_indices,
    _prepare_ray_bundle,
    _ray_weights,
    _trace_from_ray_bundle,
)
from gala_sim.config import GalaConfig, load_config
from gala_sim.gpu_measurement import (
    COMPILER_BITS,
    GpuCompilerMeasurement,
    default_gpu_measurement_path,
    load_gpu_compiler_measurement,
)
from gala_sim.identity import sha256_tree
from gala_sim.timing import CycleConfig, CycleEngine
from gala_sim.timing.bounds import analyze_cycle_lower_bounds
from gala_sim.timing.memory import CallableMemoryBackend
from gala_sim.tools.representative_packets import (
    build_representative_packet_trace,
    plan_representative_packet_groups,
)
from gala_sim.trace import (
    Trace,
    TraceWriter,
    VirtualPacketArchiveReader,
    VirtualPacketArchiveWriter,
    VirtualTracePacket,
    validate_trace,
)
from gala_sim.clamp import PrimitiveKind
from gala_sim.workspace import WorkspacePaths


MODEL_IDS = ("r2_gaussian", "fact_gs", "exact_gs", "gr_gaussian")
DATASET_IDS = ("chest", "walnut", "hdtomo_usb")
ALL_COMBINATIONS = tuple(
    (model_id, dataset_id)
    for model_id in MODEL_IDS
    for dataset_id in DATASET_IDS
)


@dataclass(frozen=True)
class CampaignResult:
    model_id: str
    dataset_id: str
    trace_root: Path
    bounds_path: Path
    ablation_path: Path
    base_cycles: int
    ablation_cycles: Mapping[str, int]
    upper_bound_status: Mapping[str, str]
    oracle_cycles: Mapping[str, int]
    oracle_speedups_vs_base_asic: Mapping[str, float]
    gpu_base_speedups: Mapping[str, float | None]


def _first_prepared_root(paths: WorkspacePaths, dataset_id: str) -> Path | None:
    metadata_paths = tuple(
        path for path in (paths.cache / "prepared" / dataset_id).glob("*/metadata.json")
        if path.is_file()
    )
    if not metadata_paths:
        return None
    # Cache identities are content hashes, so lexical order is unrelated to
    # freshness.  Prefer the newest complete metadata file and use its path as
    # a deterministic tie-breaker.
    metadata = max(
        metadata_paths,
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    )
    return metadata.parent


def prepared_dataset(paths: WorkspacePaths, dataset_id: str) -> DatasetManifest:
    """Load the newest immutable prepared dataset cache for one identifier."""

    if dataset_id not in DATASET_IDS:
        raise KeyError(f"unsupported dataset: {dataset_id}")
    root = _first_prepared_root(paths, dataset_id)
    if root is None:
        # Raw assets are accepted as a fallback; conversion remains in the
        # normal dataset adapter and is cached by the caller's workspace.
        root = paths.datasets / dataset_id
    adapter = get_dataset_adapter(dataset_id)
    manifest = adapter.load(root)
    adapter.validate(manifest)
    if manifest.initialization is None:
        raise ValueError(f"prepared dataset has no initialization: {dataset_id}")
    return manifest


def _trace_for_campaign(
    paths: WorkspacePaths,
    model_id: str,
    dataset_id: str,
    dataset: DatasetManifest,
    *,
    iterations: int = 1,
) -> tuple[Path, Trace]:
    if iterations <= 0:
        raise ValueError("campaign iteration count must be positive")
    trace_root = paths.traces / model_id / dataset_id / (
        "cpu-validation-v1"
        if iterations == 1 else f"cpu-validation-{iterations}iter-v1"
    )
    metadata_path = trace_root / "metadata.json"
    if metadata_path.is_file():
        from gala_sim.trace import TraceReader

        trace = TraceReader().read(trace_root, validate=False, mmap_mode="r")
        expected = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "capture_backend": "deterministic_cpu_model_contract",
        }
        if all(trace.metadata.get(key) == value for key, value in expected.items()):
            validate_trace(trace)
            return trace_root, trace
        raise ValueError(f"campaign trace identity mismatch: {trace_root}")

    # The graph/ray contract uses real converted projections and initialization
    # files, while remaining bounded enough for a complete CPU matrix replay.
    bundle_root = _prepare_ray_bundle(dataset, trace_root)
    trace = _trace_from_ray_bundle(
        bundle_root / "rays.npz",
        model_name=model_id,
        dataset_name=dataset_id,
    )
    if iterations > 1:
        trace = _expand_trace_iterations(trace, iterations)
    trace.metadata.update({
        "model_id": model_id,
        "model": model_id,
        "dataset_id": dataset_id,
        "dataset": dataset_id,
        "capture_backend": "deterministic_cpu_model_contract",
        "model_contract": "shared_projection_and_initialization_trace",
        "source_dataset_sha256": sha256_tree(dataset.root),
        "formal_performance_eligible": False,
        "result_scope": "quick_cpu_trace_validation",
    })
    validate_trace(trace)
    TraceWriter().write(trace, trace_root, validate=True)
    return trace_root, trace


class _TraceCountingSink:
    """Verify that a model adapter hands off every captured trace event."""

    def __init__(self, chunk_events: int) -> None:
        self.chunk_events = chunk_events
        self.event_count = 0
        self.closed = False

    def push(self, events: np.ndarray, _dependencies: np.ndarray,
             _payload: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("model adapter pushed trace events after closing its sink")
        self.event_count += len(events)

    def close(self) -> None:
        if self.closed:
            raise RuntimeError("model adapter closed its trace sink twice")
        self.closed = True


def _trace_for_gr_cpu_reference_campaign(
    paths: WorkspacePaths,
    dataset_id: str,
    dataset: DatasetManifest,
    config: GalaConfig,
    *,
    iterations: int = 1,
) -> tuple[Path, Trace]:
    """Capture GR-Gaussian from its optimized CPU reference state.

    GR-Gaussian is an independent CPU implementation, so its trace can be
    captured without consuming the shared GPU.  Unlike the generic bounded
    contract trace, this path replays the optimized radiative Gaussian state.
    """

    if iterations <= 0:
        raise ValueError("campaign iteration count must be positive")
    output_root = paths.traces / "gr_gaussian" / dataset_id / "cpu-reference-v2"
    base_trace_root = output_root / "trace"
    metadata_path = base_trace_root / "metadata.json"
    expected = {
        "model_id": "gr_gaussian",
        "dataset_id": dataset_id,
        "capture_backend": "independent_cpu_ray_bundle",
        "radiative_state_source": "optimized_reference_state",
        "ray_bundle_schema_version": _RAY_BUNDLE_SCHEMA_VERSION,
    }
    if metadata_path.is_file():
        from gala_sim.trace import TraceReader

        trace = TraceReader().read(base_trace_root, validate=False, mmap_mode="r")
        if all(trace.metadata.get(key) == value for key, value in expected.items()):
            validate_trace(trace)
        else:
            raise ValueError(
                f"GR-Gaussian campaign trace identity mismatch: {base_trace_root}"
            )
    else:
        adapter = get_model_adapter(
            "gr_gaussian", workspace=paths, output_root=output_root,
        )
        run = adapter.prepare(dataset, config)
        sink = _TraceCountingSink(int(config.value("trace.chunk_events")))
        artifact = adapter.capture_trace(run, sink)
        if not sink.closed or sink.event_count != artifact.trace.event_count:
            raise RuntimeError("GR-Gaussian trace sink handoff is incomplete")
        trace = replace(artifact.trace, metadata={
            **artifact.trace.metadata,
            "result_scope": "quick_cpu_reference_trace_validation",
            "formal_performance_eligible": False,
            "campaign_capture": "optimized_independent_cpu_reference",
        })
        validate_trace(trace)
        TraceWriter().write(trace, base_trace_root, validate=True)
    if iterations == 1:
        return base_trace_root, trace

    expanded_root = paths.traces / "gr_gaussian" / dataset_id / (
        f"cpu-reference-{iterations}iter-v2"
    )
    trace_root = expanded_root / "trace"
    metadata_path = trace_root / "metadata.json"
    if metadata_path.is_file():
        from gala_sim.trace import TraceReader

        expanded = TraceReader().read(trace_root, validate=False, mmap_mode="r")
        if (
            all(expanded.metadata.get(key) == value for key, value in expected.items())
            and expanded.metadata.get("expanded_iterations") == iterations
        ):
            validate_trace(expanded)
            return trace_root, expanded
        raise ValueError(f"GR-Gaussian campaign trace identity mismatch: {trace_root}")
    expanded = _expand_trace_iterations(trace, iterations)
    TraceWriter().write(expanded, trace_root, validate=True)
    return trace_root, expanded


def _trace_for_official_campaign(
    paths: WorkspacePaths,
    model_id: str,
    dataset_id: str,
    dataset: DatasetManifest,
    config: GalaConfig,
    capture_iteration_range: tuple[int, int],
) -> tuple[Path, Trace, Mapping[str, Any]]:
    """Capture and retain one model-specific trace through its official entrypoint."""

    start, end = capture_iteration_range
    if start <= 0 or end < start:
        raise ValueError("official capture range must have 1 <= start <= end")
    capture_root = paths.traces / model_id / dataset_id / (
        f"official-capture-{start}-{end}-v1"
    )
    trace_root = capture_root / "trace"
    reference_path = capture_root / "gpu_reference.json"
    metadata_path = trace_root / "metadata.json"
    expected = {
        "model_id": model_id,
        "dataset": dataset_id,
        "config_sha256": config.sha256,
        "capture_iteration_range": [start, end],
        "official_model_trace": True,
        "result_scope": "official_model_trace_validation",
    }
    if metadata_path.is_file():
        from gala_sim.trace import TraceReader

        trace = TraceReader().read(trace_root, validate=False, mmap_mode="r")
        if any(trace.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"official campaign trace identity mismatch: {trace_root}")
        if not reference_path.is_file():
            raise ValueError(f"official campaign GPU reference is missing: {reference_path}")
        validate_trace(trace)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not isinstance(reference, Mapping):
            raise ValueError(f"official campaign GPU reference is invalid: {reference_path}")
        return trace_root, trace, reference

    adapter = get_model_adapter(
        model_id, workspace=paths, output_root=capture_root,
        capture_iteration_range=capture_iteration_range,
    )
    run = adapter.prepare(dataset, config)
    sink = _TraceCountingSink(int(config.value("trace.chunk_events")))
    artifact = adapter.capture_trace(run, sink)
    if not sink.closed or sink.event_count != artifact.trace.event_count:
        raise RuntimeError("official model trace sink handoff is incomplete")
    trace = replace(artifact.trace, metadata={
        **artifact.trace.metadata,
        **expected,
        "formal_performance_eligible": False,
        "performance_limitations": [
            "trace_capture_overhead_is_not_a_gpu_base_measurement",
            "bounded_cpu_memory_backend_is_not_formal_memory_timing",
        ],
    })
    validate_trace(trace)
    TraceWriter().write(trace, trace_root, validate=True)
    reference_path.write_text(
        json.dumps(artifact.reference.gpu_reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trace_root, trace, artifact.reference.gpu_reference


def _cpu_packet_shape(query_count: int) -> tuple[int, int]:
    """Return a one-tile raster shape for a bounded CPU ray bundle."""

    if query_count <= 0:
        raise ValueError("CPU packet archive has no queries")
    for width in range(min(16, query_count), 0, -1):
        if query_count % width == 0 and query_count // width <= 16:
            return query_count // width, width
    raise ValueError(
        "CPU packet archive query count does not fit one bounded raster tile"
    )


def _cpu_packet_from_ray_bundle(bundle_path: Path) -> tuple[VirtualTracePacket, int]:
    """Encode the bounded ray contract as one lossless virtual packet."""

    with np.load(bundle_path, allow_pickle=False) as bundle:
        means = np.asarray(bundle["means"], dtype=np.float64)
        scales = np.asarray(bundle["scales"], dtype=np.float64)
        origins = np.asarray(bundle["ray_origins"], dtype=np.float64)
        directions = np.asarray(bundle["ray_directions"], dtype=np.float64)
    weights = _ray_weights(origins, directions, means, scales)
    shape = _cpu_packet_shape(int(weights.shape[0]))
    masks = np.zeros((len(means), 8), dtype=np.dtype("<u4"))
    for query_id, query_weights in enumerate(weights):
        for gaussian_id in _active_relation_indices(query_weights):
            masks[int(gaussian_id), query_id // 32] |= np.uint32(
                1 << (query_id % 32)
            )
    active = np.flatnonzero(np.any(masks, axis=1))
    if active.size == 0:
        raise ValueError("CPU packet archive has no active Gaussian relations")
    packet = VirtualTracePacket(
        iteration_id=1,
        template_id=1,
        query_base=0,
        query_shape=shape,
        point_ids=active.astype(np.int64, copy=False),
        point_keys=np.arange(active.size, dtype=np.uint64),
        masks=masks[active].copy(),
        state_version=0,
        field_mask=_TRACE_FIELD_MASK,
        loss_flags=1,
        backward_confirmed=True,
    )
    return packet, int(len(means))


def _packet_archive_root(paths: WorkspacePaths, model_id: str,
                         dataset_id: str) -> Path:
    return paths.traces / model_id / dataset_id / "cpu-packet-archive-v1"


def _trace_for_representative_archive_campaign(
    paths: WorkspacePaths,
    model_id: str,
    dataset_id: str,
    dataset: DatasetManifest,
    *,
    archive_chunk_bytes: int,
    iterations: int = 1,
) -> tuple[Path, Trace]:
    """Build a bounded CPU archive through the production packet replay path."""

    if iterations <= 0:
        raise ValueError("campaign iteration count must be positive")
    archive_root = _packet_archive_root(paths, model_id, dataset_id)
    manifest_path = archive_root / "manifest.json"
    expected_archive = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "capture_backend": "deterministic_cpu_packet_archive_contract",
    }
    if manifest_path.is_file():
        reader = VirtualPacketArchiveReader(archive_root)
        reader.validate()
        metadata = reader.manifest.get("metadata", {})
        if not isinstance(metadata, Mapping) or any(
            metadata.get(key) != value for key, value in expected_archive.items()
        ):
            raise ValueError(f"campaign packet archive identity mismatch: {archive_root}")
    else:
        if archive_root.exists() and any(archive_root.iterdir()):
            raise ValueError(
                f"campaign packet archive is incomplete: {archive_root}"
            )
        bundle_root = _prepare_ray_bundle(
            dataset, archive_root.parent / "cpu-packet-input-v1",
        )
        packet, initial_gaussians = _cpu_packet_from_ray_bundle(
            bundle_root / "rays.npz"
        )
        writer = VirtualPacketArchiveWriter(
            archive_root,
            max_chunk_bytes=max(archive_chunk_bytes, packet.physical_bytes),
        )
        writer.initialize_gaussians(initial_gaussians)
        writer.append_packet(packet)
        writer.close_iteration(1)
        writer.finish(metadata={
            **expected_archive,
            "model": model_id,
            "dataset": dataset_id,
            "formal_performance_eligible": False,
            "result_scope": "quick_cpu_packet_archive_validation",
            "source_dataset_sha256": sha256_tree(dataset.root),
        })
        VirtualPacketArchiveReader(archive_root).validate()

    plan_path = archive_root / "representative-plan.json"
    if not plan_path.is_file():
        plan = plan_representative_packet_groups(
            archive_root,
            None,
            expected_group_count=1,
            single_iteration=1,
        )
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    trace_root = paths.traces / model_id / dataset_id / (
        "cpu-archive-validation-v1"
        if iterations == 1 else f"cpu-archive-validation-{iterations}iter-v1"
    )
    metadata_path = trace_root / "metadata.json"
    expected_trace = {
        **expected_archive,
        "capture_backend": "deterministic_cpu_packet_archive_contract",
    }
    if metadata_path.is_file():
        from gala_sim.trace import TraceReader

        trace = TraceReader().read(trace_root, validate=False, mmap_mode="r")
        if all(trace.metadata.get(key) == value for key, value in expected_trace.items()):
            validate_trace(trace)
            return trace_root, trace
        raise ValueError(f"campaign archive trace identity mismatch: {trace_root}")
    trace = build_representative_packet_trace(
        archive_root,
        plan_path,
        window_index=0,
        max_events=65536,
        query_lanes=8,
        model_id=model_id,
        dataset_id=dataset_id,
    )
    if iterations > 1:
        trace = _expand_trace_iterations(trace, iterations)
    trace.metadata.update({
        **expected_trace,
        "model": model_id,
        "dataset": dataset_id,
        "formal_performance_eligible": False,
        "result_scope": "quick_cpu_packet_archive_validation",
    })
    validate_trace(trace)
    TraceWriter().write(trace, trace_root, validate=True)
    return trace_root, trace


def _expand_trace_iterations(trace: Trace, iterations: int) -> Trace:
    """Repeat one validated optimizer transaction with explicit state barriers."""

    if iterations <= 0:
        raise ValueError("trace iteration count must be positive")
    if iterations == 1:
        return trace
    validate_trace(trace)
    source_events = np.asarray(trace.events)
    source_dependencies = np.asarray(trace.dependencies)
    source_payload = np.asarray(trace.payload)
    event_count = int(source_events.size)
    if event_count == 0:
        raise ValueError("cannot expand an empty trace")
    query_values = source_events["query_id"]
    relation_values = source_events["relation_id"]
    query_stride = int(query_values.max(initial=-1)) + 1
    relation_stride = int(relation_values.max(initial=-1)) + 1
    if query_stride <= 0 or relation_stride <= 0:
        raise ValueError("trace has no query or relation identity to expand")
    update_begin = np.flatnonzero(
        source_events["primitive_kind"] == int(PrimitiveKind.UPDATE_BEGIN)
    )
    update_end = np.flatnonzero(
        source_events["primitive_kind"] == int(PrimitiveKind.UPDATE_END)
    )
    if update_begin.size != 1 or update_end.size != 1:
        raise ValueError("iteration expansion requires one optimizer transaction")
    source_begin_id = int(source_events[int(update_begin[0])]["event_id"])
    source_end_id = int(source_events[int(update_end[0])]["event_id"])
    event_parts: list[np.ndarray] = []
    dependency_parts: list[np.ndarray] = []
    payload_parts: list[np.ndarray] = []
    iteration_event_counts: dict[str, int] = {}
    previous_end: int | None = None
    for iteration_index in range(iterations):
        event_offset = iteration_index * event_count
        dependency_offset = sum(part.size for part in dependency_parts)
        payload_offset = sum(part.size for part in payload_parts)
        query_offset = iteration_index * query_stride
        relation_offset = iteration_index * relation_stride
        rows = source_events.copy()
        rows["event_id"] += event_offset
        rows["iteration_id"] = iteration_index + 1
        rows["state_version"] = iteration_index
        positive_query = rows["query_id"] >= 0
        rows["query_id"][positive_query] += query_offset
        rows["consumer_id"][rows["consumer_id"] >= 0] += query_offset
        positive_relation = rows["relation_id"] >= 0
        rows["relation_id"][positive_relation] += relation_offset
        rows["dependency_begin"] += dependency_offset
        rows["payload_offset"] += payload_offset
        rows["reduction_key"][
            rows["primitive_kind"] == int(PrimitiveKind.UPDATE_END)
        ] = source_begin_id + event_offset
        dependencies: list[int] = []
        for row_index, row in enumerate(rows):
            source_row = source_events[row_index]
            begin = int(source_row["dependency_begin"])
            end = begin + int(source_row["dependency_count"])
            values = [int(value) + event_offset for value in source_dependencies[begin:end]]
            kind = PrimitiveKind(int(row["primitive_kind"]))
            if previous_end is not None and kind in {
                PrimitiveKind.RELATION_CANDIDATE, PrimitiveKind.QUERY_CLOSE,
            }:
                values.append(previous_end)
            row["dependency_begin"] = dependency_offset + len(dependencies)
            row["dependency_count"] = len(values)
            dependencies.extend(values)
        event_parts.append(rows)
        dependency_parts.append(np.asarray(dependencies, dtype=source_dependencies.dtype))
        payload_parts.append(source_payload.copy())
        iteration_event_counts[str(iteration_index + 1)] = event_count
        previous_end = source_end_id + event_offset
    expanded = replace(
        trace,
        events=np.concatenate(event_parts).astype(source_events.dtype, copy=False),
        dependencies=np.concatenate(dependency_parts).astype(source_dependencies.dtype, copy=False),
        payload=np.concatenate(payload_parts).astype(source_payload.dtype, copy=False),
        metadata={
            **trace.metadata,
            **(
                {
                    "trace_window": {
                        **trace.metadata["trace_window"],
                        "iterations": list(range(1, iterations + 1)),
                    },
                }
                if isinstance(trace.metadata.get("trace_window"), Mapping)
                else {}
            ),
            "iteration_event_counts": iteration_event_counts,
            "expanded_iterations": iterations,
            "cross_iteration_barrier": "previous_update_end_on_candidate_and_query_close",
        },
    )
    validate_trace(expanded)
    return expanded


def _cycle_config(config: GalaConfig) -> CycleConfig:
    # This backend is only for deterministic CPU architecture validation.  A
    # Ramulator binding remains required for formal memory-timing evidence.
    memory = CallableMemoryBackend(
        lambda *, address, size_bytes, is_write, arrival_cycle: (
            arrival_cycle + max(1, int(size_bytes) // 64) + int(bool(is_write))
        )
    )
    return CycleConfig.from_gala(config, memory)


def _campaign_result_root(paths: WorkspacePaths, model_id: str, dataset_id: str,
                          iterations: int, *, representative_archive: bool = False,
                          official_trace: bool = False,
                          capture_iteration_range: tuple[int, int] = (1, 1)) -> Path:
    """Return a result directory whose name identifies the trace length."""

    if iterations <= 0:
        raise ValueError("campaign iteration count must be positive")
    suffix = "v1" if iterations == 1 else f"{iterations}iter-v1"
    if representative_archive and official_trace:
        raise ValueError("representative and official campaign modes are mutually exclusive")
    prefix = (
        f"official-validation-{capture_iteration_range[0]}-{capture_iteration_range[1]}"
        if official_trace else (
            "cpu-archive-validation" if representative_archive else "cpu-validation"
        )
    )
    return paths.results / model_id / dataset_id / f"{prefix}-{suffix}"


def run_campaign(
    model_id: str,
    dataset_id: str,
    *,
    workspace: WorkspacePaths | None = None,
    config: GalaConfig | None = None,
    repository: Path | None = None,
    parallel_workers: int = 1,
    iterations: int = 1,
    representative_archive: bool = False,
    official_trace: bool = False,
    capture_iteration_range: tuple[int, int] = (1, 1),
    gpu_measurement_root: Path | None = None,
) -> CampaignResult:
    """Run one complete bounded trace, bound, and seven-variant matrix.

    ``representative_archive`` exercises the packet-archive and representative
    replay path used by official captures, while retaining the explicit CPU
    contract scope until an official model trace is available. ``official_trace``
    captures that model trace before running the same bound and matrix stages.
    """

    if model_id not in MODEL_IDS:
        raise KeyError(f"unsupported model: {model_id}")
    if dataset_id not in DATASET_IDS:
        raise KeyError(f"unsupported dataset: {dataset_id}")
    if representative_archive and official_trace:
        raise ValueError("representative and official campaign modes are mutually exclusive")
    paths = (workspace or WorkspacePaths.discover(repository=repository)).ensure()
    gala_config = config or load_config(paths.repository / "configs/architecture/gala.yaml")
    dataset = prepared_dataset(paths, dataset_id)
    gpu_reference: Mapping[str, Any] | None = None
    gpu_measurement: GpuCompilerMeasurement | None = None
    if official_trace:
        if iterations != 1:
            raise ValueError("official campaigns capture their real iteration window")
        trace_root, trace, gpu_reference = _trace_for_official_campaign(
            paths, model_id, dataset_id, dataset, gala_config,
            capture_iteration_range,
        )
        measurement_root = (
            Path(gpu_measurement_root)
            if gpu_measurement_root is not None
            else paths.results / "gpu-measurements"
        )
        measurement_path = default_gpu_measurement_path(
            measurement_root, model_id, dataset_id,
        )
        if measurement_path.is_file():
            gpu_measurement = load_gpu_compiler_measurement(
                measurement_path,
                model_id=model_id,
                dataset_id=dataset_id,
                iteration_range=capture_iteration_range,
            )
        elif gpu_measurement_root is not None:
            raise ValueError(
                f"official GPU compiler measurement is missing: {measurement_path}"
            )
    elif representative_archive:
        trace_root, trace = _trace_for_representative_archive_campaign(
            paths,
            model_id,
            dataset_id,
            dataset,
            archive_chunk_bytes=int(gala_config.value("trace.archive_chunk_bytes")),
            iterations=iterations,
        )
    elif model_id == "gr_gaussian":
        trace_root, trace = _trace_for_gr_cpu_reference_campaign(
            paths, dataset_id, dataset, gala_config, iterations=iterations,
        )
    else:
        trace_root, trace = _trace_for_campaign(
            paths, model_id, dataset_id, dataset, iterations=iterations,
        )
    result_scope = "official_model_trace_validation" if official_trace else (
        "quick_cpu_packet_archive_validation"
        if representative_archive else "quick_cpu_trace_validation"
    )
    if model_id == "gr_gaussian" and not representative_archive and not official_trace:
        result_scope = "quick_cpu_reference_trace_validation"
    cycle_config = _cycle_config(gala_config)
    runs = run_matrix(trace, cycle_config, parallel_workers=parallel_workers)
    base_cycles = runs[0].result.total_cycles
    anchors = static_anchors(model_id, dataset_id)
    variant_order = [run.variant.bits for run in runs]
    cycles_by_variant = {
        run.variant.bits: run.result.total_cycles for run in runs
    }
    speedups_vs_base_asic = {
        run.variant.bits: asic_speedup(
            run.variant.bits, base_cycles=base_cycles,
            cycles=run.result.total_cycles,
        )
        for run in runs
    }
    gpu_base_speedups: dict[str, float | None] = {
        bits: None for bits in COMPILER_BITS
    }
    if gpu_measurement is not None:
        gpu_base_speedups.update(gpu_measurement.speedups_vs_gpu_base)
    targets = hardware_target_speedups()
    # Oracle policies retain the configured resources and dependencies while
    # exposing the best legal decision within one mechanism's scope.  Their
    # cycles are diagnostics for engineering coverage, not formal results.
    oracle_cycles = {
        scenario: CycleEngine(
            cycle_config, policy=f"{scenario}_oracle",
        ).run(trace).total_cycles
        for scenario in ("query", "residency")
    }
    oracle_speedups = {
        scenario: base_cycles / cycles
        for scenario, cycles in oracle_cycles.items()
    }
    result_root = _campaign_result_root(
        paths,
        model_id,
        dataset_id,
        iterations,
        representative_archive=representative_archive,
        official_trace=official_trace,
        capture_iteration_range=capture_iteration_range,
    )
    result_root.mkdir(parents=True, exist_ok=True)
    ablation_path = result_root / "ablation.json"
    target_assessment: dict[str, dict[str, Any]] = {}
    for bits in variant_order:
        if bits == "0000":
            observed_speedup = None
            assessment_status = "not_measured_platform_baseline"
        elif bits in COMPILER_BITS:
            observed_speedup = gpu_base_speedups[bits]
            if observed_speedup is None:
                assessment_status = "not_measured_cpu_only"
            elif observed_speedup >= anchors[bits].target_speedup:
                assessment_status = "target_met_gpu_measurement"
            else:
                assessment_status = "below_target_gpu_measurement"
        else:
            observed_speedup = speedups_vs_base_asic[bits]
            if (
                observed_speedup is not None
                and observed_speedup >= anchors[bits].target_speedup
            ):
                assessment_status = "target_met_cpu_validation"
            else:
                assessment_status = "below_target_cpu_validation"
        target_assessment[bits] = {
            "comparison_baseline": anchors[bits].comparison_baseline,
            "target_speedup": anchors[bits].target_speedup,
            "observed_speedup": observed_speedup,
            "status": assessment_status,
        }
    ablation_document = {
        "schema_version": "gala-campaign-ablation-v1",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "trace": paths.reference(trace_root),
        "iterations": iterations,
        "result_scope": result_scope,
        "formal_performance_eligible": False,
        "comparison_baselines": {
            bits: anchors[bits].comparison_baseline
            for bits in (run.variant.bits for run in runs)
        },
        "variant_order": variant_order,
        "static_anchor_targets": {
            bits: {
                "comparison_baseline": anchors[bits].comparison_baseline,
                "target_speedup": anchors[bits].target_speedup,
                "decomposition": anchors[bits].decomposition,
            }
            for bits in variant_order
        },
        "cycles": cycles_by_variant,
        "speedup_vs_base_asic": speedups_vs_base_asic,
        "target_assessment": target_assessment,
        "configured_oracle_diagnostics": {
            scenario: {
                "cycles": oracle_cycles[scenario],
                "speedup_vs_base_asic": oracle_speedups[scenario],
                "target_speedup_vs_base_asic": targets[scenario]
                if scenario in targets else None,
                "status": "diagnostic_only",
            }
            for scenario in oracle_cycles
        },
        "gpu_base_speedups": gpu_base_speedups,
        "gpu_reference_status": (
            "measured_uninstrumented_official_training"
            if gpu_measurement is not None else (
                "captured_with_trace_overhead_not_gpu_base"
                if gpu_reference is not None else "not_measured_cpu_only"
            )
        ),
        "trace_capture_gpu_reference": dict(gpu_reference) if gpu_reference is not None else None,
        "gpu_compiler_measurement": (
            gpu_measurement.as_campaign_reference(
                paths.reference(gpu_measurement.path),
            )
            if gpu_measurement is not None else None
        ),
    }
    ablation_path.write_text(
        json.dumps(ablation_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bound_report = analyze_cycle_lower_bounds(
        CycleEngine(cycle_config, policy="base"),
        trace,
        base_asic_cycles=base_cycles,
        targets=targets,
    )
    bounds_path = result_root / "cycle_lower_bounds.json"
    bounds_document = bound_report.as_dict()
    bounds_document["model_id"] = model_id
    bounds_document["dataset_id"] = dataset_id
    bounds_document["result_scope"] = result_scope
    bounds_document["formal_performance_eligible"] = False
    bounds_document["variant_order"] = variant_order
    bounds_document["static_anchor_targets"] = {
        bits: {
            "comparison_baseline": anchors[bits].comparison_baseline,
            "target_speedup": anchors[bits].target_speedup,
            "decomposition": anchors[bits].decomposition,
        }
        for bits in variant_order
    }
    bounds_document["gpu_compiler_bounds"] = {
        bits: {
            "status": (
                "not_measured_cpu_only"
                if gpu_base_speedups[bits] is None
                else (
                    "target_met_gpu_measurement"
                    if gpu_base_speedups[bits] >= anchors[bits].target_speedup
                    else "engineering_optimization_required"
                )
            ),
            "comparison_baseline": "gpu_base",
            "observed_speedup": gpu_base_speedups[bits],
            "target_speedup": anchors[bits].target_speedup,
        }
        for bits in COMPILER_BITS
    }
    bounds_document["base_asic_platform_bound"] = {
        "status": "not_measured_platform_baseline",
        "comparison_baseline": anchors["0000"].comparison_baseline,
        "observed_speedup": None,
        "target_speedup": anchors["0000"].target_speedup,
    }
    bounds_path.write_text(
        json.dumps(bounds_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upper_status = {
        item.scenario: item.status for item in bound_report.reachability
    }
    return CampaignResult(
        model_id=model_id,
        dataset_id=dataset_id,
        trace_root=trace_root,
        bounds_path=bounds_path,
        ablation_path=ablation_path,
        base_cycles=base_cycles,
        ablation_cycles={run.variant.bits: run.result.total_cycles for run in runs},
        upper_bound_status=upper_status,
        oracle_cycles=oracle_cycles,
        oracle_speedups_vs_base_asic=oracle_speedups,
        gpu_base_speedups=gpu_base_speedups,
    )


def run_campaigns(
    selections: Iterable[tuple[str, str]],
    *,
    workspace: WorkspacePaths | None = None,
    config: GalaConfig | None = None,
    repository: Path | None = None,
    parallel_workers: int = 1,
    iterations: int = 1,
    representative_archive: bool = False,
    official_trace: bool = False,
    capture_iteration_range: tuple[int, int] = (1, 1),
    gpu_measurement_root: Path | None = None,
) -> tuple[CampaignResult, ...]:
    """Run selected campaigns in deterministic model/dataset order."""

    unique = tuple(dict.fromkeys(selections))
    unknown = [item for item in unique if item not in ALL_COMBINATIONS]
    if unknown:
        raise ValueError(f"unsupported campaign selection: {unknown[0]}")
    return tuple(
        run_campaign(
            model_id, dataset_id, workspace=workspace, config=config,
            repository=repository, parallel_workers=parallel_workers,
            iterations=iterations, representative_archive=representative_archive,
            official_trace=official_trace,
            capture_iteration_range=capture_iteration_range,
            gpu_measurement_root=gpu_measurement_root,
        )
        for model_id, dataset_id in unique
    )
