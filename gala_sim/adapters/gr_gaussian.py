"""Independent CPU reference for graph-based radiative Gaussian splatting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from gala_sim.clamp import (
    PrimitiveKind, ResourceClass, TraceBuilder, TraceEvent, UpdateBeginKind,
)
from gala_sim.clamp.events import encode_relation_packet_flags
from gala_sim.metrics import QualityConfig
from gala_sim.trace import TraceWriter, validate_trace
from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact


_RAY_BUNDLE_SCHEMA_VERSION = "gala-gr-gaussian-ray-bundle-v2"
_TRACE_FIELD_MASK = 0b1111
_MAX_RAY_VIEWS = 4
_RAYS_PER_VIEW = 32
_MAX_INITIAL_GAUSSIANS = 128
_MIN_RETAINED_RADIATIVE_MASS = 0.99


@dataclass(frozen=True)
class GRGaussianState:
    means: np.ndarray
    scales: np.ndarray
    densities: np.ndarray
    edges: np.ndarray


def denoised_point_cloud_initialization(
    prior_volume: np.ndarray,
    gaussian_count: int,
    *,
    density_threshold: float = 0.05,
    artifact_quantile: float = 0.95,
) -> GRGaussianState:
    """Create a deterministic graph from a denoised reconstruction prior."""

    volume = np.asarray(prior_volume, dtype=np.float32)
    if volume.ndim != 3 or volume.size == 0 or not np.isfinite(volume).all():
        raise ValueError("initialization prior must be a finite non-empty 3D volume")
    if gaussian_count <= 0 or density_threshold < 0:
        raise ValueError("initialization count and density threshold must be valid")
    if not 0 < artifact_quantile <= 1:
        raise ValueError("artifact quantile must be in (0, 1]")

    padded = np.pad(volume, 1, mode="edge")
    denoised = (
        padded[1:-1, 1:-1, 1:-1]
        + padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:]
    ) / 7.0
    residual = np.abs(volume - denoised)
    eligible = denoised >= density_threshold
    if not np.any(eligible):
        raise ValueError("denoised prior contains no eligible initialization voxels")
    cutoff = float(np.quantile(residual[eligible], artifact_quantile))
    flat_candidates = np.flatnonzero(eligible & (residual <= cutoff))
    if len(flat_candidates) == 0:
        raise ValueError("artifact filtering removed every initialization voxel")
    flat_values = denoised.ravel()[flat_candidates]
    keep = min(gaussian_count, len(flat_candidates))
    if keep < len(flat_candidates):
        selected_positions = np.argpartition(-flat_values, keep - 1)[:keep]
        flat_candidates = flat_candidates[selected_positions]
        flat_values = flat_values[selected_positions]
    candidates = np.column_stack(np.unravel_index(flat_candidates, volume.shape))
    values = flat_values
    order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], -values))
    selected = candidates[order]
    shape = np.asarray(volume.shape, dtype=np.float64)
    means = (selected.astype(np.float64) + 0.5) / shape * 2.0 - 1.0
    scales = np.broadcast_to(2.0 / shape, means.shape).copy()
    densities = denoised[tuple(selected.T)].astype(np.float64)
    return GRGaussianState(means, scales, densities, build_knn_graph(means))


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx3 array")
    return array


def build_knn_graph(points: np.ndarray, neighbors: int = 8) -> np.ndarray:
    """Return a symmetric directed k-nearest-neighbor edge list."""

    from scipy.spatial import cKDTree

    points = _points(points, "points")
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(neighbors, len(points) - 1)
    tree = cKDTree(points)
    distances, nearest = tree.query(points, k=k + 1, workers=1)
    distances = np.asarray(distances).reshape(len(points), k + 1)
    nearest = np.asarray(nearest, dtype=np.int64).reshape(len(points), k + 1)
    targets = np.empty((len(points), k), dtype=np.int64)
    for source in range(len(points)):
        candidates = [
            (float(distance), int(target))
            for distance, target in zip(distances[source], nearest[source])
            if int(target) != source
        ]
        candidates.sort()
        if len(candidates) < k:
            raise RuntimeError("nearest-neighbor query omitted valid points")
        targets[source] = [target for _, target in candidates[:k]]
    sources = np.repeat(np.arange(len(points), dtype=np.int64), k)
    directed = np.column_stack((sources, targets.reshape(-1)))
    symmetric = np.concatenate((directed, directed[:, ::-1]), axis=0)
    return np.unique(symmetric, axis=0)


def _ray_weights(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    origins = _points(ray_origins, "ray_origins")
    directions = _points(ray_directions, "ray_directions")
    means = _points(means, "means")
    scales = _points(scales, "scales")
    if origins.shape != directions.shape:
        raise ValueError("ray origins and directions must have matching shape")
    if means.shape != scales.shape or np.any(scales <= 0):
        raise ValueError("Gaussian means and positive scales must match")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("ray directions must be nonzero")
    unit = directions / norms
    relative = means[None, :, :] - origins[:, None, :]
    depth = np.sum(relative * unit[:, None, :], axis=2)
    closest = origins[:, None, :] + np.maximum(depth, 0.0)[:, :, None] * unit[:, None, :]
    normalized = (closest - means[None, :, :]) / scales[None, :, :]
    radial = np.sum(normalized * normalized, axis=2)
    weights = np.exp(-0.5 * radial)
    weights[depth < 0] = 0.0
    return weights


def radiative_splat(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
) -> np.ndarray:
    """Evaluate Beer-Lambert opacity from anisotropic Gaussian ray weights."""

    weights = _ray_weights(ray_origins, ray_directions, means, scales)
    density = np.asarray(densities, dtype=np.float64)
    if density.shape != (weights.shape[1],) or np.any(density < 0):
        raise ValueError("densities must be a nonnegative vector per Gaussian")
    attenuation = weights @ density
    return -np.expm1(-attenuation)


def radiative_splat_density_gradient(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
) -> np.ndarray:
    """Return the per-ray derivative of opacity with respect to density."""

    weights = _ray_weights(ray_origins, ray_directions, means, scales)
    density = np.asarray(densities, dtype=np.float64)
    if density.shape != (weights.shape[1],):
        raise ValueError("densities must contain one value per Gaussian")
    transmission = np.exp(-(weights @ density))
    return transmission[:, None] * weights


def graph_smoothness(densities: np.ndarray, edges: np.ndarray) -> float:
    density = np.asarray(densities, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        return 0.0
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must be an Ex2 array")
    if edge_array.min() < 0 or edge_array.max() >= len(density):
        raise ValueError("graph edge index is outside the density vector")
    difference = density[edge_array[:, 0]] - density[edge_array[:, 1]]
    return float(np.mean(difference * difference))


def volumetric_total_variation(volume: np.ndarray) -> float:
    """Return anisotropic mean total variation over a three-dimensional volume."""

    values = np.asarray(volume, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("TV input must be a finite 3D volume")
    differences = [np.abs(np.diff(values, axis=axis)) for axis in range(3)]
    total = sum(float(item.sum()) for item in differences)
    count = sum(item.size for item in differences)
    return total / max(1, count)


def pixel_graph_densification_scores(
    pixel_gradients: np.ndarray,
    densities: np.ndarray,
    edges: np.ndarray,
    *,
    graph_weight: float = 1.0,
) -> np.ndarray:
    """Combine pixel gradients with local graph density contrast."""

    gradients = np.abs(np.asarray(pixel_gradients, dtype=np.float64))
    density = np.asarray(densities, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    if gradients.shape != density.shape or gradients.ndim != 1:
        raise ValueError("pixel gradients and densities must be matching vectors")
    if graph_weight < 0:
        raise ValueError("graph densification weight must be nonnegative")
    contrast = np.zeros_like(density)
    degree = np.zeros_like(density)
    if edge_array.size:
        if edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError("edges must be an Ex2 array")
        source, target = edge_array[:, 0], edge_array[:, 1]
        if edge_array.min() < 0 or edge_array.max() >= len(density):
            raise ValueError("graph edge index is outside the density vector")
        np.add.at(contrast, source, np.abs(density[source] - density[target]))
        np.add.at(degree, source, 1.0)
    contrast = np.divide(contrast, degree, out=np.zeros_like(contrast), where=degree > 0)
    return gradients + graph_weight * contrast


def gr_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    densities: np.ndarray,
    edges: np.ndarray,
    *,
    graph_weight: float = 0.01,
    volume: np.ndarray | None = None,
    tv_weight: float = 0.0,
) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shape")
    if graph_weight < 0 or tv_weight < 0:
        raise ValueError("regularization weights must be nonnegative")
    if tv_weight and volume is None:
        raise ValueError("TV regularization requires a volume")
    tv = volumetric_total_variation(volume) if volume is not None else 0.0
    return (
        float(np.mean((prediction - target) ** 2))
        + graph_weight * graph_smoothness(densities, edges)
        + tv_weight * tv
    )


def density_loss_gradient(
    prediction: np.ndarray,
    target: np.ndarray,
    jacobian: np.ndarray,
    densities: np.ndarray,
    edges: np.ndarray,
    *,
    graph_weight: float = 0.01,
) -> np.ndarray:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    gradient = 2.0 * jacobian.T @ residual / max(1, residual.size)
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size and graph_weight:
        graph_gradient = np.zeros_like(np.asarray(densities, dtype=np.float64))
        source, target_index = edge_array[:, 0], edge_array[:, 1]
        difference = densities[source] - densities[target_index]
        np.add.at(graph_gradient, source, 2.0 * difference / len(edge_array))
        np.add.at(graph_gradient, target_index, -2.0 * difference / len(edge_array))
        gradient += graph_weight * graph_gradient
    return gradient


def optimize_densities(
    state: GRGaussianState,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    target: np.ndarray,
    *,
    steps: int = 100,
    learning_rate: float = 0.05,
    graph_weight: float = 0.01,
) -> GRGaussianState:
    if steps < 0 or learning_rate <= 0:
        raise ValueError("optimization steps and learning rate must be valid")
    densities = np.asarray(state.densities, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    for _ in range(steps):
        prediction = radiative_splat(
            ray_origins, ray_directions, state.means, state.scales, densities,
        )
        jacobian = radiative_splat_density_gradient(
            ray_origins, ray_directions, state.means, state.scales, densities,
        )
        gradient = density_loss_gradient(
            prediction, target, jacobian, densities, state.edges,
            graph_weight=graph_weight,
        )
        densities = np.maximum(0.0, densities - learning_rate * gradient)
    return GRGaussianState(state.means, state.scales, densities, state.edges)


def adaptive_gaussian_step(
    means: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    density_gradients: np.ndarray,
    *,
    split_gradient: float,
    prune_density: float,
    graph_contrast_weight: float = 0.0,
) -> GRGaussianState:
    """Prune low density and split high-gradient Gaussians deterministically."""

    means = _points(means, "means")
    scales = _points(scales, "scales")
    density = np.asarray(densities, dtype=np.float64)
    gradients = np.asarray(density_gradients, dtype=np.float64)
    if density.shape != (len(means),) or gradients.shape != density.shape:
        raise ValueError("density and gradient vectors must match Gaussian count")
    if prune_density < 0 or split_gradient < 0:
        raise ValueError("adaptive thresholds must be nonnegative")
    scores = pixel_graph_densification_scores(
        gradients, density, build_knn_graph(means),
        graph_weight=graph_contrast_weight,
    )
    output_means: list[np.ndarray] = []
    output_scales: list[np.ndarray] = []
    output_densities: list[float] = []
    for mean, scale, value, score in zip(means, scales, density, scores, strict=True):
        if value <= prune_density:
            continue
        if score >= split_gradient:
            axis = int(np.argmax(scale))
            offset = np.zeros(3)
            offset[axis] = 0.25 * scale[axis]
            output_means.extend((mean - offset, mean + offset))
            output_scales.extend((scale / 1.6, scale / 1.6))
            output_densities.extend((0.5 * value, 0.5 * value))
        else:
            output_means.append(mean)
            output_scales.append(scale)
            output_densities.append(float(value))
    if not output_means:
        raise ValueError("adaptive update pruned every Gaussian")
    mean_array = np.asarray(output_means)
    return GRGaussianState(
        mean_array, np.asarray(output_scales), np.asarray(output_densities),
        build_knn_graph(mean_array, neighbors=min(8, len(mean_array) - 1))
        if len(mean_array) > 1 else np.empty((0, 2), dtype=np.int64),
    )


@dataclass(frozen=True)
class GRGaussianAdapter:
    source_root: Path
    output_root: Path
    descriptor: Any

    def build_command(self, dataset_root: Path, output_root: Path) -> tuple[str, ...]:
        bindings = {
            # Resolving a venv's ``bin/python`` follows its symlink to the
            # system interpreter and drops the venv's installed packages.
            "python_executable": sys.executable,
            # The independent implementation consumes a compact, immutable
            # ray bundle derived from the shared prepared dataset.
            "dataset_root": str(_bundle_root(Path(output_root))),
            "output_root": str(Path(output_root)),
        }
        return tuple(
            argument.format(**bindings) for argument in self.descriptor.commands["train"]
        )

    def prepare(self, dataset: Any, config: Any) -> PreparedRun:
        dataset_root = Path(getattr(dataset, "root", dataset)).resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"GR-Gaussian dataset root is missing: {dataset_root}")
        _prepare_ray_bundle(dataset, self.output_root)
        return PreparedRun(
            model_name=self.descriptor.display_name,
            dataset_name=getattr(dataset, "id", "dataset"),
            source_root=self.source_root.resolve(),
            dataset_root=dataset_root,
            config_sha256=config.sha256,
            quality_config=QualityConfig.from_gala(config),
            seed=0,
            official_command=self.build_command(dataset_root, self.output_root),
            output_root=self.output_root.resolve(),
        )

    def run_reference(self, run: PreparedRun) -> ReferenceArtifact:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                run.official_command, cwd=run.source_root, check=False,
                text=True, capture_output=True, timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("GR-Gaussian CPU reference exceeded the five-minute limit") from error
        if completed.returncode:
            raise RuntimeError(
                "GR-Gaussian reference command failed: "
                + (completed.stderr[-2000:] or completed.stdout[-2000:])
            )
        state_path = run.output_root / "gr_gaussian_state.npz"
        if not state_path.is_file():
            raise RuntimeError("GR-Gaussian reference did not produce a state artifact")
        return ReferenceArtifact(
            run.output_root, state_path, {}, {
                "execution": "independent_cpu_reimplementation",
                "wall_seconds": time.monotonic() - started,
                "command": list(run.official_command),
            },
        )

    def capture_trace(self, run: PreparedRun, sink: Any) -> TraceArtifact:
        reference = self.run_reference(run)
        if reference.volume_path is None:
            raise RuntimeError("GR-Gaussian reference state is missing")
        bundle_path = _bundle_root(run.output_root) / "rays.npz"
        trace = _trace_from_ray_bundle(
            bundle_path,
            model_name=run.model_name,
            dataset_name=run.dataset_name,
            optimized_state_path=reference.volume_path,
        )
        validate_trace(trace)
        trace_root = run.output_root / "trace"
        TraceWriter().write(trace, trace_root)
        from .r2_gaussian import _push_trace_chunks

        _push_trace_chunks(trace, sink)
        return TraceArtifact(trace_root, trace, reference)

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact:
        raise RuntimeError("GR-Gaussian reduction replay requires a captured trace")


def _bundle_root(output_root: Path) -> Path:
    return Path(output_root).resolve() / "input"


def _bundle_identity(dataset: Any) -> dict[str, Any]:
    root = Path(dataset.root).resolve()
    initialization = getattr(dataset, "initialization", None)
    if initialization is None:
        raise ValueError("GR-Gaussian requires a prepared Gaussian initialization")
    initialization = Path(initialization).resolve()
    if not initialization.is_file():
        raise FileNotFoundError(f"GR-Gaussian initialization is missing: {initialization}")
    geometry = dataset.geometry
    return {
        "schema_version": _RAY_BUNDLE_SCHEMA_VERSION,
        "dataset_id": str(dataset.id),
        "dataset_root": str(root),
        "initialization": str(initialization),
        "initialization_size": initialization.stat().st_size,
        "projection_count": len(dataset.projections),
        "detector_shape": list(geometry.detector_shape),
        "volume_shape": list(geometry.volume_shape),
        "DSO": float(geometry.source_object_distance),
        "DSD": float(geometry.source_detector_distance),
        "view_limit": _MAX_RAY_VIEWS,
        "rays_per_view": _RAYS_PER_VIEW,
        "gaussian_limit": _MAX_INITIAL_GAUSSIANS,
    }


def _prepare_ray_bundle(dataset: Any, output_root: Path) -> Path:
    """Derive a bounded CPU ray bundle from a prepared public dataset."""

    identity = _bundle_identity(dataset)
    root = _bundle_root(output_root)
    bundle_path = root / "rays.npz"
    manifest_path = root / "manifest.json"
    if bundle_path.is_file() and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("GR-Gaussian ray bundle manifest is invalid") from error
        if existing == identity:
            with np.load(bundle_path, allow_pickle=False) as bundle:
                required = {
                    "means", "scales", "densities", "ray_origins",
                    "ray_directions", "target", "view_indices",
                }
                if required.issubset(bundle.files):
                    return root
        raise ValueError("GR-Gaussian ray bundle belongs to different prepared inputs")
    if bundle_path.exists() or manifest_path.exists():
        raise ValueError("GR-Gaussian ray bundle is incomplete")
    root.mkdir(parents=True, exist_ok=True)
    means, scales, densities = _compact_initial_gaussians(Path(identity["initialization"]))
    origins, directions, target, view_indices = _sample_dataset_rays(dataset)
    temporary_bundle = root / ".rays.npz.tmp.npz"
    np.savez(
        temporary_bundle, means=means, scales=scales, densities=densities,
        ray_origins=origins, ray_directions=directions, target=target,
        view_indices=view_indices,
    )
    temporary_bundle.replace(bundle_path)
    temporary_manifest = root / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return root


def _compact_initial_gaussians(initialization: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(np.load(initialization, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4 or not np.isfinite(values[:, :4]).all():
        raise ValueError("GR-Gaussian initialization must be finite Nx4 values")
    count = min(_MAX_INITIAL_GAUSSIANS, len(values))
    if count <= 0:
        raise ValueError("GR-Gaussian initialization is empty")
    density = np.maximum(values[:, 3], 0.0)
    selected = np.argpartition(-density, count - 1)[:count]
    selected.sort()
    coordinates = values[selected, :3]
    lower = np.quantile(values[:, :3], 0.01, axis=0)
    upper = np.quantile(values[:, :3], 0.99, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    means = np.clip(2.0 * (coordinates - lower) / span - 1.0, -1.5, 1.5)
    selected_density = density[selected]
    scale = max(float(np.quantile(selected_density, 0.95)), 1.0e-12)
    densities = np.clip(selected_density / scale, 0.0, 1.0)
    scales = np.full_like(means, 2.0 / max(2.0, count ** (1.0 / 3.0)))
    return means, scales, densities


def _sample_dataset_rays(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    available = tuple(dataset.train_indices) or tuple(range(len(dataset.projections)))
    if not available:
        raise ValueError("GR-Gaussian dataset has no training projections")
    selected_positions = np.linspace(
        0, len(available) - 1, min(_MAX_RAY_VIEWS, len(available)), dtype=np.int64,
    )
    selected_views = tuple(available[int(index)] for index in selected_positions)
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    target: list[float] = []
    view_ids: list[int] = []
    height, width = (int(value) for value in dataset.geometry.detector_shape)
    pixel_indices = np.linspace(
        0, height * width - 1, min(_RAYS_PER_VIEW, height * width), dtype=np.int64,
    )
    for view_index in selected_views:
        projection = dataset.projections[int(view_index)]
        values = np.asarray(np.load(projection.path, mmap_mode="r", allow_pickle=False))
        if values.shape != (height, width):
            raise ValueError("GR-Gaussian prepared projection shape changed")
        source, detector_center, tangent, vertical = _cone_beam_frame(
            float(projection.angle_radians), dataset.geometry.source_object_distance,
            dataset.geometry.source_detector_distance,
        )
        for flat_index in pixel_indices:
            row, column = divmod(int(flat_index), width)
            u = (column + 0.5) / width * 2.0 - 1.0
            v = (row + 0.5) / height * 2.0 - 1.0
            detector = detector_center + tangent * u + vertical * v
            origins.append(source)
            directions.append(detector - source)
            target.append(float(values[row, column]))
            view_ids.append(int(view_index))
    raw_target = np.asarray(target, dtype=np.float64)
    lower, upper = np.quantile(raw_target, (0.01, 0.99))
    normalized = np.clip((raw_target - lower) / max(upper - lower, 1.0e-12), 0.0, 1.0)
    return (
        np.asarray(origins, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
        normalized,
        np.asarray(view_ids, dtype=np.int64),
    )


def _cone_beam_frame(angle: float, dso: float, dsd: float) -> tuple[np.ndarray, ...]:
    if not 0.0 < dso < dsd:
        raise ValueError("GR-Gaussian cone-beam geometry requires 0 < DSO < DSD")
    radial = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    source = radial * 2.0
    detector_center = source - radial * (2.0 * dsd / dso)
    tangent = np.asarray([-np.sin(angle), np.cos(angle), 0.0], dtype=np.float64)
    vertical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return source, detector_center, tangent, vertical


def _trace_from_ray_bundle(
    bundle_path: Path,
    *,
    model_name: str,
    dataset_name: str,
    optimized_state_path: Path | None = None,
):
    """Build a trace from immutable rays and optionally optimized Gaussian state.

    A capture following a GR-Gaussian reference run must use that run's final
    Gaussian state.  The immutable ray bundle remains the source for scanner
    geometry and targets, while the state file supplies the trained radiative
    parameters used to select active relations.
    """

    with np.load(bundle_path, allow_pickle=False) as bundle:
        means = np.asarray(bundle["means"], dtype=np.float64)
        scales = np.asarray(bundle["scales"], dtype=np.float64)
        densities = np.asarray(bundle["densities"], dtype=np.float64)
        origins = np.asarray(bundle["ray_origins"], dtype=np.float64)
        directions = np.asarray(bundle["ray_directions"], dtype=np.float64)
        targets = np.asarray(bundle["target"], dtype=np.float64)
    state_source = "initial_ray_bundle"
    if optimized_state_path is not None:
        state_path = Path(optimized_state_path)
        if not state_path.is_file():
            raise FileNotFoundError(f"GR-Gaussian optimized state is missing: {state_path}")
        with np.load(state_path, allow_pickle=False) as state:
            required = {"means", "scales", "densities"}
            if not required.issubset(state.files):
                raise ValueError("GR-Gaussian optimized state lacks radiative fields")
            means = np.asarray(state["means"], dtype=np.float64)
            scales = np.asarray(state["scales"], dtype=np.float64)
            densities = np.asarray(state["densities"], dtype=np.float64)
        state_source = "optimized_reference_state"
    if (
        means.ndim != 2
        or means.shape[1] != 3
        or scales.shape != means.shape
        or densities.shape != (len(means),)
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(densities).all()
        or np.any(scales <= 0)
        or np.any(densities < 0)
    ):
        raise ValueError("GR-Gaussian trace state is not a finite nonnegative Gaussian set")
    if not len(means) or not np.any(densities > 1.0e-8):
        raise ValueError("GR-Gaussian trace state has no radiatively active Gaussian")
    weights = _ray_weights(origins, directions, means, scales)
    builder = TraceBuilder()
    gradients: dict[int, list[int]] = {}
    relation_id = 0
    relation_records: dict[int, list[tuple[int, int, int, int, int]]] = {
        query_id: [] for query_id in range(len(weights))
    }
    # Hardware packets span query lanes for one Gaussian.  Keep the relation
    # selection ray-specific, but group a Gaussian's active lanes under one
    # shared candidate event and a precise partial lane mask.
    for query_base in range(0, len(weights), 8):
        query_ids = tuple(range(query_base, min(query_base + 8, len(weights))))
        members_by_gaussian: dict[int, list[tuple[int, int]]] = {}
        for query_id in query_ids:
            for gaussian_id in _active_relation_indices(
                weights[query_id], densities=densities,
            ):
                members_by_gaussian.setdefault(int(gaussian_id), []).append(
                    (query_id, query_id - query_base)
                )
        for gaussian_id, members in sorted(members_by_gaussian.items()):
            lane_mask = sum(1 << lane for _query_id, lane in members)
            candidate = builder.emit(TraceEvent(
                iteration_id=1, primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
                gaussian_id=gaussian_id, state_version=0,
                resource_class=int(ResourceClass.RELATION), template_id=1,
                field_mask=_TRACE_FIELD_MASK,
            ))
            for query_id, lane in members:
                packet_flags = encode_relation_packet_flags(
                    lane=lane, lane_mask=lane_mask,
                )
                relation = builder.emit(TraceEvent(
                    iteration_id=1, primitive_kind=int(PrimitiveKind.RELATION),
                    query_id=query_id, gaussian_id=gaussian_id, relation_id=relation_id,
                    state_version=0, reduction_key=gaussian_id,
                    resource_class=int(ResourceClass.RELATION), template_id=1,
                    field_mask=_TRACE_FIELD_MASK, flags=packet_flags,
                ), dependencies=[candidate])
                request = builder.emit(TraceEvent(
                    iteration_id=1, primitive_kind=int(PrimitiveKind.CACHE_REQUEST),
                    query_id=query_id, gaussian_id=gaussian_id, relation_id=relation_id,
                    state_version=0, address_token=gaussian_id * 128, data_bytes=128,
                    resource_class=int(ResourceClass.CACHE), template_id=1,
                    field_mask=_TRACE_FIELD_MASK, flags=packet_flags,
                ), dependencies=[relation])
                returned = builder.emit(TraceEvent(
                    iteration_id=1, primitive_kind=int(PrimitiveKind.CACHE_RETURN),
                    query_id=query_id, gaussian_id=gaussian_id, relation_id=relation_id,
                    state_version=0, address_token=gaussian_id * 128, data_bytes=128,
                    resource_class=int(ResourceClass.CACHE), template_id=1,
                    field_mask=_TRACE_FIELD_MASK, flags=packet_flags,
                ), dependencies=[request])
                forward = builder.emit(TraceEvent(
                    iteration_id=1, primitive_kind=int(PrimitiveKind.FORWARD),
                    query_id=query_id, gaussian_id=gaussian_id, relation_id=relation_id,
                    state_version=0, reduction_key=-1,
                    address_token=gaussian_id * 128,
                    resource_class=int(ResourceClass.ISSUE), template_id=1,
                    field_mask=_TRACE_FIELD_MASK, flags=packet_flags,
                ), dependencies=[relation, returned])
                relation_records[query_id].append(
                    (gaussian_id, relation_id, relation, forward, packet_flags)
                )
                relation_id += 1
    for query_id, records in relation_records.items():
        query_base = query_id // 8 * 8
        group_count = min(8, len(weights) - query_base)
        close_flags = encode_relation_packet_flags(
            lane=query_id - query_base, lane_mask=(1 << group_count) - 1,
        )
        relation_event_ids = [record[2] for record in records]
        forward_events = [record[3] for record in records]
        close = builder.emit(TraceEvent(
            iteration_id=1, primitive_kind=int(PrimitiveKind.QUERY_CLOSE),
            query_id=query_id, state_version=0, resource_class=int(ResourceClass.RELATION),
            template_id=1, field_mask=_TRACE_FIELD_MASK, flags=close_flags,
        ), dependencies=relation_event_ids)
        reduction = builder.emit(TraceEvent(
            iteration_id=1, primitive_kind=int(PrimitiveKind.QUERY_REDUCTION),
            query_id=query_id, state_version=0, resource_class=int(ResourceClass.QUERY),
            template_id=1, field_mask=_TRACE_FIELD_MASK,
        ), dependencies=[close, *forward_events])
        consumer = builder.emit(TraceEvent(
            iteration_id=1, primitive_kind=int(PrimitiveKind.CONSUMER),
            query_id=query_id, consumer_id=query_id, state_version=0,
            reduction_key=query_id,
            resource_class=int(ResourceClass.QUERY), template_id=1,
            field_mask=_TRACE_FIELD_MASK,
        ), dependencies=[reduction], payload=[float(targets[query_id])])
        for gaussian_id, emitted_relation_id, _relation, forward, packet_flags in records:
            adjoint = builder.emit(TraceEvent(
                iteration_id=1, primitive_kind=int(PrimitiveKind.ADJOINT),
                query_id=query_id, gaussian_id=gaussian_id,
                relation_id=emitted_relation_id, state_version=0,
                reduction_key=-1, address_token=gaussian_id * 128,
                resource_class=int(ResourceClass.ISSUE),
                template_id=1, field_mask=_TRACE_FIELD_MASK, flags=packet_flags,
            ), dependencies=[consumer, forward])
            gradient = builder.emit(TraceEvent(
                iteration_id=1, primitive_kind=int(PrimitiveKind.GRADIENT_REDUCTION),
                query_id=query_id, gaussian_id=gaussian_id,
                relation_id=emitted_relation_id, state_version=0,
                reduction_key=gaussian_id, resource_class=int(ResourceClass.QUERY),
                template_id=1, field_mask=_TRACE_FIELD_MASK, flags=packet_flags,
            ), dependencies=[adjoint])
            gradients.setdefault(gaussian_id, []).append(gradient)
    begin = builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.UPDATE_BEGIN), state_version=0,
        resource_class=int(ResourceClass.UPDATE), flags=int(UpdateBeginKind.OPTIMIZER),
        field_mask=_TRACE_FIELD_MASK,
    ))
    commits = [
        builder.emit(TraceEvent(
            iteration_id=1, primitive_kind=int(PrimitiveKind.UPDATE_COMMIT),
            gaussian_id=gaussian_id, state_version=0,
            resource_class=int(ResourceClass.UPDATE), field_mask=_TRACE_FIELD_MASK,
        ), dependencies=[begin, *gradient_events])
        for gaussian_id, gradient_events in sorted(gradients.items())
    ]
    builder.emit(TraceEvent(
        iteration_id=1, primitive_kind=int(PrimitiveKind.UPDATE_END), state_version=0,
        reduction_key=begin, resource_class=int(ResourceClass.UPDATE),
        flags=int(UpdateBeginKind.OPTIMIZER), field_mask=_TRACE_FIELD_MASK,
    ), dependencies=commits or [begin])
    return builder.finish(metadata={
        "model_id": "gr_gaussian",
        "model": model_name,
        "dataset_id": dataset_name,
        "dataset": dataset_name,
        "initial_gaussian_count": int(len(densities)),
        "ray_bundle_schema_version": _RAY_BUNDLE_SCHEMA_VERSION,
        "radiative_state_source": state_source,
        "relation_selection": {
            "method": "cumulative_radiative_mass",
            "minimum_retained_mass": _MIN_RETAINED_RADIATIVE_MASS,
            "minimum_relations_per_ray": min(map(len, relation_records.values())),
            "maximum_relations_per_ray": max(map(len, relation_records.values())),
        },
        "capture_backend": "independent_cpu_ray_bundle",
        "formal_performance_eligible": False,
        "result_scope": "quick_cpu_trace_validation",
        "trace_window": {
            "schema_version": "gala-iteration-window-v1",
            "result_scope": "quick_trace_validation",
            "formal_performance_eligible": False,
            "iterations": [1],
            "selection": "deterministic_cpu_ray_bundle",
        },
        "iteration_event_counts": {"1": builder._next_event_id},
    })


def _active_relation_indices(
    weights: np.ndarray, *, densities: np.ndarray | None = None,
) -> np.ndarray:
    effective = np.asarray(weights, dtype=np.float64)
    if effective.ndim != 1 or not len(effective) or not np.isfinite(effective).all():
        raise ValueError("GR-Gaussian relation weights must be finite and non-empty")
    if densities is not None:
        density = np.asarray(densities, dtype=np.float64)
        if (
            density.shape != effective.shape
            or not np.isfinite(density).all()
            or np.any(density < 0)
        ):
            raise ValueError("GR-Gaussian relation densities must match finite weights")
        effective = effective * density
    if np.any(effective < 0):
        raise ValueError("GR-Gaussian radiative contributions must be nonnegative")
    active = np.flatnonzero(effective > 0.0)
    if not len(active):
        raise ValueError("GR-Gaussian ray has no radiatively active relation")
    ordered = active[np.argsort(-effective[active], kind="stable")]
    cumulative = np.cumsum(effective[ordered], dtype=np.float64)
    retained = _MIN_RETAINED_RADIATIVE_MASS * cumulative[-1]
    count = int(np.searchsorted(cumulative, retained, side="left")) + 1
    return ordered[:count]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.evaluate is not None:
        state = np.load(args.evaluate, allow_pickle=False)
        print({"gaussians": int(len(state["densities"]))})
        return 0
    if args.dataset is None or args.output is None:
        raise SystemExit("--dataset and --output are required for optimization")
    bundle = np.load(args.dataset / "rays.npz", allow_pickle=False)
    means = np.asarray(bundle["means"], dtype=np.float64)
    state = GRGaussianState(
        means, np.asarray(bundle["scales"], dtype=np.float64),
        np.asarray(bundle["densities"], dtype=np.float64), build_knn_graph(means),
    )
    state = optimize_densities(
        state, bundle["ray_origins"], bundle["ray_directions"], bundle["target"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "gr_gaussian_state.npz", means=state.means, scales=state.scales,
        densities=state.densities, edges=state.edges,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
