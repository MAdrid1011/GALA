"""Independent CPU reference for graph-based radiative Gaussian splatting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from gala_sim.metrics import QualityConfig
from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact


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

    points = _points(points, "points")
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(neighbors, len(points) - 1)
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(squared, np.inf)
    nearest = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
    edges = {
        (source, int(target))
        for source, row in enumerate(nearest)
        for target in row
    }
    edges |= {(target, source) for source, target in tuple(edges)}
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


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
            "python_executable": str(Path(sys.executable).resolve()),
            "dataset_root": str(Path(dataset_root)),
            "output_root": str(Path(output_root)),
        }
        return tuple(
            argument.format(**bindings) for argument in self.descriptor.commands["train"]
        )

    def prepare(self, dataset: Any, config: Any) -> PreparedRun:
        dataset_root = Path(getattr(dataset, "root", dataset)).resolve()
        return PreparedRun(
            self.descriptor.display_name, getattr(dataset, "id", "dataset"),
            self.source_root.resolve(), dataset_root, config.sha256,
            QualityConfig.from_gala(config), 0,
            self.build_command(dataset_root, self.output_root),
        )

    def run_reference(self, run: PreparedRun) -> ReferenceArtifact:
        raise RuntimeError("GR-Gaussian reference execution requires a prepared ray bundle")

    def capture_trace(self, run: PreparedRun, sink: Any) -> TraceArtifact:
        raise RuntimeError("GR-Gaussian trace capture requires configured CLAMP hooks")

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact:
        raise RuntimeError("GR-Gaussian reduction replay requires a captured trace")


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
