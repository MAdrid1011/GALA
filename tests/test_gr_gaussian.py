from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from gala_sim.adapters.gr_gaussian import (
    GRGaussianAdapter, adaptive_gaussian_step, build_knn_graph, denoised_point_cloud_initialization,
    density_loss_gradient, graph_smoothness, gr_loss,
    pixel_graph_densification_scores, radiative_splat,
    radiative_splat_density_gradient, volumetric_total_variation,
    _active_relation_indices, _trace_from_ray_bundle,
)
from gala_sim.adapters.datasets import DatasetManifest, DatasetProjection, ScannerGeometry
from gala_sim.adapters.registry import model_descriptors
from gala_sim.clamp import PrimitiveKind
from gala_sim.clamp.events import has_relation_packet_metadata
from gala_sim.config import load_config
from gala_sim.trace import validate_trace
from gala_sim.timing.packets import RelationPacketPlan


ROOT = Path(__file__).resolve().parents[1]


def test_knn_graph_is_symmetric_without_self_edges() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    edges = build_knn_graph(points, neighbors=1)
    assert np.all(edges[:, 0] != edges[:, 1])
    assert {tuple(edge) for edge in edges} == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_knn_graph_scales_without_a_quadratic_distance_matrix() -> None:
    axis = np.linspace(-1.0, 1.0, 20)
    points = np.asarray(np.meshgrid(axis, axis, axis)).reshape(3, -1).T

    edges = build_knn_graph(points, neighbors=8)

    assert edges.shape[0] <= len(points) * 16
    assert np.all(edges[:, 0] != edges[:, 1])
    assert {tuple(edge[::-1]) for edge in edges} == {tuple(edge) for edge in edges}


def test_radiative_splat_matches_beer_lambert_composition() -> None:
    origins = np.asarray([[0.0, 0.0, -2.0]])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    means = np.asarray([[0.0, 0.0, 0.0]])
    scales = np.asarray([[1.0, 1.0, 1.0]])
    densities = np.asarray([0.5])
    prediction = radiative_splat(origins, directions, means, scales, densities)
    assert np.allclose(prediction, 1.0 - np.exp(-0.5), atol=1e-12)


def test_density_gradient_matches_finite_difference() -> None:
    origins = np.asarray([[0.0, 0.0, -2.0]])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    means = np.asarray([[0.0, 0.0, 0.0]])
    scales = np.asarray([[1.0, 1.0, 1.0]])
    densities = np.asarray([0.4])
    analytic = radiative_splat_density_gradient(
        origins, directions, means, scales, densities,
    )[0, 0]
    epsilon = 1e-6
    upper = radiative_splat(origins, directions, means, scales, densities + epsilon)[0]
    lower = radiative_splat(origins, directions, means, scales, densities - epsilon)[0]
    assert np.isclose(analytic, (upper - lower) / (2 * epsilon), rtol=1e-5)


def test_graph_regularizer_and_adaptive_update_are_deterministic() -> None:
    edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    assert graph_smoothness(np.asarray([1.0, 3.0]), edges) == 4.0
    means = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    scales = np.ones((2, 3))
    densities = np.asarray([0.8, 0.001])
    gradients = np.asarray([2.0, 0.0])
    updated = adaptive_gaussian_step(
        means, scales, densities, gradients,
        split_gradient=1.0, prune_density=0.01,
    )
    assert updated.means.shape == (2, 3)
    assert np.all(updated.densities > 0.01)


def test_graph_loss_density_gradient_matches_finite_difference() -> None:
    densities = np.asarray([0.4, 0.7])
    edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    prediction = np.asarray([0.3, 0.6])
    target = np.asarray([0.2, 0.5])
    jacobian = np.asarray([[0.7, 0.2], [0.1, 0.8]])
    analytic = density_loss_gradient(
        prediction, target, jacobian, densities, edges, graph_weight=0.2,
    )
    epsilon = 1e-6
    numeric = []
    residual = prediction - jacobian @ densities
    for index in range(2):
        upper = densities.copy()
        lower = densities.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_prediction = residual + jacobian @ upper
        lower_prediction = residual + jacobian @ lower
        numeric.append((
            gr_loss(upper_prediction, target, upper, edges, graph_weight=0.2)
            - gr_loss(lower_prediction, target, lower, edges, graph_weight=0.2)
        ) / (2 * epsilon))
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_denoised_initialization_rejects_an_artifact_and_builds_graph() -> None:
    prior = np.zeros((3, 3, 3), dtype=np.float64)
    prior[1, 1, 1] = 1.0
    prior[1, 1, 2] = 0.8
    prior[0, 0, 0] = 100.0
    state = denoised_point_cloud_initialization(
        prior, 2, density_threshold=0.01, artifact_quantile=0.5,
    )
    assert state.means.shape[1] == 3
    assert np.all(state.densities < 100.0)
    assert np.all(state.edges[:, 0] != state.edges[:, 1])


def test_pixel_graph_score_and_total_variation_are_analytic() -> None:
    scores = pixel_graph_densification_scores(
        np.asarray([0.1, 0.2]), np.asarray([0.0, 1.0]),
        np.asarray([[0, 1], [1, 0]]), graph_weight=0.5,
    )
    assert np.allclose(scores, [0.6, 0.7])
    ramp = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    assert np.isclose(volumetric_total_variation(ramp), 7.0 / 3.0)


def test_relation_selection_retains_bounded_radiative_mass_without_fixed_count() -> None:
    diffuse = np.asarray([0.40, 0.30, 0.20, 0.09, 0.01])
    concentrated = np.asarray([0.995, 0.003, 0.001, 0.0009, 0.0001])

    diffuse_selected = _active_relation_indices(diffuse)
    concentrated_selected = _active_relation_indices(concentrated)

    for values, selected in (
        (diffuse, diffuse_selected), (concentrated, concentrated_selected),
    ):
        assert values[selected].sum() >= 0.99 * values.sum()
        assert len(selected) == 1 or values[selected[:-1]].sum() < 0.99 * values.sum()
    assert len(diffuse_selected) != len(concentrated_selected)


def test_adapter_executes_cpu_bundle_and_captures_a_valid_trace(tmp_path: Path) -> None:
    dataset_root = tmp_path / "prepared"
    projection_root = dataset_root / "projections"
    projection_root.mkdir(parents=True)
    projection = projection_root / "view.npy"
    np.save(projection, np.arange(16, dtype=np.float32).reshape(4, 4))
    initialization = dataset_root / "init_fixture.npy"
    np.save(initialization, np.asarray([
        [-1.0, -1.0, -1.0, 0.2],
        [-0.2, 0.1, 0.0, 0.4],
        [0.3, -0.2, 0.5, 0.8],
        [1.0, 1.0, 1.0, 0.6],
    ]))
    dataset = DatasetManifest(
        "fixture", dataset_root,
        (DatasetProjection(projection, 0.0, (4, 4), "float32"),),
        ScannerGeometry((4, 4), (4, 4, 4), 5.0, 7.0),
        (0,), (), None, "identity", {}, initialization,
    )
    output = tmp_path / "output"
    adapter = GRGaussianAdapter(ROOT, output, model_descriptors()["gr_gaussian"])
    run = adapter.prepare(dataset, load_config(ROOT / "configs/architecture/gala.yaml"))
    assert run.dataset_root == dataset_root.resolve()
    assert run.official_command[0] == sys.executable
    assert (output / "input/rays.npz").is_file()

    class Sink:
        chunk_events = 32

        def __init__(self) -> None:
            self.events = 0
            self.closed = False

        def push(self, events, _dependencies, _payload) -> None:
            self.events += len(events)

        def close(self) -> None:
            self.closed = True

    sink = Sink()
    artifact = adapter.capture_trace(run, sink)
    report = validate_trace(artifact.trace)
    assert (output / "gr_gaussian_state.npz").is_file()
    assert artifact.reference.gpu_reference["execution"] == "independent_cpu_reimplementation"
    assert report.counts["UPDATE_COMMIT"] > 0
    assert artifact.trace.metadata["trace_window"]["result_scope"] == "quick_trace_validation"
    assert artifact.trace.metadata["radiative_state_source"] == "optimized_reference_state"
    relation_stage_kinds = {
        PrimitiveKind.RELATION, PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN,
        PrimitiveKind.FORWARD, PrimitiveKind.ADJOINT, PrimitiveKind.GRADIENT_REDUCTION,
    }
    assert all(
        has_relation_packet_metadata(int(row["flags"]))
        for row in artifact.trace.events
        if PrimitiveKind(int(row["primitive_kind"])) in relation_stage_kinds
    )
    assert RelationPacketPlan.from_trace(artifact.trace, query_lanes=8).relation_packet_count > 0
    assert sink.events == artifact.trace.event_count
    assert sink.closed


def test_trace_capture_uses_optimized_radiative_state_for_relation_selection(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "rays.npz"
    np.savez(
        bundle,
        means=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        scales=np.ones((2, 3), dtype=np.float64),
        densities=np.asarray([1.0, 0.0]),
        ray_origins=np.asarray([[0.0, 0.0, -2.0]]),
        ray_directions=np.asarray([[0.0, 0.0, 1.0]]),
        target=np.asarray([0.5]),
    )
    optimized = tmp_path / "optimized.npz"
    np.savez(
        optimized,
        means=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        scales=np.ones((2, 3), dtype=np.float64),
        densities=np.asarray([0.0, 1.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )

    trace = _trace_from_ray_bundle(
        bundle,
        model_name="GR-Gaussian",
        dataset_name="fixture",
        optimized_state_path=optimized,
    )

    relations = trace.events[
        trace.events["primitive_kind"] == int(PrimitiveKind.RELATION)
    ]
    assert set(relations["gaussian_id"].tolist()) == {1}
    forwards = trace.events[
        trace.events["primitive_kind"] == int(PrimitiveKind.FORWARD)
    ]
    adjoints = trace.events[
        trace.events["primitive_kind"] == int(PrimitiveKind.ADJOINT)
    ]
    consumers = trace.events[
        trace.events["primitive_kind"] == int(PrimitiveKind.CONSUMER)
    ]
    assert np.all(forwards["reduction_key"] == -1)
    assert np.all(adjoints["reduction_key"] == -1)
    assert np.array_equal(forwards["address_token"], forwards["gaussian_id"] * 128)
    assert np.array_equal(adjoints["address_token"], adjoints["gaussian_id"] * 128)
    assert np.array_equal(consumers["reduction_key"], consumers["query_id"])
    assert trace.metadata["radiative_state_source"] == "optimized_reference_state"
