from __future__ import annotations

import numpy as np

from gala_sim.adapters.gr_gaussian import (
    adaptive_gaussian_step, build_knn_graph, denoised_point_cloud_initialization,
    density_loss_gradient, graph_smoothness, gr_loss,
    pixel_graph_densification_scores, radiative_splat,
    radiative_splat_density_gradient, volumetric_total_variation,
)


def test_knn_graph_is_symmetric_without_self_edges() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    edges = build_knn_graph(points, neighbors=1)
    assert np.all(edges[:, 0] != edges[:, 1])
    assert {tuple(edge) for edge in edges} == {(0, 1), (1, 0), (1, 2), (2, 1)}


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
