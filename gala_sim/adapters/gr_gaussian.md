# gr_gaussian.py

## External Interfaces

This module is an independent implementation of the mathematical core described
by GR-Gaussian. It is not author-released source.

`build_knn_graph()` constructs a symmetric spatial graph. `radiative_splat()`
evaluates anisotropic Gaussian attenuation with Beer-Lambert composition, and
`radiative_splat_density_gradient()` returns its analytic density Jacobian.

`graph_smoothness()`, `gr_loss()`, `density_loss_gradient()`, and
`optimize_densities()` implement graph-regularized density fitting.
`denoised_point_cloud_initialization()` removes high-residual prior artifacts,
`pixel_graph_densification_scores()` combines pixel gradients and neighborhood
density contrast, and `volumetric_total_variation()` supplies the volume
regularizer. `adaptive_gaussian_step()` performs deterministic density pruning
and graph-aware splitting. `GRGaussianAdapter` exposes the common model adapter
boundary.

## Internal Helpers

Ray weights project each Gaussian mean to the forward half-ray and evaluate its
anisotropic radial distance. Input validators reject non-finite arrays,
non-positive scales, invalid graph indices, and invalid optimization bounds.
