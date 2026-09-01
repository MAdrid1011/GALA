# Quality Validation and Acceptance

## Numerical Reference

An official-source model adapter executes its pinned author path with the
catalogued dataset, training configuration, and random state. The independent
GR-Gaussian implementation executes the published equations under its own
identified implementation provenance.

Trace capture uses the same numerical inputs. With tracing disabled, an
official-source adapter must follow the uninstrumented author path. Functional
replay follows the legal reduction and update order emitted by the cycle model.

## Unified Quality Protocol

PSNR is evaluated over the complete reference volume using the data range in
the dataset manifest. SSIM uses the configured 3D window, sigma, covariance,
and boundary policy. LPIPS uses configured orthogonal slices, normalization,
network, and weight identities.

The default common settings are a data range of `[0, 1]`, SSIM window 11,
sigma 1.5, reflected boundaries, and LPIPS AlexNet version 0.1. Dataset
manifests may provide a different physical input range, but normalization and
the evaluated range are recorded before execution.

Model-specific metrics may be added but do not replace the common protocol.
Predictions are not silently clipped unless the dataset manifest explicitly
defines clipping as part of preprocessing.

## Trace Validation

Validation checks unique IDs, existing dependencies, relation lineage,
monotonic state versions, query and gradient closure, consumer ordering,
release after final use, update after old-version drain, and complete set
mutation. Structural checks apply to every event; payload sampling is an
additional numerical audit.

Streaming and materialized validation must agree. A malformed or incomplete
packet stops the run and cannot be counted as complete evidence.

## Cycle Validation

Module tests cover port limits, queue capacity, bank conflicts, backpressure,
completion order, cache release, update barriers, and memory return ordering.
Repeated replay of the same trace and configuration must produce identical
cycles.

The canonical ablation set is
`0000,1000,1010,0100,0101,1100,1111`. C requires A, D requires B, every row
preserves the mathematical event set, and `1111` matches the full entry point.

## Acceptance Gates

| Gate | Condition |
| --- | --- |
| Input | Source, data, configuration, and license identities validate |
| Functional | The model completes its configured numerical path |
| Quality | Common metric differences remain within configured limits |
| Trace | Dependencies, versions, lineage, and event coverage validate |
| Cycle | Module accounting closes and resource limits are respected |
| Ablation | Canonical variants, prerequisites, and baselines are correct |

Quality failure invalidates a performance conclusion. A result that violates a
resource bound, exceeds its constrained Oracle, or changes the event set is a
modeling error and must be diagnosed before use.
