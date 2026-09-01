# Reproduction Targets and Evidence

## Workload Catalog

The adapter catalog covers four model implementations:

| Identifier | Provenance |
| --- | --- |
| `r2_gaussian` | Pinned R2-Gaussian author repository |
| `fact_gs` | Pinned FaCT-GS author repository |
| `exact_gs` | Pinned Exact-GS author repository |
| `gr_gaussian` | Independent implementation of DOI `10.1109/TCI.2026.3701585` |

The dataset catalog covers Chest, the FIPS Walnut cone-beam CT dataset, and the
USB sample from HDTomo record `4822516`. Model and dataset adapters are
composable through a shared geometry and projection contract.

An author repository and an independent implementation are different evidence
types. Manifests preserve that distinction, and public documentation does not
attribute independently written code to a paper's authors.

## Execution Scope

A workload includes relation construction, forward contributions, query
reduction, local consumers, adjoint computation, optimizer updates, Gaussian
set mutation, and final volume generation. All variants use the same complete
event set and numerical configuration.

Dataset conversion records the source archive identity, geometry convention,
view partition, preprocessing, and reference-volume identity. Conversion does
not overwrite raw downloads.

## Performance Contract

Every hardware result reports absolute end-to-end cycles. Time conversion uses
the configured GALA clock. Compiler-only speedup uses measured end-to-end GPU
time for the same model, dataset, training configuration, and input state.
Hardware speedup uses Base ASIC cycles from the same trace.

Published targets are comparison references, not calibration inputs. Hardware
parameters come from the design, module characterization, or public memory
timing. No global scale factor is applied to match an expected speedup.

An AGX Orin comparison requires a matching measured calibration vector for
arithmetic, memory, atomics, launch, and synchronization behavior. Public peak
specifications may be reported as device metadata but do not produce a measured
end-to-end time.

## Quality Contract

Each numerical run reports PSNR, SSIM, and LPIPS using the common protocol in
[Quality Validation](09_VALIDATION_AND_ACCEPTANCE.md). A scheduled replay must
remain within the configured difference limits from its software reference.

Scheduling and residency do not change the mathematical relation set. Allowed
differences are limited to documented FP32 operation semantics and legal
reduction order.

## Evidence Qualification

A qualified result binds:

- model implementation and source identity
- dataset archive and prepared-manifest identity
- training and random-state configuration
- trace schema and event coverage
- architecture and memory configuration
- quality metrics and comparison baseline

The repository defines how these records are produced but does not version
generated performance tables or local execution histories.
