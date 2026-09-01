# Implementation Workflow

## Acquire and Validate Assets

Select model and dataset identifiers from the asset catalog. Acquire immutable
sources into the ignored workspace, verify repository commits or archive
checksums, validate licenses, and prepare the dataset without modifying raw
files.

## Freeze Inputs

Bind the model implementation, dataset manifest, training configuration,
random state, Python environment, CUDA environment, quality protocol, and
architecture configuration into a local input manifest. Portable references
are used for files below the repository or workspace roots.

## Execute the Numerical Path

Run the model's configured reference path and produce reconstruction and metric
inputs. Trace hooks remain side-band instrumentation. Short fixture runs may be
used to debug an adapter but are never substituted for catalogued execution.

## Capture and Validate Events

Capture candidate, relation, forward, reduction, consumer, adjoint, gradient,
update, and set-mutation events through bounded device buffers. Validate event
identity, dependencies, lineage, state versions, and lifecycle closure before
cycle replay.

## Establish Base ASIC

Replay `0000` through every hardware module using the frozen architecture and
memory configuration. Record cycles, stalls, memory requests, and peak resource
occupancy. Base ASIC retains the complete event set and numerical path.

## Evaluate Mechanisms

Evaluate compiler A and matching hardware C, then compiler B and matching
hardware D. Compiler-only paths compare with the same-work GPU base; hardware
paths compare with Base ASIC on the same trace. Bounds retain every physical
resource and remain separate from implemented results.

When shared engineering is needed, apply it uniformly, repeat input and quality
checks, and regenerate every affected variant.

## Run the Canonical Ablation

Execute `0000`, `1000`, `1010`, `0100`, `0101`, `1100`, and `1111` in that
order. Validate mechanism prerequisites, identical mathematical event counts,
quality limits, and exact equality between `1111` and the full GALA entry point.

## Add Integrations

A model integration adds a provenance manifest, registered adapter, command and
stage mapping, and fixtures. A dataset integration adds a source manifest,
registered parser, normalization logic, and validation fixtures. Both use the
same hardware configuration and output contracts.

Generated runs, acquisition reports, traces, profiles, and measurements remain
under `workspace/`. The public repository documents interfaces and methods,
not local execution progress.
