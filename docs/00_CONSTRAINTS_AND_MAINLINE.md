# Project Constraints and Mainline

## Evaluation Boundary

GALA Simulator evaluates reconstruction quality and execution performance. It
models the complete forward, reduction, consumer, adjoint, optimizer, and
Gaussian-set mutation path. Shortening training, replacing inputs, reducing the
relation set, or omitting backward or update work changes the workload and is
not a valid comparison.

The repository does not model or report area, power, energy, or energy
efficiency. Generated measurements and run histories remain outside version
control.

## Canonical Mechanisms

Variant bits use `ABCD` order. A supplies compiler query-load metadata and C is
the hardware that consumes it. B supplies semantic-workset metadata and D is
the hardware residency mechanism that consumes it. C therefore requires A,
and D requires B.

The canonical configurations are `0000`, `1000`, `1010`, `0100`, `0101`,
`1100`, and `1111`. Other bit patterns are rejected by the public runner. The
full GALA entry point and `1111` must produce identical cycle counts.

## Architecture Boundary

The modeled hardware contains these modules:

- relation constructor
- overlap-guided fusion issue unit
- semantic cache and residency controller
- reconfigurable Gaussian compute Pods
- bidirectional query execution unit
- reconstruction update unit
- shared SRAM, segmented interconnect, and off-chip memory interface

The resource envelope fixes the Pod count, execution paths, on-chip SRAM, and
off-chip interface. Queue depths, bank organization, table sizes, and internal
pipeline registers are configuration parameters and remain identical across
models, datasets, and ablations.

Adapters, trace buffers, event schedulers, validators, and writers are software
support. They do not add simulated capacity or bypass module latency.

## Input Authenticity

Model and dataset inputs come from catalogued public sources or a clearly
identified independent implementation. Each source is bound to a repository
commit, DOI, publisher record, or content checksum. Random tensors and reduced
fixtures are permitted only in unit tests and are never substituted for a
catalogued workload.

The cycle model executes every relation, dependency, queue operation, bank
access, conflict, memory request, and state-version update. Batching may improve
host execution efficiency but may not replace events with average counts, hit
rates, or prescribed throughput.

## Numerical Correctness

The functional and cycle paths consume the same relation and task identities.
Scheduling may alter only a legal execution order. It may not alter the model's
loss, optimizer, initialization, training schedule, densification, split, or
prune rules.

When order affects FP32 reduction, the cycle path emits a submission order and
the functional path replays that order. Quality validation precedes use of any
performance output.

## Parameter Provenance

Values that affect cycles, resources, scheduling, memory behavior, data
partitioning, or quality come from typed configuration. Each parameter records
its unit, source, scope, and allowed range where applicable. Constants derived
solely from array dimensions or protocol encodings may remain in code.

Hardware parameters are workload-independent. A dataset-specific result may
not be matched by tuning latency, capacity, bandwidth, or event counts.

## Simulation Efficiency

The timing engine advances to observable state changes rather than polling idle
cycles. Structured arrays, bounded packet streams, and compiled numerical
kernels are used on hot paths. Modules wake when input arrives, a resource is
released, memory returns, or control state changes.

Trace capture uses preallocated device buffers and asynchronous chunk transfer.
Per-event GPU synchronization and per-event Python callbacks are prohibited in
the production path.

## Fair Optimization

Mechanism bounds retain real dependencies, issue widths, ports, banks, queues,
capacity, and memory bandwidth. They may change candidate selection but cannot
create events, remove consumers, or expand resources.

Engineering optimizations to data loading, kernel organization, allocation,
synchronization, trace handling, or simulator execution apply equally to the
base and every variant. They must preserve the algorithm, event set, numerical
output, and resource envelope.

Compiler-only variants compare with the same workload's GPU software base.
Hardware variants compare with Base ASIC on the same trace. Times from different
workloads are never divided, and configured targets never populate result
fields.

## Repository Policy

Git contains implementation, tests, schemas, configuration, and design
documentation. The ignored `workspace/` contains upstream checkouts, datasets,
builds, traces, profiles, checkpoints, and generated results. Tracked files use
portable paths and contain no machine-specific execution history.
