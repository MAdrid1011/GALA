# Parameter Registry

## Registry Rules

`configs/architecture/gala.yaml` is the authoritative typed parameter
registry. Every entry contains a value, unit, source, scope, and freeze state;
parameters with legal tuning freedom also define an allowed range.

A configuration is accepted only when every required entry is present, units
and types match the schema, values lie within their ranges, and the aggregate
resources fit `configs/architecture/gala-resource-usage.json`.

Source code must not duplicate a configurable latency, width, capacity, bank
count, quality threshold, or runtime bound. Derived values may be computed from
registered parameters and input dimensions.

## Architecture Defaults

The registry groups hardware values by responsibility:

| Group | Examples |
| --- | --- |
| `clock` and `top` | frequency, Pod count, total shared SRAM |
| `shared_sram` | region allocation and bank ports |
| `relation` | seed FIFO and support lanes |
| `issue` | query-state entries, banks, candidate width, FIFO capacity |
| `cache` | instances, directory banks, fill credits, multicast scope |
| `compute` | clusters, lanes, templates, initiation intervals |
| `query` and `update` | banks, ports, queues, commit latency |
| `interconnect` | segments, widths, arbitration, transfer latency |
| `memory` | channels, transaction bytes, queue limits |

These values are shared by every workload and ablation. Internal organization
may change only within an explicitly declared allowed range and without
exceeding the top-level resource envelope.

## Software and Trace Parameters

Software-scoped entries control capture and simulator execution rather than
modeled hardware. They include trace chunk bytes, events per chunk, in-flight
chunks, archive compression, worker counts, and diagnostic reporting periods.
Changing them may affect host runtime or storage but must not change events,
cycles, quality, or resource accounting.

## Run Strategy

Run-strategy values define warmup, measurement repetition, utilization
sampling, and long-run preflight. They qualify measurement procedure and never
alter a model's training schedule or the accelerator's cycle behavior.

The command layer may override a strategy value for diagnostics. Such an
override is written into the local output manifest and does not modify the
tracked architecture configuration.

## Runtime Throughput Diagnostic

Throughput diagnostics sample completed events, simulated cycles, and wall
time at configured event and time intervals. Stability requires multiple
quiescent iteration boundaries and simultaneous convergence of cycle projection
and processing rate. Diagnostic early termination is explicitly identified in
its local output and does not create a complete cycle result.

## Validation and Provenance

`gala-sim config-check` resolves the registry and reports configuration
readiness. Resource closure verifies that regional SRAM allocations sum to the
declared total and that replicated structures fit their budgets.

Sources refer to design-document headings, public specifications, or
machine-readable resource files. Generated run histories are not parameter
sources. A configuration change updates its design source and validation tests
in the same change.
