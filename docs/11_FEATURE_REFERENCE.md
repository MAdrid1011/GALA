# Feature Reference

This page summarizes the implemented compiler, execution, and evaluation
features. The simulator operates on lowered event and task metadata; it does
not include a source-language frontend or RTL implementation.

## Dataflow Primitives

CLAMP programs are represented by nine primitive semantics. Trace events,
dependencies, task packets, and compute templates carry their lowered effects
through the simulator.

| Primitive | Modeled behavior | Primary implementation |
| --- | --- | --- |
| `EXPAND` | Generate candidate and accepted query-Gaussian relations | `gala_sim/clamp/events.py`, `gala_sim/trace/virtual.py` |
| `ROUTE` | Preserve query, Gaussian, relation, and reduction ownership across forward and adjoint paths | `gala_sim/timing/packets.py`, `gala_sim/timing/engine.py` |
| `FOLD` | Close keyed query and Gaussian reductions after all inputs retire | `gala_sim/clamp/tasks.py`, `gala_sim/timing/engine.py` |
| `AWAIT` | Release a consumer only after its query dependencies complete | `gala_sim/clamp/tasks.py` |
| `HOLD` | Retain a versioned Gaussian state while declared uses remain | `gala_sim/clamp/worksets.py`, `gala_sim/timing/modules/hardware.py` |
| `RELEASE` | Close residency after the last declared use and active read | `gala_sim/clamp/worksets.py`, `gala_sim/timing/modules/hardware.py` |
| `TRANSFORM` | Execute coordinate, matrix, state, and update transforms | `gala_sim/timing/config.py`, model compute templates in `configs/architecture/gala.yaml` |
| `EVALUATE` | Execute relation, projection, loss, and adjoint operations | `gala_sim/timing/modules/hardware.py`, model compute templates |
| `COMBINE` | Combine attributes, reductions, and gradients with explicit resource use | `gala_sim/timing/modules/hardware.py`, model compute templates |

`PrimitiveKind` defines the stable event schema used by captured and virtual
traces. `RelationPacketPlan` binds scalar event identities into physical
relation packets without changing their dependencies.

## Compiler Metadata

Query-load metadata tracks outstanding forward, consumer, and adjoint work for
each logical query. `FusionIssueScheduler` combines predicted remaining work,
exact readiness, conflicts, age, and bounded queue-head visibility when
selecting work.

Semantic-workset metadata groups cache requests by `(gaussian_id,
state_version)`. Each request records its ordinal, total uses, remaining uses,
and last-use marker. `SemanticPlacement` assigns the resulting work to the
configured Compute Pod topology using measured template demand.

The CUDA overlay tools build isolated extensions below the local workspace.
They expose independent query and semantic compiler controls and preserve a
common uninstrumented GPU baseline. Model-specific probes use CUDA events,
serial variants, GPU isolation checks, numerical-equivalence checks, and an
inactivity watchdog.

## Architecture Model

The event-driven cycle engine models these bounded units:

- A relation constructor with a Seed FIFO, support lanes, and relation-window
  backpressure.
- A fusion issue unit with query-state counters, load forecasts, conflict
  checks, bounded candidate FIFOs, and per-path issue ports.
- Four Gaussian-semantic caches with versioned directories, miss merging,
  active records, use-counted release, and bounded scope multicast.
- Four reconfigurable Compute Pods. Template stages reserve FMA,
  transcendental, reduction, register, scratch, and cluster-issue resources.
- A bidirectional query unit for forward reductions, consumer readiness, and
  adjoint replay.
- A reconstruction update unit that waits for old-version reads and gradients
  before update commits and set mutation.
- Partitioned, banked shared SRAM, a segmented interconnect, and an optional
  native Ramulator 2 memory backend.

The authoritative resource values are in
`configs/architecture/gala.yaml`. The checked resource summary is in
`configs/architecture/gala-resource-usage.json`.

## Evaluation Variants

Variant bits use `ABCD` order:

| Bit | Feature |
| --- | --- |
| A | Compiler query-load rules |
| B | Compiler semantic worksets |
| C | Overlap-guided hardware issue |
| D | Gaussian-semantic hardware residency |

The supported matrix is `0000`, `1000`, `1010`, `0100`, `0101`, `1100`, and
`1111`. C requires A, and D requires B. Compiler-only variants compare with the
same workload's GPU base; hardware variants compare with Base ASIC on the same
trace. The matrix runner verifies identical event counts and rejects a combined
configuration that is slower than either component.

The representative experiment selects phase-stratified, dependency-complete
trace windows and runs the full seven-variant matrix on every selected window.
The matrix audit checks all model and dataset combinations against the same
strategy, baseline, composition, and target rules.

## Integrations

The adapter registry supports R2-Gaussian, FaCT-GS, Exact-GS, and an independent
GR-Gaussian implementation. Dataset adapters support Chest, FIPS Walnut, and
HDTomo-USB. Model and dataset selection is composable, producing twelve
registered combinations under one architecture configuration.

Generated model checkouts, datasets, extensions, traces, profiles, and results
remain under the ignored `workspace/` tree. The public package includes the
simulator, configuration, interface documentation, and tests, but no generated
performance, RTL, synthesis, area, or power artifacts.
