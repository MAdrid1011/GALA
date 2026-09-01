# GALA Hardware Contract

## Resource Envelope

The modeled accelerator contains four Pods, 20 compute clusters, 320 FP32 FMA
lanes, 40 transcendental lanes, eight external memory channels, and 2.75 MiB of
shared SRAM. `configs/architecture/gala.yaml` and
`configs/architecture/gala-resource-usage.json` are the machine-readable
resource definitions.

All variants share the same top-level resources. Enabling a compiler or
hardware mechanism changes metadata and legal scheduling behavior, not the
number of Pods, execution lanes, SRAM bytes, memory channels, or ports.

## Relation Construction

The relation constructor receives candidate Gaussian/query pairs, evaluates
the configured support test, and emits each valid relation exactly once.
Backpressure from the relation window and downstream queues stops admission;
candidates are not dropped or summarized.

The seed FIFO and support lanes are bounded resources. Candidate, accepted,
rejected, and blocked counts are reported by the timing model.

## Fusion Issue

The base issue policy observes physical queue heads and issues one legal task
per cycle. Compiler query metadata enables bounded load information. Hardware C
uses that metadata for overlap prediction and conflict-free issue while
retaining candidate width, queue capacity, bank ports, and execution resources.

Candidate selection may inspect only the configured bounded head window. It may
not scan arbitrary future events or bypass dependencies. C is invalid without
compiler mechanism A.

## Semantic Cache

Each Pod has a directory keyed by `(gaussian_id, state_version)`. Entries track
resident state, a fill in progress, remaining uses, active readers, and closing
state. A miss allocates a bounded fill; requests for the same key may merge with
that fill. A resident entry is released only after its declared uses and active
reads reach zero.

Compiler semantic worksets provide reuse and release metadata. Hardware D uses
that metadata for residency, fill merging, and scope multicast. D is invalid
without compiler mechanism B. The base path fetches state through the ordinary
memory path without semantic retention.

## Semantic Fusion Bundle

Semantic bundle metadata identifies tasks that share a Gaussian state and can
legally reuse one resident entry. Bundle metadata occupies the configured
control SRAM budget and does not remove events, reads, writes, or dependencies.

Multicast is permitted only among ready consumers in the same declared scope.
Every destination still consumes its configured queue and execution capacity.

## Compute Pods

Compute Pods execute exact template IDs for forward, query reduction, local
consumer, adjoint, gradient reduction, and update work. Templates declare FMA,
transcendental, reduction, register, and scratch requirements. Arbitration
selects ready physical queue heads that fit the available resources.

Pipeline latency and initiation interval are configuration values. A task holds
its resources until the modeled completion event releases them.

## Query and Update Units

The query unit models banked query state, reduction dependencies, consumer
completion, and adjoint feedback. The update unit commits a new Gaussian state
version only after all reads and gradients for the old version have drained.
Set mutation preserves parent/child lineage and never reuses an active identity.

## Shared SRAM

The shared SRAM allocation is:

| Region | Bytes |
| --- | ---: |
| Active Gaussian state | 524,288 |
| Relation window | 655,360 |
| Query volume | 524,288 |
| Gradient and update state | 655,360 |
| Index graph | 262,144 |
| Control metadata | 262,144 |

Each access maps to a physical bank and competes for the configured read or
write port. A logical hit does not bypass bank contention.

## Compressed Relation Records

The relation window stores compact physical records plus separately budgeted
control metadata. Records retain producer, forward, consumer, and adjoint
ownership until every reference completes. Capacity is enforced on physical
records, not an average relation count.

## Capacity Continuation Windows

Large producer packets are divided into bounded continuation windows. A
continuation changes storage and streaming boundaries only: it does not split,
duplicate, or omit mathematical events. Dependencies and state versions remain
global across windows.

## Off-Chip Memory

Logical requests are decomposed into transactions accepted by the native
Ramulator 2 frontend. A request completes only after all of its transactions
return. The bridge records address, operation, bytes, arrival cycle, and return
cycle and uses the memory configuration without a fixed-latency shortcut.

## Required Counters

Every module reports accepted work, blocked cycles by cause, queue occupancy,
bank conflicts, issued operations, and completions. Cache modules additionally
report hits, misses, merged fills, multicast destinations, and releases.
