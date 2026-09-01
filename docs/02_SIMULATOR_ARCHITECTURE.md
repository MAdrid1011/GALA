# Simulator Software Architecture

## 1. Overall Structure

The simulator separates functional execution from cycle execution behind one
event contract. The functional executor runs the real model and produces
dynamic relations, primitive tasks, numerical results, and the final
reconstruction. The cycle executor consumes the same events and simulates GALA.
Both paths share event IDs, dependencies, state versions, and configuration
identity.

```text
Official Model and Dataset
        |
        v
Model Adapter -> CLAMP Task Builder -> Functional GPU Executor
        |                    |                  |
        |                    |                  +-> Reconstruction and Metrics
        |                    v
        +------------> Device Trace Buffers
                             |
                             v
                    Async Chunk Transfer
                             |
                             v
                 Event-driven Cycle Simulator
                             |
                             v
              Cycles, stalls, module counters
```

Functional execution never reads cycle results to alter the model's
mathematical path. Cycle execution never fabricates values that approximate the
functional path. When schedule order can affect numerical output, the cycle
executor emits a submission order and the functional executor replays legal
reductions in that order.

## 2. Python Package Boundary

The initial package layout and responsibilities are:

```text
gala_sim/
  cli/                 command entry points
  config/              typed configuration and validation
  adapters/            model and dataset adapters
  clamp/               combinators, primitives, task packets, analyses
  functional/          GPU numerical executor and quality path
  trace/               device buffers, schemas, chunk writer and reader
  timing/
    kernel/            event queue, clock domains and backpressure
    modules/           one implementation per hardware module
    memory/            on-chip banks and Ramulator 2 bridge
  ablation/            switch semantics and matrix runner
  metrics/             cycle, stall, PSNR, SSIM and LPIPS
  results/             manifest and table writers
  tools/               conversion and preflight utilities
tests/
configs/
```

`timing/modules` may contain only modules named by the hardware contract.
`trace`, `results`, and `tools` are simulator support layers and are never
counted as simulated hardware.

## 3. Functional Executor

Model adapters take ownership of these boundaries around the official training
entry point:

1. Read projections, scan geometry, initial Gaussians, and training settings.
2. Map relation generation, contribution evaluation, query reduction,
   consumers, adjoint work, updates, and set mutation to CLAMP tasks.
3. Run the original numerical formulas for the original number of steps.
4. Write real dynamic events on the device instead of reconstructing average
   events in Python.
5. Save the final volume, model checkpoints, and metric inputs.

The official CUDA extension remains the numerical reference. The adapter adds a
side-band trace output without removing original computation. The upstream
commit and minimal extension patch are recorded, and disabling tracing must
restore output identical to upstream.

## 4. Trace Layer

The trace layer uses preallocated structured device buffers. Every event
contains at least:

| Field | Meaning |
| --- | --- |
| `event_id` | Globally unique event ID |
| `iteration_id` | Reconstruction iteration |
| `primitive_kind` | CLAMP primitive type |
| `query_id` | Query ID, or the typed empty value |
| `gaussian_id` | Gaussian ID, or the typed empty value |
| `state_version` | Gaussian state version |
| `relation_id` | Relation ID shared by related forward and adjoint work |
| `consumer_id` | Local-consumer instance ID |
| `reduction_key` | Query or Gaussian reduction key |
| `resource_class` | Target execution resource |
| `dependency_begin`, `dependency_count` | Range in the dependency array |
| `template_id` | Exact arithmetic-template ID |
| `field_mask` | Gaussian fields accessed |
| `address_token` | Deterministic SRAM or DRAM address-layout token |
| `payload_offset` | Numerical payload range needed for replay |

When a device buffer fills, a CUDA stream copies the completed chunk to pinned
host memory. The CPU consumes the preceding chunk while the GPU produces the
next. The columnar NumPy or Arrow IPC format is frozen by field type as
`gala-clamp-events-v2`. `UPDATE_BEGIN` and `UPDATE_END` enclose an optimizer
commit or set mutation. `UPDATE_END.field_mask == 0` is a no-op version barrier:
it closes the old version and advances state without a zero-mask write. Per-event
JSON output and per-event Python callbacks are prohibited.

For dense R²-Gaussian raster and voxel queries, the producer may also emit
`gala-trace-virtual-packet-v1` work buffers containing the official CUDA
`point_list`, `point_key`, and complete `uint32` valid masks. A set bit denotes
a real Gaussian/query relation. `VirtualTracePacket` is a producer-side work
package, not a cycle trace and not a substitute for lifecycle validation.

The event materializer assigns globally contiguous `event_id` values, preserves
global dependency IDs and cross-packet versions, and emits candidate, relation,
cache, forward, query, consumer, adjoint, and gradient chains. Update and set
mutation events still come from the upper lifecycle materializer.
`VirtualTraceLifecycleValidator` tracks per-iteration candidate/relation/adjoint
counts, active Gaussians, versions, and updates, and writes a compact ledger at
closure. It rejects open cross-iteration work, inactive Gaussians, duplicate
lineage, and version jumps. The ledger never replaces real dependencies, cache
traffic, or memory returns.

Production/consumption distance is bounded by the configured in-flight packet
count. Produced and consumed packets, relations, physical bytes, and peak bytes
are recorded. An incomplete work-buffer packet fails the run and may not be
treated as a complete formal trace.

## 5. Cycle Executor

The cycle executor advances to the next module state transition. The global
kernel maintains a small wakeup heap; each hardware module maintains its input,
in-flight, completion, and resource-availability state.

```python
class CycleModule(Protocol):
    def next_wakeup(self) -> int | None: ...
    def accept(self, batch: EventBatch, cycle: int) -> AcceptResult: ...
    def advance(self, cycle: int) -> ModuleOutputs: ...
    def snapshot_counters(self) -> CounterBlock: ...
```

`accept` reports acceptance, rejection, and blocking causes. `advance` emits
only pipeline completion, resource release, state update, and output events that
occur at the given cycle. A module may not inspect global future state to bypass
ports, queues, or backpressure.

## 6. Numerical and Cycle Replay

Most optimizations change event start cycles without changing numerical output.
Because reduction submission order can affect FP32 rounding, the cycle executor
records query reduction, Gaussian gradient reduction, and update order. The
functional executor performs deterministic replay from that log and produces
the full GALA quality result.

Debug mode permits per-event comparison on a short real sample. Formal mode
stores submission order, module statistics, and optional sampled traces without
retaining every numerical payload.

## 7. Operating Modes

| Mode | Purpose | Allowed simplification |
| --- | --- | --- |
| `native-reference` | Run official CUDA and produce reference quality and end-to-end time | No GALA cycle simulation |
| `trace-capture` | Run real numerical work and produce complete events | Optional visualization and intermediate checkpoints may be disabled |
| `cycle-replay` | Run one or more variants from a validated trace | Does not repeat GPU numerical execution |
| `functional-replay` | Replay reductions in selected cycle order and produce quality output | Does not recapture events |
| `campaign` | Sequence reference, trace, seven variants, and quality summary | May not bypass an acceptance gate |

One validated trace can serve all seven canonical variants and avoid repeating
expensive forward and backward model execution. Reuse requires identical model
commit, dataset checksum, training configuration, adapter version, and CLAMP
schema. Validation covers stable Gaussian IDs, parent/child lineage,
cross-update dependencies, and state-version caching. Query closure does not
release cached state; an old version is released only after its update writeback
completes. For one memory-mapped trace, the cycle entry point performs one
structural validation and then resets the seven variants in canonical order
using the compact NumPy dependency index.
