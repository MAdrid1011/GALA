# Performance Engineering

## Correctness-Preserving Optimization

Host and GPU engineering may improve data loading, kernel launch structure,
allocation reuse, synchronization, trace transfer, compression, and timing
engine implementation. It must preserve model inputs, numerical operations,
events, dependencies, state versions, and modeled hardware resources.

An optimization is applied to every affected variant. A change that benefits
only one mechanism by altering its workload is not a fair comparison.

## Profiling Boundaries

GPU profiling records end-to-end time and phase boundaries from the same model,
dataset, and configuration used by reference execution. Warmup and repetition
counts come from configuration. Profiler replay and instrumentation overhead
are excluded from raw application timing.

CPU simulator profiling separates trace decoding, validation, packetization,
event scheduling, module advancement, memory integration, and output writing.

## Compact Packet Archive

Compact archives use bounded chunks whose target byte size is configured by
`trace.archive_chunk_bytes`. Compression runs asynchronously with a bounded
number of in-flight chunks. Archive metadata records schema, packet counts,
event counts, uncompressed bytes, compressed bytes, and checksums.

Changing chunk or worker parameters may change host throughput and storage but
must not change materialized events or cycle results.

## Stream Validator Configuration

`trace.chunk_events` bounds the event count in one validation chunk and
`trace.max_inflight_chunks` bounds producer/consumer distance. Validation state
for dependencies, relations, Gaussian versions, and open updates persists
across chunks.

The configuration is a software memory bound, not simulated SRAM. Validation
must produce identical acceptance and summary values across legal chunk sizes.

## Long-Run Preflight

Before a long model or replay task, a bounded sample estimates input rate,
event rate, host memory, device memory, and wall time. The preflight checks for
unrelated GPU compute processes before launching a GPU workload and reports
actionable failures without terminating other processes.

## Runtime Throughput Diagnostic

Cycle replay can emit low-frequency progress containing completed events,
simulated cycles, wall time, and projected completion. Diagnostic stability is
evaluated only at quiescent iteration boundaries and requires a configured
sequence of converged windows.

Stopping after diagnostic convergence produces a diagnostic artifact, not a
complete cycle result. The ordinary replay path always consumes the full event
stream.

## Parallel Work

Independent ablation variants and archive chunks may execute concurrently when
they do not share mutable state. Worker counts are bounded explicitly. Parallel
execution must produce byte-equivalent manifests and cycle values to serial
execution, apart from wall-clock metadata.
