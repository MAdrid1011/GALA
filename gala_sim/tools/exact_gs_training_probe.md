# exact_gs_training_probe.py

## External Interface

`run_probe(...)` executes an uninstrumented Exact-GS training prefix with one
compiler variant. It measures the post-warmup interval with CUDA events,
captures the final optimizer state and gradients, records GPU isolation and
watchdog state, and stops before evaluation, checkpoint, or save operations.

`run_matrix(...)` executes `gpu_base`, `1000`, `0100`, and `1100` serially with
rotated repeat order. It writes per-sample records, a matrix summary, and a
standard `gala-gpu-compiler-measurement-v1` artifact. A matrix is eligible for
performance comparison only when GPU isolation, workload identity, and
numerical equivalence pass.

```bash
python -m gala_sim.tools.exact_gs_training_probe --help
```

## Internal Helpers

The training overlay removes per-iteration host synchronization and scalar
telemetry from the measured interval without changing the optimizer
transaction. `_ExactStageProfiler` optionally characterizes CUDA stages outside
the primary matrix. `summarize_matrix()` computes medians, gradient checks,
compiler speedups, and combined-mechanism monotonicity. The watchdog interrupts
only its owning process after the configured inactivity interval and enforces a
host-memory reserve.
