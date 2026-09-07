# representative_ablation.py

## External Interface

`run_representative_ablation(...)` executes the canonical seven-variant cycle
matrix over every phase-stratified window in a representative plan. It accepts
a packet archive, plan, validated architecture configuration, workload
identity, output path, optional reusable trace directory, optional GPU compiler
measurement, and bounded replay settings.

The function returns and writes a `gala-representative-ablation-v1` document.
The document contains the common strategy, selected windows, per-window event
counts and cycles, median aggregate cycles, GPU and Base ASIC speedups,
composition checks, and static-anchor assessments.

Errors are raised for malformed or non-adjacent windows, unsupported replay
limits, changed event sets, incomplete variant matrices, and a combined
mechanism that regresses either component.

The command-line interface is available through:

```bash
gala-sim representative-ablation --help
```

## Internal Helpers

`_window_entries()` validates and deduplicates planned iteration windows.
`_simulation_cycle_config()` constructs the calibrated portable memory timing
configuration. `_median()` applies the experiment's integer median convention,
and `_gpu_speedups()` normalizes supported GPU measurement schemas.
