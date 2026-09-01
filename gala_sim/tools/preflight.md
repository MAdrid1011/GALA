# preflight.py

## External Interfaces

`predict_runtime()` projects a complete run from measured work after warmup.
`decide_long_run()` applies configured duration and utilization gates.
`sample_gpustat()` captures GPU utilization, memory, and compute-process
ownership. `run_native_preflight()` validates an input manifest and runs a
bounded upstream calibration.

`ComputeProcess`, `GpuSample`, `PreflightDecision`, and
`NativePreflightReport` are the machine-readable record types.

## Internal Helpers

Helpers inspect process CPU time, parse available timing output, and supervise
only the launched process group. An unrelated compute process prevents launch;
the preflight never terminates it.
