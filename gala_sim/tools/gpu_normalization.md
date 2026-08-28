# GPU stage normalization

`python -m gala_sim.tools.gpu_normalization` combines CUDA Event stage times,
NSYS kernel-coverage evidence, NCU counters, and matching local/AGX Orin
calibration vectors. Stage weights are derived from measured work and sum to
one. Exhaustive profiles use calibration rates only within their measured
batch range. Representative sampling may use a nearest calibration endpoint
outside that range and records the extrapolation mode. Hashes are retained as
metadata and never gate normalization.

`--nsys-profile` is required. Every kernel launched inside each selected outer
training-iteration NVTX range must belong to exactly one detailed `gala_stage`
range. CUDA Event residual time is retained as a boundary and host-idle timing
diagnostic; it is not used as the kernel-coverage gate.

The `--orin-calibration` argument is optional so a run can be recorded before
the Orin vector is available.  Missing Orin data, missing counter evidence,
incomplete NSYS coverage, out-of-range batches, or unclassified XU instructions produce
`provisional_normalization` and no Orin-equivalent time.  Such output is not a
formal performance result. A representative NCU report can still produce
`local_sampling_status=passed` when all local stage vectors close; this status
does not claim an AGX Orin conversion or exhaustive NCU launch coverage.
