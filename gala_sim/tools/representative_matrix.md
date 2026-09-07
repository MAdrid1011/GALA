# representative_matrix.py

## External Interface

`audit_representative_matrix(...)` selects one completed representative result
for each registered model and dataset combination. It checks the common
strategy metadata, complete seven-variant shape, joint non-regression, source
selection, target status, and the frozen R2-Gaussian/Chest calibration.

GPU compiler evidence is loaded from uninstrumented measurement documents.
Qualifying measurements must pass platform isolation, workload identity,
numerical equivalence, canonical ordering, baseline, and static-anchor checks.
Selection favors broader iteration coverage and repeat count rather than a
larger speedup.

The function writes a `gala-representative-matrix-audit-v1` JSON report. The
command-line interface is available through:

```bash
gala-sim representative-matrix-audit --help
```

## Internal Helpers

Candidate discovery is restricted to the selected workload directories.
`_select_result()` prefers documents carrying the complete strategy metadata.
`_select_gpu_measurement()` rejects incomparable measurements before ranking
eligible candidates. `_composition_ok()` requires monotonic combined software
and hardware mechanisms.
