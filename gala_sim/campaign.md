# campaign.py

`run_campaign(model_id, dataset_id, workspace=...)` executes one bounded CPU
validation campaign. It prepares the selected dataset, writes a model- and
dataset-bound trace, replays Base ASIC and the seven canonical mechanism
variants, and computes the necessary hardware upper-bound report.

`run_campaigns(selections, ...)` applies the same workflow to an explicit list
of `(model_id, dataset_id)` pairs. Generated traces and reports belong under the
ignored workspace. The CPU memory backend is deterministic and suitable for
engineering validation; formal GPU and Ramulator evidence uses the existing
official capture and replay paths.

Pass `representative_archive=True`, or invoke `gala-sim campaign-ablation
--representative-archive`, to construct a bounded virtual packet archive and
replay its representative packet window. The canonical seven-variant
representative experiment is exposed by `gala-sim representative-ablation`.
It uses the calibrated R2+Chest window-selection, admission, arbitration, and
median aggregation policy for every model/dataset combination. It is a complete
representative-window simulation, not a reduced variant matrix; it remains
explicitly separate from formal GPU performance evidence.

After the twelve representative documents exist, audit that one policy was
used everywhere with:

```bash
gala-sim representative-matrix-audit \
  --results-root workspace/results \
  --output workspace/results/representative-ablation-matrix-v1/audit.json
```

Pass `official_trace=True`, or invoke `gala-sim campaign-ablation
--official-trace`, to run the selected model's pinned entrypoint with its trace
hooks and then feed that model-specific trace into the same necessary-bound and
seven-variant replay stages. Official capture and representative archive modes
are mutually exclusive. A capture-overhead wall time is retained for audit but
is never used as the GPU Base latency.

For a substantially faster development pass, add `--fast-capture` to
`--official-trace`. This captures compact virtual packets, validates the full
packet/lifecycle stream, selects a dependency-closed representative tile/brick,
and runs all seven variants on that end-to-end trace. The result scope is
`representative_speedup_validation` and is never formal GPU evidence; omit the
flag for the complete official trace workflow.

Compiler-only GPU timing is supplied separately as
`<measurement-root>/<model>/<dataset>/gpu-compiler-measurement.json`. Pass the
root with `--gpu-measurement-root`; when omitted, official campaigns look below
`workspace/results/gpu-measurements/`. The artifact contains uninstrumented
CUDA-event medians for `gpu_base`, `1000`, `0100`, and `1100`. The campaign
rejects a different model, dataset, iteration window, operation set, variant
order, comparison baseline, non-isolated GPU, failed numerical-equivalence
check, or trace-instrumented timing. Source and repository hashes may be kept
as provenance but are not acceptance gates.

All four model adapters emit this artifact from their serial timing matrices:

```bash
python -m gala_sim.tools.r2_gaussian_training_matrix --help
python -m gala_sim.tools.fact_gs_training_probe --help
python -m gala_sim.tools.exact_gs_training_probe --help
python -m gala_sim.tools.gr_gaussian_training_probe --help
```

R2-Gaussian and Exact-GS use isolated CUDA extension overlays. Generate and
build them below `workspace/build/` without editing the pinned source checkout:

```bash
python -m gala_sim.tools.r2_gaussian_compiler_overlay --help
python -m gala_sim.tools.exact_gs_compiler_overlay --help
```

Use the same measured iteration window for the GPU matrix and official trace.
For example, a matrix with two warm-up iterations and eight requested
iterations measures iterations `3:8`, so its matching campaign uses
`--capture-iteration-range 3:8`.

Official captures are stored below
`workspace/traces/<model>/<dataset>/official-capture-<start>-<end>-v1/`. The
default official window is `1:1`; `--capture-iteration-range START:END` selects
a later or full window, and execution stops immediately after that window. The process runner
requires an idle GPU, terminates only its own process group after five minutes
without trace, CPU, or GPU progress, and preserves at least 1 GiB of free GPU
memory and 8 GiB of available host memory. Its process report and logs remain
under the ignored capture directory.

Official adapters use bounded raw-column streaming and avoid materializing a
second copy of the event columns. For fast development captures, the pinned
runner also supports compact packets:

R²-Gaussian virtual captures disable its unconditional image-evaluation pass;
that pass concatenates every train/test projection and can consume several GiB
of otherwise unused device memory. The R² child process also defaults to
fragmentation-resistant CUDA allocation (`expandable_segments` with a bounded
split size). These guards apply only to trace capture and do not change the
un-instrumented reference or compiler timing paths.

```bash
python -m gala_sim.adapters.trace_runner \
  --virtual-capture --virtual-capture-audit-only \
  --packet-archive-root workspace/traces/<model>/<dataset>/virtual-archive \
  --capture-config configs/architecture/gala.yaml \
  --capture-iteration-range START:END --stop-after-capture-range ...
```

Validate the archive, select a representative packet window, and replay the
seven variants with `trace-archive-validate`, `representative-packet-plan`,
`representative-packet-trace`, and `ablation --quick-validation`. Compact
replay accepts `archive-ablation --max-events` and `--prefetch-chunks` to bound
memory while overlapping archive decompression with cycle simulation.
Registered upstream adapters prefer
`workspace/build/envs/<model>/bin/python` when present. An explicitly supplied
Python executable has higher priority; the invoking interpreter is only a
fallback for workspaces without a model environment.
