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
replay its representative packet window. This is explicitly scoped as
`quick_cpu_packet_archive_validation`; it validates the production archive
contract but is not GPU performance evidence.

Pass `official_trace=True`, or invoke `gala-sim campaign-ablation
--official-trace`, to run the selected model's pinned entrypoint with its trace
hooks and then feed that model-specific trace into the same necessary-bound and
seven-variant replay stages. Official capture and representative archive modes
are mutually exclusive. A capture-overhead wall time is retained for audit but
is never used as the GPU Base latency.

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
Registered upstream adapters prefer
`workspace/build/envs/<model>/bin/python` when present. An explicitly supplied
Python executable has higher priority; the invoking interpreter is only a
fallback for workspaces without a model environment.
