# Ablation Matrix and Output Contract

## 1. Optimization Controls

The simulator defines four bits, but they are not four independent mechanisms.
A and B produce compiler metadata; C and D are the hardware mechanisms that
consume the corresponding metadata.

| Bit | Configuration key | Enabled behavior | Disabled behavior |
| --- | --- | --- | --- |
| A | `compiler.query_load_rules` | Generate query-load and progress rules | Preserve legal dependencies and use arrival order |
| B | `compiler.semantic_worksets` | Generate Gaussian residency, reuse, and release metadata | Provide no cross-task semantic residency metadata |
| C | `architecture.overlap_guided_issue` | Use historical overlap prediction and conflict-free fused issue | Issue one legal task per cycle in arrival order |
| D | `architecture.semantic_residency` | Enable the semantic directory, residency, miss merging, and scope multicast | Serve every state request through the base memory path |

Bits use `ABCD` order. `0000` is Base ASIC and `1111` is full GALA. Disabling a
control removes only that optimization; it may not remove tasks, dependencies,
queues, numerical work, or hardware state required for correctness.

Hardware C is valid only with A, and hardware D is valid only with B. The runner
must reject `C=1, A=0`, `D=1, B=0`, and any bit pattern outside the canonical set.

## 2. Canonical Matrix

One validated trace is replayed in this fixed order:

| Variant | Meaning |
| --- | --- |
| `0000` | Base ASIC |
| `1000` | Compiler A only |
| `1010` | Compiler A with hardware C |
| `0100` | Compiler B only |
| `0101` | Compiler B with hardware D |
| `1100` | Compiler A and B together |
| `1111` | Full GALA |

Every row uses the same hardware resources, memory mapping, input events, and
numerical semantics. The runner rejects missing, duplicate, reordered, and
noncanonical rows. Configuration hashes are audit records, not execution gates;
uniform configuration is enforced through resolved parameter snapshots and
resource-closure checks.

The query-scheduling closure point is `1010`, semantic-residency closure is
`0101`, and the joint compiler point is `1100`. Variants `1000`, `0100`, and
`1100` enable neither C nor D and must not be described as standalone hardware
experiments without their compiler prerequisites. Full-entry GALA cycles must
exactly match the `1111` row.

## 3. Speedup Baselines

Compiler-only variants `1000`, `0100`, and `1100` execute on the GPU software
path and report speedup only over the same workload's GPU base:

```text
speedup_vs_gpu_base = gpu_base_seconds / gpu_variant_seconds
```

If the GPU base uses a credible AGX Orin estimate, the method and uncertainty
interval must be recorded and the value labeled as an estimate rather than a
measurement. ASIC cycles may not replace GPU time for a compiler-only variant.

Base ASIC reports absolute cycles, converted time, and platform-level speedup
over an AGX Orin GPU base for the same work when that measurement is available.
Only hardware variants `1010`, `0101`, and `1111` report speedup over Base ASIC:

```text
speedup_vs_base_asic = cycles_0000 / cycles_variant
```

Every result row names its comparison baseline explicitly. For `1000`, `0100`,
and `1100`, `speedup_vs_base_asic` is empty. For `1010`, `0101`, and `1111`,
`speedup_vs_gpu_base` is empty. Oracle results are labeled separately and are
excluded from geometric means of implemented mechanisms.

## 4. Output Files

Each run directory contains:

| File | Contents |
| --- | --- |
| `manifest.json` | Code, model, data, configuration, device, and environment identities |
| `cycles.json` | End-to-end cycles and per-module cycle decomposition |
| `stalls.parquet` | Stall causes and cycle intervals |
| `memory_requests.parquet` | Address, operation, byte count, arrival cycle, and Ramulator 2 return cycle for every off-chip request |
| `events.json` | Event counts, cache statistics, and issue statistics |
| `quality.json` | PSNR, SSIM, and LPIPS for the CUDA reference and functional replay |
| `gpu_reference.json` | Local GPU time, calibration items, and matching Orin conversion when available |
| `status.json` | Run state, failure reason, and acceptance result |

The active matrix is summarized in `ablation.csv`. Each row binds the model,
dataset, variant bits, baseline, applicable speedup, quality difference, and
configuration identity. Any AGX Orin estimate retains its estimate status and
uncertainty. Result generation reads only machine-readable files; manuscript
tables may not be filled manually.

## 5. Consistency Assertions

- `1111` end-to-end cycles equal the full GALA entry-point result.
- All seven variants have identical dynamic valid-relation, numerical-task, and
  update-commit counts.
- The variant set and order are exactly
  `0000,1000,1010,0100,0101,1100,1111`.
- Every variant with C also has A, and every variant with D also has B.
- Only scheduling order, cache events, stall causes, and cycles may change.
- `1000`, `0100`, and `1100` compare only with the same workload's GPU base.
- `1010`, `0101`, and `1111` compare only with the same trace's Base ASIC.
- Base ASIC and GPU base platform comparisons use identical work.
- A variant that exceeds any quality limit is excluded from performance
  aggregation.
