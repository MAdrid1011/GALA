# exact_gs_compiler_overlay.py

## External Interface

`prepare_overlay(source_root, output_root, glm_root=...)` copies the pinned
Exact-GS CUDA extension into a new output directory and applies deterministic
compiler transformations. The source checkout is not modified. The generated
manifest records source and output hashes, transformations, GLM provenance,
and runtime controls.

`build_overlay(output_root, python_executable, cxx=..., nvcc=...)` builds the
isolated extension with one compiler job and records the selected toolchain.
It returns the unique generated extension path.

The generated extension reads two environment controls:

- `GALA_QUERY_WARP_REDUCE` enables query-side compiler transformations.
- `GALA_SEMANTIC_WARP_REDUCE` enables semantic-side compiler transformations.

The query kernel selects a compile-time specialization from detector size. A
large grid uses a higher-occupancy launch bound, medium grids use numerically
guarded pairwise mean-gradient accumulation, and smaller grids retain direct
atomic accumulation. The dispatch depends on `W * H`, not a dataset name.

```bash
python -m gala_sim.tools.exact_gs_compiler_overlay --help
```

## Internal Helpers

`render_exact_backward_overlay()` applies source-fragment-checked CUDA
transformations and emits all compiler-control launch variants. Accumulation
helpers preserve direct atomics when their feature is disabled. Fast arithmetic
is selected at compile time for enabled paths. `_resolve_glm_root()` validates
the required header tree.
