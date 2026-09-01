# GPU Compiler Ablation Probe

`python -m gala_sim.tools.gpu_compiler_ablation` measures the GPU software base
and compiler variants `1000`, `0100`, and `1100` for identical work. It warms
each path, rotates execution order, and reports repeated medians.

The runner verifies matching query, Gaussian, candidate, relation, consumer,
loss, and gradient semantics. Compiler variants report speedup only against the
same-work GPU base. Probe output remains a local diagnostic and is not a
hardware-cycle result.
