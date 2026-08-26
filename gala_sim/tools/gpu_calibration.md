# GPU calibration

`python -m gala_sim.tools.gpu_calibration --config <yaml> --output <json>` runs the same float32 FMA, EXP, LOG, RCP, SQRT, memory-copy, atomic, launch, and synchronization suite on the selected CUDA device. Run the unchanged suite in both the local and AGX Orin software environments; a local vector alone cannot produce a formal Orin normalization.

Calibration batch selection uses each real kernel launch's `grid * block` work size, not an aggregate stage counter. Values outside the measured batch range remain provisional. Atomic calibration applies only when its operation type, address space, and contention semantics match the profiled instruction.
