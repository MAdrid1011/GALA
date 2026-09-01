# GPU profiling campaign

`python -m gala_sim.tools.gpu_profile_campaign --campaign <yaml> --output <json>` validates and materializes the frozen profiling plan. Supplying one full CUDA Event stage profile plus all `--nsys-profile` files validates measured evidence instead.

Each campaign defines its complete CUDA Event interval and representative NSYS
and NCU iterations for initialization, densification phases, periodic
evaluation, collection, and final reconstruction. Representative evidence
never replaces the complete CUDA Event time series.

For bounded NSYS collection, run `stage_runner` in `representative` mode with
`--capture-profiler-api`, and configure NSYS with
`--capture-range=cudaProfilerApi --capture-range-end=repeat:N`. The non-shutdown
form lets the complete configured process finish naturally after the last
range. NCU representative runs also require `--capture-profiler-api` and
`--profile-from-start off`. The model executes one continuous training process;
only profiler collection starts and stops at selected iteration boundaries.
Process-wide `gala_ncu_stage` ranges cover autograd kernels launched from worker
threads. Campaign mode requires a verified input freeze and rejects training
argument changes other than the output directory.

The NCU signature plan requests the `SourceCounters` section in addition to the explicit aggregate metrics. This is required for dynamic SASS opcode classification; source text alone is not sufficient because it has no executed instruction counts. Evidence remains ineligible for formal performance until every requested iteration, exact NSYS kernel assignment, required stage-role, NCU counter, dynamic SASS classification, and local/Orin calibration gate passes.
