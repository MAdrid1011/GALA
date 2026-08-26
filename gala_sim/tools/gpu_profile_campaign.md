# GPU profiling campaign

`python -m gala_sim.tools.gpu_profile_campaign --campaign <yaml> --output <json>` validates and materializes the frozen profiling plan. Supplying one full CUDA Event stage profile plus all `--nsys-profile` files validates measured evidence instead.

The R2-Gaussian + Chest campaign times every iteration from 1 through 30,000. NSYS and NCU evidence is collected at versioned representative iterations for initialization, pre/early/late/post densification, every periodic evaluation, collection, and final reconstruction. Representative evidence never replaces the full CUDA Event time series.

For bounded NSYS collection, run `stage_runner` in `representative` mode with `--capture-profiler-api`, and configure NSYS with `--capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown:N`. The model still executes one continuous official training process; only profiler collection starts and stops at selected outer iteration boundaries. Campaign mode requires a verified input-freeze and rejects manual iteration ranges or training-argument changes other than the output directory.

Evidence remains ineligible for formal performance until every requested iteration, exact NSYS kernel assignment, required stage-role, NCU counter, and local/Orin calibration gate passes.
