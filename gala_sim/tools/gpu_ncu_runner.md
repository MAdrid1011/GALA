# NCU invocation runner

`python -m gala_sim.tools.gpu_ncu_runner preflight ... ` runs the configured short calibration with a formal v5 invocation-job filter and only the iteration-1 profiler window, samples `gpustat` at the frozen interval, and combines measured profiler overhead with a passed native preflight to predict the formal job duration. It profiles up to the configured launch maximum from one real kernel name; sparse groups use every available iteration-1 launch and record the actual count. The utilization floor is enforced only when that prediction reaches the frozen long-run threshold. `run` executes the complete frozen official training command for either an invocation or NVTX-range job, exports details and SourceCounters pages, binds the observed launches, and performs dynamic SASS classification.

Both modes reject existing output directories, dirty worktrees, plans without
repeated NSYS stability, and plans containing `--launch-count` or `--kill`.
Plan, implementation, commit, and artifact hashes are audit metadata rather
than execution gates. Jobs use the complete campaign command and configured
CUDA Profiler API windows; the runner never terminates a target after a profile
count. A configured watchdog acts only on the launched process group after GPU,
stdout, and report progress all remain idle for the declared interval.
