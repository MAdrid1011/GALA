# NCU invocation runner

`python -m gala_sim.tools.gpu_ncu_runner preflight ...` runs the configured short calibration with the formal v3 job filter and only the iteration-1 profiler window, samples `gpustat` at the frozen interval, and combines measured profiler overhead with a passed native preflight to predict the formal job duration. The utilization floor is enforced only when that prediction reaches the frozen long-run threshold. `run` executes the complete frozen official training command, exports details and SourceCounters pages, binds every launch to the selected plan job, and performs dynamic SASS classification.

Both modes reject existing output directories and plans containing `--launch-count` or `--kill`. Formal jobs always use the frozen 30,000-iteration command and CUDA Profiler API representative windows; the runner never terminates the target after a requested profile count.
