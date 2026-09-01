# native_reference.py

## External Interfaces

`run_native_reference(config, freeze, preflight, output)` validates matching
input and preflight manifests, executes the bound upstream command, and writes
reconstruction, quality, timing, utilization, and process-accounting output to
the local workspace.

## Internal Helpers

Command binding preserves the frozen interpreter, source directory, dataset,
and output directory. The watchdog observes real process and GPU progress and
terminates only the launched process group after the configured inactivity
condition. TensorBoard timing is reported only when the upstream run emits it.
