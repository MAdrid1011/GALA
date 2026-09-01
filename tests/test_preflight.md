# Preflight Tests

Validate long-run projection, GPU utilization gating, compute-process ownership,
external-job rejection, isolated short-training projection, and configuration and
resource gates for cycle runs. When an external compute process exists, the test
requires `failed_preflight` to be written before any calibration model directory
is created.
