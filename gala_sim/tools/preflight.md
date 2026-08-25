# preflight.py

实现官方数值路径和长任务的运行时间预测及 GPU 门控。

## External Interfaces

- `predict_runtime(...)`：由 warmup 后的短测量预测总迭代时间。
- `decide_long_run(...)`：按冻结阈值和本任务 GPU 利用率决定是否允许长任务。
- `sample_gpustat(gpu_index=0)`：组合 `gpustat` 与 compute-app 清单，区分外部计算进程。
- `run_native_preflight(...)`：验证 input-freeze，运行隔离的 60 次迭代校准，写出 `preflight.json` 和 `status.json`。
- `ComputeProcess`、`GpuSample`、`PreflightDecision`、`NativePreflightReport`：机器可读记录类型。

## Internal Helpers

内部逻辑派生短测量命令、采集 `/proc` CPU 与读取计数、提取 TensorBoard 测量区间，并在采样、启动、训练或测量失败时终止短任务和写出 `failed_preflight`。
