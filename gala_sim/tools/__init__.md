# tools/__init__.py

导出周期预检、原生官方预检、GPU 采样和运行时间预测接口，不包含被模拟硬件。

## External Interfaces

公开 `run_native_preflight`、`sample_gpustat`、`run_cycle_preflight` 及其记录类型和决策 helpers。

## Internal Helpers

无内部实现；具体行为分别位于 `preflight.py` 和 `cycle_preflight.py`。
