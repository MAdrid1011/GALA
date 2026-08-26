# tools/__init__.py

导出周期预检、原生官方预检、GPU 采样、运行时间预测、可移植 GPU 校准、Nsight 阶段产物解析和 local/Orin 阶段归一化接口，不包含被模拟硬件。

## External Interfaces

公开 `run_native_preflight`、`sample_gpustat`、`run_cycle_preflight` 及其记录类型和决策 helpers。

## Internal Helpers

无内部实现；具体行为分别位于 `preflight.py`、`cycle_preflight.py`、`gpu_calibration.py`、`gpu_profile_artifacts.py` 和 `gpu_normalization.py`。
