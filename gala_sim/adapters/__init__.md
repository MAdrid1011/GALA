# adapters/__init__.py

导出模型适配器合同、Chest 数据清单、R²-Gaussian 追踪适配器、native-reference 运行器和带 NVTX 边界的阶段 GPU profiler。

## External Interfaces

公开 `ModelAdapter`、`PreparedRun`、`ReferenceArtifact`、`TraceArtifact`、`R2GaussianChestAdapter`、`run_native_reference` 及其错误类型。

## Internal Helpers

具体模型入口和追踪实现分别位于 `r2_gaussian.py`、`trace_capture.py`、`stage_profile.py` 和 `native_reference.py`；R²-Gaussian 适配器的 `python_executable` 字段用于绑定冻结解释器；该包不定义额外硬件模块。

`python -m gala_sim.adapters.trace_runner --capture-iteration-range START:END` 在完整执行此前训练迭代的前提下，只记录闭区间内的事件。窗口开始时以当时真实 Gaussian 集合建立局部稳定 ID epoch；产物固定标记为 quick trace validation，不能进入正式性能结果。
