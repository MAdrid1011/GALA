# adapters/__init__.py

导出模型适配器合同、Chest 数据清单、R²-Gaussian 追踪适配器和 native-reference 运行器。

## External Interfaces

公开 `ModelAdapter`、`PreparedRun`、`ReferenceArtifact`、`TraceArtifact`、`R2GaussianChestAdapter`、`run_native_reference` 及其错误类型。

## Internal Helpers

具体模型入口和追踪实现分别位于 `r2_gaussian.py`、`trace_capture.py` 和 `native_reference.py`；R²-Gaussian 适配器的 `python_executable` 字段用于绑定冻结解释器；该包不定义额外硬件模块。
