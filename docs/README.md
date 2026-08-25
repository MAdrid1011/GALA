# GALA 模拟器文档

本目录给出 GALA 模拟器的实现合同。文档按开发顺序组织，后续实现不需要重新推导论文中的模块边界、开关语义或实验口径。

1. [项目约束与实现主线](00_CONSTRAINTS_AND_MAINLINE.md)
2. [复现目标与证据等级](01_REPRODUCTION_TARGETS.md)
3. [模拟器软件架构](02_SIMULATOR_ARCHITECTURE.md)
4. [GALA 硬件合同](03_HARDWARE_CONTRACT.md)
5. [参数注册表](04_PARAMETER_REGISTRY.md)
6. [模型适配与真实数据](05_MODELS_AND_DATASETS.md)
7. [周期模型与事件语义](06_CYCLE_MODEL.md)
8. [消融矩阵与输出合同](07_ABLATION_AND_OUTPUTS.md)
9. [性能工程与长任务门控](08_PERFORMANCE_ENGINEERING.md)
10. [质量验证与验收](09_VALIDATION_AND_ACCEPTANCE.md)
11. [唯一执行工作流](10_IMPLEMENTATION_WORKFLOW.md)
12. [实验记录规范](11_EXPERIMENT_RECORDS.md)

若文档之间出现冲突，优先级依次为项目约束、硬件合同、参数注册表、周期模型、模型适配、实验规范。实现不得用局部便利覆盖上层合同。
