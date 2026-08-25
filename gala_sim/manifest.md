# manifest.py

生成并校验 R²-Gaussian + Chest 的内容寻址 input-freeze 记录。

## External Interfaces

- `source_record(...)`：验证固定上游提交、许可和源码树身份。
- `dataset_record(...)`：记录 Chest 文件清单、几何和参考体身份。
- `training_record(...)`：对照固定上游源码校验训练快照，并绑定绝对 Python 解释器、数据和输出路径。
- `environment_snapshot(python_executable)`：用冻结解释器查询 Python、CUDA 和依赖版本。
- `build_freeze_record(...)`：组合输入身份并生成 self-hash。
- `verify_freeze_record(record)`：拒绝内容与 self-hash 不一致的记录。

## Internal Helpers

AST helpers提取上游参数默认值、运行时参数、调度追加和 `safe_state` 随机状态。质量检查器核对参考体范围与 LPIPS 切片边界。
