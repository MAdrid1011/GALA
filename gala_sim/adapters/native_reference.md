# native_reference.py

执行并记录冻结的 R²-Gaussian + Chest 官方数值参考路径。

## External Interfaces

`run_native_reference(config, freeze, preflight, output)` 要求匹配的 `gala-input-freeze-v3`、通过的 `gala-native-preflight-v1` 和空的官方模型输出目录。它运行原始命令，记录日志、GPU 样本、TensorBoard 迭代时间、统一 PSNR/SSIM/LPIPS、官方附加指标和状态文件。

## Internal Helpers

命令解析器绑定数据与模型输出路径；R²-Gaussian 适配器可接收冻结的绝对 Python 解释器，使 trace 命令不回退到环境 PATH；TensorBoard 读取器只报告实际 `train/iter_time` 标量，并把未单独测量的剩余 wall time 标为 residual，不填充虚构阶段值。
