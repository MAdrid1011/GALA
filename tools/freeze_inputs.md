# freeze_inputs.py

生成首个 R²-Gaussian + Chest input-freeze，不下载数据或启动训练。

## External Interface

命令要求固定上游源码、Chest 数据或不可用原因，以及官方训练的绝对 Python 解释器。存在真实数据时还要求 `--model-output`，输出为带 self-hash 的 `gala-input-freeze-v3` JSON。

## Internal Helpers

数据检查器验证文件完备性、几何、投影、参考体和初始化数组；manifest helpers校验源码、训练参数、质量协议与环境身份。
