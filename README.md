# GALA Simulator

本仓库用于开发 GALA 的 Python 体系结构模拟器。模拟器以真实高斯体重建模型和真实数据集为输入，同时执行数值功能路径与事件驱动周期路径，输出完整重建结果、端到端周期数、PSNR、SSIM、LPIPS 和全部优化组合的消融结果。

实现约束、硬件合同、数据获取、周期模型和验收流程见 [docs](docs/README.md)。当前只推进 R²-Gaussian + Chest，并严格遵循输入冻结、官方软件、真实 trace、Base ASIC、Oracle、实际机制和完整消融的顺序。

输入冻结后，使用同一记录执行官方短训练门控：

```bash
gala-sim native-preflight \
  --config configs/architecture/gala.yaml \
  --freeze <input-freeze.json> \
  --output <preflight-output>
```

目标 GPU 存在外部计算进程时，该命令写出 `failed_preflight` 且不启动训练。预检通过只允许进入论文配置训练，不构成正式时间或质量结果。
