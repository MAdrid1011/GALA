# GPU 编译消融探针

`python -m gala_sim.tools.gpu_compiler_ablation` 对同一 CUDA 工作量运行 GPU Base、`1000`、
`0100` 和 `1100`。入口先逐项预热，再轮换执行顺序并取重复中位数，避免冷启动顺序被误写为
机制收益。

runner 必须输出同步边界内的 `relation_generation_ms` 和 `primitive_ms`。工具逐项检查 query、
Gaussian、候选、关系和 consumer 数量完全一致，并以小容差检查 loss 与梯度。输出固定为
`development_gpu_compiler_probe`，不能进入正式性能表；`1000`、`0100`、`1100` 只生成
`speedup_vs_gpu_base`，不生成相对 Base ASIC 的加速比。
