# 消融矩阵与输出合同

## 1. 优化开关

模拟器定义四个独立布尔开关。

| 位 | 配置名 | 启用行为 | 关闭行为 |
| --- | --- | --- | --- |
| A | `compiler.query_load_rules` | 生成查询负载与推进规则 | 保留合法依赖，使用到达顺序 |
| B | `compiler.semantic_worksets` | 生成高斯驻留、复用与释放信息 | 不提供跨任务语义驻留信息 |
| C | `architecture.overlap_guided_issue` | 使用历史重叠预测和无冲突融合发射 | 每周期按到达顺序发射一个合法任务 |
| D | `architecture.semantic_residency` | 启用语义目录、驻留、Miss 合并与作用域多播 | 每个状态请求经基础存储路径完成 |

位序固定为 `ABCD`。`0000` 是 Base ASIC，`1111` 是完整 GALA。关闭开关只移除对应优化，不能删除任务、依赖、队列、数值计算或正确性所需的硬件状态。

## 2. 组合运行

同一已验证 trace 运行从 `0000` 到 `1111` 的十六个组合。每个组合使用相同的硬件资源配置、内存映射、输入事件和数值语义。运行器拒绝缺少组合、重复组合或配置哈希不一致的结果。

两类机制还输出聚合点。查询调度为 `A=1,C=1` 且 B、D 关闭。语义驻留为 `B=1,D=1` 且 A、C 关闭。完整 GALA 的周期必须与消融矩阵中的 `1111` 逐周期一致。

## 3. 加速比口径

每个组合首先报告相对于 Base ASIC 的加速比。

```text
speedup_vs_base_asic = cycles_0000 / cycles_variant
```

平台级结果再报告相对于 AGX Orin 的加速比。

```text
speedup_vs_orin = orin_normalized_seconds * clock_hz / cycles_variant
```

结果表必须同时包含绝对周期、相对于 Base ASIC 的加速比、本地 GPU 原始时间、AGX Orin 换算时间和相对于 Orin 的加速比。Oracle 结果独立标记，不能与实际机制结果混入同一几何平均。

## 4. 输出文件

每个运行目录包含以下文件。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 代码、模型、数据、配置、设备与环境哈希 |
| `cycles.json` | 端到端周期与模块周期分解 |
| `stalls.parquet` | 阻塞原因与周期区间 |
| `memory_requests.parquet` | 每个片外请求的地址、读写、字节数、到达周期和 Ramulator 2 返回周期 |
| `events.json` | 事件计数、缓存和发射统计 |
| `quality.json` | CUDA 参考和功能重放的 PSNR、SSIM、LPIPS |
| `gpu_reference.json` | 本地 GPU 时间、校准项和 Orin 换算 |
| `status.json` | 运行状态、失败原因和验收结果 |

活动组合汇总为 `ablation.csv`。表中每行绑定模型、数据集、开关位、绝对周期、两种加速比、质量差异和配置哈希。结果生成器只读取这些机器可读文件，不允许手工填写论文表格。

## 5. 一致性断言

- `1111` 的端到端周期等于完整 GALA 入口输出
- 所有组合的动态有效关系数、数值任务数和更新提交数一致
- 仅允许调度顺序、缓存事件、停顿原因和周期数变化
- 相对于 Base ASIC 的加速比统一使用同一任务的 `0000`
- 相对于 Orin 的加速比统一使用同一任务的分阶段换算时间
- 任一质量越界时，该组合不进入性能汇总
