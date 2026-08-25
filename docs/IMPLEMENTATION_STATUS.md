# 当前实现状态

本文件记录 `docs/10_IMPLEMENTATION_WORKFLOW.md` 的当前入口和证据，不替代正式实验结果。

## 已通过

- 固定的 R²-Gaussian 提交 `f2579bfddd9aac009cb797c8503bef8119bbd022` 可核验，官方 Chest 数据清单包含 153 个文件、元数据哈希和文件级 SHA-256。
- 官方 CUDA 扩展在 `gaussian-slam-official` 环境中使用 CUDA 12.1 工具链和 GCC 11 编译并通过最小 GPU kernel 调用。
- 官方 Chest 1/2 迭代真实 smoke 已在仓库外保存；1 迭代评估输出了官方路径的 PSNR、SSIM 和体重建文件。
- 配置加载器检查参数元数据、状态、值域和不可变快照。
- CLAMP 事件 schema、批量 NumPy trace 存储、依赖/版本/释放校验和模块拆分的离散事件周期内核已通过 30 项单元测试。
- 资源包络、长任务 GPU 利用率门和十六项消融矩阵的结构检查已通过单元测试。
- `49bf36d` 为十六项变体使用显式 `variant:<bits>` 策略，并在周期内核中落实模块在途容量和关系种子 FIFO 反压。
- `cde1a93` 为每个消融变体隔离记录内存完成表的消费游标，避免同一外部 Ramulator 记录被首个变体消耗。
- 真实 CUDA trace 旁路已接入官方 `R²-Gaussian` rasterizer/voxelizer：只在官方查询边界处于 grad-enabled 的训练路径捕获，排除 `no_grad` 质量评估、保存和报告调用；Chunked sink 传输和依赖偏移重建已通过单元测试。
- Chest 1 迭代 grad-gated smoke 产生 1,048,191 个结构化事件并通过 `validate_trace`；与未插桩官方运行的 `vol_pred.npy` 逐元素一致，PSNR/SSIM 输出一致。

## 当前入口

首个组合仍停在输入与工具链之后。`configs/architecture/gala.yaml` 中的 `relation.seed_fifo_entries`、各模块时序、trace chunk 容量尚未由权威设计或模块微基准冻结，因此正式周期运行必须拒绝并标记 `failed_preflight`。

R²-Gaussian 的官方 CUDA 扩展仍没有导出完整 CLAMP 关系、消费者、伴随和更新事件缓冲区；当前旁路只读官方返回的 CUDA work buffer，并在官方 Python 调用边界映射这些事件。该旁路尚未证明长程增密 ID 稳定性、所有真实队列操作和正式 30k 训练的流式最终存储，因此不能视为正式 trace 闭环。

## 下一步入口条件

1. 审计旁路事件与每个真实队列操作、消费者依赖、版本更新和稳定增密 ID 的一一对应关系。
2. 为正式 30k 训练完成不将完整事件数组聚合到 Python 内存的流式最终存储，并测量关系种子突发、模块时序和 trace chunk，更新冻结配置及运行清单。
3. 用完整真实 trace 通过依赖/状态/释放检查后，运行 `0000` Base ASIC 和两个受资源约束的 Oracle。

当前不报告面积、功耗、能量、能效、正式周期或论文加速比。
