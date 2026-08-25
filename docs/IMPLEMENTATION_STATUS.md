# 当前实现状态

本文件记录 `docs/10_IMPLEMENTATION_WORKFLOW.md` 的当前入口和证据，不替代正式实验结果。

## 已通过

- 固定的 R²-Gaussian 提交 `f2579bfddd9aac009cb797c8503bef8119bbd022` 可核验，官方 Chest 数据清单包含 153 个文件、元数据哈希和文件级 SHA-256。
- 官方 CUDA 扩展在 `gaussian-slam-official` 环境中使用 CUDA 12.1 工具链和 GCC 11 编译并通过最小 GPU kernel 调用。
- 官方 Chest 1/2 迭代真实 smoke 已在仓库外保存；1 迭代评估输出了官方路径的 PSNR、SSIM 和体重建文件。
- 配置加载器检查参数元数据、状态、值域和不可变快照；`config-check` 现在输出逐项 pending 参数并以非零状态阻止未冻结配置。
- `cycle-preflight` 已生成 `preflight.json` 与 `status.json`，逐项检查配置冻结、Ramulator 2 binding 和资源使用快照；失败状态使用工作流规定的 `failed_preflight`。
- `CycleConfig` 在提供资源使用快照时由注册配置推导顶层资源包络并强制校验 SRAM、Pod、计算通路和片外通道闭合。
- Ramulator 2 桥接边界已改为异步 `try_issue/tick/drain_completions`：周期内核与内存模型共同推进，支持并发到达、前端反压、64 B transaction 拆分和返回唤醒；每个逻辑请求写入 `memory_requests.parquet` 供逐请求核对。
- 仓库自带的 C ABI bridge 已对固定 Ramulator 2 v2.1.0 提交 `38c51d40a976c6b07fbc09de869a7e08dc187d29` 完成仓库外构建；真实 LPDDR5-6400 八通道配置 smoke 中，一个 128 B 读取拆为两个 64 B transaction 并在周期 36 返回。该配置尚未冻结为 canonical 正式配置。
- Ramulator bridge smoke 记录保存在仓库外 `GALA-runtime/records/ramulator2_bridge_smoke_20260826.json`，SHA-256 为 `f152cbc3cd7b75f0f0516e5189d05c0f752e7e50bdf7c9eab1b9d0329ac0c657`；更新后的首组合输入冻结记录 SHA-256 为 `0a7ce43f629d6bea5229cb0a0889004dae857fdb8e7a5fbca005052ad0a5b7fc`。
- 融合发射前向、消费者和伴随三类端口已使用注册配置中的独立端口数与独立 II 状态，不再由单个聚合端口互相错误阻塞。
- CLAMP 事件 schema、批量 NumPy trace 存储、依赖/版本/释放校验和模块拆分的离散事件周期内核已通过 51 项单元测试。
- 资源包络、长任务 GPU 利用率门和十六项消融矩阵的结构检查已通过单元测试。
- `49bf36d` 为十六项变体使用显式 `variant:<bits>` 策略，并在周期内核中落实模块在途容量和关系种子 FIFO 反压。
- `cde1a93` 为每个消融变体隔离记录内存完成表的消费游标，避免同一外部 Ramulator 记录被首个变体消耗。
- 真实 CUDA trace 旁路已接入官方 `R²-Gaussian` rasterizer/voxelizer：只在官方查询边界处于 grad-enabled 的训练路径捕获，排除 `no_grad` 质量评估、保存和报告调用；Chunked sink 传输和依赖偏移重建已通过单元测试。
- Chest 1 迭代 grad-gated smoke 产生 1,048,191 个结构化事件并通过 `validate_trace`；与未插桩官方运行的 `vol_pred.npy` 逐元素一致，PSNR/SSIM 输出一致。
- Trace validator 已对真实事件链执行候选→关系→关闭、关系→缓存→前向→归约→消费者→伴随→梯度以及梯度→更新提交的前置依赖检查；capture audit 同时记录官方调用、排除的 `no_grad` 调用、CUDA 候选和有效关系计数。
- 事件、依赖和 payload 三列均支持磁盘 chunk 合并为最终 mmap 文件；同路径 writer 不再重复截断，避免正式长任务在 `finish()` 阶段聚合整份数组。
- 周期内核已改为依赖计数反向唤醒和有界就绪窗口，不再逐周期扫描全部 pending 事件；在真实 1,048,191-event trace 上两次 `variant:0000` smoke 均为 997,127 周期，停顿记录按同周期/模块/原因合并。
- event-driven 周期内核新增 4,096 事件依赖链、重复运行确定性和停顿计数合并回归测试；长链测试按真实服务延迟完成且无死锁。
- 语义驻留状态已接入 `variant:0001` 的逐请求目录、容量反压、填充、读完成和释放路径；实验 smoke 完成且未凭空产生命中。

## 当前入口

首个组合仍停在正式 Base ASIC 之前。`configs/architecture/gala.yaml` 中的 `relation.seed_fifo_entries`、各模块时序、trace chunk 容量和 `memory.ramulator_config_sha256` 尚未由权威设计或模块微基准冻结。原生 Ramulator 2 binding 已可用，但正式入口会拒绝未与 canonical 配置哈希一致的 YAML。当前周期 smoke 使用显式实验 timing，只能验证内核行为，正式运行必须拒绝并标记 `failed_preflight`。

R²-Gaussian 的官方 CUDA 扩展仍没有导出完整 CLAMP 关系、消费者、伴随和更新事件缓冲区；当前旁路只读官方返回的 CUDA work buffer，并在官方 Python 调用边界映射这些事件。该旁路尚未证明长程增密 ID 稳定性、所有真实队列操作和正式 30k 训练的流式最终存储，因此不能视为正式 trace 闭环。

## 下一步入口条件

1. 审计旁路事件与每个真实队列操作、消费者依赖、版本更新和稳定增密 ID 的一一对应关系。
2. 继续把 mmap trace 接入流式 `TraceReader`/周期消费，避免周期重放再次要求完整事件数组；同时测量关系种子突发、模块时序和 trace chunk，更新冻结配置及运行清单。
3. 用完整真实 trace 通过依赖/状态/释放检查后，运行 `0000` Base ASIC 和两个受资源约束的 Oracle。

当前不报告面积、功耗、能量、能效、正式周期或论文加速比。
