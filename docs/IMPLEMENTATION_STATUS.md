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
- Ramulator bridge smoke 记录保存在仓库外 `GALA-runtime/records/ramulator2_bridge_smoke_20260826.json`，SHA-256 为 `c2e402375fc15ebed2e1027315e1817b9c05c9842bac455a3ddb6680748f633c`；更新后的首组合输入冻结记录 SHA-256 为 `e51bdf00cc36e47decd24a8ff4f28621e698d82f14667eda564c0a9266070478`。
- 融合发射前向、消费者和伴随三类端口已使用注册配置中的独立端口数与独立 II 状态，不再由单个聚合端口互相错误阻塞。
- CLAMP 事件 schema v2、批量 NumPy trace 存储、依赖/版本/释放校验和模块拆分的离散事件周期内核已通过 65 项测试（1 项环境依赖跳过）。
- 资源包络、长任务 GPU 利用率门和十六项消融矩阵的结构检查已通过单元测试。
- `49bf36d` 为十六项变体使用显式 `variant:<bits>` 策略，并在周期内核中落实模块在途容量和关系种子 FIFO 反压。
- `cde1a93` 为每个消融变体隔离记录内存完成表的消费游标，避免同一外部 Ramulator 记录被首个变体消耗。
- 真实 CUDA trace 旁路已接入官方 `R²-Gaussian` rasterizer/voxelizer：只在官方查询边界处于 grad-enabled 的训练路径捕获，排除 `no_grad` 质量评估、保存和报告调用；Chunked sink 传输和依赖偏移重建已通过单元测试。
- 旧版 Chest 1 迭代 grad-gated smoke 产生 1,048,191 个结构化事件，且与未插桩官方运行的 `vol_pred.npy` 逐元素一致、PSNR/SSIM 输出一致；后续审计确认该 trace 把完整 kernel 调用当作 query，并按 Gaussian 合并 mask，故只能保留为旧粒度插桩一致性证据，不能用于正式模板周期。
- CUDA decoder 已在设备端把 raster 16×16 与 voxel 8×8×8 mask 压紧为每个有效 bit 一个 `(candidate_index, local_query)`，候选和关系通过单个 packed tensor 批量传回主机。2-candidate CUDA 微型对照中 packed relation 数 12 与 mask popcount 12 完全一致。
- 捕获路径现为每个 pixel/voxel 分配稳定全局 `query_id`，每个有效 Gaussian–query 配对分配独立 `relation_id`；每条 relation 只依赖其 tile–Gaussian candidate，并产生独立缓存请求、前向、伴随和梯度事件。
- consumer 已从 relation 粒度改为 query 粒度，并从实际 loss 调用捕获 L1、11×11 SSIM 与 3D TV：投影 consumer 依赖对应 SSIM 邻域的 query reduction，体 consumer 依赖轴向相邻 query reduction，所有同 query adjoint 依赖同一 consumer。
- Trace validator 已对真实事件链执行候选→关系→关闭、关系→缓存→前向→归约→查询消费者→伴随→梯度以及梯度→更新事务的前置依赖检查；同时重建 active Gaussian 集合、稳定 ID、Clone/Split/Prune lineage、跨迭代更新屏障和关闭版本访问。capture audit v3 分开记录官方/捕获 kernel 调用、逻辑 query、CUDA 候选、有效关系、backward relation 数以及 begin/end、no-op 和集合修改事务。
- 事件、依赖和 payload 三列均支持磁盘 chunk 合并为最终 mmap 文件；同路径 writer 不再重复截断，避免正式长任务在 `finish()` 阶段聚合整份数组。
- 周期内核已改为依赖计数反向唤醒和有界就绪窗口，不再逐周期扫描全部 pending 事件；在真实 1,048,191-event trace 上两次 `variant:0000` smoke 均为 997,127 周期，停顿记录按同周期/模块/原因合并。
- event-driven 周期内核新增 4,096 事件依赖链、重复运行确定性和停顿计数合并回归测试；长链测试按真实服务延迟完成且无死锁。
- 语义驻留状态已接入 `variant:0001` 的逐请求目录、容量反压、填充、读完成和状态版本级释放路径；query close 不再提前释放，同版本 no-op 更新仍可复用，状态写入结束后才关闭旧版本。

## 当前入口

首个组合仍停在正式 Base ASIC 之前。`configs/architecture/gala.yaml` 中的 `relation.seed_fifo_entries`、各模块时序、trace chunk 容量和 `memory.ramulator_config_sha256` 尚未由权威设计或模块微基准冻结。原生 Ramulator 2 binding 已可用，但正式入口会拒绝未与 canonical 配置哈希一致的 YAML。当前周期 smoke 使用显式实验 timing，只能验证内核行为，正式运行必须拒绝并标记 `failed_preflight`。

R²-Gaussian 的官方 CUDA 扩展仍没有直接导出完整 CLAMP 事件缓冲区；当前旁路从官方 CUDA work buffer 重建逐 query 关系，并在官方 loss、backward 和 optimizer 边界映射其余事件。新版真实粒度 Chest smoke 因 GPU 正被外部作业持续占用而尚未执行，因此还没有重新证明插桩/未插桩体数据逐元素一致。该旁路也尚未证明长程增密 ID 稳定性、所有真实队列操作和正式 30k 训练的流式最终存储，不能视为正式 trace 闭环。

## 下一步入口条件

1. GPU 空闲后运行新版 Chest 1 迭代真实 smoke，要求 validator 通过、relation 数等于全部 mask popcount，并重新核对插桩/未插桩体数据逐元素一致。
2. GPU 空闲后运行新版 Chest 1 迭代真实 smoke；随后继续把 mmap trace 接入流式 `TraceReader`/周期消费，避免周期重放再次要求完整事件数组。
3. 在真实关系粒度上实现 `template_id + primitive_kind` 的精确计算 Pod 资源序列，并测量关系种子突发、模块时序和 trace chunk，更新冻结配置及运行清单。
4. 用完整真实 trace 通过依赖/状态/释放检查后，运行 `0000` Base ASIC 和两个受资源约束的 Oracle。

当前不报告面积、功耗、能量、能效、正式周期或论文加速比。
