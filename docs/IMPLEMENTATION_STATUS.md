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
- Ramulator bridge smoke 记录保存在仓库外 `GALA-runtime/records/ramulator2_bridge_smoke_20260826.json`，SHA-256 为 `c2e402375fc15ebed2e1027315e1817b9c05c9842bac455a3ddb6680748f633c`。首组合 input-freeze v3 已按提交 `53949ba` 和配置哈希 `8f9a249ffbc74b749a97313647f9a98785f873a75c128ec0716133e5fa1a6c50` 重新生成；训练快照 SHA-256 为 `0d607a04e68d61f1fb05cc98a7dbb8255fcd19bbf7501388ef6caa19a84c1482`，官方命令 SHA-256 为 `abbd3983f18ac492a7abd4ab3178d6f99f3689d35896e69febef437f88d2538d`，run manifest SHA-256 为 `70754a896e624f45643f04515eb984a30482a44e651d506a3a9bdc5583327cf7`，仓库外文件 SHA-256 为 `551f26223f223b8be80fae581b0355bdf4bbfcddd17d08bfd8fd03d7e89023d9`。其状态为 `planned`，且配置因硬件与运行策略参数仍 pending 而未 ready。
- 官方 native-reference 预检已按同一冻结身份执行，记录位于 `GALA-runtime/records/r2_gaussian_chest_native_preflight/`。预检记录 SHA-256 为 `fa2c3cc5a8fb1255908a58af8baacaaa9d0f9b216436d9e8102ed7ba5236e722`，状态为 `failed_preflight`，原因是 `gpu_busy_external`：`gpustat` 与 `nvidia-smi` 均确认外部 `gdesmond` PID `2594715` 占用目标 GPU。`calibration.launched=false`，因此没有启动任何 GALA 训练或争用 GPU；报告 self-hash 为 `ae155fb0aedc7d401b0fc77ea3c5fbd1b42b696da6ce6739254f74e0cf3f28e0`，状态文件 SHA-256 为 `6373c15f46af145c8a59e56a57012d9fdfb4432560200a36904bd0913b7cd1f`。
- `native-reference` 入口已要求同一 input-freeze 与 `passed` native-preflight 才能启动官方完整命令；它写出流式 stdout/stderr、GPU 样本、TensorBoard `train/iter_time` 序列、统一质量指标、官方附加指标和运行 manifest。所有进入执行阶段的失败记录都绑定配置哈希、freeze manifest、仓库提交和官方命令复现信息；预启动复采样发现外部 compute 进程时也会写出 `failed_preflight/status.json` 并拒绝创建模型输出。当前使用 `failed_preflight` 记录的拒绝演练返回退出码 2，未创建模型输出。
- 融合发射前向、消费者和伴随三类端口已使用注册配置中的独立端口数与独立 II 状态，不再由单个聚合端口互相错误阻塞。
- CLAMP 事件 schema v2、批量 NumPy trace 存储、依赖/版本/释放校验和模块拆分的离散事件周期内核及步骤 2 入口已通过 90 项测试（1 项环境依赖跳过）。正式 `CycleConfig.from_gala` 会在周期执行前校验 trace 的 `config_sha256` 身份。
- 资源包络、长任务 GPU 利用率门和十六项消融矩阵的结构检查已通过单元测试。
- `49bf36d` 为十六项变体使用显式 `variant:<bits>` 策略，并在周期内核中落实模块在途容量和关系种子 FIFO 反压。
- `cde1a93` 为每个消融变体隔离记录内存完成表的消费游标，避免同一外部 Ramulator 记录被首个变体消耗。
- 真实 CUDA trace 旁路已接入官方 `R²-Gaussian` rasterizer/voxelizer：只在官方查询边界处于 grad-enabled 的训练路径捕获，排除 `no_grad` 质量评估、保存和报告调用；捕获完成后绑定配置哈希、模型提交、数据清单哈希和 GALA 仓库提交，才交给周期 sink；Chunked sink 传输和依赖偏移重建已通过单元测试。
- 旧版 Chest 1 迭代 grad-gated smoke 产生 1,048,191 个结构化事件，且与未插桩官方运行的 `vol_pred.npy` 逐元素一致、PSNR/SSIM 输出一致；后续审计确认该 trace 把完整 kernel 调用当作 query，并按 Gaussian 合并 mask，故只能保留为旧粒度插桩一致性证据，不能用于正式模板周期。
- CUDA decoder 已在设备端把 raster 16×16 与 voxel 8×8×8 mask 压紧为每个有效 bit 一个 `(candidate_index, local_query)`，候选和关系通过两遍 CUDA kernel 写入预分配 packed tensor，避免 `nonzero/stack/cat` 临时张量。1/2-candidate GPU full-mask 对照中 packed relation 行数分别为 256/512，local query 与 mask popcount 完全一致。
- 捕获路径现为每个 pixel/voxel 分配稳定全局 `query_id`，每个有效 Gaussian–query 配对分配独立 `relation_id`；每条 relation 只依赖其 tile–Gaussian candidate，并产生独立缓存请求、前向、伴随和梯度事件。
- consumer 已从 relation 粒度改为 query 粒度，并从实际 loss 调用捕获 L1、11×11 SSIM 与 3D TV：投影 consumer 依赖对应 SSIM 邻域的 query reduction，体 consumer 依赖轴向相邻 query reduction，所有同 query adjoint 依赖同一 consumer。
- Trace validator 已对真实事件链执行候选→关系→关闭、关系→缓存→前向→归约→查询消费者→伴随→梯度以及梯度→更新事务的前置依赖检查；同时重建 active Gaussian 集合、稳定 ID、Clone/Split/Prune lineage、跨迭代更新屏障和关闭版本访问。capture audit v4 分开记录官方/捕获 kernel 调用、逻辑 query、CUDA 候选、有效关系、backward relation 数、关系记录设备批次与 D2H 批次，以及 begin/end、no-op 和集合修改事务。
- 事件、依赖和 payload 三列均支持磁盘 chunk 合并为最终 mmap 文件；同路径 writer 不再重复截断，避免正式长任务在 `finish()` 阶段聚合整份数组。
- 批量 trace writer 已支持 `stream_only` raw-column 模式：事件、依赖和 payload 在有界 chunk 刷满时直接追加到仓库外 raw 列，`finish()` 只写 chunk manifest，`TraceReader` 可按冻结 dtype 直接 mmap，不再复制数十 GB 的最终数组。capture 在最后一个 query backward 到达时立即转移并释放 pending decoder records；真实 v9 1-iteration capture 已完整结束，证据见“当前入口”。
- 周期内核已改为依赖计数反向唤醒和有界就绪窗口，不再逐周期扫描全部 pending 事件；在真实 1,048,191-event trace 上两次 `variant:0000` smoke 均为 997,127 周期，停顿记录按同周期/模块/原因合并。
- event-driven 周期内核新增 4,096 事件依赖链、重复运行确定性和停顿计数合并回归测试；长链测试按真实服务延迟完成且无死锁。
- 语义驻留状态已接入 `variant:0001` 的逐请求目录、容量反压、填充、读完成和状态版本级释放路径；query close 不再提前释放，同版本 no-op 更新仍可复用，状态写入结束后才关闭旧版本。
- mmap trace 周期入口已使用 NumPy 压缩反向依赖索引；同一 trace 的十六项消融只执行一次结构校验。旧版 1,048,191-event 规模样本的索引构建使用约 21.8 MB 连续数组，结果仅作为软件可扩展性证据，不作为正式周期。
- Chest 统一质量口径已在查看任何 GALA 功能重放结果前冻结：参考范围 `[0,1]`，三维 SSIM 使用 `11/1.5/reflect`，LPIPS 使用三轴各 `[42,85,128,170,213]`、`alex` v0.1 及两份受检权重。R²-Gaussian 参考产物统一输出 `psnr`、`ssim`、`lpips`，官方 `psnr_3d/ssim_3d` 仅作附加指标；input-freeze v3 展开保存参考体统计、质量参数和质量依赖版本。
- R²-Gaussian + Chest 论文训练协议已冻结并由生成器直接对照固定上游源码：完整 41 项解析参数、30,000 次迭代、评估/保存/检查点调度、Python/NumPy/PyTorch seed 0、`cuda:0` 设备选择、实际工作目录和完整命令均进入 v3 记录。参数、调度、提交或 seed 漂移时生成器拒绝输出。
- 已保存的一迭代官方体在 `numpy 1.26.4/scikit-image 0.21.0/torch 2.1.2/torchvision 0.16.2/lpips 0.1.4` 的纯 CPU 质量 smoke 中得到 PSNR `20.2852146768`、SSIM `0.2686132998`、LPIPS `0.6325289289`，两份 LPIPS 权重哈希通过。该值只验证统一入口，不是论文配置质量结果。
- 大型 raw-column validator 的历史事件键已按实际 query、Gaussian、relation 和 event domain 压缩，并支持把临时索引显式放到独立目录；扫描时不再逐块丢弃仍会被后续依赖引用的历史索引。真实 v9 一迭代 trace 连续两次完成标准全量验证，第二次为 577,268,206 个事件、895,439,361 条依赖，耗时 `213.56 s`、峰值 RSS `8,459,036 KB`、进程 swap 0。仓库外日志 SHA-256 为 `5972ce75c9c06a40aad6827eb9808807e9d94d28335a3eac548ab15178a90107`，结构化记录 SHA-256 为 `b4f54dfa468cf98f93b7fdbecc8d1726a0e9a05904c093dd9706ad5873a8aac5`。

## 当前入口

本轮 v9 stream-only 真实 Chest 1-iteration capture 已生成完整 raw-column manifest：577,268,206 个事件、895,439,361 条依赖；capture audit 记录 294,912 个逻辑 query、690,928 个 CUDA candidate、95,948,757 条有效 relation，实际 D2H transfer 12 次。插桩与未插桩官方运行的 `vol_pred.npy` SHA-256 和逐元素值完全一致，PSNR `20.285214676826495`、SSIM `0.268613299848576`、LPIPS `0.6325289289156596` 完全一致。该证据仍是 1 iteration smoke，不是正式 30k trace。

`TraceReader` 现会从旧式 raw-column `chunk_manifest.json` 的 `storage_format` 补入 `trace_storage_format`，标准 `read(validate=True)` 可自动选择流式校验入口。上述大 trace 的标准 validator 已连续两次 PASS；第二次使用 4,000,000-event 扫描块和 `/dev/shm` 临时紧凑索引，耗时 `3:33.56`、峰值 RSS 约 8.07 GiB、无进程 swap。扫描块和临时目录只改变验证软件的吞吐与资源位置，不改变全量检查集合。该 trace 不含正式 30k 路径所需的 optimizer/update/collection 状态迁移，因此仍不能视为正式 trace 闭环。

快速 trace 路径现支持显式 query ranges、事件上限、扫描块和 `cpu/cuda/auto` 后端。对同一 v9 q0 闭包，CUDA 与 CPU 输出的 events/dependencies/payload 逐元素一致；CUDA 用时 `88.65 s`、峰值 RSS `884,720 KB`，CPU 用时 `110.77 s`、峰值 RSS `476,364 KB`。中心 raster query `131328` 与中心 voxel query `279056` 的联合闭包包含 406,008 个事件、700,903 条依赖、99,831 条真实 relation 和两种 template，并通过独立 validator。四策略周期 smoke 已完成，但使用实验 timing，且两个 Oracle 均返回 `heuristic_unproven`；这些结果只证明快速路径可运行，不是正式 Base ASIC/Oracle 周期。

首个组合仍停在正式 Base ASIC 之前。`configs/architecture/gala.yaml` 的质量参数已冻结，当前配置哈希为 `8f9a249ffbc74b749a97313647f9a98785f873a75c128ec0716133e5fa1a6c50`；`relation.seed_fifo_entries`、各模块时序、trace chunk 容量和 `memory.ramulator_config_sha256` 仍未冻结。原生 Ramulator 2 binding 已可用，但正式周期入口会拒绝带 pending 参数或未与 canonical 配置哈希一致的 YAML。当前周期 smoke 使用显式实验 timing，只能验证内核行为，正式运行必须拒绝并标记 `failed_preflight`。

仓库外 `r2_gaussian_chest_freeze.json` 使用冻结解释器 `/home/madrid/anaconda3/envs/gaussian-slam-official/bin/python3.10`、`CUDA_HOME=/usr`、`/usr/bin/nvcc` 12.0、GCC/G++ 11、PyTorch CUDA 12.1 与冻结质量依赖生成；记录中的参考体范围、切片边界、训练默认值、有效调度、随机状态和配置哈希已由生成器交叉检查。冻结的官方训练命令为 `/home/madrid/anaconda3/envs/gaussian-slam-official/bin/python3.10 train.py -s /home/madrid/Desktop/GALA-runtime/data/chest/extracted/cone_ntrain_50_angle_360/0_chest_cone -m /home/madrid/Desktop/GALA-runtime/official/r2_gaussian_chest_30000`，工作目录为固定上游源码根目录。

R²-Gaussian 的官方 CUDA 扩展仍没有直接导出完整 CLAMP 事件缓冲区；当前旁路从官方 CUDA work buffer 重建逐 query 关系，并在官方 loss、backward 和 optimizer 边界映射其余事件。v9 已重新证明一迭代插桩/未插桩体数据逐元素一致，但尚未证明长程增密 ID 稳定性、所有真实队列操作和正式 30k 训练的流式最终存储，不能视为正式 trace 闭环。

关系记录 decoder 输出现按迭代边界合并，每个非空 flush 最多执行一次 D2H；flush 在最后一个 query 的 backward 到达、下一迭代、optimizer step、Gaussian 集合修改和最终写出之前发生，并要求每个捕获 query 恰有一次类型匹配且已关联 loss 的 backward。v9 decoder 的 raw terminal CUDA scanner 已通过 CPU 逐索引对照；完整一迭代质量与全量 validator 均已通过，仍待含 optimizer/update/collection 的大 trace 状态迁移验证。

## 下一步入口条件

1. 外部 `gdesmond` 释放 GPU 后，重新运行相同 input-freeze 的 native-preflight；门控通过后按论文配置完成官方参考训练、重建与 PSNR/SSIM/LPIPS，先闭合工作流步骤 2。
2. 为大型 raw-column validator 补齐 optimizer/update/collection 状态迁移 pass，并用跨迭代增密 smoke 验证稳定 ID、版本屏障、lineage 和关闭版本访问；一迭代 v9 的全量结构检查、relation 计数与插桩一致性已通过。
3. 在 smoke 通过后把 raw-column manifest 接入 trace producer/consumer 与周期内核的实时 chunk 交接；当前 `TraceReader` 已可 mmap raw 列，但周期 replay 仍需等待捕获完成后开始。
4. 在真实关系粒度上实现 `template_id + primitive_kind` 的精确计算 Pod 资源序列，并测量关系种子突发、模块时序和 trace chunk，更新冻结配置及运行清单。
5. 用完整真实 trace 通过依赖/状态/释放检查后，运行 `0000` Base ASIC 和两个受资源约束的 Oracle。

当前不报告面积、功耗、能量、能效、正式周期或论文加速比。
