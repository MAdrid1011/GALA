# 参数注册表

## 1. 配置规则

所有影响结果的参数由 `GalaConfig` 加载。配置使用 YAML 作为外部格式，启动时转换为冻结的 Python dataclass。未知字段、缺失必需字段、单位错误和越界值直接终止运行。

每个参数必须包含 `value`、`unit`、`source` 和 `scope`。运行清单保存完整解析配置和 SHA-256 哈希。模块代码只能读取配置对象，不得定义重复默认值。

## 2. 架构默认配置

| 配置路径 | 默认值 | 单位 | 来源 |
| --- | ---: | --- | --- |
| `clock.frequency` | 500000000 | Hz | 论文 Implementation |
| `top.num_pods` | 4 | count | GALA Architecture |
| `top.shared_sram_bytes` | 2883584 | byte | 2.75 MiB 设计预算 |
| `shared_sram.read_ports_per_bank` | 1 | port | 每个 Shared SRAM Bank 的独立读端口（1R1W） |
| `shared_sram.write_ports_per_bank` | 1 | port | 每个 Shared SRAM Bank 的独立写端口（1R1W） |
| `shared_sram.active_gaussian_bytes` | 524288 | byte | 四个 Pod 的活动高斯记录 |
| `shared_sram.relation_window_bytes` | 524288 | byte | 十六 Bank 关系包与关系链 |
| `shared_sram.query_volume_bytes` | 524288 | byte | 查询结果、梯度、统计与体素 |
| `shared_sram.gradient_update_bytes` | 655360 | byte | 局部梯度、规范化梯度与更新队列 |
| `shared_sram.index_graph_bytes` | 262144 | byte | 桶、候选与边 |
| `shared_sram.control_metadata_bytes` | 393216 | byte | 状态、依赖、任务与窗口元数据 |
| `relation.seed_fifo_entries` | 必须由设计配置给出 | entry | 关系构造器实现参数 |
| `relation.support_lanes` | 8 | lane | 八条五级支持域流水，由枚举器动态分配空闲 lane |
| `issue.query_state_entries` | 2048 | entry | 查询状态 SRAM |
| `issue.query_state_banks` | 8 | bank | 八路查询状态事件更新 Bank |
| `issue.candidate_lanes` | 3 | lane | Forecast、Conflict、Issue 三级结构 |
| `issue.candidate_fifo_entries` | 32 | entry_per_source_fifo | 前向、消费者、伴随三条候选 FIFO 的独立容量 |
| `issue.forward_ports` | 1 | port | 三输入三输出交叉开关的前向端口；当前每类仅观察一个真实队首 |
| `issue.consumer_ports` | 1 | port | 三输入三输出交叉开关的消费者端口；当前每类仅观察一个真实队首 |
| `issue.adjoint_ports` | 1 | port | 三输入三输出交叉开关的伴随端口；当前每类仅观察一个真实队首 |
| `cache.instances` | 4 | cache | 每 Pod 一个缓存 |
| `latency.semantic_cache.ports` | 1 | port_per_cache_instance | 每个 Pod 缓存各自独立的目录/Active SRAM 接纳端口，不是四个实例共享的全局端口 |
| `cache.directory_banks_per_instance` | 4 | bank | 语义驻留目录 |
| `cache.directory_entries_per_bank` | 256 | entry | 语义驻留目录 |
| `cache.active_records_per_instance` | 1024 | record | Active SRAM |
| `cache.active_record_bytes` | 128 | byte | Active SRAM 记录 |
| `cache.active_sram_bytes_per_instance` | 131072 | byte | 128 KiB Active SRAM |
| `cache.miss_merge_entries_per_instance` | 32 | entry | 敏感性默认点 |
| `cache.multicast_destinations` | 4 | destination | 作用域多播器 |
| `cache.sector_bytes` | 64 | byte | 两个扇区传输 128 B 记录 |
| `compute.clusters_per_pod` | 5 | cluster | 可重构计算 Pod |
| `compute.fma_lanes_per_cluster` | 16 | lane | 可重构计算 Pod |
| `compute.fma_groups_per_cluster` | 4 | group | 四个四路算术组 |
| `compute.fma_lanes_per_group` | 4 | lane | 四个四路算术组 |
| `compute.transcendental_lanes_per_cluster` | 2 | lane | 可配置超越函数流水 |
| `compute.reduction_inputs_per_cluster` | 16 | input | 十六输入归约树 |
| `compute.microcontexts_per_cluster` | 4 | context | 计算簇微上下文 |
| `compute.relations_per_microcontext` | 8 | query lane | 每个物理微上下文内 RelationPacket 的八个查询 lane，不是八个独立 context |
| `compute.cluster_issue_slots_per_cluster` | 2 | slot | 二维 interaction 的两条配对算术路径 |
| `compute.microcontext_bytes_per_cluster` | 6656 | byte | 6.5 KiB 设计值 |
| `compute.owner_gradient_slots_per_cluster` | 2 | slot | 每个 owner cluster 的并发梯度 epoch |
| `compute.template_profiles[1].paths.forward` | pack 4，first 17，last 20，2 lane/cycle | cycle | GALA `ffadc13d`: `interaction_2d` |
| `compute.template_profiles[1].paths.adjoint` | pack 8，first 27，last 34，1 cycle/lane | cycle | GALA `ffadc13d`: `interaction_2d_adjoint` |
| `compute.template_profiles[2].paths.forward` | pack 8，first 27，last 34，1 lane/cycle | cycle | GALA `ffadc13d`: `interaction_3d` |
| `compute.template_profiles[2].paths.adjoint` | pack 24，first 47，last 68，3 cycle/lane | cycle | GALA `ffadc13d`: `interaction_3d_adjoint` |
| `query.reduction_banks` | 64 | bank | 双向查询执行单元 |
| `query.partial_sum_groups_per_bank` | 4 | group | 交错部分和 |
| `query.loss_fma_lanes` | 32 | lane | 查询损失单元 |
| `query.loss_queries_per_cycle` | 16 | query/cycle | 两条 FP32 FMA 对应一个 L1/L2 查询 |
| `query.adjoint_replay_lanes` | 8 | lane | 伴随重放流水 |
| `query.replay_queue_entries` | 256 | entry | 伴随重放队列 |
| `query.relation_window_entries` | 256 | window | 并发关系作用域窗口，不是关系记录数 |
| `query.relation_window_entry_bytes` | 32 | byte | 窗口基址、计数与三类引用 |
| `query.relation_store_banks` | 16 | bank | 关系链存储 |
| `query.relation_store_records` | 16384 | record | 512 KiB / 32 B 关系记录 |
| `query.query_volume_banks` | 16 | bank | 查询结果与梯度 SRAM |
| `query.query_volume_word_bytes` | 16 | byte | 查询 SRAM Bank 端口宽度 |
| `update.inflight_contexts` | 20 | context | 更新控制 FSM |
| `memory.channels` | 8 | channel | LPDDR5 接口 |
| `memory.channel_width_bits` | 32 | bit | LPDDR5 接口 |
| `memory.data_rate` | 6400 | MT/s | LPDDR5-6400 |
| `memory.peak_bandwidth_bytes_per_second` | 204800000000 | byte/s | 八通道总带宽 |
| `memory.ramulator_version` | v2.1.0 | version | Ramulator 2 固定版本 |
| `memory.ramulator_commit` | `38c51d40a976c6b07fbc09de869a7e08dc187d29` | git_commit | Ramulator 2 v2.1.0 发布提交 |
| `memory.ramulator_config_sha256` | 待冻结 | sha256 | canonical LPDDR5 配置身份 |

`relation.seed_fifo_entries` 等尚未在论文正文给出具体数值的字段不能在代码中猜测。首个实现应从总体架构源图、冻结配置或模块微基准配置中读取。若权威来源仍无值，使用带名称的实验参数，并在结果中标记 `design_parameter_pending_freeze`。不得根据端到端目标调节该值。

## 3. 允许调整的硬件参数

下列范围用于首个真实组合的资源闭合。参数只允许在范围内选择一次，冻结后覆盖全部模型、数据集和消融组合。

| 配置路径 | 默认值 | 允许范围 | 约束 |
| --- | ---: | ---: | --- |
| `issue.query_state_entries` | 2048 | 1024 至 4096 | 查询状态 SRAM 计入 2.75 MiB |
| `cache.active_records_per_instance` | 1024 | 512 至 2048 | 四个实例总量计入 2.75 MiB |
| `cache.miss_merge_entries_per_instance` | 32 | 16 至 64 | 端口与等待者存储同时闭合 |
| `query.relation_window_entries` | 256 | 128 至 512 | 32 B 窗口表项计入 2.75 MiB；关系记录使用独立 512 KiB 存储 |
| `relation.seed_fifo_entries` | 待冻结 | 默认值的 0.5 至 2 倍 | 由真实突发深度确定 |
| `pipeline.register_stages` | 模块配置 | 模块内重定时 | 不改变模块边界和吞吐资源 |

共享 SRAM 区域可以在总量内重新分配。每项容量必须由记录宽度推导字节数，不能只记录表项数量。Pod 数、每 Pod 计算簇数、FMA 数、超越函数流水数、融合发射候选宽度、共享 SRAM 总量和片外通道组织不属于可调参数。

## 4. 流水与存储时序

FMA、EXP、LOG、RCP、SQRT、SRAM、CAM、互连和寄存器流水延迟不写入模块源码。它们由 `configs/architecture/gala.yaml` 的 `latency` 与 `initiation_interval` 字段提供。

时序值按以下优先级确定。

1. 论文或冻结硬件设计中的周期数
2. 对应开源 RTL 的周期数
3. 固定频率下的模块微基准
4. 有来源的开源实现配置

不得使用端到端结果拟合时序。Ramulator 2 管理 LPDDR5 命令时序，因此 Python 配置不重复保存固定读写延迟。

## 5. 数值参数

| 配置路径 | 默认值 | 单位 | 来源 |
| --- | ---: | --- | --- |
| `numeric.dtype` | `fp32` | enum | 论文 Implementation |
| `numeric.rounding` | `round_to_nearest_even` | enum | 论文 Implementation |
| `numeric.flush_subnormal_input` | `true` | bool | 论文 Implementation |
| `numeric.flush_subnormal_output` | `true` | bool | 论文 Implementation |
| `numeric.exp_max_relative_error` | 7.49e-5 | ratio | 论文 Implementation |
| `numeric.log_max_absolute_error` | 1.69e-7 | absolute | 论文 Implementation |
| `numeric.rcp_max_relative_error` | 1.45e-7 | ratio | 论文 Implementation |
| `numeric.sqrt_max_relative_error` | 1.83e-7 | ratio | 论文 Implementation |

近似函数系数存放在独立只读表中，表文件包含生成脚本、定义域、误差扫描和哈希。禁止在 kernel 中散布系数。

## 6. 质量参数

统一质量口径使用下列参数；首个 Chest 参考结果验收前必须冻结，且不得在查看 GALA 功能重放结果后调整。

| 配置路径 | 默认值 | 单位 | 来源 |
| --- | ---: | --- | --- |
| `quality.data_min` | 0.0 | scalar | Chest `vol_gt.npy` 冻结范围 |
| `quality.data_max` | 1.0 | scalar | Chest `vol_gt.npy` 冻结范围 |
| `quality.ssim_window` | 11 | voxel | Chest 统一质量协议 |
| `quality.ssim_sigma` | 1.5 | voxel | Chest 统一质量协议 |
| `quality.ssim_boundary` | `reflect` | mode | scikit-image 0.21.0 三维 Gaussian SSIM |
| `quality.lpips_slices` | 每轴 `[42,85,128,170,213]` | index list per axis | 上游固定五切片位置扩展到三个正交轴 |
| `quality.lpips_network` | `alex` | model name | LPIPS 0.1.4 预训练网络 |
| `quality.lpips_version` | `0.1` | version | LPIPS 校准权重版本 |
| `quality.lpips_backbone_sha256` | `7be5be79...cdee02` | SHA-256 | torchvision AlexNet ImageNet checkpoint |
| `quality.lpips_calibration_sha256` | `df73285e...835c0` | SHA-256 | LPIPS v0.1 alex calibration 权重 |

缺少任一值时，质量入口必须拒绝生成正式 `quality.json`，不得回退到模型官方指标或隐式默认值。

## 7. 运行策略参数

| 配置路径 | 默认值 | 单位 | 作用 |
| --- | ---: | --- | --- |
| `preflight.long_run_threshold_seconds` | 3600 | second | 触发小时级任务预检 |
| `preflight.gpu_utilization_floor_percent` | 60 | percent | 触发工程效率审计 |
| `preflight.gpustat_interval_seconds` | 1 | second | GPU 利用率采样周期 |
| `ncu.watchdog_inactivity_seconds` | 300 | second | GPU、stdout 和分析器报告均无推进时的终止门限 |
| `ncu.watchdog_termination_grace_seconds` | 10 | second | watchdog 发出 SIGTERM 后等待精确进程组退出的时间 |
| `ncu.maximum_capture_job_count` | 61 | job | 非 NVTX range 的 Nsight Compute invocation 采集组上限；range 组单独计入 |
| `ncu.maximum_single_kernel_group_launch_count` | 64 | launch | 单一内核采集组的实际启动序号上限；超过后按序号分区 |
| `ncu.isolated_invocation_ordinals` | `CUDAFunctor_add<float>: [4862]` | launch ordinal list | 有实测 watchdog 证据的启动序号必须单独采集 |
| `preflight.warmup_iterations` | 10 | iteration | 排除首次编译和缓存建立 |
| `preflight.measure_iterations` | 50 | iteration | 预测总运行时间 |
| `preflight.inactivity_timeout_seconds` | 300 | second | 官方长任务无 GPU、日志或进程 CPU 推进时的终止门限 |
| `trace.chunk_events` | 自动调优后冻结 | event | 设备 trace chunk 容量 |
| `trace.archive_chunk_bytes` | 67108864 | byte | 紧凑 packet 归档单个压缩块的未压缩数组字节上限；单个超大 packet 可独占一块并超过该值 |
| `trace.inactivity_timeout_seconds` | 300 | second | 虚拟 trace 数据包或日志无推进时的终止门限 |
| `trace.progress_interval_seconds` | 30 | second | 虚拟 trace 结构化吞吐日志的最长间隔 |
| `diagnostic.throughput_report_interval_events` | 100000 | completed_event | 周期开发模式的事件采样间隔 |
| `diagnostic.throughput_report_interval_seconds` | 30 | second | 周期开发模式的最长静默间隔 |
| `diagnostic.inactivity_timeout_seconds` | 300 | second | 周期入口无完成事件或迭代推进时的终止门限 |
| `diagnostic.throughput_warmup_samples` | 3 | sample | 吞吐稳定判定前丢弃的预热样本数 |
| `diagnostic.throughput_stability_window_samples` | 5 | sample | 吞吐与周期投影同时判稳的连续窗口数 |
| `diagnostic.throughput_required_stable_windows` | 2 | window | 允许提前停止前连续通过的稳定窗口数 |
| `diagnostic.throughput_stability_relative_span` | 0.02 | ratio | 窗口内吞吐和周期投影允许的最大相对跨度 |
| `diagnostic.throughput_minimum_completion_fraction` | 0.55 | ratio | 越过主要增密阶段后允许诊断提前停止的最小迭代完成比例 |
| `trace.max_inflight_chunks` | 自动调优后冻结 | chunk | GPU 与 CPU 重叠深度 |
| `trace-sample.query_ranges` | 命令显式指定 | query range | quick-validation terminal 选择 |
| `trace-sample.max_events` | 命令显式指定 | event | quick-validation 闭包事件上限 |
| `trace-sample.max_total_events` | 命令显式指定 | event | quick-validation 单次回放总事件上限 |
| `trace-sample.max_dependencies` | 命令显式指定 | dependency | quick-validation 中间与输出依赖上限 |
| `trace-sample.scan_events` | 命令显式指定 | event | CPU/CUDA 顺序扫描块 |
| `trace-sample.scan_backend` | `auto` | enum | `cpu`、`cuda` 或记录实际回退的 `auto` |
| `trace-validate.scan_events` | trace capture chunk | event | 全量流式 validator 顺序扫描块 |
| `trace-validate.index_directory` | source trace directory | path | 全量 validator 的临时紧凑索引目录；不改变检查集合 |

自动调优只允许在首个真实样例上搜索 trace chunk 和软件缓冲参数。调优目标是减少主机同步与内存峰值，不改变任何硬件参数、事件内容或周期结果。冻结值记录在运行环境配置中。

## 8. 参数审计

持续集成执行以下静态检查。

- 扫描 `timing/modules` 中影响控制流的数字字面量
- 验证配置字段均在注册表中出现
- 验证单位换算只在配置加载层完成
- 验证同一参数没有模块私有副本
- 验证正式结果清单包含配置哈希

数字字面量检查只报告潜在魔数，不对代码生成形式证书，也不扩展为大规模审计任务。
