# 当前实现状态

本文件记录 `docs/10_IMPLEMENTATION_WORKFLOW.md` 的当前入口和证据，不替代正式实验结果。

## 最新性能主线

2026-08-30 完成语义驻留单机制的最小真实闭环。Raster+Voxel 双包样本包含 `505,391` 个事件、
`779,583` 条依赖、`11,698` 个物理状态请求和 `217` 个唯一 `(gaussian_id, state_version)` 键。
旧实现的 Base/`0101` 为 `30,505/26,793 cycles`，加速 `1.13854x`；实际结果已等于旧
Residency Oracle，证明容量和替换策略不是差距来源。返回路径原来仍逐请求读取 Active SRAM，
且只比较四个全局队首，无法看到同键请求在每个缓存实例内约 `42--50` 项的中位间隔。

修正后的作用域多播保持最老真实就绪请求为 leader，以每个活动槽的 ready-head 索引选择最多三个
同槽 follower。请求和返回均只合并真实就绪事件，不删除事件或依赖。`0101` 的片外状态请求由
`11,698` 降到 `217`，Shared SRAM 访问降到 `3,208`，完成 `2,961` 次真实广播；四目的理想
读取下限约为 `3,142`，当前只高 `66` 次。该阶段约为 `1.22x`，说明缓存机制已接近自身路径上限，
但状态路径尚未充分落到端到端关键路径。

随后发现公共 Fusion 路径错误地让 C 关闭的 Base 使用了三宽无冲突发射和 `2/1/2` 类端口，
违反 Base 每周期按到达顺序只发射一个合法任务的合同。因此此前 Base `23,164 cycles`、语义驻留
`15,221 cycles` 和 `1.521845x` 加速均受 Base 污染，现全部撤销，不再作为性能结论或后续门槛。
三条公共 FIFO、Bank 队首索引和逐查询依赖继续保留；三宽发射与扩展类端口只在 C 开启时使用。

优化前，合同修复后的相同 Raster+Voxel 双包样本中，Base `0000=30,497 cycles`，语义驻留
`0101=24,947 cycles`，实际加速为 `1.222472x`。语义驻留把片外状态请求从 `11,698` 降至
`217`，Shared SRAM 访问从 `23,396` 降至 `3,208`，完成 `2,961` 次真实广播，且两者逐类型
事件计数完全相同；查询负载规则和历史候选比较在两者中均为零。相对 `1.443210x` 派生参考，
目标周期预算约为 `21,131 cycles`，当时还差 `3,816 cycles`。旧必要下界的限制项是 C 关闭时
`24,164` 个 Fusion 任务的全局单项发射，因此进一步提高缓存命中率无法达到参考。

语义驻留现增加三目的同状态前向 Fusion 束：只有 B 与 D 同时开启、C 关闭时，同一
`(gaussian_id, state_version, template_id)`、已就绪且位于不同查询状态 Bank 的真实前向任务
才共享一次物理控制提交；Base 和其他单机制不享受。相同样本上 `0101=19,210 cycles`，相对
Base 加速 `1.587559x`，超过 `1.443210x` 派生参考约 `10.0%`。运行形成 `3,090` 个语义束并
合并 `5,841` 个 follower，Fusion 仍完整接纳并完成 `24,164/24,164` 个逻辑任务，缓存片外请求
仍为 `217`；Base 对照完整复现 `30,497 cycles`。四目的试验因越过 FIFO 到达链引入约
`4.6M` 次查询归约 Bank 冲突，退化到 `31,848 cycles`，已否决并保持三目的设计。结果位于仓库外
`GALA-runtime/records/r2_gaussian_chest_semantic_contractfix_v1_*`，仍标记为
`quick_cycle_validation`；三目的结果位于
`GALA-runtime/records/r2_gaussian_chest_semantic_bundle_probe_v4_0101`，Base 对照位于相邻的
`semantic_bundle_probe_v1_0000`。两者均不是完整三万轮正式性能结果。

同一 trace 和冻结资源上的必要下界已按语义束重算。`11,698` 个前向任务形成至少 `4,009` 次
三目的提交，`768` 个消费者和 `11,698` 个伴随保持单项提交，总计至少 `16,475` 次物理 Fusion
提交；计入服务尾部后 Residency 下界为 `16,477 cycles`，相对 Base 的理论最高加速为
`1.850883x`。因此 `1.443210x` 参考不再被必要下界排除，当前 `1.587559x` 实测位于可解释
区间。新下界记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_semantic_bundle_v1_bounds`。

语义驻留闭环后，同一 `505,391` 事件样本的 Full `1111` 快速复验完成全部事件，得到
`15,313 cycles`，相对同一 Base `30,497 cycles` 为 `1.991576x`。缓存仍只有 `217` 次片外
请求和 `2,961` 次多播，Fusion 与 ComputePod 分别完整完成 `24,164` 和 `35,094` 个逻辑任务；
但 C 开启时按当前合同不叠加语义束，记录到 `9,713` 次 Fusion 输入队列停顿，双向查询数据通路
与 ComputePod 资源仍是主要等待来源。相对 `2.648560x` 派生 Full 参考的周期预算约为
`11,514 cycles`，当前还需减少 `3,799 cycles`，即当前周期的 `24.81%`。同一必要下界为
`10,898 cycles`，目标尚未被排除，但只比下界高 `616 cycles`；下一步先分解联合路径并判断是否
需要让语义束与 C 的通用调度以不增加候选宽度的方式协同，不启动十六项消融。结果位于仓库外
`GALA-runtime/records/r2_gaussian_chest_full_after_semantic_bundle_v1_1111`。

2026-08-30 在提交 `2ce6eb1` 中完成并推送了共同周期路径的两项实现。查询伴随路径现在按
真实八 lane RelationPacket 在九条重放流水上做 work-conserving 分派：已接纳 packet 的剩余
lane 可在后续周期继续占用空闲 replay lane，直到全部 lane 完成；离线 `CycleEngine` 与在线
`CycleReplaySession` 使用同一逻辑。ReadyCandidate 有界队列增加 O(1) 最差可见项索引，避免
容量满时反复扫描所有候选。这两项都只改变模拟器的合法调度和软件索引，不改变事件、依赖、
物理 packet、端口、Bank、FIFO 容量或顶层资源包络。新增 packet 部分完成、逐 lane 周期、在线/
离线一致性和队列顺序回归；全仓结果为 `365 passed, 1 skipped, 2 warnings`。

同一真实 `600:601` quick trace（`631,244` 事件、`933,642` 依赖、`15,600` 物理伴随 packet）
的单个 Full 快速复验完成全部事件，得到 `16,875 cycles`，Base 为 `32,727 cycles`，相对
Base 加速 `1.939378x`。Fusion `31,328` 个 packet 和查询 `119,450` 个事件均完整完成，
九 lane replay 共分派 `103,594` 个伴随 lane，pending packet 最终为零；结果目录为
`/tmp/gala-r2-1111-restored-shared-fifo-v1`，作用域仍是 `quick_cycle_validation`，不能作为
正式三万轮性能结果。相对派生 Full 参考 `2.648560x`，当前还需减少约 `4,518 cycles`，
即当前周期的 `26.78%`；该参考不被当作 quick trace 的闭包上限或通过门槛。

同日验证了一个共享 FIFO 入队 Bank 平衡方案后予以否决：它保持 `144` 项总容量但按驻留
Bank 数选择待入队任务，完整事件虽完成，周期恶化为 `41,886`，墙钟约 `187.8 s`，并破坏
跨迭代完成顺序。代码、测试和结果均未保留为主线；该结果只作为失败实验记录，防止后续重复。

2026-08-30 完成在线周期模拟器的共同软件路径优化。事件类型、模块阶段和 Fusion task 类型改为
有界前沿内缓存；语义缓存与关系构造分区按输入 packet 批量计算；NumPy 事件行保留 packet 视图，
不再逐事件复制；QueryReplayTracker 与 OwnerGradientTracker 只批量扫描相关事件；Query State
释放由全表扫描改为状态变化时登记成熟候选，并实时维护物理 pack 的 lane 占用。上述修改同时作用于
Base 和全部消融，不改变事件、依赖、模块、端口、Bank、队列或硬件周期。相同两百万前沿 profile
中，约 30 秒内接受的事件由 `1,418,658` 增至 `1,689,645`，提高 `19.1%`；相同约 48.5 秒
调度 profile 中完成事件由 `195,766` 增至 `275,547`，提高 `40.8%`。两者都是模拟器软件吞吐，
不能相加，也不是 ASIC 加速比。

后续 profile 发现在线入口先运行独立精确关系波前预检，再由流式 expander 重建一次实际
query-pack 调度计划，两次都遍历相同真实关系。在线路径现直接生成并复用实际调度计划；独立
`relation_store_wavefront()` 仍用于离线容量审计，报告口径不变。真实完整 Raster packet 的
30 秒 cProfile 中，接受事件由 `2,039,757` 增至 `2,551,863`，再提高 `25.1%`；重复扫描原占
`5.36 s`，现已退出在线热路径。实际调度计划峰值为 `196,980` 条关系记录，仍低于冻结容量
`218,448`；该计划比离线精确波前峰值 `196,918` 更保守 62 条，不会放宽容量检查。

关系调度计划的每个 query pack 原先还会把已经生成的候选列再次执行 `np.unique`，只为取得逻辑
关系数和物理记录数。当前实现直接复用同一真实 mask 活动矩阵，以非零元素数和非空候选行数取得
两个精确计数；关系列生成仍从该矩阵恢复原有 query-major 顺序。完整 Raster packet 的计划生成
由约 `3.00 s` 降至 `1.48 s`；逻辑关系仍为 `90,769,547`、物理记录仍为 `13,170,672`、
峰值仍为 `196,980`。Voxel packet 的物理记录仍为 `735,147`、峰值仍为 `24,437`。相同
30 秒真实 Raster profile 接受事件由 `2,551,863` 增至 `2,679,556`，再提高约 `5.0%`。
新增 Raster/Voxel 稀疏 lane 与边界 pack 回归逐包对照新计数和实际生成列；Native Ramulator 2
重跑仍得到 Raster Base `1,344 cycles` 和 Voxel Base `26,739 cycles`，事件全部完成且前沿
quiescent。完整回归为 `359 passed, 1 skipped, 2 warnings`。该修改只减少共同模拟器软件工作，
不改变任何事件、依赖、关系、资源或硬件周期，并公平作用于 Base 和全部消融。

随后对在线事件登记做了同样的共同软件路径优化。完整 Raster 的前 `5,001,891` 个真实事件中，
`677,229` 个为零依赖、`3,491,588` 个为单依赖，合计占 `83.34%`；高扇入仍集中在真实
Query Reduction、Query Close 和 Consumer。登记器现在直接读取 packet 的依赖 offset/count 列，
为零依赖和单依赖使用直接路径，按 packet 批量累计事件类型、接受数和前沿峰值，并利用全局连续
事件流合同消除每行三次不可能命中的重复 ID 查询。多依赖事件、Consumer 的全部 Query Reduction
依赖与 credit、生命周期事务依赖，以及 Residency 开启时 Cache Return 的唯一 Cache Request
依赖均原样保留。相同 cProfile 条件下，新路径在 `25.05 s` 已接受 `2,797,571` 个事件，超过旧
路径 `30 s` 的 `2,679,556` 个；按这两个保守采样点计算，事件登记率提高约 `25.0%`。不带
profiler 时 `25.00 s` 接受 `4,704,129` 个事件。Native Ramulator 2 再次精确复现 Raster Base
`1,344 cycles` 和 Voxel Base `26,739 cycles`，前沿峰值分别仍为 `1,021` 和 `131,035`；
完整回归为 `360 passed, 1 skipped, 2 warnings`。这些都是模拟器墙钟优化，不是 ASIC 加速比。

当前代码用 Native Ramulator 2 重跑 `631,244` 事件、`933,642` 依赖的跨迭代样本，Base 和
完整 GALA 分别为 `32,727` 和 `17,148 cycles`，相对 Base 加速仍为 `1.908502x`；墙钟分别为
约 `39.2 s` 和 `32.0 s`。四路并行十六项消融逐项精确复现既有矩阵，所有变体事件计数相同，
`1111` 与完整 GALA 周期完全一致。新记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_fifo144_runtimeopt_v1_ablation.csv`；
资源快照固定为 `configs/architecture/gala-resource-usage.json`，并由回归测试逐字段对照配置注册表。
本次完整回归为 `357 passed, 1 skipped, 2 warnings`。哈希仅随结果记录，不参与执行通过条件；
同套件 AGX Orin 实测仍不可用，因此全部 `speedup_vs_orin` 保持 `unavailable`。

2026-08-30 修复了在线容量 continuation 的三个软件侧闭合问题。前沿容量触发排空时，若当前
ready 工作只等待尚未摄取的后续 continuation，执行器现在把控制权返回给紧凑 packet 生产器；
普通事件包、迭代关闭和 `finish()` 仍使用严格死锁检查。Base 阶段屏障移走被阻塞的伴随事件后，
调度器会在同一模拟周期继续扫描后续消费者；严格 replay 队列还会在消费者尚未实际就绪时阻塞
伴随物理包，而不是在预约时抛出不变量错误。这些修改不改变事件集合、依赖、物理 packet、硬件
资源或周期语义。新增回归覆盖多窗口 Raster 形态和 stage-gated replay，当前完整回归为
`355 passed, 1 skipped, 2 warnings`。

同日用真实一迭代 compact archive 做了两类在线端到端验证。Raster tile 0 包含 `9` 个候选、
`816` 条关系、`256` 个查询和 `5,673` 个逻辑事件。最新代码与干净提交 `dccf41f` 在相同
Native Ramulator 2、冻结配置和 `1,024` 个软件前沿槽下均得到 Base `1,344 cycles`；完整
GALA 为 `845 cycles`，相对 Base 为 `1.590533x`。两者均完成 `5,673/5,673` 个事件，峰值前沿
分别为 `1,021/1,022`，四个关系窗口全部释放。此前记录的 `891 cycles` 没有保存策略和复现命令，
对应结果文件也已被另一实验覆盖，现无法在其声称的同配置下复现，因此撤销该周期，不能作为 Base、
完整 GALA 或回归基准。Voxel tile 0 包含 `212` 个候选、`80,063` 条关系、`512` 个查询和
`482,126` 个逻辑事件；在 `131,072` 个软件前沿槽下，最新代码与干净提交 `f37beea` 均完成
`482,126/482,126` 事件，Base 周期 `26,739`，峰值前沿 `131,035`，八个关系窗口全部释放，
关系记录峰值 `5,304`。旧 `26,623` 同样缺少可复现策略记录，由该 A/B 结果取代。两次结果均为
真实 packet 的在线回放，不是合成关系
计数；软件前沿上限与硬件关系 SRAM 容量分别记录。

同日以四路并行、真实 Ramulator 2 后端复验 `600:601` quick trace 的十六项消融。`631,244`
个事件和 `933,642` 条依赖在每个变体中完全一致，周期为：`0000=32,727`、`0001=31,812`、
`0010=20,147`、`0011=18,111`、`0100=32,727`、`0101=31,812`、`0110=20,147`、
`0111=18,111`、`1000=32,522`、`1001=31,581`、`1010=30,875`、`1011=17,148`、
`1100=32,522`、`1101=31,581`、`1110=30,875`、`1111=17,148`。相对 `0000` 的 Full/`1111`
加速为 `1.908502x`，所有 Orin 字段继续保持 `unavailable`。

同日尝试以真实 Ramulator 2 做完整一迭代 Raster+Voxel 在线回放。该 archive 的逻辑事件总数为
`577,268,206`；在五分钟门控内持续推进，`270 s` 时完成 `6,107,277` 个事件，吞吐约
`22,618 events/s`，没有形成完整结果，进程按门控停止。该运行证明流式路径可持续推进，但按
当前 CPU 周期调度吞吐完成一迭代约需七小时，不能作为正式三万轮结果，也不再无条件重复。
正式性能入口仍要求完整 archive；在取得可接受的事件内核吞吐或稳定吞吐诊断前，不标记
`formal_performance_eligible`。

2026-08-29 在同一 `631,244` 事件、`933,642` 依赖的完整 quick trace 上完成候选 FIFO
容量扫描。`candidate_fifo_entries=144` 是当前稳定最优点；`160` 及以上会触发跨 Bank
反压并退化。以 144 项冻结值重跑的十六项并行消融全部完成，Base 为 `32,727 cycles`，
Query 为 `20,147 cycles`（`1.624411x`），Residency 为 `31,812 cycles`（`1.028763x`），
Full/`1111` 为 `17,148 cycles`（`1.908502x`）。所有变体保持 `631,244` 个事件，Full
与 `1111` 周期完全一致。该结果仍属于 `quick_cycle_validation`，但不再以闭包上限作为优化
退出条件；后续性能判断以真实周期、事件完整性和资源合同为准。

同日将离线依赖反向索引改为稳定 NumPy 排序构造。它保持每个依赖的原有 dependent-event
顺序，将单次索引构造从约 `1.5 s` 降至约 `0.12 s`，只减少模拟器前处理时间，不改变周期
语义或事件集合。144 配置加载检查为 `ready=true`，当前配置 SHA-256 为
`f342ef47a3b7e9e8c1be4946a8b94a8bf46665033656313df61e83644eccf89a`。

同日完成当前 144 配置的两个受资源约束 Oracle 回放。查询 Oracle 的 Base、实际机制和
future-visible 成员分别为 `32,727`、`30,875` 和 `27,214 cycles`，portfolio 选择
future-visible 成员，相对 Base 为 `1.202580x`；Residency 的三个成员为 `32,727`、
`31,812` 和 `31,812 cycles`，portfolio 选择实际成员，相对 Base 为 `1.028763x`。两次
回放均完整处理 `631,244` 个事件和 `933,642` 条依赖，结果状态为 `passed`，但作用域仍是
`quick_cycle_validation`，因此 `formal_performance_eligible=false`。按本记录约定，查询的
机制可覆盖周期为 `5,513`、不可覆盖周期为 `27,214`；Residency 的机制可覆盖周期为
`915`、不可覆盖周期为 `31,812`。对应的机器可读分解分别位于
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_fifo144_v1_query_oracle/oracle_scope.json`
和
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_fifo144_v1_residency_oracle/oracle_scope.json`。
静态锚点派生目标只作为诊断字段记录，不再作为快速回放的完成门槛；Residency 目标在该样本
上已被必要下界排除，Query 目标仍未被必要下界排除。

## 最新运行时优化

2026-08-29 将离线 `CycleEngine` 和在线 `CycleReplaySession` 的 Fusion 待入队缓冲按
FORWARD、CONSUMER、ADJOINT 三个源分别维护。每个源仍保持原有到达顺序和 32 项有界 FIFO，
但某个源满时不再扫描其他源的阻塞条目；该修改只优化模拟器软件路径，不改变硬件队列、
事件集合、调度选择或周期语义。同一 `631,244` 事件完整 trace 和同一逐请求内存返回表复验
得到 `18,163 cycles`，事件计数与 Fusion/ComputePod 停顿计数均与优化前一致；墙钟由约 `27.2 s`
降至 `24.6 s`（约 `9.5%`）。优化后的完整回归为 `344 passed, 1 skipped, 2 warnings`。

同日完成哈希门控审计：冻结记录、NCU plan、NCU profile binding 和周期消融入口均把哈希
作为可选审计字段，不再因哈希缺失或内容不一致阻塞执行；结构、启动序列、进程归属、GPU
利用率 watchdog、事件集合和资源约束检查仍保持启用。哈希字段仍可随结果写出，但不参与
通过或拒绝判断。

当前 R²-Gaussian + Chest 的静态锚点已按 2026-08-29 外部规范分解为两套端点：
CLAMP-CUDA 软件端点为 A0B0/A1B0/A0B1/A1B1 `1.000x/1.254x/1.282x/1.482x`，
匹配 GALA 硬件端点为 `2.430x/3.513x/3.507x/6.436x`（均相对 CUDA_OPT）。硬件
端点已经包含匹配的编译元数据与体系结构支持，不能再乘软件端点。硬件端点相对 Base ASIC
的派生参考约为 `1.446x`、`1.443x` 和 `2.649x`；它们是 `anchor_not_measurement`，
不替代同套件实测，也不构成当前 quick trace 的理论上限或硬周期目标。

## 当前跨迭代完整闭包结果

2026-08-29 当前性能入口继续使用真实 `600:601` CUDA 窗口中的八个连续查询包，完整保留
`631,244` 个事件、`933,642` 条依赖、`103,594` 条逻辑关系和 `15,600` 个物理
RelationPacket。扫描和回放均不设置事件数或依赖数闭包上限；只要完整闭包能在当前分钟级入口
完成，就不引入截断结果。该样本仍是 `quick_cycle_validation`，不能冒充三万轮正式结果。
最近一次 `Full` 无上限复验在约 `35.3 s` 墙钟内完成全部事件，结果目录为
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_banked_fifo_v2_full_recheck/`。

当前冻结的低成本资源调整为：前向和伴随各两个可选 Fusion 出口，消费者仍为一个出口；消费者
候选槽空闲时在前向与伴随之间轮转借用，但总候选和总发射宽度仍为三。每个 cluster 的
owner-gradient 槽由两个增至三个，并将预约点修正为 owner ComputePod 实际接纳；伴随重放流水
由八条增至九条，八 lane RelationPacket 保持不变。上述参数统一用于 Base、实际机制和全部消融，
没有按变体单独调参。

同一闭包、同一 32 项配置的十六项四路并行消融已经作为上一版基准全部通过。该历史 Banked
FIFO 结果为 Full `18,163 cycles`；144 项容量扫描是在此基础上得到的当前配置，仓库外记录位于
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_banked_fifo_v1_*`。

必要下界也使用完整 `631,244` 事件闭包。Query 和 Full 均为 `11,514 cycles`，相对 Base 的
必要条件理论最大加速为 `2.842366x`；Residency 为 `31,330 cycles`，理论最大加速仅
`1.044590x`。因此 Full 的 `2.648560x` 静态派生参考不再被资源必要条件排除，但当前实际
`1.743116x` 仍未达到：实际周期还需从 `18,775` 降到约 `12,356`，即再减少 `6,419 cycles`
（当前周期的 `34.19%`）。Query 已超过其 `1.445679x` 派生参考；Residency 的 `1.443210x`
派生参考在这个样本上仍被必要下界排除。下界报告位于仓库外
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_borrow3slot9_v1_bounds/`；它只用于
排除不可能配置，不再作为优化必须逼近的目标。

已排除的继续扩容方向包括候选/发射总宽度、超过 144 项的 FIFO、重放流水、owner-gradient 槽、ComputePod
接纳能力、microcontext、关系构造、语义缓存端口和内存延迟。当前 Full 的主要剩余时间在依赖
就绪到 Fusion 发射之间，而不是 Fusion 后端。闭包必要下界仅保留为诊断信息，不再作为快速回放
或工程优化的阻塞条件；下一性能任务是减少物理 RelationPacket/Fusion 工作量或改善跨查询完成顺序，
同时保持完整事件集合和低成本资源包络。
当前冻结配置校验为 `ready=true`、无 pending 参数，配置 SHA-256 为
`f342ef47a3b7e9e8c1be4946a8b94a8bf46665033656313df61e83644eccf89a`；完整测试正在按本次
配置变更复验。144 项容量扫描的原始记录位于当前 quick-trace 实验目录，历史 32 项记录
仍只作为基准对照。

在提交 `63e848d` 修正 Query future-visible 路径的冻结端口语义后，同一完整闭包的 Query
portfolio 为 Base `32,727 cycles`、实际 Query `20,362 cycles`、future-visible Query
`19,863 cycles`；未来信息只比实际机制再减少 `499 cycles`。Residency portfolio 为 Base
`32,727 cycles`、实际 Residency `31,812 cycles`、future-visible Residency
`31,812 cycles`；未来驻留选择不再增加收益。将 future-visible Query 与实际语义工作集、目录、
容量、Bank 和 Ramulator 路径联合后得到 `18,213 cycles`，比实际 Full 的 `18,775 cycles`
再减少 `562 cycles`，相对 Base 为 `1.797012x`，但离静态 Full 派生参考的约
`12,356-cycle` 预算仍差 `5,857 cycles`。该联合运行保持三候选总宽度、三条独立 32 项 FIFO、
`2/1/2` Fusion 出口、八个查询状态 Bank、三个 owner-gradient 槽、九条重放流水及冻结内存接口，
完成全部 `631,244` 个事件；记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_borrow3slot9_v4_joint_oracle_diagnostic/`。
它是 best-known 受约束诊断而非已证明的数学上界，现有证据已排除“继续扩大 Query/Residency
候选搜索”作为补齐 `5,857 cycles` 的主路径，下一项转向所有变体共享的关系、查询重放和
ComputePod 执行路径。

随后完成的 Banked Fusion FIFO 工程优化保持每个 F/C/A 源共享 32 项容量、总候选宽度 3、
`2/1/2` 出口和八个查询状态 Bank，只增加每源 `224 B` 的 Bank 队首索引与控制元数据，
并将 Bank/source 轮转指针限制为真实 Fusion 发射提交后推进。全套回归为 `343 passed, 1 skipped,
2 warnings`。同一完整 `631,244` 事件闭包的严格完整回放得到 `18,163 cycles`，相对旧 Full
`18,775 cycles` 少 `612 cycles`，约为 `1.801850x` Base；Fusion 实际记录了 `4,429` 次
Bank 队首选择，其中 `3,910` 次绕过第二个全局到达项。四路并行的十六项完整闭包消融已通过：
`0000=32727`、`0001=31812`、`0010=20608`、`0011=18856`、`0100=32727`、`0101=31812`、
`0110=20608`、`0111=18856`、`1000=32466`、`1001=31581`、`1010=19867`、`1011=18163`、
`1100=32466`、`1101=31581`、`1110=19867`、`1111=18163`。记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_banked_fifo_v1_*`；其
`quick_cycle_validation` 作用域保持不变，配置哈希更新为
`61ab78bf4f78c05b53dd8fee1b9d60b411720cbcdd641aca162eb57befc0a39b`。距离约 `12,356-cycle`
静态参考仍差 `5,807 cycles`，因此该优化是已验证的工程收益，不是最终目标达成。

- 2026-08-29 查询调度最小真实跨迭代验证已扩大到每轮四个十六查询物理行。样本包含
  `631,244` 个事件、`933,642` 条依赖、`103,594` 条源 trace 真实逻辑关系和 `15,600`
  个物理关系包；CUDA 顺序扫描约两分半钟完成，不遍历 optimizer 依赖闭包，也不设置闭包
  事件或依赖上限。Base 为 `34,174 cycles`，仅查询负载规则为 `34,030 cycles`
  （`1.004232x`），仅重叠引导发射为 `23,235 cycles`（`1.470798x`），联合 Query 为
  `22,750 cycles`（`1.502154x`），Query portfolio 为 `22,365 cycles`（`1.528013x`）。
  联合机制取得 portfolio 可消除周期的 `96.7398%`，并比 `1.445679x` 静态派生参考的
  `23,638.72-cycle` 比例点低约 `888.72 cycles`。联合运行恢复 `64` 个跨轮查询历史，执行
  `17,558` 次历史候选比较、`22,256` 次负载规则评估并改变 `3,425` 次选择。五组运行均
  完成全部事件且状态为 `passed`，记录位于仓库外
  `GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_query_v1_*`；该样本仍固定为
  `quick_cycle_validation`，不冒充完整 30,000 iteration 正式性能。

- 2026-08-29 修正 Base 的阶段隔离：A1 关闭时，消费者梯度先写已有 query-volume SRAM，
  不提前占用 256 项 query replay queue；只有当前关系窗口的全部前向和消费者完成后，物理
  ADJOINT packet 才预约相关 query 并进入重放。离线与在线执行器使用同一合同。修复前 Base
  错误允许少量消费者/伴随重叠；修复后同一 `846,012` 事件、`1,289,964` 依赖 quick trace
  的 Base 为 `46,743 cycles`，Residency 为 `43,162 cycles`，相对 Base 为 `1.082966x`，
  两者均只增加 `36 cycles`。这证明隔离错误真实存在但不是当前性能缺口的主因。权威 C++
  当前按 relation window 发出 full-stage completion，但尚未完整体现文字规范要求的“全部消费者
  完成后才开始伴随”，因此保留为后续 C++ 对拍缺口，不能把这 36 cycles 宣称为性能优化。

- 修复后的必要下界分析计入 C 关闭时 Fusion 的冻结全局单发射合同。Residency trace 含
  `19,777` 个物理 FORWARD、`768` 个 CONSUMER 和 `19,777` 个物理 ADJOINT 任务，共
  `40,322` 个 Fusion 任务；latency `3`、II `1` 给出 `40,324-cycle` 必要下界。相对修复后
  Base，Residency 的理论最大加速仅为 `46,743/40,324=1.159186x`，所以完整 Chest 锚点派生的
  `1.443210x` 在这个双窗口 quick trace 上已被资源合同排除。报告位于仓库外
  `GALA-runtime/records/r2_gaussian_chest_strict_stage_gate_v1_bounds/`，明确标记为
  `quick_cycle_validation`；Query 和 Full 的共同必要下界为 `19,779 cycles`，分别对应乐观
  最大加速 `2.363264x`，但这不是可达调度预测。完整回归为
  `329 passed, 1 skipped, 2 warnings`。

- 2026-08-29 修正 Query future-visible 验证路径的两个作用域错误：前向、消费者和伴随现在各自
  使用独立的 32 项候选 FIFO 容量，不再误共享一个总 32 项容量；future 权重只改变 Fusion 候选
  选择，不再重排关系构造、ComputePod、查询归约等机制覆盖范围外的模块。修正后的 future 成员
  在同一真实 quick trace 上完成全部 `846,012` 个事件，周期由旧实现约 `45,728` 降为
  `34,675 cycles`，但仍比实际 Query 的 `34,471 cycles` 慢 `204 cycles`。因此当前 portfolio
  只能称为 `best_known_not_proven_upper_bound`，不能冒充文档要求的最优 Oracle；资源必要下界
  `19,779 cycles` 仍只证明完整 Chest 的 Query 派生参考在资源上未被排除。结果位于仓库外
  `GALA-runtime/records/r2_gaussian_chest_query_oracle_scope_fix_v1_future/`。

- 查询调度器此前虽然定义了 `R_s/R_p/H`，周期引擎却从未推进 round boundary，且查询状态释放
  会丢失历史，因此 A 的上一迭代支持域预测实际上始终无效。当前实现按真实 `iteration_id` 前进
  时将本轮 `R_s` 固化为下一轮 `R_p`，在活动查询状态释放后由历史 sidecar 保留，并在下一轮
  重新分配同一查询时恢复 `R_p/H`；若迭代前进时仍有未闭合 F/C/A，直接拒绝而不混合两轮状态。
  定向测试证明历史预测在真实 forward/adjoint 两队首发生目标冲突时改变 A 的选择，而关闭 A 的
  Base 仍按到达顺序单发射。当前 `r2_gaussian_chest_median_packets_v1` 的全部事件都属于唯一的
  `iteration_id=1`，只能验证 C 的跨阶段发射，不能测量 A 的历史预测收益；下一验证入口必须是
  相邻真实迭代的同查询样本，不能把单迭代结果包装成完整 A1B0 结论。

- 实际 Query 的 ComputeTelemetry 在同一 quick trace 上精确复现 `34,471 cycles`。FORWARD 从
  依赖就绪到 Fusion 平均等待 `2,618.26 cycles`，Fusion 到 ComputePod 平均仅 `3.04 cycles`；
  ADJOINT 从依赖就绪到 Fusion 平均等待 `1,828.13 cycles`，Fusion 到 query replay 平均
  `665.89 cycles`。四个 Pod 的 active-microcontext 面积为
  `112,621/134,140/107,609/134,231`，至少一簇活跃周期为
  `25,794/26,629/24,451/26,889`，最轻与最重 Pod 的工作面积相差约 `24.7%`。这说明当前 Query
  机制尚未充分改善负载均衡，优先分析 Fusion 排队、伴随 replay 和 Pod 映射，而不是增加前向
  ComputePod 接口。遥测位于仓库外 `GALA-runtime/records/r2_gaussian_chest_query_telemetry_v1/`。

- 2026-08-29 已将关系窗口记录生命周期改为物理 packet 级回收：每条 RelationPacket 的全部伴随
  lane 完成后立即释放对应关系记录，窗口表项仍保持 producer/forward/consumer/adjoint 四类引用
  全部清零才释放。对存在多个同时活动窗口的情况，较新窗口还必须预留较老窗口尚未追加的记录
  信用，防止 future-visible 调度挤占老窗口的最后记录而死锁。该修改不改变事件集合、依赖、关系
  packet 数、关系存储容量或片外通道；新增生命周期和信用预留回归后定向测试为 `106 passed`。

- 同一 `846,012` 事件、`1,289,964` 依赖真实双窗口 quick trace 的当前结果为：Base `46,743 cycles`，
  Query 实际机制 `34,471 cycles`（相对 Base `1.356x`），Residency 实际机制从修复前的
  `43,126 cycles` 变为 `43,162 cycles`（`1.083x`），Full `32,290 cycles`（`1.448x`）。关系
  存储容量停顿分别由旧 Base 的约 `8.28M`
  降至 `6.12M`，Full 降至约 `4.51M`；Full 较合同修正后的 `34,187 cycles` 减少 `1,897` 周期。
  Query 端到端反而比旧 portfolio 的 `33,246 cycles` 多 `1,225` 周期，原因是提前追加改变了
  Ramulator 请求到达序列；因此该优化只作为共同生命周期/可闭合性修复，不宣称单机制收益。
  future-visible Query 由关系容量死锁改为完整 `45,728 cycles`，仍不作为可达上界。

- 若仅把最新静态端点相对 Base 的比例机械应用到这个 quick trace，会得到 Query `32,333`、
  Residency `32,388`、Full `17,648 cycles`。这些数值现在只保留为无验收资格的比例投影：外部
  规范给出的是完整同套件端到端锚点，没有给出这个双窗口样本的周期分解；该样本也不包含完整
  30,000 轮训练、更新和集合修改。尤其 Full 投影低于冻结单端口合同的 `19,779-cycle` 资源必要
  下界，不能被称为该样本的目标预算。临时 `forward=2/adjoint=2` 多头 Fusion 试验只有 `27 cycles`
  收益，经验水位 `9,216` 的公平限制得到 Query `33,336`、Full `34,152`，均未形成共同收益，
  均已回退，不进入冻结配置。

- 2026-08-29 对当前 Full 单点增加了 ComputePod 遥测；同一 `846,012` 事件完整闭合并精确复现
  `32,290 cycles`。前向事件从依赖就绪到 Fusion 发射平均等待 `5,891.90 cycles`，Fusion 到
  ComputePod 发射平均仅 `3.04 cycles`；伴随事件从依赖就绪到 Fusion 发射平均等待
  `3,305.54 cycles`，Fusion 到查询重放平均等待 `358.43 cycles`。四个 Pod 的 active-microcontext
  占用面积为 `112,621/134,140/107,609/134,231`，但各 Pod 至少一个 cluster 活跃的周期为
  `29,409/29,565/29,443/29,593`，尾部不是由单个闲置 Pod 造成。物理包计划把 `140,520` 个伴随
  lane 准确合并为 `19,777` 个 ADJOINT stage，其中 `14,719` 个是满八 lane 包；RELATION、FORWARD、
  ADJOINT 和 GRADIENT_REDUCTION 的物理 stage 数及 lane 分布完全一致。因此权威 C++ 的伴随 lane
  合并已在 Python 模型中兑现，不能再次计作优化。遥测保存在仓库外
  `GALA-runtime/records/r2_gaussian_chest_relation_record_reuse_v2_full_telemetry/`。

- 2026-08-29 已将 ComputePod 资源占用回收改为按退休周期的堆/桶增量清理（提交
  `63ba797`）。该软件路径优化不改变事件集合、资源包络或周期结果，完整回归为
  `320 passed, 1 skipped, 2 warnings`。在完整物理 packet quick trace 上，Pod 内负载感知选路、
  关系窗口整窗容量预留，以及仅增加 Fusion 前向/伴随端口并开放多头观察的旧实验均未降低端到端周期，
  已回退；当前仅保留物理 packet 级记录回收与较老窗口信用预留，冻结顶层配置保持不变。

- 2026-08-29 从外部规范仓库 `main` 快进拉取后，修正了 Python 查询归约资源模型：权威 C++ 的
  `64` 个物理 Bank 每周期各接收一个输入，`partial_sum_groups_per_bank=4` 只表示 Bank 内活动
  partial entry 容量，不再被展开成 `256` 个发射槽；同时按 `query_tag & (banks - 1)` 固定映射并
  增加 2 的幂配置校验。定向回归覆盖 `query_tag=0/64` 同 Bank 冲突和 `0/1` 异 Bank 并行，
  全套测试为 `322 passed, 1 skipped, 2 warnings`。

- 2026-08-29 在同一 `846,012` 事件、`1,289,964` 依赖的中位物理 packet quick trace、同一冻结
  GALA/Ramulator 配置和资源快照上重跑合同修正后的周期。Base 为 `46,707 cycles`；Query portfolio
  的实际成员为 `33,246 cycles`（相对 Base `1.404891x`），future-visible 成员在关系窗口容量
  达到冻结上限后死锁，不能作为已证明上界。Residency portfolio 的实际成员为 `43,126 cycles`
  （`1.083036x`），future-visible 成员为 `43,132 cycles`；Full 联合入口为 `34,187 cycles`
  （`1.366221x`）。结果目录位于仓库外 `GALA-runtime/records/r2_gaussian_chest_reduction_bank_contract_v1_*`，
  全部标记为 `quick_cycle_validation`，不具备正式 30,000 iteration 性能资格，也不生成
  `speedup_vs_orin`。

- 上述合同修正后的 Full 仍完成全部事件、两个关系窗口和 `19,777` 个物理关系 packet。按最新
  Full 静态锚点相对 Base 的派生参考 `2.648560x` 换算，`46,707-cycle` Base 对应目标预算约为
  `17,635 cycles`，实际 Full 仍多 `16,552 cycles`（`48.43%`）。这不是千倍级收益，也不是正式
  30,000 iteration 性能结果；差距必须按周期分解继续做共同工程路径优化。

- 合同修正后的当前配置资源必要下界仍为 `19,779 cycles`，限制项是 Fusion 前向和伴随各一个
  真实队首、各一个发射端口。该下界证明对 quick trace 机械换算得到的 Full `17,635-cycle`
  比例投影与冻结资源合同不相容，但不否定完整同套件 `ASIC_A1B1=6.436x` 静态锚点。这个下界
  假设其余冲突全部消失，不是可达预测、Oracle 或静态锚点。当前实际差距主要集中在 Fusion
  输入队列、查询重放和 ComputePod issue；下一步按真实遥测分解逐项优化并对 Base、两个 Oracle
  和所有消融公平重跑，不通过扩大冻结的三候选发射宽度追赶比例投影。

- 缓存物理 RelationPacket 的命中或 miss-merge 路径现会完成 packet 的全部逻辑 lane，同时只由
  packet head 更新一次目录、Active SRAM 和填充状态。该修复消除了 Full 在 `cycle 3337` 留下
  非 head `CACHE_RETURN` 依赖的死锁。完整回归为 `319 passed, 1 skipped, 2 warnings`。

- 历史阶段门控修复（旧资源计费口径）的同一双窗口 quick trace 四点对照曾闭合：Base `125,214 cycles`，
  编译侧 A（`variant:1000`）`124,878 cycles`，架构侧 C（`variant:0010`）`124,938 cycles`，
  联合 AC（`variant:1010`）`124,633 cycles`，事件集合均为 `846,012/846,012`。相对 Base
  的实际加速依次为 `1.002691x`、`1.002209x` 和 `1.004662x`。停顿计数的主要项仍是
  ComputePod `compute_resource`、查询 `reduction_bank` 和关系存储 `relation_store_capacity`；
  因此当前只证明门控语义和周期闭合，尚未达到新锚点，也不具备正式 30,000 iteration 资格。

- 2026-08-29 已从外部模拟器 `main` 快进拉取并确认带分解锚点提交 `2fd7866` 已包含在已推送的
  `260f1c0` 中。当前静态目标按 A/B 配置分别记录软件端点与匹配 GALA 硬件端点：R²-Gaussian
  + Chest 的硬件端点为 A0B0 `2.430x`、A1B0 `3.513x`、A0B1 `3.507x`、A1B1 `6.436x`；
  相对 Base ASIC 的派生参考为 Query `1.445679x`、Residency `1.443210x`、Full `2.648560x`。
  这些值仍标记为 `anchor_not_measurement`，不参与周期参数拟合，也不冒充可达上限。

- 历史周期边界模型（旧 Base 口径）曾补齐查询损失 16 query/cycle 合同、query-volume SRAM 读写 Bank
  必要工作量、ComputePod 按 Pod 的资源下界，以及关系窗口/记录存储、回放队列、owner-gradient
  槽位和共享 SRAM 容量诊断。对同一 `846,012` 事件双窗口 quick trace 的静态计算约 16 秒完成，
  资源受限必要下界为 `23,280 cycles`，相对 Base `144,550 cycles` 的乐观最大加速为 `6.209x`。
  该值只用于排除不可能目标和定位工程优化空间，仍不等同于可达调度或正式性能结果；新增回归后
  全套测试为 `309 passed, 1 skipped, 2 warnings`。

## 已通过

- 2026-08-29 生命周期与 lineage 修复后的最小真实双窗口 Base 回放已重新完成：`846,012/846,012` 个事件、`1,289,964` 条依赖，原生 LPDDR5-6400 Ramulator 2 返回 `144,550 cycles`（500 MHz 下 `0.289100 ms`），记录位于仓库外 `GALA-runtime/records/r2_gaussian_chest_median_packets_base_post_lineage_v1/`。该结果只证明最新依赖实现仍能闭合同一 quick trace，不是正式 30,000 iteration 周期，也不是绝对性能锚点。

- 2026-08-29 虚拟集合修改 lineage 依赖已对齐 raw capture：`CLONE` 的 child 修改依赖 parent，`SPLIT` 的 parent 修改等待全部 child 修改完成；collection transaction 仍共享 begin frontier，optimizer commit 仍保持事务内并行。新增 clone/split 顺序回归，全套测试 `297 passed, 1 skipped, 2 warnings`。这只修正事件依赖真实性，不改变顶层硬件资源或性能锚点口径。

- 2026-08-29 在线虚拟生命周期事务依赖已修正：`UPDATE_BEGIN` 只连接 backward/state frontier；同一事务内各个 `UPDATE_COMMIT` 共享 begin 依赖，不再通过前一个 commit 或修改事件链式串行化；`UPDATE_END` 汇合本事务全部 commit/修改事件。新增依赖图回归，定向生命周期测试 4 项通过，全套测试 `296 passed, 1 skipped, 2 warnings`。该修复只消除模拟器人为串行化，不构成绝对性能锚点或硬件加速承诺。

- 2026-08-29 查询负载规则的独立接线已通过回归：A 单独开启时使用三条有界 F/C/A FIFO 的真实负载排序，同时保持 Base 单发射宽度；在线 `CycleReplaySession` 会排空调度 FIFO 和待融合输入。定向测试 2 项、全套测试 `295 passed, 1 skipped, 2 warnings`。当前没有可用于绝对性能合理性、数量级检查或 ASIC 达标判断的静态锚点；双窗口 `144,550 cycles` 仅是同一真实 trace、同一 Ramulator 后端和同一资源配置下的 Base ASIC 比较分母，不能解释为外部锚点、可达目标或性能上界。`speedup_vs_orin` 继续保持 `unavailable`。

- 2026-08-29 在同一 `r2_gaussian_chest_median_packets_v1` 双窗口真实物理包 trace、同一 LPDDR5-6400 Ramulator 和冻结资源包络上完成 Base、两类 Oracle、两类实际机制与联合入口对照。Base `0000` 为 `144,550 cycles`；实际 Query `143,663 cycles`（`1.006174x`），Query future-visible Oracle `144,177 cycles`，portfolio 胜者为实际 Query；实际 Residency `144,549 cycles`（`1.0000069x`），Residency Oracle `144,550 cycles`；完整 `full` 为 `143,662 cycles`，与十六项矩阵 `1111` 完全一致。所有事件均完成且结果目录状态为 `passed`，但 trace 仍是 `quick_cycle_validation`，不具备正式 30,000 iteration 性能资格。Base 的主要停顿为双向查询 `reduction_bank=15,212,818`、`query_datapath=587,748`，以及 ComputePod `compute_resource=473,396`；实际 Query 将前者降至 `15,199,619`，但端到端只减少 `887 cycles`。实际 Residency 的语义缓存请求由 `19,777` 降至 `547`，却只减少 `1 cycle`。这组分解说明当前不是千倍级可达提升，资源必要下界也不能充当目标或锚点。

- 查询调度运行时已对齐权威 Fusion Issue 合同：配置显式冻结每条候选源 FIFO 为 32 项；Query State 按八 lane pack 分配真实物理槽并在完整 F/C/A、生成关闭和归约可读后复用；候选冲突同时检查查询状态、归约键和目标资源；年龄由实际入队周期计算。离线与在线 consumer 均在最后一个查询归约依赖完成时选择真实完成时间最晚的 credit owner，支持一个归约被多个消费者共享。新增容量、槽复用、目标冲突、年龄和 owner 生命周期回归；编译与全量测试为 `293 passed, 1 skipped`。该实现尚未形成正式周期结果，下一入口仍是同一双窗口真实 trace 的同后端 Base/Oracle/实际机制对照。

- 固定的 R²-Gaussian 提交 `f2579bfddd9aac009cb797c8503bef8119bbd022` 可核验，官方 Chest 数据清单包含 153 个文件、元数据哈希和文件级 SHA-256。
- 官方 CUDA 扩展在 `gaussian-slam-official` 环境中使用 CUDA 12.1 工具链和 GCC 11 编译并通过最小 GPU kernel 调用。
- 官方 Chest 1/2 迭代真实 smoke 已在仓库外保存；1 迭代评估输出了官方路径的 PSNR、SSIM 和体重建文件。
- 配置加载器检查参数元数据、状态、值域和不可变快照；`config-check` 现在输出逐项 pending 参数并以非零状态阻止未冻结配置。
- `cycle-preflight` 已生成 `preflight.json` 与 `status.json`，逐项检查配置冻结、Ramulator 2 binding 和资源使用快照；失败状态使用工作流规定的 `failed_preflight`。
- `CycleConfig` 在提供资源使用快照时由注册配置推导顶层资源包络并强制校验 SRAM、Pod、计算通路和片外通道闭合。
- Ramulator 2 桥接边界已改为异步 `try_issue/tick/drain_completions`：周期内核与内存模型共同推进，支持并发到达、前端反压、64 B transaction 拆分和返回唤醒；每个逻辑请求写入 `memory_requests.parquet` 供逐请求核对。
- 仓库自带的 C ABI bridge 已对固定 Ramulator 2 v2.1.0 提交 `38c51d40a976c6b07fbc09de869a7e08dc187d29` 完成仓库外构建；真实 LPDDR5-6400 八通道配置 smoke 中，一个 128 B 读取拆为两个 64 B transaction 并在周期 36 返回。该配置尚未冻结为 canonical 正式配置。
- Ramulator bridge smoke 记录保存在仓库外 `GALA-runtime/records/ramulator2_bridge_smoke_20260826.json`，SHA-256 为 `c2e402375fc15ebed2e1027315e1817b9c05c9842bac455a3ddb6680748f633c`。首组合 input-freeze v3 已按提交 `53949ba` 和配置哈希 `8f9a249ffbc74b749a97313647f9a98785f873a75c128ec0716133e5fa1a6c50` 重新生成；训练快照 SHA-256 为 `0d607a04e68d61f1fb05cc98a7dbb8255fcd19bbf7501388ef6caa19a84c1482`，官方命令 SHA-256 为 `abbd3983f18ac492a7abd4ab3178d6f99f3689d35896e69febef437f88d2538d`，run manifest SHA-256 为 `70754a896e624f45643f04515eb984a30482a44e651d506a3a9bdc5583327cf7`，仓库外文件 SHA-256 为 `551f26223f223b8be80fae581b0355bdf4bbfcddd17d08bfd8fd03d7e89023d9`。其状态为 `planned`，且配置因硬件与运行策略参数仍 pending 而未 ready。
- 官方 native-reference 预检的早期 `gpu_busy_external` 记录仍保留为历史失败证据（`gdesmond` PID `2594715`，记录 SHA-256 `fa2c3cc5a8fb1255908a58af8baacaaa9d0f9b216436d9e8102ed7ba5236e722`），不再作为当前入口阻塞。释放 GPU 后按同一 input-freeze 重跑的 v3 预检已通过：记录位于 `GALA-runtime/records/r2_gaussian_chest_native_preflight_v3/`，`preflight.json` SHA-256 为 `216546e8cb0d13dec77ec91a9042dc2cb40e72508453a02681eaeb66245460d8`，报告 self-hash 为 `8b8e5d3be18444c204479a7c1c8d2387b105df72a7ee81a4c2514432e96d0d46`，60 iteration calibration 预测 30,000 iteration 为 `1300.9847366809845 s`，低于 `3600 s` 长任务门限。
- `native-reference` 入口已要求同一 input-freeze 与 `passed` native-preflight 才能启动官方完整命令；它写出流式 stdout/stderr、GPU 样本、TensorBoard `train/iter_time` 序列、统一质量指标、官方附加指标和运行 manifest。所有进入执行阶段的失败记录都绑定配置哈希、freeze manifest、仓库提交和官方命令复现信息；预启动复采样发现外部 compute 进程时也会写出 `failed_preflight/status.json` 并拒绝创建模型输出。当前使用 `failed_preflight` 记录的拒绝演练返回退出码 2，未创建模型输出。
- 融合发射前向、消费者和伴随三类端口已使用注册配置中的独立端口数与独立 II 状态，不再由单个聚合端口互相错误阻塞。
- CLAMP 事件 schema v2、批量 NumPy trace 存储、依赖/版本/释放校验和模块拆分的离散事件周期内核及步骤 2 入口已通过测试。哈希字段只用于记录，不再作为周期入口或实验入口的拒绝条件。
- 资源包络、长任务 GPU 利用率门和十六项消融矩阵的结构检查已通过单元测试。
- `49bf36d` 为十六项变体使用显式 `variant:<bits>` 策略，并在周期内核中落实模块在途容量和关系种子 FIFO 反压。
- `cde1a93` 为每个消融变体隔离记录内存完成表的消费游标，避免同一外部 Ramulator 记录被首个变体消耗。
- 真实 CUDA trace 旁路已接入官方 `R²-Gaussian` rasterizer/voxelizer：只在官方查询边界处于 grad-enabled 的训练路径捕获，排除 `no_grad` 质量评估、保存和报告调用；捕获完成后记录配置哈希、模型提交、数据清单哈希和 GALA 仓库提交，但不以哈希一致性阻塞周期 sink；Chunked sink 传输和依赖偏移重建已通过单元测试。
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
- 大型 raw-column 生命周期 pass 已实现当前版本、active/known Gaussian、cache read 闭合、gradient→optimizer commit 集合、update begin/end、Clone/Split/Prune lineage 和后继查询状态屏障检查。小型 raw trace 覆盖 optimizer、no-op optimizer、collection 及故障注入；真实一迭代 v9 的结构+生命周期全量回归耗时 `325.30 s`、峰值 RSS `8,803,612 KB`、进程 swap 0。该真实 trace 没有 optimizer/collection 事务，因此只证明无事务大规模回归；仓库外日志 SHA-256 为 `9d7638ba704efddbf02528a6777353a3f946de03b05ebfe464448814d3514721`，结构化记录 SHA-256 为 `6e67709cd45283dfcf8bf2836d4dae4f78301184d50325fb7732c5b77a652676`。
- 官方 30,000 iteration native reference 已在通过 v3 preflight 后完成，记录位于 `GALA-runtime/records/r2_gaussian_chest_native_reference/`，状态为 `passed`。端到端 wall time 为 `1629.5491471290588 s`，TensorBoard iteration time 总和为 `1494400.0303459167 ms`，GPU 样本数 `1356`、利用率中位数 `96%`、峰值显存 `2043674624 bytes`；统一质量为 PSNR `35.200574854081545`、SSIM `0.9380215966065076`、LPIPS `0.08493900671601295`。manifest、status、quality、gpu_reference 的 SHA-256 分别为 `ddb3cfb6b2cfd95140e2957a5588005c0a52cc815a7f7cee3f48c21e1ac11c1e`、`2bfdc8988831355d70f7c383dbb65db4b181147711e9edf026a4263aa63de7fa`、`9be91364fb69e989191d567160a934f1445dc4977c9002bb969d43c19c82c5b5` 和 `519404be6acb02dd3f53382dc03a954af41a4f3300fe5403c3019d1ca462eb06`。该结果闭合了官方训练与质量，但阶段 kernel/访存权重和 AGX Orin 校准向量仍 pending。
- 中间迭代 NSYS smoke 已使用 `iteration_overhead` 重新采集并导出；iteration 2 的 302 个 CUDA kernel 全部且唯一归属到详细 `gala_stage`，`kernel_coverage.status=complete`，未归属与多重归属均为 0。CUDA Event 的 0.1756 ms 边界残差只保留为诊断，不作为 kernel 覆盖 gate；归一化器现要求独立 NSYS exact-coverage 证据。
- 正式 GPU profiling campaign 已版本化：CUDA Event 计时覆盖完整 `1:30000`，9 个离散代表迭代覆盖初始化、增密前、早/晚增密、增密后、全部五个评估点、collection 和最终重建。NSYS bounded capture 使用同一连续官方训练进程内的 CUDA Profiler API 边界；双窗口 smoke 分别导出 iteration 2/3，两个窗口均为 302/302 kernel 唯一归属，对应 inventory JSON SHA-256 为 `6136d7bde9c833531be64f5520106c4046d4131cc4cb24eb33360f0d14abe2ab` 和 `9d776592250482cef0e99da2ed9481b24c1ecc9b71cd97d10b7bf6ba414a367a`。仓库外 campaign manifest SHA-256 为 `0f11a8f558e4cb2a7f4db37bff3bdf47c29ec8d08255604fe4c248c2e6a39980`；runner 绑定 input-freeze，拒绝手工 range、训练参数漂移、源码/数据身份漂移和请求迭代未实际执行。
- NCU v3 Job 1 已让官方训练自然完成 30,000 iteration，最终官方指标为 PSNR3D `35.183`、SSIM3D `0.936`、PSNR2D `49.752`、SSIM2D `0.985`；但 runner 正确拒绝该证据并记录 `failed_preflight/ncu_evidence_binding_failed`：NCU 实测 `3623` 个 launch，而 v3 固定计划预期 `3581` 个，触发 multiplicity 和 selected-launch-count mismatch。该历史结果位于 `GALA-runtime/profiles/r2_gaussian_chest_ncu_v3_job1/`，耗时 `2641.936972618103 s`，不能用于正式归一化，也未继续运行 v3 Job 2。
- 跨线程探针确认 PyTorch NVTX push/pop 栈不会传播到 autograd worker，而进程级 NVTX start/end range 可以覆盖 worker 发出的 kernel。正式插桩因此在调用实际 stage 前创建带 iteration、stage、call 和 campaign 身份的 `gala_ncu_stage` range，在调用返回后关闭；invocation jobs 使用全局 kernel ordinal 并排除动态 `collection`，collection 单独使用精确 range 全量捕获。该设计不按固定 multiplicity 展开，也不依赖不同训练间完全相同的动态 grid。
- 第二套正式稳定性运行 `GALA-runtime/profiles/r2_gaussian_chest_nsys_stability_v3/` 使用 `--capture-range-end=repeat:9` 并自然完成 30,000 iteration；退出码和 stage 状态均通过，9 个请求窗口完整，最终官方指标为 PSNR3D `35.179`、SSIM3D `0.935`、PSNR2D `49.700`、SSIM2D `0.985`。`stages.json` SHA-256 为 `48c0797507b63186d20bcfbdf3984ebc80d8f5dfba82353b42b82e473fae5651`；九份 inventory SHA-256 依次为 `ce240ede9a24d39e18f88f590db698d8b714df753a4f240038b0f354d62fb31e`、`b9a243370f3eb745abc98fc9978c340400273372e25d3d742a4bcf717e1c2652`、`03aeafe727d41b964efaaa420f75ea6bd012b5d1f02737e8d8ddca63a541e2e6`、`4756cbecbc3056ad7519fe9bc246a28c30263ef35cc56ab0551dfe448e5e41bb`、`0b20a3efc87198b8a55837b3f1440c90062b1a6bc1bbcfe57124bf43d0902cea`、`c6bc113d483b0f4ad28f2fa959572aaa4b7833c3f95ec5a8b70b34e3e1cdd635`、`3b5636bf8d523a1cd264f8d0f15b35c6ac0d37d33842610aad1d979bf6dea810`、`04606b4d77d67b5228b338fc045628a6d8b64b8a4713d5fcf4045d1d83c4db93`、`e5e2f6fbb9b8a0e9a05e57d63a7fdbbde0c1fc33ef5a18dd4e670672ab8fd90c`。两套 NSYS 的 `173097` 个非 collection launch 按 iteration/stage/call/kernel ordinal/name/block 得到相同选择序列 SHA-256 `234575b549d7ff79a7c876ff956b8256fef63c056daeb68600a6090f36894079`，同时保留 `4420` 处真实动态 grid 差异；collection 在 iteration 600/5000/10000/14900 的两次 launch 数分别为 `305/305`、`409/409`、`305/389`、`305/400`，后面三个窗口的精确 signature set 不同，因此只能按实际 range launch 全量测量。
- 正式 v4 plan 位于 `GALA-runtime/records/r2_gaussian_chest_ncu_signature_plan_v4.json`，内容 SHA-256 为 `cf17491c691c7d1731ab6bcf24e51a2cf641be7e958aa44c6d5166d2e766434b`，文件 SHA-256 为 `53e5960c8ee9d09f78d96ad772467dadfa90364b2a5dc3232551822805344834`。计划稳定性状态为 `passed`，覆盖 `174421` 个 NSYS launch、`1173` 个全局 signature、`2400` 个代表迭代 signature 和 `5418` 个必测样本；共 7 个顺序作业，其中 6 个 invocation job 和 1 个 collection range job，预测选择 `8939` 个 launch。runner 要求 clean worktree、逐文件实现哈希一致和自然完成的双 NSYS 稳定性，并拒绝 `--launch-count`、`--kill` 或提前终止。
- v4 Job 1--3 均自然完成 30,000 iteration 并逐项通过 launch binding、SourceCounters 和动态 SASS gate。Job 1 为 `1435/1435`，耗时 `2257.096601009369 s`；Job 2 为 `1634/1634`，耗时 `2068.0485558509827 s`，status 文件 SHA-256 为 `dcc584abb9246aa2deaed2b79b830a72640d61609290d9da8d65b75b08e3528c`；Job 3 为 `964/964`，耗时 `1835.250658750534 s`，status 文件 SHA-256 为 `5767d558789f34055309de867db6e2d326cbc249a2efe402d0a7bf3c8ef5df85`。三项累计覆盖 `2338/5418` 个必测样本，但在完整计划闭合前仍明确 `formal_performance_eligible=false`。
- v4 Job 4 的同一 direct-copy kernel replay 点连续两次进入 GPU 0%、stdout 和 report 均不更新的停滞状态。两次均只终止精确进程树并保留现场；首次 `watchdog_stop.json` SHA-256 为 `51a0346583fd965bb40202759e8e582639dd8b8faaeab7f6a0528d78397d0181`，第二次在连续 300 秒无推进后以 `watchdog_inactivity_timeout` 退出，`watchdog.json` SHA-256 为 `b86d3d7af321a1c770539fb44fa04ccdc64733a1b2696576d9a798a339a4c8b5`。600-iteration 定向 kernel-replay 诊断用正式 metrics、SourceCounters 和短 ordinal 窗口自然完成 `14/14`，证明 kernel 可采，失败来自 v4 多名称/ordinal 交叉分组产生的额外 replay 压力。
- v5 将 invocation 作业上限从 6 提到 16，候选计划保持 `5418` 个必测样本不变，把选择量从 `8939` 降至 `5824`、额外 launch 从 `3521` 降至 `406`，并将 direct-copy 独立为 `127/127`、0 extra 的作业。正式 runner 同时加入计划绑定的 300 秒 inactivity watchdog；只有本作业进程组 GPU 活动、stdout 或 report 任一推进才续时，超时后先 SIGTERM、再按冻结 grace 必要时 SIGKILL，并写入正式 status。短门对稀疏组使用一个真实内核名称的全部可用 iteration-1 launch（上限 16），并记录实际数量；v5 正式计划必须在该实现形成 clean Git 提交后重新生成；v4 Job 1--3 仅作为历史证据，未经逐字段等价检查和显式 plan-lineage 重绑定不得并入 v5。
- 针对第十三组 `CUDAFunctor_add<float>` 的 `239` 个实际启动序号在同一回放组内触发停滞，分组生成器现支持配置化的 `maximum_single_kernel_group_launch_count`：只对超过上限的单一内核按真实 invocation ordinal 连续分区，并禁止受保护分区再次合并。128 和 64 启动分片均在序号 `4862` 处触发 watchdog，因此依据真实诊断将普通分片上限冻结为 `64`，并通过 `isolated_invocation_ordinals` 将该序号单独隔离；真实 inventory 代价比较显示 invocation 组上限设为 `61` 时无额外启动并保持 `5418` 个必测样本。新的正式计划尚待重新生成。
- 稀疏 invocation 分片的短门现可从同一计划中复用同名内核的 iteration-1 真实启动作为校准源，并记录来源采集组；因此没有 iteration-1 启动的后续分片不会被错误拒绝，也不会为短门运行到后期迭代。该 runner 修订已通过 GPU profiling 测试。
- revision9 第 55 组已自然完成完整 30,000 iteration，并单独捕获 `CUDAFunctor_add<float>` 的真实 invocation ordinal `4862`。实际捕获 `1/1`，launch binding、SourceCounters 和动态 SASS 分类均通过；耗时 `1645.9276325702667 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `fed6da0e9886dcfad7a76411bb7f15415c29ffd4531427a2de712676de637f5a`、`ffbf42667194c8178cf45794834fa901739a70be7b68c5e171de2e0f55fdb3a0`、`8b2ee6cc72debefbd58c75c0b8560e1150361d35725b33442f8bda969fbff2df` 和 `bfe2bc327d64e8e558492a1ec3c907549757d9e9c41c4c9dbc1508129afb7f0e`。该结果证明序号 `4862` 的单独隔离可以越过此前停滞窗口，但 revision9 全部采集组闭合前仍不产生正式性能结论。
- revision9 第 56 组已自然完成完整 30,000 iteration，采集 `AUnaryFunctor<float, ..., MulFunctor<float>>` 的 `45` 个真实启动序号，实际捕获 `45/45`；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1668.1216790676117 s`，GPU 利用率中位数 `94%`、采样数 `1392`，watchdog 观测到的最大无进展仅 `1.2039127501193434 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `1c07991b3cf712b6c744c73a4753029a20c715ccd1afb0a7d5b045c0e43afaaa`、`622afd6e5909f3dd2d5dbf6ed14ce37b9abaa223bed30d079e02ae43a9aeba9c`、`0ced7a4414c38c3a31f69b1c4fb50ca5037544afb50fcc5301f79a6d9f1414b9` 和 `1cbf094e49909f737672a9689f928619a2650648083f88a809b9d90068195921`。该结果仍只闭合正式采集组，不改变完整计划闭合前 `formal_performance_eligible=false` 的限制。
- revision9 第 57 组已自然完成完整 30,000 iteration，采集评估指标幂运算内核的 `4` 个真实启动序号，实际捕获 `4/4`；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1675.1750276088715 s`，GPU 利用率中位数 `94%`、采样数 `1401`，watchdog 观测到的最大无进展仅 `1.1997600109316409 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `86561afa9d278d69622c81e218f53a44b892e3d87d815ea59292b46e02f10475`、`cf15d4f8c488aabaf42d4d12aec6881a1bc7a124daf1b53a2961500b152ff27a`、`1d326f180a1b449dfeb77f3a2ba9703fbb0b9797e366c034b9796aba1048ed96` 和 `7ab2cf3cfbc884a017d1a698183a0e62b73c781b614a43584a7a7b4cdef9b71b`。该结果仍只闭合正式采集组，不改变完整计划闭合前 `formal_performance_eligible=false` 的限制。
- revision9 第 58 组已自然完成完整 30,000 iteration，采集 `CUDAFunctorOnSelf_add<float>` 的 `59` 个真实启动序号，实际捕获 `59/59`；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1668.6396832466125 s`，GPU 利用率中位数 `94%`、采样数 `1395`，watchdog 观测到的最大无进展仅 `1.1983098699711263 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `36461c884d59c983fcf1710b2cad1d7c9a9dc2d3cef3ce3f915620f1e1839056`、`d3ac44b024bed2fb2e3882d08f9d9f37c4a51b7989a836e4a09f43ae2139d2cc`、`7e229175daa3053f5880f5309eaae245e3bf3fa1af953f52159ed92daad007b4` 和 `14fd39ad529f8e29e462ec7e07e4a5d3b1b6cfb526a297c7169cc818f78cd741`。该结果仍只闭合正式采集组，不改变完整计划闭合前 `formal_performance_eligible=false` 的限制。
- revision9 第 59 组已自然完成完整 30,000 iteration，采集三维数组版本 `CUDAFunctor_add<float>` 的 `64` 个真实启动序号，实际捕获 `64/64`；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1677.9132273197174 s`，GPU 利用率中位数 `94%`、采样数 `1404`，watchdog 观测到的最大无进展仅 `2.391245927894488 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `a2048490419627ba39179b7988d1284aba3b87287d767cd56855060c5c9c442f`、`3b4905cc0b7cf0cbb2ec3b23cbf5047c7ed83adb2a8601e427b711f8247c6c46`、`c98d9576f1a0b637fd924466161ddae632de3a4d8f73456090eade44f9ab3cff` 和 `d578fc6350a9417b6e0eaa481b614411da228e39d54afafcecef3760b9b85a12`。该结果仍只闭合正式采集组，不改变完整计划闭合前 `formal_performance_eligible=false` 的限制。
- revision9 第 60 组已自然完成完整 30,000 iteration，采集三维乘法内核的 `38` 个真实启动序号，实际捕获 `38/38`；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1656.4200429916382 s`，GPU 利用率中位数 `94%`、采样数 `1386`，watchdog 观测到的最大无进展仅 `1.2076716478914022 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `aea43455d7d0ea7215dd69ee4148bba60a562159a989e67c7b2bfadf8d0420e5`、`bd5af38cc9512868f0266c0c3ce81e7e9842f78fb3ee38b1d641ee70a9e8952e`、`1b2fe8606b2f152868c3599a611f72414a9e359336504fdcc9a73f7d99ff8621` 和 `a5dd0f4b93c20d7b45807629fb1398e8571e9899dfba20089de581c23853dae2`。该结果仍只闭合正式采集组，不改变完整计划闭合前 `formal_performance_eligible=false` 的限制。
- revision9 第 61 组已自然完成完整 30,000 iteration，采集三维加法内核的 `46` 个真实启动序号，实际捕获 `46/46`；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1648.3301060199738 s`，GPU 利用率中位数 `95%`、采样数 `1379`，watchdog 观测到的最大无进展仅 `1.211710151983425 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `3bbde0dc88faf1e8ec0cf13fcfbdbcdd4b7d2a829d0227c8d03a99fabeb81e57`、`77d75a0affce3a263197ff297184bc80d196b9403dacafc4bdf0289409ce758e`、`a2e9b476b4acd23ecde8c7f3b6160839ba98cc983f054a6fc13dc7d83ed9a1cf` 和 `7c60848107c4a65d0c5b365060c0a95aa3b85bfe5a78b8959f32bcfbec444dee`。该结果仍只闭合正式采集组，不改变完整计划闭合前 `formal_performance_eligible=false` 的限制。
- revision9 第 62 组已自然完成完整 30,000 iteration，执行四个 collection 范围窗口，实际捕获 `1532` 个范围启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1917.9410190582275 s`，GPU 利用率中位数 `94%`、采样数 `1600`，watchdog 观测到的最大无进展仅 `1.2078223710414022 s`。仓库外状态、报告、SourceCounters 和动态 SASS 证据哈希分别为 `349a968459f8f636b49b5b9fd827af4b712b1d807058b377c799bc317cfd438b`、`bab010017385eef57338b837228c7eaf36b38b538ff5a16ec7eb18665adcb07f`、`16ff355059d29d98d04369d67715373bde4222f722f06511e01b75c8278aa69f` 和 `bcb486633efaf03f22398f78a2ac1d134f671f83e2eaa27e0bcd19f7ca9d21e8`。范围组允许真实动态启动数，`1324` 的计划预测与 `1532` 的实际观察均保留；该结果仍只闭合正式采集组，不改变完整计划合并验证前 `formal_performance_eligible=false` 的限制。
- revision9 第 1 组已自然完成完整 30,000 iteration，实际捕获 `609/609` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `2091.564104795456 s`，GPU 利用率中位数 `94%`、采样数 `1748`，watchdog 观测到的最大无进展仅 `1.2197624929249287 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `b5c5cc1245ec48c21d4a37bb5ad7150b48870840d5044097ed3caece181a9f30`、`4832bc7be55c9a9d449b04ffc717a4374d60ce789930aaa5eb3a7f1ff60ae02a`、`821e280b0795e24741a02233a9aa311cfb6448dd20d756f5ec946da4465a727a` 和 `1fa1dc5384fe8e1c415525c1ede9c6df77727ec343381a65240ea2fbf55c1473`。第 1 组结果已加入 revision9 当前合并检查；当前已纳入第 1、55--62 组，合并仍为 `provisional_ncu_evidence`，并保留 `51` 个 `counter_reuse_disagreement`，不能生成正式权重。
- revision9 第 2 组已自然完成完整 30,000 iteration，实际捕获 `96/96` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1694.6952993869781 s`，GPU 利用率中位数 `94%`、采样数 `1418`，watchdog 观测到的最大无进展仅 `1.210273328004405` 秒。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `0f30ecf4a050ee8d84da3b039a6399777b46cc524fa78a44f9a2430ae4308df8`、`ca0643c8beac2f95b26fb779dcbcf381ebd9a758b5985267e3410debbfbe7308`、`d583ae47288462f19cf07b0544758555527dd8c39bcce259fe4db8492c8fb5e9` 和 `0e285fe046c286f2dbcd4c05b11e157a45eb12d8f590fb1a6975ff3e337a4312`。第 2 组结果已加入 revision9 当前合并检查；已纳入第 1、2、55--62 组，合并仍为 `provisional_ncu_evidence`，并保留计数器复用不一致，不能生成正式权重。
- revision9 第 3 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1654.0583519935608 s`，GPU 利用率中位数 `94%`、采样数 `1382`，watchdog 观测到的最大无进展仅 `1.2065768900793046 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `f43a3072f1452c279f9da8b71c5d03b187cae619d706463056b4a56cd134b629`、`20c0b9cbf62d1ae9bbfb2da89e248a4fb1d0251a3dac4f93bb5a24e2ae2f5f95`、`3c6c1a9c1d5a63efc40ca1b85b2f33aaadb6594ccf18132bbfca3bf270c0901a` 和 `2e0eead8d1c4553f555d71a4a43207d8e100040058c777cfab685e9c67361fc7`。第 3 组结果已加入 revision9 当前合并检查；已纳入第 1--3、55--62 组，合并仍为 `provisional_ncu_evidence`，计数器复用门槛尚未通过。
- revision9 第 4 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1685.6357839107513 s`，GPU 利用率中位数 `93%`、采样数 `1401`，峰值显存 `2171600896 bytes`，watchdog 观测到的最大无进展仅 `1.2003006781451404 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `acb7373a545103e9839c269a026397a3a06ca8fad4190a21b129cb4f60b94312`、`8fa2887901ad9b0fd5eaed220528ce39a21dea27b71318bdccb709d1d0690bb2`、`f0d7c65bc349c690d9737747b00c8822db011347499c1f1d167006ff36f25c86` 和 `970ac120a28b4731a647093002d6fd21a04fbadb511024dbeaf8b7ea0736442e`。第 4 组已加入 revision9 当前合并检查；当前纳入第 1--4、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current12_evidence.json` 的 SHA-256 为 `d17039552b76e582e9954f0d7bb80c0e9d389e30834f4e89d107508d845c3078`，尚有 `101` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 5 组已自然完成完整 30,000 iteration，实际捕获 `60/60` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1682.8784902095795 s`，GPU 利用率中位数 `93%`、采样数 `1399`，峰值显存 `3977248768 bytes`，watchdog 观测到的最大无进展仅 `2.4125981491524726 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `6f2c8296bc00d92cc43c5a35482d5717703e2a41240bd9ab5d105849e5ea675b`、`2e89af1f5b3644295382ec50ae99e94d276d9e5db98fc05de3361af232ff64fa`、`af78cc55975c428d2f2916f545e16173a916bce4c9b07c80f4dd27491fd86cc0` 和 `d2e7b6c17d15976d0c365b04d8506aea2259707dd13761461cd9ce5bee124935`。第 5 组已加入 revision9 当前合并检查；当前纳入第 1--5、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current13_evidence.json` 的 SHA-256 为 `652cc5cb209f3cf07e5af92e5ae537a00b227bba7c3b72862ba8163fe09351c1`，尚有 `106` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 6 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1670.8949275016785 s`，GPU 利用率中位数 `94%`、采样数 `1390`，峰值显存 `3559915520 bytes`，watchdog 观测到的最大无进展仅 `2.38465956505388 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `f47d21d279025f77fba869eabb4f4296204c3805f25194adacf60f91e07ea55e`、`dccfb1de1265fdf7c3f4ccdd84f4eb2ddf0fae5d46757df169eca84017e94dd9`、`0e58d226980ff7784f06f7833424556524d20a3be1ecf1bd0154893077c0fbd3` 和 `58925848f14795a0ea97074835dfdfbbc5aba011dee56b56aaab8b5867b03471`。第 6 组已加入 revision9 当前合并检查；当前纳入第 1--6、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current14_evidence.json` 的 SHA-256 为 `034a9cfc6532c82db5d02d0c7193c124acdacb7049c19247312bb478e0ef508a`，尚有 `120` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 7 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1663.5583536624908 s`，GPU 利用率中位数 `94%`、采样数 `1388`，峰值显存 `4004511744 bytes`，watchdog 观测到的最大无进展仅 `1.2038618091028184 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `ba687fec613db272a9b09e4915adad04d4d7cbfc31dc954f8398d631b1657458`、`642d104975749176e3d8946024e18fcebf5b41d58f8b534da40473ebb98265fc`、`3562e10db1dd1f61a3543021a21799c72ac58d05cde94b6069adf202340b6d28` 和 `6221438c6a29e83d3b382c911867c03e34d9d5dd527c0dcdb85cc030e2a028e6`。第 7 组已加入 revision9 当前合并检查；当前纳入第 1--7、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current15_evidence.json` 的 SHA-256 为 `6a986cdaa0facb7e0e3ab98ee20b36cb7eb743f14ef458427a4c6f61f0788246`，尚有 `134` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 8 组已自然完成完整 30,000 iteration，实际捕获 `48/48` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1674.0549321174622 s`，GPU 利用率中位数 `94%`、采样数 `1395`，峰值显存 `2224029696 bytes`，watchdog 观测到的最大无进展仅 `1.2090907180681825 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `649276c8d1804327529231ad7abb3fae5de2bd2c39957ce33cfc2756b2733070`、`1701aa43eca6dbca6d9a4792e76bb35b429f41257718a0456fc763fbbeee31df`、`0bbb87c9d0aae26bd221de9868751aa017558ce402c953efc2726cb6095a1ef7` 和 `1ced9d7361e089949512db9879aa96b27a16fd39d7ac9d276b401069d58d1434`。第 8 组已加入 revision9 当前合并检查；当前纳入第 1--8、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current16_evidence.json` 的 SHA-256 为 `0fb057d8265e2bfb8c1eb0895d57fa78c3914e86ce30d82adf7245bfae1422e4`，尚有 `139` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 9 组已自然完成完整 30,000 iteration，实际捕获 `30/30` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1679.332267999649 s`，GPU 利用率中位数 `94%`、采样数 `1398`，峰值显存 `4050649088 bytes`，watchdog 观测到的最大无进展仅 `1.2216715328395367 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `f09e9477a8668d6b7ee774a0a272237b550e458fb0ea53479301455d191334e2`、`c1e6a4bdc87c461c98c77d23e09ae6d058f546b9febbad488d32698a0f9cdb3e`、`cce04c24fcd2b098debc686617ac2f44b2a736817702d561a06147b332811dae` 和 `e8cddc99ad9a2168ebaadeb4cb49a9be3ba4f41c3cdc74e36f4b8c7bc3acf2a0`。第 9 组已加入 revision9 当前合并检查；当前纳入第 1--9、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current17_evidence.json` 的 SHA-256 为 `e3bc8116e476d79ccf7f38adf5e9062cb52326cc0a92de8b4299369097d8ba5e`，尚有 `149` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 10 组已自然完成完整 30,000 iteration，实际捕获 `27/27` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1649.908768415451 s`，GPU 利用率中位数 `94%`、采样数 `1378`，峰值显存 `4067426304 bytes`，watchdog 观测到的最大无进展仅 `1.2082525701262057 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `2bbe0ec07297d73dc6dd37881c3afe1f41a735cfd03aaddea851991362b16962`、`1602624f1ff4a60d80399f8ec2645ff6274b68fdea6f6c8aa5bdebe623930dd4`、`e2c1d713c9fdac9c66836cf9ec59b79ade829eff84aafeedbdfa68a5036fb102` 和 `7edb3a581bfadab99ac6c58c21c59c18dd5735a6f79ee9f1f4dc935a215920cd`。第 10 组已加入 revision9 当前合并检查；当前纳入第 1--10、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current18_evidence.json` 的 SHA-256 为 `d3fd0e38cd53545025821b9cecbbe21fdb26458e540665b3373833ae5321781b`，尚有 `158` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 11 组已自然完成完整 30,000 iteration，实际捕获 `234/234` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1677.5682332515717 s`，GPU 利用率中位数 `94%`、采样数 `1403`，峰值显存 `2513436672 bytes`，watchdog 观测到的最大无进展仅 `1.2089175339788198 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `a51005631c19ed0b368a9491a47d943f76b8e2f016919c9f6ae7eb8e8a08a8ca`、`15ca4f8c4cdf1e97967eed7face7902b4a566421057c2906adf2c2c3cdc16f86`、`72954a58b76fcd6c4466363a5ffec77a5dd2e035fc41da690a57814c9cc5fec9` 和 `55084d1a7edce41909790b46032e5d92d03eb6a77bbc909f615e9905a3290d85`。第 11 组已加入 revision9 当前合并检查；当前纳入第 1--11、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current19_evidence.json` 的 SHA-256 为 `8e427758fbfb5399b61cf8b1d1dd044de7d3df1fb7fcb5750a4833e3acf92364`，尚有 `188` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 12 组已自然完成完整 30,000 iteration，实际捕获 `59/59` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1666.7222836017609 s`，GPU 利用率中位数 `94%`、采样数 `1390`，峰值显存 `3379560448 bytes`，watchdog 观测到的最大无进展仅 `1.2115375918801874 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `8304031224ad9c53e7badc5fdd7841c96ee6b8b1dba904c0a5e15935ed4d5ffa`、`3a81c414bd4492e71688b84322ea157fbbc10b086146caed936ef10b28b6d188`、`fe35a9e3823f440d0539bc218a768e4faac51bb179c08eca88cbb8669b66101d` 和 `f7fd38401ec089b288e52d44828b8c4a273c845e3f248558715947efb2870b99`。第 12 组已加入 revision9 当前合并检查；当前纳入第 1--12、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current20_evidence.json` 的 SHA-256 为 `11bdd61d7f0efcfa9fc141fe135111832fe3ff95f45eb595a57f824ed2d479b9`，尚有 `193` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 13 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1673.9013845920563 s`，GPU 利用率中位数 `94%`、采样数 `1399`，峰值显存 `2794455040 bytes`，watchdog 观测到的最大无进展仅 `1.2072965460829437 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `d6feb072e48a0f14b9bddc0b726ae065a3c207e8a813d442345dc7445af21563`、`2a01e36723c17f600d094df9c737d9bfec8c8fa34d08e295812d148c4bd0a8a0`、`a5cbcc25bcc705a55f690854066842e0e7bbeffc4b3dd9bcb8bd8132de6dae94` 和 `9e1d287be02ff376ef394caff27eb0571266dbda537fae6e8fb2cb58da5e5fc7`。第 13 组已加入 revision9 当前合并检查；当前纳入第 1--13、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current21_evidence.json` 的 SHA-256 为 `be6778db46d6408a925953e52737819a5a2b75e0515d559c6208c7e0f0db8d57`，尚有 `197` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 14 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1669.6496560573578 s`，GPU 利用率中位数 `94%`、采样数 `1395`，峰值显存 `2794455040 bytes`，watchdog 观测到的最大无进展仅 `1.1995460151229054 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `8b0253a2812ae5f7c43f93281cf16cb65fd7a9540319fdf79ceb67cc97a7e4b6`、`5057c73fded992a91f43855c3e04e5fe6c53649191d620c77b0980d4c6f0a981`、`f717ddef4f03765403694f376039586b9e099bb4c11475f489704e5ed44cfb89` 和 `5eeedfd4ed2a7d5d0110aff3cddba05cf10db15be12b9f31b78f2178cabb4587`。第 14 组已加入 revision9 当前合并检查；当前纳入第 1--14、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current22_evidence.json` 的 SHA-256 为 `a49449d4bc4438121b8dbaf7623626935852fe42f6cc9f895c8bf11a9e22890c`，尚有 `206` 个计数器复用不一致和未采集启动，不能生成正式权重。
- revision9 第 15 组已自然完成完整 30,000 iteration，实际捕获 `64/64` 个计划启动；launch binding、SourceCounters 和动态 SASS 分类均通过。耗时 `1667.0764219760895 s`，GPU 利用率中位数 `94%`、采样数 `1394`，峰值显存 `3780116480 bytes`，watchdog 观测到的最大无进展仅 `1.2049168690573424 s`。仓库外状态、报告、绑定结果和动态 SASS 证据哈希分别为 `3010de89d9ac67d5d573440c1d9793348b2b1bebe7a3ebee338ed18cbfff4e0d`、`e787b9446eef08b4b3d8d8fe210441ae1563d3d8b8e145e474dad841406fe4d0`、`f91405ef1828866fa0f71ab4e24b4a5ca96a96731b7d6c8eb93ed93349681f0f` 和 `3adfb48cc2e0f1e26ad52650cb526aca52a10e7abb9ccab1f756e728e593412c`。第 15 组已加入 revision9 当前合并检查；当前纳入第 1--15、55--62 组，合并仍为 `provisional_ncu_evidence`，汇总文件 `r2_gaussian_chest_ncu_revision9_current23_evidence.json` 的 SHA-256 为 `b84214dad07067a3cbfd75bab1c87db74dd2e77835fc0ea4ac6896b15298f925`，当前观察到 `3400/4094` 个计划启动，仍有 `224` 个计数器复用不一致和未采集启动，不能生成正式权重。
- 修订后的正式计划位于 `GALA-runtime/records/r2_gaussian_chest_ncu_signature_plan_v5_revision6.json`，内容 SHA-256 为 `1609031dde955a9cd5d4ef0351913f159ce430638b8ffe26e28256fc14bee188`，文件 SHA-256 为 `3a4580eb64de7002cee10e173a8c1a8d94d25ec6e2584d3aa1a6a61b0df9ce6f`，绑定提交 `b35134f`。计划稳定性为 `passed`，共 36 个 invocation 组和 1 个 collection range 组，必测样本 `5418`、选择启动 `5418`、额外启动 `0`。第 17 组与第 36 组的同名 `CUDAFunctor_add<float>` 短门均通过，状态文件 SHA-256 分别为 `86fc74b7c9070adcd2aa5d89a71082d35fe091612374fcce6c38e2a7881aa32d` 和 `f8a2b78efcbda35d9055006f611fb15dc2a0747726ca36bea4bbdf214fe3b990`；完整作业预测时间分别为 `1312.0653946399689 s` 和 `1311.1520919799805 s`。这些短门只证明采集入口和开销预检通过，不是正式计数证据。该计划第 17 组于完整运行中在序号 `4862` 触发 `watchdog_inactivity_timeout`，状态文件保留在 `GALA-runtime/profiles/r2_gaussian_chest_ncu_v5r4_job17/status.json`，不能作为通过证据。
- revision9 第 16 组已自然完成完整 30,000 iteration，状态为 `passed`，实际捕获 `64/64` 个计划启动；GPU 利用率中位数 `94%`、采样数 `1400`，watchdog 最大无进展 `2.38716679206118 s`。性能采样在此终止，不再启动第 17--54 组。代表采样汇总位于 `GALA-runtime/records/r2_gaussian_chest_ncu_sampled16_performance_v1.json`：按迭代、阶段、调用位置、启动序号和内核名称匹配，不校验计划、提交或产物哈希；精确内容命中 `3238/4094`，覆盖率 `79.09135319980459%`，按真实 signature 出现次数加权覆盖率 `85.88074894423358%`，collection 保留 `1532` 个实测启动，其余缺口按同阶段同内核、同内核或同阶段的最近工作规模外推并逐阶段报告离散度。本地归一化报告 `r2_gaussian_chest_gpu_normalization_sampled16_v1.json` 的 13 个阶段权重和均为 `1.0`，`local_sampling_status=passed`；AGX Orin 校准向量不可用，因此 Orin 换算保持 provisional，不影响本地性能采样继续用于模拟器参数化。
- `600:601` 跨迭代真实窗口 trace 已完成 quick validation，结构化记录位于 `GALA-runtime/records/r2_gaussian_chest_trace_window_600_601.json`，SHA-256 为 `b80547658c5fbe4af69ae7864309fd5c97ed6563712a146316ecdba0ca59c947`。捕获包含 `872588146` 个事件、`1430920031` 条依赖、`589824` 个 query、`144911313` 条真实 relation，并覆盖一次 collection、clone/prune、optimizer/no-op optimizer、update begin/end 和后继 query 屏障；全量 validator 修复后 PASS。插桩与两次未插桩 601-iteration 运行的质量差异为 PSNR `-0.000180522587961 dB`、SSIM `-0.000003659142978`、LPIPS `-0.000018147627513`，低于冻结阈值；由于上游 backward CUDA 使用 `atomicAdd`，不宣称 bitwise identical。该窗口明确 `formal_performance_eligible=false`、`quality_eligible=false`，不能作为正式完整 trace、Base ASIC、Oracle、消融或论文结果。
- 2026-08-28 曾在 GPU 空闲时启动一次完整 30,000 iteration `stream_only` trace 补采，5 分钟门点仍停在首轮 CPU 序列化，GPU 利用率为 `0%`、无训练日志或 manifest 推进，`events.raw` 已达约 `45.8 GB`、进程 RSS 约 `9.8 GB`，遂按门控停止并保留现场 `GALA-runtime/trace-smoke/r2_gaussian_chest_trace_v11_full30k/`。该目录没有完成 manifest，不能进入 validator 或周期；结合已通过的一迭代 `577,268,206` events 证据，继续到 30,000 iteration 会产生不可用规模，因此不再重复该补采。
- 有界虚拟捕获已接入官方 CUDA raster/voxel、loss、backward、optimizer 和集合修改钩子。真实 Chest 一迭代短门在约 `9.02 s` 内自然完成，直接从 point list、point key 和完整 valid mask 得到 `2` 个查询包、`294,912` 个 query、`690,928` 个 candidate、`95,948,757` 条 relation 和 `577,268,206` 个逻辑事件，与旧完整一迭代 raw-column manifest 逐项一致；物理包流为 `33,602,912 bytes`，峰值单包 `32,506,992 bytes`，未生成约 `61.19 GB` 的事件列。optimizer 使用当前全部活动 Gaussian 的精确批量提交记录，Clone/Split/Prune 保留稳定 ID、父子关系和版本事务。该短门没有附加周期包消费者，故 manifest 固定 `formal_performance_eligible=false`、`event_stream_validated=false`，不能作为正式 trace 或周期结果。
- `CycleEngine.run_virtual()` 已提供有界快速回放入口：多个虚拟数据包共享一个事件扩展器、全局事件编号、依赖、缓存、融合、内存等待和 Ramulator 状态；调用方必须显式提供 `max_events` 与 `max_total_events`，入口会在展开前按 `candidates + 6*relations + 3*queries` 拒绝超限，并拒绝复用已有输出目录。两包测试得到连续 `20` 个事件并成功回放，超出上限时不写 chunk manifest。该入口明确标记 `result_scope=quick_cycle_validation`、`formal_performance_eligible=false`；它仍会在上限内生成 mmap raw columns，不能用于三万次正式周期。正式路径仍需持久化周期状态、增量依赖 frontier、跨包语义工作集和最终 quiescence 检查。
- `CycleReplaySession` 已提供不写 expanded raw columns 的增量事件消费：每个 `VirtualEventPacket` 到达后注册全局依赖、立即推进模块/融合/cache/Ramulator 状态，完成事件行从内存 frontier 释放；生命周期事件进入同一全局事件编号空间，结束时要求 packet frontier、生命周期 ledger 和内存等待均 quiescent。新增 resident frontier 上限由冻结的 `trace.chunk_events × trace.max_inflight_chunks` 推导，完成事件 ID 和 semantic workset bookkeeping 会在连续完成或迭代闭合后压缩回收；该入口仍因正式 30k 逐字段闭合尚未完成而不具备正式性能资格。
- 在线周期内核现会在 `RELATION_CANDIDATE` 完成时释放 relation-seed FIFO 槽位，并在当前 cycle 的 Bank 冲突唤醒点推进到下一 cycle；同时回收过期 Bank reservation，避免大 packet 因反压误报 deadlock 或保留逐 cycle 状态。
- 在线周期软件路径已完成一轮不改变硬件语义的优化：Ramulator 2 后端缓存绑定元数据，避免每个 `CACHE_REQUEST` 重读配置并重复计算审计哈希；在线事件只在进入 frontier 时解析一次 `PrimitiveKind`，阶段映射使用缓存；重复 stall 记录先在内部累加，输出时再恢复原有不可变记录。对同一真实中心 raster/voxel 闭包（`406008` events、`700903` dependencies）逐包回放仍完成 `406008/406008` events，周期保持 `416629`；带 profile 的墙钟从 `34.69 s` 降至 `26.44 s`，改善约 `23.8%`，无 profile 实测为 `12.01 s`。该优化只影响 CPU 软件开销，未改变事件集合、Ramulator 请求、Bank 冲突或周期语义；全仓测试 `214 passed, 1 skipped`。
- `VirtualCaptureConsumer` 已支持把 query packet、lifecycle record、iteration close 和 finish 统一分发给对象式在线消费者；`TraceSession` 可通过 `virtual_packet_consumer` 或延迟 factory 注入该消费者。`trace_runner --virtual-capture` 在提供真实周期配置和 Ramulator 绑定时自动连接 `BufferedVirtualCycleConsumer`，缺少这些输入必须显式使用 `--virtual-capture-audit-only`；原有低层 audit-only API 保持兼容。
- 虚拟 packet 新增独立 `backward_confirmed` 标记；生命周期 validator 不再把 relation 数自动当成 backward 数，而是在真实 backward hook 确认后计数，并检查 query base 连续性和非零 optimizer 的 ordered active Gaussian 提交集合。正式 capture 仍需把 gradient 依赖 sidecar 进一步传入周期事件。
- 修复 voxel 有界 decoder 的 point-key 布局解析：现在读取与 `voxel_masks_kernel`/relation kernel 相同的第一个 uint64 key 数组，不再跳过到后续辅助数组；新增边界 tile 回归覆盖 query `512` 的映射。
- `CycleReplaySession` 现在可接收精确 `semantic_workset_totals[(gaussian_id, state_version)]` sidecar，在事件注册顺序中生成并校验每个 cache request 的 ordinal、total、remaining 与 last-use；cache fill/return/release 使用该 sidecar，缺少 sidecar 时仍保持保守模式并不升级正式资格。
- 新增 `BufferedVirtualCycleConsumer`：按迭代暂存紧凑 point list/key/mask，计算该 state version 的精确 Gaussian 使用总数后再送入 `CycleReplaySession`，并将后续生命周期记录接到同一全局事件前沿；暂存内容不包含 expanded raw columns，迭代边界仍受生命周期和 quiescence 门控制。在线结果清单记录 packet batch、resident frontier 峰值、完成标记、semantic workset 精确模式和 300 秒无进展门限。
- `VirtualTraceStream` 的驻留资源统计已改为队列中等待包与当前 consumer 持有包的总字节数，而非最大单包；变长包和并行生产消费回归已通过。
- 生命周期 sidecar 已增加 `dependency_ids` 与有序 `active_ids`：在线会话优先使用生产端给出的真实前置，`all_active` optimizer commit 按稳定 Gaussian 展开为独立 UPDATE_COMMIT；官方捕获已记录 optimizer/collection begin 的活动快照，backward/gradient terminal 和父子修改依赖仍待 hook 侧补齐。
- 在线 query 扩展已把每轮真实 GRADIENT_REDUCTION terminal 作为后续 optimizer/collection begin 默认依赖，并将上一 UPDATE_END 作为下一状态版本首个 candidate 的外部屏障；Clone/Split 的 parent 与 child 均生成 SET_MODIFICATION。生产端显式 `dependency_ids` 仍可覆盖默认前沿，用于逐字段对照冻结后的精确拓扑。
- canonical Ramulator preflight 与中心 raster/voxel 真实依赖闭包 quick 周期已重新执行：`0000` Base ASIC 为 `415477` cycles，`query_oracle` 为 `415554`，`residency_oracle` 为 `406180`，联合 `1111` 为 `406208`。十六项消融在同一配置上全部通过，`1111` 与联合入口一致、事件集合一致；新增 `--parallel-workers 4` 后 16 个独立变体约 2 分钟完成，逐项进度写入 stderr。上述结果固定为 `quick_cycle_validation`，不升级为完整 30k 正式周期。
- 长周期重放已加入显式开发诊断：按完成事件数或 30 秒墙钟间隔输出运行健康状态，并只在完整连续迭代闭合时采样墙钟吞吐、每事件模拟周期、总周期投影和按配置时钟换算的 ASIC 秒数。默认仍完整运行；只有显式 `--stop-when-throughput-stable` 才能在预热、最小完成比例、连续稳定窗口和多指标跨度同时通过后提前结束。提前结束要求独立空目录，固定 `formal_performance_eligible=false`，不写正式 `cycles.json` 或消融结果。公开规格 extrapolation 已从周期诊断和消融入口移除，不再生成 Orin 锚点或加速比。
- A/B/C/D 已由周期引擎独立解析，不再把 A 与 C 合并或丢弃 B；`query`、`residency` 和 `full` 别名分别严格映射到 `1010`、`0101` 和 `1111`。C 关闭时融合发射使用全局单发射合同；C 打开时调度器以带查询/高斯目标域的真实归约键执行冲突选择，并只在周期引擎确认端口、Bank 和队列接纳后提交发射状态。中心 raster/voxel 闭包的修复后十六项 quick 矩阵位于 `GALA-runtime/records/r2_gaussian_chest_ablation_mechanism_fix_v1.csv`：Base `0000=415477` cycles，`1000=415558`，`0100=415477`，`0010=415704`，`0001=406180`，`1111=406364`，完整 GALA 相对 Base 为 `1.02243x`。A、B、C 单独位尚未形成目标收益：A 仍缺冻结的真实查询负载计划，B 尚未生成语义工作集，C 尚未使用三个真实有界队首和跨轮历史，因此该矩阵只验证独立开关与冲突合同，不是机制闭环或正式性能结果。
- C 的三输入队首和成功提交路径已完成并通过反例测试：融合前向、消费者、伴随分别进入有界输入队列，每周期只将三个实际队首交给调度器；被冲突、端口、Bank 或队列拒绝的任务留在原队列，只有真正进入在途状态的任务才更新调度器历史。第二次中心闭包十六项 quick 矩阵位于 `GALA-runtime/records/r2_gaussian_chest_ablation_three_heads_v1.csv`，`0000=415477`、`0010=415458`、`1111=406186` cycles，`1111` 与完整入口一致；该结果仍是 quick validation，C 的跨迭代重叠状态和 A 的真实查询负载规则尚未完成。
- B 已生成带类型的精确语义工作集：每条真实 `CACHE_REQUEST` 绑定 `uint64 event_id`、`int64 gaussian_id`、状态版本、同键序号、总用途、剩余用途和最后使用标志，不把事件编号或使用数写入 FP32 payload，也不使用平均复用率。`0100` 只输出工作集提示和审计计数，仍走基础存储；`0001` 保留无编译器提示的在线目录、Miss 合并和版本关闭；`0101` 使用精确剩余用途并在最后一次真实读取完成后释放。当前实现仍作用于内存中的有限 trace；正式 30k 路径需要把相同 typed workset 字段接入有界虚拟 trace 包流。

## 当前入口

在线周期入口已修正同一 CUDA work-buffer packet 的事件子包调度边界：
`CycleReplaySession.accept_query_packet()` 现在先将该 packet 展开的候选、关系、缓存、
前向、归约、消费者、伴随和梯度事件登记到有界 frontier，再推进调度器；达到 frontier
容量时才因真实反压 drain。这样不会在同一 packet 的子包之间提前 drain 并人为串行化，
同时保持有界内存。新增回归在代表性单 packet 上与离线展开回放保持完全相同的事件计数，
周期差为最多 1 cycle；frontier 受限时的额外周期明确归因于容量边界。
使用仓库外真实 capture 原始记录重建的栅格 tile 0（9 个候选、816 条关系、256 个 query）
已接入同一 LPDDR5-6400 Ramulator：离线 `5806 cycles`、在线 `5807 cycles`，事件计数逐项
一致，周期差 `0.0172%`，在线结束 `pending=0` 且 `quiescent=true`。该短门仍是
`quick_cycle_validation`，不提升完整 30,000 iteration 结果的正式资格。
在完整真实栅格 packet 的有界在线探针中，已登记 `12,677,229` 个事件、完成
`10,565,757` 个事件，最后稳定吞吐约 `37,545 events/s`，模拟周期为 `10,565,765`；
按已完成前缀投影该栅格 packet 约 `546,081,356 cycles`（500 MHz 下 `1.092 s`），
此前将这一迭代级 ASIC 投影直接除以完整 30,000 iteration 的 Orin proxy，得到的
`3495x` 属于工作量不一致的无效计算，已撤销。探针在 `281 s` 主动停止，未写正式周期结果；
日志期间持续更新，CPU/Ramulator 满载，GPU 空闲属于周期模拟的正常特征。

2026-08-28 已加入 `gala_sim.trace.virtual` 的有界工作缓冲区和全局事件包原型。
`VirtualTracePacket` 保留官方 raster/voxel 的 point list、point key 和完整 valid mask，
并以稳定的全局 query 顺序惰性枚举真实 relation。`VirtualQueryEventExpander` 现在能
在全局连续 ID 下生成候选、关系、缓存请求/返回、前向、查询归约、消费者、伴随和梯度
归约事件；`VirtualEventStreamValidator` 检查包号、事件连续性、外部依赖和前向依赖；
`VirtualTraceLifecycleValidator` 维护活动 Gaussian、状态版本、更新事务和每迭代计数
ledger。
`VirtualTraceStream` 使用有界生产/消费队列、物理包字节和峰值驻留统计，并在配置的不活动
期限内停止无进展运行。`TraceSession --virtual-capture` 已接入正式 CUDA work buffer、
loss/backward 确认、UPDATE_BEGIN/COMMIT/END、Clone/Split/Prune lineage 和迭代级状态
ledger；捕获时不再展开完整事件链，而是把仍驻留于有界内存的精确数据包交给在线
`BufferedVirtualCycleConsumer`。在线 consumer 已能持久化全局周期、依赖 frontier、
ready/in-flight 队列、cache/Ramulator waiters、融合调度和关闭版本状态，并由真实 Ramulator
配置驱动；当前仍缺正式 30k packet 流的逐字段等价、完整生命周期 sidecar 对照和最终资源门，
因此尚不能提升三万次 trace、Base ASIC、Oracle 或消融结果的资格。

使用中止的真实 CUDA 捕获现场 `.capture_records` 重建局部 packet 后，逐字段比较器已通过两类真实
对照：栅格 packet 为 `677,229` 个 candidate、`90,769,547` 条 relation，体 packet 为 `13,699`
个 candidate、`5,179,210` 条 relation；candidate 的类型、序号、Gaussian ID、point key 以及
relation 的类型、candidate 序号、local query、reserved 字段均为 `0` mismatches。进一步选取
真实 tile 0 的 `9/816` 栅格记录和 `212/82,995` 体记录接入原生 Ramulator 2 在线回放，得到
`514,125` cycles，约 `21.7 s` 完成；结束时 pending event、resident completion marker 和
semantic workset 均为 `0`。这些是局部真实短门，尚不代表完整一迭代或 canonical 30,000 iteration
packet 流，也不能提升正式性能资格。

随后以同一官方 CUDA 训练入口启动真实一迭代 online packet 诊断，在 `240.9 s` 内接受
`6,313,325` 个事件、完成 `6,247,789` 个事件，吞吐由 `22,500` 稳定到 `25,934 events/s`，
模拟周期达到 `6,248,173`；周期回放阶段 CPU/Ramulator 满载而 GPU 为 `0%`，但日志每 30 秒
持续推进，因此按五分钟门控在吞吐稳定后主动停止。按已完成前缀投影，一迭代
`577,268,206` 事件约为 `571,310,937 cycles`、`1.1426 s`（500 MHz），在线回放墙钟约
`6.18 h`。旧记录把完整 30,000 iteration 的 Orin proxy `3817.435 s` 直接与这一迭代
ASIC 时间相除，产生的 `3341x`（区间 `2500x--4182x`）是工作量不一致的无效计算，已撤销。
即使按同一迭代归一化，公开规格 extrapolation 得到的 `0.12725 s/iteration` 也没有端到端
Orin 实测或跨平台缩放依据，不能作为合理性锚点，不能与 ASIC 投影相除，也不能形成方向判断。
仓库外旧 `online_throughput_diagnostic.json` 中相关字段已作废；当前只保留 ASIC 自身的周期投影。

R²-Gaussian + Chest 已结束正式 GPU 计数器采集。性能分析固定使用前 16 组代表采样、已有 collection 报告和历史可用报告，不再追求 62 组穷举覆盖；这 16 组是本组合的性能采样上限，不再启动第 17--54 组。采样结果明确标注 `representative_gpu_performance_estimate`，保留精确内容覆盖率、按真实出现频次加权覆盖率、外推模式和逐阶段离散度，不能解释为穷举计数或正式 Orin 实测。哈希只记录，不参与任何通过或拒绝判断。

本轮决策后的入口是：先冻结统一硬件/时序/trace 配置，随后只对质量与完整性执行一次全量 trace 验证，性能迭代使用可配置的代表性依赖闭包，最后运行 Base ASIC 和两个受资源约束 Oracle。16 组采样已经足以支撑该阶段的本地 GPU 性能参数化；增加到 60 多组只会扩大重复采集时间，不改变当前采样估计的定义。

本轮 v9 stream-only 真实 Chest 1-iteration capture 已生成完整 raw-column manifest：577,268,206 个事件、895,439,361 条依赖；capture audit 记录 294,912 个逻辑 query、690,928 个 CUDA candidate、95,948,757 条有效 relation，实际 D2H transfer 12 次。插桩与未插桩官方运行的 `vol_pred.npy` SHA-256 和逐元素值完全一致，PSNR `20.285214676826495`、SSIM `0.268613299848576`、LPIPS `0.6325289289156596` 完全一致。该证据仍是 1 iteration smoke，不是正式 30k trace。随后 `600:601` 窗口已补齐真实增密、更新和释放事务，但仍只用于 quick validation。

`TraceReader` 现会从旧式 raw-column `chunk_manifest.json` 的 `storage_format` 补入 `trace_storage_format`，标准 `read(validate=True)` 可自动选择流式校验入口。上述大 trace 的结构 validator 已连续两次 PASS；加入生命周期 pass 并复用 relation/gradient 索引后再次全量 PASS，使用 4,000,000-event 扫描块和 `/dev/shm` 临时紧凑索引，耗时 `5:25.30`、峰值 RSS 约 8.40 GiB、无进程 swap。扫描块和临时目录只改变验证软件的吞吐与资源位置，不改变全量检查集合。该 trace 不含正式 30k 路径所需的 optimizer/update/collection 事务，因此仍不能视为正式 trace 闭环。

快速 trace 路径现支持显式 query ranges、事件上限、扫描块和 `cpu/cuda/auto` 后端。对同一 v9 q0 闭包，CUDA 与 CPU 输出的 events/dependencies/payload 逐元素一致；CUDA 用时 `88.65 s`、峰值 RSS `884,720 KB`，CPU 用时 `110.77 s`、峰值 RSS `476,364 KB`。中心 raster query `131328` 与中心 voxel query `279056` 的联合闭包包含 406,008 个事件、700,903 条依赖、99,831 条真实 relation 和两种 template，并通过独立 validator。四策略周期 smoke 已完成，但使用实验 timing，且两个 Oracle 均返回 `heuristic_unproven`；这些结果只证明快速路径可运行，不是正式 Base ASIC/Oracle 周期。

首个组合仍停在正式 Base ASIC 之前。`configs/architecture/gala.yaml` 的质量参数和本轮周期验证所用配置已冻结；正式周期结果仍需以 canonical trace、Ramulator 2 和资源快照为入口。原生 Ramulator 2 binding 已可用。正式周期入口不得因 canonical 配置、trace、bridge、库或产物哈希不同而拒绝；当前代表性周期 smoke 只验证内核行为，不冒充完整 30k 性能。

仓库外 `r2_gaussian_chest_freeze.json` 使用冻结解释器 `/home/madrid/anaconda3/envs/gaussian-slam-official/bin/python3.10`、`CUDA_HOME=/usr`、`/usr/bin/nvcc` 12.0、GCC/G++ 11、PyTorch CUDA 12.1 与冻结质量依赖生成；记录中的参考体范围、切片边界、训练默认值、有效调度、随机状态和配置哈希已由生成器交叉检查。冻结的官方训练命令为 `/home/madrid/anaconda3/envs/gaussian-slam-official/bin/python3.10 train.py -s /home/madrid/Desktop/GALA-runtime/data/chest/extracted/cone_ntrain_50_angle_360/0_chest_cone -m /home/madrid/Desktop/GALA-runtime/official/r2_gaussian_chest_30000`，工作目录为固定上游源码根目录。

R²-Gaussian 的官方 CUDA 扩展仍没有直接导出完整 CLAMP 事件缓冲区；当前旁路从官方 CUDA work buffer 重建逐 query 关系，并在官方 loss、backward 和 optimizer 边界映射其余事件。`600:601` 窗口已证明长程增密 ID 稳定性、真实队列操作和 collection/update 生命周期，但正式 30k 训练的完整流式 trace 尚未获取，不能视为正式 trace 闭环。

关系记录 decoder 输出现按迭代边界合并，每个非空 flush 最多执行一次 D2H；flush 在最后一个 query 的 backward 到达、下一迭代、optimizer step、Gaussian 集合修改和最终写出之前发生，并要求每个捕获 query 恰有一次类型匹配且已关联 loss 的 backward。v9 decoder 的 raw terminal CUDA scanner 已通过 CPU 逐索引对照；完整一迭代和 `600:601` 跨迭代窗口的质量与结构+生命周期 validator 均已通过，下一项 trace 工作是正式完整训练 capture，而不是重复 smoke。

本轮新增的真实虚拟包生命周期短门使用固定官方入口执行到第 601 次迭代，仅捕获第 600--601 窗口。虚拟包 manifest 位于仓库外
`GALA-runtime/trace-smoke/r2_gaussian_chest_virtual_capture_600_601_v3/virtual_trace_manifest.json`，状态为 `passed`：4 个 packet、589,824 个 query、1,331,283 个 candidate、144,922,250 条 relation，逻辑展开事件公式为 `candidate + 6*relation + 3*query = 872,634,255`；窗口包含一次 collection、9,706 个 clone parent、9,706 个 clone child、145 个 prune、一次 optimizer 和一次 no-op optimizer。该结果没有展开或写出数十 GB 事件列，仍明确 `formal_performance_eligible=false`。

对同一窗口的旧 raw-column trace 已有独立全量结构+生命周期 PASS，但其来自另一 CUDA 运行，关系数为 144,911,313；两次运行的差异属于官方 backward/排序的非确定性，哈希和跨运行计数均只作记录，不作为拒绝条件。以旧窗口直接启动完整周期回放时，300 秒内持续报告 `validation`/`iteration_index` 而完成事件为 0，随后按长任务门控停止；这证明重复全量验证索引是当前入口瓶颈，不能把该探针称为 Base ASIC 周期结果。依赖闭包批量读取修复已在提交 `3c29841` 推送，完整测试为 `217 passed, 1 skipped`。

周期引擎现支持捕获 manifest 提供的精确 `iteration_event_counts`：字段覆盖检查通过时直接建立吞吐诊断迭代索引，缺少字段仍回退到严格全量扫描；依赖反向边构建同时报告 `dependency_count` 和 `dependency_fill` 阶段进度。该优化只消除重复软件扫描，不改变硬件周期或事件语义，相关回归与捕获 metadata 写入已通过。

## 下一步入口条件

在没有 AGX Orin 实机的条件下，曾用 16 组本地归一化报告和显式公开规格参考生成规格 extrapolation
`GALA-runtime/records/r2_gaussian_chest_agx_orin_proxy_v1.json`：总估算为 `3817.435 s`，
区间为 `2856.733--4778.137 s`。该记录按阶段和类别保存权重、换算比与不确定性，
状态为 `proxy_estimate` 且 `formal_performance_eligible=false`。公开峰值规格无法预测真实 kernel、访存、
调度和软件开销，因此该数值不能用于性能比较、数量级检查、方向判断或 ASIC 达标门槛。不改变正式归一化的
`agx_orin_estimate.status=unavailable`，也不填充正式 `speedup_vs_orin`。

1. 将有界虚拟数据包直接接入跨包 CycleEngine，并在一迭代和 `600:601` 窗口逐字段对照旧完整 trace；通过后运行一次 canonical 30k 虚拟包流的质量、结构、生命周期和资源门。
2. 在真实 trace 上通过依赖、状态、释放和动态事件计数检查，再运行 `0000` Base ASIC 和两个受资源约束 Oracle；性能采样继续固定使用 16 组结果。
3. Base ASIC、两个 Oracle 和覆盖外路径冻结后，完成 A/B/C/D、联合运行和十六项消融；`1111` 必须与完整 GALA 周期完全一致。
4. AGX Orin 若要形成正式换算，必须在同一校准套件、数据类型和批量范围下取得 Orin 实测向量；在此之前结果块保持 `agx_orin_estimate.status=unavailable`，不得以峰值规格比补值。

当前不报告面积、功耗、能量、能效、正式周期或论文加速比。上述双窗口结果仅用于同一工作量下的机制覆盖范围、Oracle 和相对 Base ASIC 诊断；在正式 30,000 iteration trace 逐字段闭合前，不将其升级为论文端到端周期。

## 2026-08-28 完整物理包技术样本诊断

使用真实 CUDA capture 记录中的栅格 tile 0 和体 brick 0 重建了前向、消费者、伴随和梯度均完整的
物理 `RelationPacket` 快速样本。该样本包含 `505,391` 个事件、`779,583` 条依赖、`83,811`
条逻辑关系和 `11,698` 个物理关系包，平均每包 `7.16456` 条有效 lane；栅格前向与伴随均为
`816` 条关系，体前向与伴随均为 `82,995` 条关系。该结果通过结构验证和物理包计划，但仍固定为
`quick_cycle_validation`、`formal_performance_eligible=false`、`quality_eligible=false`。

当前周期引擎得到 Base ASIC `36,093 cycles`、future-visible Query Oracle `36,091 cycles`
和 Residency Oracle `36,093 cycles`，对应相对 Base ASIC 的诊断性加速比约为 `1.000055x` 和
`1.000000x`。Residency Oracle 将语义缓存片外请求从 `11,698` 降到 `217`，但没有缩短端到端
周期。资源必要下界为 `24,935 cycles`，其中当前限制项是
`bidirectional_query.issue_ports`；该下界只表示任何实现不能更快，不是可达预测、机制上界或性能
锚点。

这组三个周期暂不提升为有效 Base/Oracle 结论。硬件合同声明双向查询执行单元包含 64 个查询归约
Bank、每 Bank 四组部分和、32 条 loss FMA lane 和 8 条伴随重放流水；当前周期引擎却把
`QUERY_REDUCTION`、`CONSUMER`、`ADJOINT` 和 `GRADIENT_REDUCTION` 共用一个通用 issue port，
且需要继续核对前向贡献进入查询 Bank、伴随 replay 回到 Pod 的阶段映射。该合同一致性问题会人为
形成约 `24,935-cycle` 的串行瓶颈，使两类机制失去可优化空间。下一入口是修正或证明上述资源与阶段
映射，并公平重跑同一 Base、两个 Oracle 和必要下界；确认之前不启动十六项消融。

上述 `36,093-cycle` 结果已被后续硬件合同修正取代，不能继续作为当前配置下的 Base 或 Oracle
结果。周期引擎现已分别建模查询归约、loss、伴随重放、query-volume SRAM、关系窗口、关系记录
SRAM、owner-gradient epoch 槽和 ComputePod 模板资源；共享 SRAM 也改为 active Gaussian、关系
窗口、query volume、gradient/update、index/graph 和 control metadata 六区账本，总量
`2,883,584 bytes`。新的 Ramulator 2 native bridge 预检已在四个 Pod、二十个 cluster、八个
LPDDR5 channel 和该六区资源包络下通过。

真实中位 raster tile 最小样本包含 `356,353` 个事件、`554,381` 条依赖、`59,201` 条逻辑关系和
`8,202` 个物理关系 packet。最初 Base 在 `cycle 64,324` 停于 `239,195/356,353` 个完成事件；
诊断证明关系 SRAM 仅使用 `8,202/16,384` 条记录，而 owner-gradient 等待队列中有 `8,124` 个
已 ready 的伴随 packet，其中 `1,156` 个可复用活动 epoch，但固定队首扫描看不到它们。调度器现按
权威 C++ 的 owner-gradient 可接受性扫描等待集合，并把“八条 replay lane 暂忙”作为正常反压而非
非法 packet 宽度。修复后 Base 完整闭合 `356,353/356,353` 个事件，结果为 `70,596 cycles`
（500 MHz 下 `0.141192 ms`）；该单 tile 结果是 `quick_cycle_validation`，不外推到完整训练，也不与
Orin 时间相除。可选 ComputePod 遥测同步保存 `177,603` 条事件时序和二十个 cluster 的无损占用
区间。同一修复版本下，实际 Query 机制为 `70,593 cycles`，相对 Base ASIC 为
`1.000042x`；future-visible、同资源 Query Oracle 为 `70,546 cycles`，对应 `1.000709x`。
实际 Residency 机制为 `70,563 cycles`，对应 `1.000468x`；future-visible Residency Oracle
没有优于 Base，因此 portfolio 保留实际机制为该组最佳已知成员。三者都是相同最小真实 trace 的
周期比，不外推到完整训练。资源必要下界为 `8,616 cycles`，即只从必要条件看最多仍有 `8.19x`
未排除空间；该下界假设其余冲突全部消失，不是可达性能、静态锚点或机制上界。当前真实 Oracle
仅约 `1.0007x`，说明下一技术任务是解释并缩小必要下界与受约束可达调度之间的差距，而不是用
Orin 外推构造达标数字。后续双窗口验证发现该阶段把 owner-gradient 槽错误持有到同一 Gaussian
在整次 trace 的全部梯度归约完成，因此上述 `70,596` Base、两类实际机制和 Oracle 周期均已由
在途 packet 生命周期修正取代，不能继续作为当前性能结果。

owner-gradient 等待集合的全扫描已替换为语义等价的增量就绪索引。索引分别维护普通候选、
单 owner-key 候选和多 key packet；cluster 有空 epoch 槽时取该 cluster 最早候选，槽满时只从
活动 key 中取最早候选，并在所有类别之间保持原有 `(event_id, stage)` 全局顺序。该结构只优化
模拟器软件路径，offline 和 online replay 共用，不增加被模拟硬件资源。带 ComputePod 精确遥测的
同一最小真实 trace 从 `251.00 s` 降至 `82.14 s`，软件墙钟提速 `3.06x`，Base 仍精确为
`70,596 cycles`；最终回归记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_raster_ready_index_telemetry_v2/`。运行期间最大进展报告间隔
约 `30.0 s`，未触发五分钟无进展门控。该 `3.06x` 只说明软件选择器提速；其 `70,596 cycles`
随后因上述硬件生命周期错误失效。

同时包含 raster 和 volumetric packet 的中位双窗口 trace 含 `846,012` 个事件、`1,289,964`
条依赖和 `19,777` 个物理 relation packet。首次重放在 replay queue 的 256 个 query 项与每个 owner
cluster 的两个错误常驻 Gaussian 槽之间形成循环等待。权威 C++ 实现证明 owner 槽在 partial 被
gradient-completion 接收后立即释放，未来 relation 不占槽；两个真实 trace 中全部 adjoint 与
gradient-reduction packet 均按 relation-ID 一一配对，且所有 `140,520` 个 logical gradient event
都直接依赖配对 adjoint。周期模型现按已预约 relation 维护在途引用，同一 packet 穿过查询和 Pod
阶段只计一次，配对 gradient packet 完成后释放。修正后最小 raster Base 为 `64,682 cycles`，双窗口
Base 首次完整闭合为 `144,550 cycles`（500 MHz 下 `0.289100 ms`），墙钟 `283.41 s`，峰值主机
内存约 `1.07 GiB`，最大进展间隔约 `30.01 s`。双窗口记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_median_packets_owner_inflight_fix_v1_base/`。两者仍是
`quick_cycle_validation`，不外推到正式 30K；下一入口是在同一双窗口 trace 上重跑实际机制、两类
同资源 Oracle 和必要下界。

完整测试当前为 `284 passed, 1 skipped, 2 warnings`。`3341x`、`3495x` 和所有公开规格
Orin extrapolation 已彻底退出性能判断，不能作为正式结果、静态锚点、数量级检查、方向判断或
ASIC 达标门槛。Orin 平台结果保持 `unavailable`，直到取得同套件实测向量。

双窗口 Query portfolio 已在修正后的 owner-gradient 生命周期上闭合。Base ASIC 为 `144,550`
cycles，实际 Query 机制为 `143,782` cycles，future-visible Query 为 `144,416` cycles；portfolio
保留实际机制为最佳已知成员，对 Base 的加速为 `1.005341x`。future 策略比实际机制慢 `634`
cycles，因此状态为 `portfolio_best_known_not_proven_upper_bound`，不能称作已经证明的 Oracle 上界。

双窗口实际 Residency 最初在 `cycle 107,390` 死锁。失败时两个 relation window 均未释放，关系
存储占满 `16,384` 条；根因不是硬件容量不足，而是 Residency 候选排序错误地按 Gaussian ID
重排了关系构造、前向和反向等非缓存模块，使两个窗口交错占满存储。排序现只在语义缓存候选原有
槽位内聚合相同 Gaussian，非缓存候选的相对顺序保持不变。修复后实际 Residency 完成全部
`846,012` 个事件和两个窗口，得到 `144,549` cycles，相对 Base 为 `1.0000069x`。它把片外状态
请求从 `19,777` 降到 `547`、Ramulator 请求累计延迟从 `2,215,553` 降到 `37,416` cycles，但端到端
只缩短一个周期。future-visible Residency 与 Base 都是 `144,550` cycles，因此当前证据说明语义
驻留覆盖的内存等待不是该双窗口端到端关键路径。

同一 trace 的资源必要下界已在修正后的 `144,550-cycle` Base 上重算。Query、Residency 和联合
情形的共同必要下界均为 `21,284` cycles，限制项是 ComputePod microcontext 槽的总需求，必要
条件最多仍未排除 `6.79x` 空间。该数值假设全局池化并忽略每簇碎片、依赖耦合和可达调度，只是
不可突破的乐观下界，不是可达性能、目标、Oracle 或锚点。Base 的真实停顿记录显示双向查询归约
Bank 从 cycle `448` 到 `129,299` 几乎持续冲突；当前 Python 实现仍用 `query_id` 排序代替权威
实现的实时 `F/C/A` 负载状态和 `(released_work, completed_queries, -remaining_work)` 比较规则。
下一入口是按权威 C++ 状态机补齐查询负载规则和每 Pod 有界就绪队列，再重跑同一 Base、Query
实际机制和受资源约束 portfolio；在此之前不启动十六项消融。

完整测试更新为 `285 passed, 1 skipped, 2 warnings`。上述结果均为
`quick_cycle_validation`，不外推到正式 30K；`speedup_vs_orin` 继续保持 `unavailable`。

## 2026-08-29 跨迭代状态版本消融样本

从真实 `600:601` CUDA 窗口提取了八个连续查询包，不设置事件数或依赖数闭包上限。样本保留源状态
版本 `0/1`，包含 `631,244` 个事件、`933,642` 条依赖、`103,594` 条逻辑关系和 `15,600`
个物理 RelationPacket；独立 validator 通过。CUDA 源扫描约两分半钟，单次完整周期回放约
32--54 秒，因此无需以闭包上限截断。该样本固定为 `gala-query-packet-sample-v2`、
`quick_cycle_validation`、`formal_performance_eligible=false`，不能代替完整三万轮正式结果。

状态版本本身没有破坏查询调度：Base ASIC 为 `34,174 cycles`，查询负载规则与重叠引导联合为
`22,750 cycles`，相对 Base 为 `1.502154x`。此前语义缓存按 Gaussian ID 重排四个真实队首，
使 Full 退化到 `32,820 cycles`；该排序不属于硬件合同。恢复真实就绪顺序后，Full 和消融
`1111` 均为 `21,704 cycles`，相对 Base 为 `1.574548x`，并比查询调度联合再缩短 `1,046`
cycles。未提供语义工作集时，目录驻留新增每缓存实例一个 FIFO 替换指针，仅淘汰活动读取数与
剩余使用数均为零的最早装入表项；`0001` 不再因容量耗尽死锁，本样本发生 `585` 次替换并完成于
`33,184 cycles`。

同一配置的十六项四路并行消融已全部通过，仓库外 CSV 为
`GALA-runtime/records/r2_gaussian_chest_iter600_601_q177888_pack64_stateful_readyfifo_v1_ablation`。
逐项周期为：`0000=34174`、`0001=33184`、`0010=23235`、`0011=22329`、
`0100=34174`、`0101=33184`、`0110=23235`、`0111=22329`、`1000=34030`、
`1001=32994`、`1010=22750`、`1011=21704`、`1100=34030`、`1101=32994`、
`1110=22750`、`1111=21704`。B 在该样本中只提供精确工作集与释放信息，不单独改变周期；D
在所有查询调度组合上提供约 3%--4.6% 的额外收益。

必要下界使用相同 `631,244` 个事件和冻结资源重算。Query 下界为 `15,602 cycles`，目标
`1.445679x` 未被排除；Residency 下界为 `31,330 cycles`，理论最多 `1.090776x`，因为关闭
查询调度后 `31,328` 个 Fusion 任务受基线单发射规则约束；Full 下界为 `15,602 cycles`，理论
最多 `2.190360x`，限制项是单前向端口和单伴随端口各自处理 `15,600` 个物理包。因此
Residency `1.443210x` 与 Full `2.648560x` 的旧静态参考在当前样本和资源合同下均不可达；突破
Full 下界需要修改 RelationPacket 宽度或融合输出组织，不能通过继续扩大闭包或缓存调参实现。

完整测试现为 `335 passed, 1 skipped, 2 warnings`。AGX Orin 同套件实测仍不可用，所有
`speedup_vs_orin` 字段继续保持 `unavailable`。

## 2026-08-30 有界在线回放的正式分包等价门

在线周期回放现将各模块的 ready 候选限制在冻结的硬件 `queue_capacity` 内，容量外事件留在
有序上游等待区；新到达的较早事件可替换硬件可见队列中的较晚事件。离线与在线候选均按全局
事件 ID 稳定排序。streaming continuation 在任何 transport fragment 发射前完整登记该
query-pack 的物理 RelationPacket、query replay 和 owner-gradient 元数据，完成事件释放的
frontier 空位会在同一模拟周期发射前补入后续输入。上述逻辑同时作用于 Base、实际机制和消融，
不删除事件、依赖、状态版本或资源访问。

使用完整 compact archive 中真实 Voxel tile 0 的 `212` 个 candidate、`80,063` 条 relation、
`512` 个 query 和 `482,126` 个事件执行原生 Ramulator 2 三档门。固定正式
`max_frontier_events=8,000,000`，只改变 transport `max_events=65,536/131,072/2,000,000`，
三档均得到 `26,614 cycles`、`11,542` 个 Ramulator 请求，并且逐事件 completion cycle、事件
计数、模块计数、stall、内存请求和最终 quiescence 完全一致。定向周期测试为 `83 passed`，全仓
测试为 `361 passed, 1 skipped, 2 warnings`。

减小 `max_frontier_events` 会显式改变软件输入反压，因此不再作为正式分包等价维度：同一 tile 的
`131,072` frontier 为 `26,671 cycles`，而冻结的 `8,000,000` frontier 为 `26,614 cycles`。
文档冻结配置仍由 `trace.chunk_events=4,000,000` 与 `trace.max_inflight_chunks=2` 推导正式
frontier；不得把更小诊断容量产生的额外周期混入 Base ASIC、Oracle 或十六项消融。

同日的完整捕获物理包最小样本包含真实 Raster 与 Voxel packet，共 `505,391` 个事件、`779,583`
条依赖、`11,698` 个前向物理包和 `217` 个状态键。合同修复后的 Base 为 `30,497 cycles`；B 与 D
联合开启的语义驻留最初为 `24,947 cycles`。三目的同状态前向 Fusion 束首先把结果降到 `19,210`
cycles，并消除 `5,841` 次独立 Fusion 提交。进一步复用同一 144 项前向 FIFO 语义链，为已经形成
三目的链设置高于两目的链的就绪优先级，相同满度选择最老 leader，没有成束链时回退到 Base 到达
顺序；该调整不增加候选宽度、Fusion 端口、计算通路、缓存容量或片外接口。

正常代码路径复跑完成全部 `505,391` 个事件，得到 `17,840 cycles`，相对 Base 为 `1.709473x`；
记录位于仓库外 `GALA-runtime/records/r2_gaussian_chest_semantic_ready_group_v1_0101/`。运行形成
`4,350` 个语义束、消除 `7,238` 次独立 Fusion 提交，达到相同三目的不同 Bank 结构机会
`7,680` 次的约 `94%`。该结果比旧 `19,210-cycle` 版本再减少 `1,370 cycles`，高于
`1.443210x` 静态参考约 `18.4%`，距离当前 `16,477-cycle` 必要下界仍有 `1,363 cycles`；下界
假设全部同状态前向工作同时就绪，不能称为可达上限。定向 Fusion/在线周期测试为 `13 passed`，
完整 `tests/test_trace_cycle.py` 为 `97 passed`。该样本仍是 `quick_cycle_validation`，不外推到
正式三万轮训练。当前全仓回归为 `380 passed, 1 skipped, 2 warnings`。

同一最小真实样本随后完成语义放置与伴随语义束闭环。编译器按每个
`(iteration_id, gaussian_id)` 的真实前向、伴随和梯度归约 cluster-issue demand 执行确定性
LPT 放置，并把同一映射用于缓存实例、ComputePod 路由和 gradient owner；Base 与 B 单开不使用
该映射。前向和伴随 FIFO 分别增加同状态就绪链，最多三个不同查询状态 Bank 的同状态物理任务
共享一次 Fusion 控制提交，但伴随重放、ComputePod 工作、梯度归约、事件和依赖仍逐项保留。
两个 144 项索引及共享优先编码控制共占 `640 B` control metadata，不增加候选宽度、端口、Pod、
计算通路、数据 SRAM 容量或片外接口。

四点公平复测均完成 `505,391` 个事件：Base `0000=30,436`、B 单开 `0100=30,436`、语义驻留
`0101=15,604`、Full `1111=11,806 cycles`。语义驻留相对 Base 为 `1.950526x`，高于
`1.443210x` 静态参考约 `35.15%`；Full 为 `2.578011x`，相对 `2.648560x` 参考仍差约
`314 cycles`。`0101` 共形成 `8,056` 个语义束并合并 `10,950` 个 follower，其中前向为
`4,418/7,152`、伴随为 `3,638/3,798`；`1111` 中前向和伴随 follower 分别为 `4,074/756`。
四个变体的事件类型计数完全一致，Base 与 B 单开的语义束计数均为零。结果位于仓库外
`GALA-runtime/records/r2_gaussian_chest_semantic_adjoint_bundle_impl_v1_*`。
`full` 别名另行复跑也为 `11,806 cycles`，与 `variant:1111` 的全部 `505,391` 个事件完成周期、
模块计数和事件计数逐项一致。相关定向回归为 `165 passed`，全仓回归为
`387 passed, 1 skipped, 2 warnings`。

必要下界已同步覆盖前向和伴随语义束：两类各至少 `4,009` 次提交，加 `768` 次消费者提交，
语义场景理想 Fusion 提交下限为 `8,786` 次、计服务尾部为 `8,788 cycles`；当前整个样本的共同
必要下界由 ComputePod cluster issue 限制在 `9,828 cycles`。该下界假设同状态工作同时就绪且
查询 Bank 理想分布，只是必要条件，不是可达 Oracle。记录位于仓库外
`GALA-runtime/records/r2_gaussian_chest_semantic_adjoint_bundle_v1_bounds`。上述结果仍为
`quick_cycle_validation`，不外推到正式三万轮训练，AGX Orin 继续保持 `unavailable`。

## 2026-08-30 正式消融合同修正

当前正式消融不再把 A、B、C、D 解释为四个相互独立的硬件开关。A、B 分别生成查询负载规则和
语义工作集信息；硬件 C 必须依赖编译 A，硬件 D 必须依赖编译 B。正式配置集合与汇总顺序固定为
`0000`、`1000`、`1010`、`0100`、`0101`、`1100`、`1111`。其中 `1100` 只联合两类编译信息，
不启用硬件 C 或 D。此前记录的十六项矩阵仅保留为历史调试证据，不再构成当前正式消融、验收或
论文结果；后续正式运行只接受上述七项，且 `1111` 仍必须与完整
GALA 入口逐周期一致。

2026-08-30 完成最新七项正式消融和低成本公共地址散列复测。针对真实 R²-Gaussian + Chest
完整物理包样本（`505,391` 个事件、`779,583` 条依赖），为所有没有编译语义放置覆盖的
ComputePod、owner cluster 和语义缓存 fallback 路径统一采用
`gaussian_id ^ (gaussian_id >> 6) ^ (gaussian_id >> 13)` 后取模；该实现只使用 XOR 和移位，
不增加 Pod、cluster、端口、FIFO 或 SRAM 容量。三份已有物理 packet 样本的离线 demand 统计中，
最坏 Pod/平均负载比为 `1.063`，直接低位取模为 `1.102`。

同一 Ramulator 2、资源快照和 trace 下完成七项矩阵：`0000=31,205`、`1000=31,125`、
`1010=19,520`、`0100=31,205`、`0101=15,604`、`1100=31,125`、`1111=11,601 cycles`。
相对 Base 的加速比分别为 `1.000000x`、`1.002570x`、`1.598617x`、`1.000000x`、
`1.999808x`、`1.002570x` 和 `2.689854x`；查询参考 `1.445679x`、驻留参考 `1.443210x`、
Full 参考 `2.648560x` 均已达到。`1100` 只启用 A/B 两类编译信息，不启用 C/D，事件集合与
`1000` 一致。`full` 别名实际复跑同样为 `11,601 cycles`，与 `variant:1111` 逐项一致。
结果 CSV 和七份模块分解位于仓库外
`GALA-runtime/records/r2_gaussian_chest_xor6_13_ablation_v1.csv`，范围仍是
`quick_cycle_validation`，不能冒充完整三万轮正式性能结果；AGX Orin 实测缺失时仍为
`unavailable`。定向回归和全仓回归分别为 `148 passed` 与 `393 passed, 1 skipped, 2 warnings`。
