# 模拟器软件架构

## 1. 总体结构

模拟器采用功能执行与周期执行分离、事件合同统一的结构。功能执行器运行真实模型，生成动态关系、原语任务、数值结果和最终重建体。周期执行器读取相同事件，模拟 GALA 的队列、流水、Bank、端口、缓存、存储请求和反压。两条路径共享事件编号、依赖、状态版本和配置哈希。

```text
Official Model and Dataset
        |
        v
Model Adapter -> CLAMP Task Builder -> Functional GPU Executor
        |                    |                  |
        |                    |                  +-> Reconstruction and Metrics
        |                    v
        +------------> Device Trace Buffers
                             |
                             v
                    Async Chunk Transfer
                             |
                             v
                 Event-driven Cycle Simulator
                             |
                             v
              Cycles, stalls, module counters
```

功能执行器不读取周期结果来改变模型数学路径。周期执行器不生成数值近似来代替功能执行。需要模拟调度顺序对归约数值的影响时，周期执行器输出确定的提交顺序，功能执行器按该顺序执行归约重放。

## 2. Python 包边界

后续代码使用以下包结构。目录名和职责在首个代码里程碑中冻结。

```text
gala_sim/
  cli/                 command entry points
  config/              typed configuration and validation
  adapters/            model and dataset adapters
  clamp/               combinators, primitives, task packets, analyses
  functional/          GPU numerical executor and quality path
  trace/               device buffers, schemas, chunk writer and reader
  timing/
    kernel/            event queue, clock domains and backpressure
    modules/           one implementation per hardware module
    memory/            on-chip banks and Ramulator 2 bridge
  ablation/            switch semantics and matrix runner
  metrics/             cycle, stall, PSNR, SSIM and LPIPS
  results/             manifest and table writers
  tools/               download, conversion and preflight utilities
tests/
  real_smoke/          one licensed real sample per available adapter
  unit/                arithmetic and state-transition tests only
configs/
  architecture/
  models/
  datasets/
  campaigns/
```

`timing/modules` 只能包含硬件合同中的模块。`trace`、`results` 和 `tools` 是软件支撑层，不进入被模拟硬件的模块清单。

## 3. 功能执行器

模型适配器从官方训练入口接管以下边界。

1. 读取投影、扫描几何、初始高斯和训练配置。
2. 将关系生成、贡献计算、查询归约、消费者、伴随、更新和集合修改映射为 CLAMP 任务。
3. 运行原模型的数值公式和训练步数。
4. 在设备端写出真实动态事件，不在 Python 中重建平均事件。
5. 保存最终体数据、模型检查点和指标输入。

官方 CUDA 扩展仍作为数值真实性参考。适配器可以增加旁路 trace 输出，但不得删减原有计算。修改 CUDA 扩展时保留上游提交号和最小补丁，并使未启用 trace 时的输出与上游一致。

## 4. 追踪层

追踪层使用预分配的结构化设备缓冲区。每个事件至少包含以下字段。

| 字段 | 含义 |
| --- | --- |
| `event_id` | 全局唯一事件编号 |
| `iteration_id` | 重建迭代编号 |
| `primitive_kind` | CLAMP 原语类型 |
| `query_id` | 查询编号，无查询时使用类型化空值 |
| `gaussian_id` | 高斯编号，无高斯时使用类型化空值 |
| `state_version` | 高斯状态版本 |
| `relation_id` | 前向与伴随共享的关系编号 |
| `consumer_id` | 局部消费者实例编号 |
| `reduction_key` | 查询或高斯归约键 |
| `resource_class` | 目标执行资源 |
| `dependency_begin`、`dependency_count` | 依赖数组范围 |
| `template_id` | 精确算术模板编号 |
| `field_mask` | 访问的高斯字段 |
| `address_token` | 可确定地映射到 SRAM 或 DRAM 地址的布局令牌 |
| `payload_offset` | 数值重放所需数据范围 |

设备缓冲区满时，使用 CUDA stream 将完整 chunk 异步复制到固定页内存。CPU 周期线程消费上一个 chunk，GPU 同时生成下一个 chunk。追踪格式使用按列 NumPy 数组或 Arrow IPC，字段类型由 `gala-clamp-events-v2` schema 固定。`UPDATE_BEGIN` 与 `UPDATE_END` 包围优化器提交或集合修改事务；`UPDATE_END.field_mask == 0` 是不推进状态版本的控制屏障，非零掩码才关闭旧版本并推进状态。禁止逐事件 JSON 和逐事件 Python 回调。

对于 R²-Gaussian 的密集 raster/voxel 查询，追踪层还提供
`gala-trace-virtual-packet-v1` 工作缓冲区表示。每个包保留官方 CUDA 输出的
`point_list`、`point_key` 和完整 `uint32` valid-mask，不保存展开后的关系事件；
置位 bit 与真实 Gaussian-query relation 一一对应。`VirtualTracePacket` 只属于
生产端的有界工作包，不能直接作为周期 trace 或跳过生命周期检查。后续的有界
事件展开器必须为所有事件分配全局连续 `event_id`，保留全局依赖 ID、跨包状态版本、
未完成缓存读和语义工作集 sidecar，再将全局事件包交给 validator 和周期执行器。
当前实现已经覆盖一个查询包的候选、关系、缓存请求/返回、前向、查询归约、消费者、
伴随和梯度归约前缀；更新事务、集合修改和跨迭代状态仍必须由上层生命周期展开器提供。
`VirtualTraceLifecycleValidator` 保留当前迭代的候选/关系/反向计数、活动 Gaussian
集合、状态版本和更新事务，并在迭代关闭时写出有界 ledger；它拒绝跨迭代的开放事务、
失效 Gaussian、重复 lineage 或版本跳变。该 ledger 只记录状态和计数，不替代事件包中的
真实依赖、缓存读写或 Ramulator 返回。
生产线程与消费线程之间最多保留配置数量的包；生产、消费、关系数量、物理包字节数
和峰值驻留字节数写入运行记录。任一生产或消费方向连续五分钟没有进展时，运行必须
停止并标记失败，不得把工作缓冲包当作已完成的正式 trace。

## 5. 周期执行器

周期执行器以模块下一次状态变化为调度粒度。全局内核维护少量模块唤醒项，每个硬件模块维护自己的输入队列、在途项、完成队列和可用资源周期。调度器从当前最早唤醒周期跳到下一周期，并批量处理同周期事件。

模块实现统一提供以下接口。

```python
class CycleModule(Protocol):
    def next_wakeup(self) -> int | None: ...
    def accept(self, batch: EventBatch, cycle: int) -> AcceptResult: ...
    def advance(self, cycle: int) -> ModuleOutputs: ...
    def snapshot_counters(self) -> CounterBlock: ...
```

`accept` 必须返回接纳、拒绝和阻塞原因。`advance` 只处理该周期发生的流水完成、资源释放、状态更新和输出事件。模块不得通过读取全局未来状态绕过端口、队列或反压。

## 6. 数值与周期重放

绝大多数优化只改变事件起始周期，不改变数值结果。归约事件的提交顺序可能影响 FP32 舍入，因此周期执行器为查询归约、高斯梯度归约和更新提交输出有序日志。功能执行器以该日志执行确定性重放，得到完整 GALA 质量结果。

调试模式允许在短真实样例上逐事件对照。正式模式只保存归约提交顺序、模块统计和可选采样 trace，不保存全部数值 payload，以控制磁盘与内存占用。

## 7. 运行模式

| 模式 | 用途 | 允许的简化 |
| --- | --- | --- |
| `native-reference` | 运行官方 CUDA 并产生参考质量和端到端时间 | 不产生 GALA 周期 |
| `trace-capture` | 运行真实数值并生成完整事件 | 可关闭非必要可视化和中间检查点 |
| `cycle-replay` | 从已验证 trace 运行一个或多个消融配置 | 不重复 GPU 数值执行 |
| `functional-replay` | 按选定周期顺序重放归约并输出质量 | 不重新捕获关系 |
| `campaign` | 串联参考、追踪、七项正式消融和质量汇总 | 不允许跳过验收门 |

同一个经过校验的 trace 可以用于七项正式消融，从而避免重复运行昂贵的模型前向和反向。正式位型固定为 `0000`、`1000`、`1010`、`0100`、`0101`、`1100`、`1111`；硬件 C 必须消费编译 A 生成的信息，硬件 D 必须消费编译 B 生成的信息。trace 只有在模型提交、数据校验值、训练配置、适配器版本和 CLAMP schema 完全一致时才能复用。稳定 Gaussian ID、父子 lineage、跨迭代 `UPDATE_END` 依赖和状态版本级缓存驻留均由 validator 校验；查询关闭不会释放缓存，只有真实写入的更新结束才释放旧版本。周期入口对同一份 mmap trace 只执行一次结构校验，随后用 NumPy 压缩反向依赖索引按固定顺序重放七个正式变体。
