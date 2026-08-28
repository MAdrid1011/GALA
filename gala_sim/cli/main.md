# main.py

提供配置、原生预检、trace 校验、周期重放和消融的机器可读命令行入口。包级入口使用懒加载，避免 `python -m gala_sim.cli.main` 污染输出。

## External Interface

`gala-sim native-preflight --config <yaml> --freeze <json> --output <dir>` 验证冻结身份并执行官方短训练门。成功返回 0；门控或运行失败返回 2，并在输出目录写入状态记录。

`native-reference --config <yaml> --freeze <json> --preflight <json> --output <dir>` 只在预检通过后运行官方完整训练并写出质量、GPU 参考和状态文件。

`trace-sample --trace <dir> --output <dir> --query-range START:COUNT --max-events N --max-dependencies N --scan-events N --scan-backend cpu|cuda|auto` 从一个或多个 query range 的 consumer/gradient terminal 出发，抽取完整传递依赖闭包。输出保留真实 relation、地址、字节数和依赖，只能用于快速周期验证。

`trace-packetize --trace <quick-dir> --output <new-dir> --query-domain TEMPLATE:BASE:DIMxDIM[xDIM] --query-lanes N` 为旧版 dependency-closed quick trace 派生物理 RelationPacket lane/mask 元数据。命令只允许 quick trace，只修改 packetized event 的 `flags`，写盘前验证其他事件列、依赖和 payload 完全不变，并验证物理包计划；输出不提升正式性能或质量资格。

`trace-captured-packets --manifest <json> --output <dir> --max-events N --query-lanes N --initial-gaussian-count N` 从 CUDA 捕获器保存的 candidate-major 四列记录抽取 manifest 指定的 canonical tile/brick，重建原始 sparse mask，并完整展开前向、查询消费者、伴随和梯度链。输出前执行结构验证和物理包计划验证，结果固定为 quick validation，不作为正式性能或质量结果。

`trace-validate --trace <dir> [--scan-events N] [--index-directory DIR]` 对完整 trace 执行结构和生命周期校验。显式扫描块和临时索引目录只改变软件验证吞吐、临时空间和峰值，不改变检查集合；省略扫描块时沿用 trace capture chunk。`--index-directory` 必须与 `--scan-events` 一起使用。

`gala-r2-trace-runner --virtual-capture --packet-archive-root <dir> --capture-config <gala.yaml>` 在不展开事件列的情况下保存有序、紧凑的 packet/lifecycle 归档；在线周期模式可复用 `--online-cycle-config`，无需重复传入 `--capture-config`。归档按配置的未压缩数组字节上限分块，可由多个独立周期会话重复读取。归档只有在完整 `1..30000` 迭代和显式验证证据同时满足时才允许标记正式资格。

其余入口为 `config-check`、`cycle-preflight`、`trace-validate`、`cycle-replay` 和 `ablation`。query sample 和显式迭代窗口 trace 默认均被周期入口拒绝；调用者必须显式传入 `--quick-validation`，输出 manifest 会固定 `formal_performance_eligible=false`。`ablation --parallel-workers N` 可并行运行独立变体；父进程按固定 `0000..1111` 顺序收集并验证结果。消融进度只输出绝对周期和相对 Base ASIC 的加速比；没有同套件 AGX Orin 实测校准时，不生成 Orin 比较。

`cycle-replay --throughput-progress` 按完整迭代输出运行健康状态、墙钟事件吞吐、每事件模拟周期、总周期投影和按配置时钟换算的 ASIC 秒数。只有显式增加 `--stop-when-throughput-stable` 才允许稳定窗口提前结束；该路径要求输出目录不存在或为空，只写 `throughput.json`、非正式 manifest 和标明非正式资格的状态记录，不写 `cycles.json`。不带早停选项时始终保留完整重放语义。公开规格 extrapolation 不接入周期诊断，也不生成任何 `speedup_vs_orin` 字段。

`cycle-replay --compute-telemetry` 额外保存每个 ComputePod 事件的依赖就绪、融合发射、查询发射、
计算发射和完成周期，以及二十簇 active-microcontext 占用的无损区间编码。该开关只采集诊断，不能
改变周期决策。ComputePod 资源停顿在 `stalls.parquet` 中同时记录首个阻塞资源、Pod、簇、冲突
周期、已用量、需求量和容量。Oracle portfolio 的 Base、actual 和 future 成员分别保存到
`oracle_members/`，主目录仍表示胜者。

所有 `cycle-replay` 运行都启用配置化 inactivity watchdog。准备阶段以阶段切换或显著进程 CPU 时间作为推进；正式 replay 阶段只有完成事件数或完成迭代数增长才续时，周期心跳与日志打印本身不续时。连续达到门限后输出固定为 `failed_cycle/watchdog_inactivity_timeout`，不写 `cycles.json`。

## Internal Helpers

解析 query range、Ramulator binding、资源使用快照和子命令参数；异常统一转换为非零退出状态。
