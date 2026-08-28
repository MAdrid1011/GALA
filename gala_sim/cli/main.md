# main.py

提供配置、原生预检、trace 校验、周期重放和消融的机器可读命令行入口。包级入口使用懒加载，避免 `python -m gala_sim.cli.main` 污染输出。

## External Interface

`gala-sim native-preflight --config <yaml> --freeze <json> --output <dir>` 验证冻结身份并执行官方短训练门。成功返回 0；门控或运行失败返回 2，并在输出目录写入状态记录。

`native-reference --config <yaml> --freeze <json> --preflight <json> --output <dir>` 只在预检通过后运行官方完整训练并写出质量、GPU 参考和状态文件。

`trace-sample --trace <dir> --output <dir> --query-range START:COUNT --max-events N --max-dependencies N --scan-events N --scan-backend cpu|cuda|auto` 从一个或多个 query range 的 consumer/gradient terminal 出发，抽取完整传递依赖闭包。输出保留真实 relation、地址、字节数和依赖，只能用于快速周期验证。

`trace-validate --trace <dir> [--scan-events N] [--index-directory DIR]` 对完整 trace 执行结构和生命周期校验。显式扫描块和临时索引目录只改变软件验证吞吐、临时空间和峰值，不改变检查集合；省略扫描块时沿用 trace capture chunk。`--index-directory` 必须与 `--scan-events` 一起使用。

其余入口为 `config-check`、`cycle-preflight`、`trace-validate`、`cycle-replay` 和 `ablation`。query sample 和显式迭代窗口 trace 默认均被周期入口拒绝；调用者必须显式传入 `--quick-validation`，输出 manifest 会固定 `formal_performance_eligible=false`。`ablation --parallel-workers N` 可并行运行独立变体；父进程按固定 `0000..1111` 顺序收集并验证结果。正式范围的 `ablation --orin-anchor <json>` 会在每个变体完成时输出相对于 Base ASIC 和静态 Orin 锚点的加速比，并写出非正式比较 sidecar；快速验证 trace 禁止使用该选项。

## Internal Helpers

解析 query range、Ramulator binding、资源使用快照和子命令参数；异常统一转换为非零退出状态。
