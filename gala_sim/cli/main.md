# main.py

提供配置、原生预检、trace 校验、周期重放和消融的机器可读命令行入口。包级入口使用懒加载，避免 `python -m gala_sim.cli.main` 污染输出。

## External Interface

`gala-sim native-preflight --config <yaml> --freeze <json> --output <dir>` 验证冻结身份并执行官方短训练门。成功返回 0；门控或运行失败返回 2，并在输出目录写入状态记录。

`native-reference --config <yaml> --freeze <json> --preflight <json> --output <dir>` 只在预检通过后运行官方完整训练并写出质量、GPU 参考和状态文件。

其余入口为 `config-check`、`cycle-preflight`、`trace-validate`、`cycle-replay` 和 `ablation`。

## Internal Helpers

解析 Ramulator binding、资源使用快照和子命令参数；异常统一转换为非零退出状态。
