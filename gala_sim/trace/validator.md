# validator.py

验证结构化 trace 的事件编号、依赖范围、查询/关系链、缓存请求、消费者、反向梯度、状态版本、更新事务和 capture audit。

## External Interfaces

`TraceValidationConfig(scan_events, index_directory)` 配置全量 raw trace 的软件扫描块和临时紧凑索引目录。

`validate_trace(trace, *, config=None)` 返回 `TraceValidationReport`；普通 trace 使用完整生命周期状态机，capture-audit v4 raw trace 使用磁盘紧凑索引和分块向量检查。任何结构、身份、版本或审计不一致均抛出 `TraceValidationError`。

## Internal Helpers

流式路径把跨事件回查限制为按实际 domain 压缩的 kind/query/Gaussian/relation 字段，并使用紧凑 relation index。含 optimizer 或 collection 的大 trace 在完整状态迁移 pass 实现前会明确拒绝，不会降级检查。
