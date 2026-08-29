# Trace Sample Tests

验证 query 快速样本保留完整候选、关系、缓存、前向、归约、consumer、伴随和梯度依赖链，重建 dense event ID 和 payload offset，并固定 quick-only 元数据。

测试还检查显式事件上限和 CLI 防误用门：sample trace 未传 `--quick-validation` 时不得进入周期入口。

`test_real_query_packet_sample_preserves_supports_and_rebases_history` 构造两个连续迭代的来源 trace，验证查询关系包保留每条真实 relation/Gaussian 映射、记录源状态版本、归一化查询调度状态，并声明严格的策略白名单。

`test_cli_writes_real_query_packet_sample` 验证 `trace-query-packets` 写出可校验的快速样本及关系和物理包计数。
