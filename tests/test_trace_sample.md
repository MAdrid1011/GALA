# Trace Sample Tests

验证 query 快速样本保留完整候选、关系、缓存、前向、归约、consumer、伴随和梯度依赖链，重建 dense event ID 和 payload offset，并固定 quick-only 元数据。

测试还检查显式事件上限和 CLI 防误用门：sample trace 未传 `--quick-validation` 时不得进入周期入口。
