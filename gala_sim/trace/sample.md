# sample.py

从超大 raw trace 中构造依赖闭合的 query 快速验证样本。该接口不估算或替代正式全量周期。

## External Interfaces

`QueryRange(start, count)` 定义半开 query ID 区间。

`TraceSampleConfig(query_ranges, max_events, max_dependencies, scan_events, scan_backend)` 保存全部 quick-validation 选择和资源上限。

`dependency_closed_query_sample(trace, config, *, source_identity, progress)` 选择区间内的 consumer 和 gradient terminal，递归加入全部真实前置依赖，重排 event/dependency/payload offset，并返回通过结构校验的独立 `Trace`。超过显式事件或依赖上限、区间为空或 CUDA 后端不可用时抛出 `ValueError` 或 `RuntimeError`。

## Internal Helpers

CPU 后端使用 NumPy 大块扫描；CUDA 后端将连续 raw event bytes 交给 v9 decoder 的 terminal-mask kernel。闭包阶段保留原始 ID 集合，最终使用排序索引把依赖映射到样本内 dense event ID。mmap 页在每个扫描块后回收。
