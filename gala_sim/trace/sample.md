# sample.py

从超大 raw trace 中构造依赖闭合的 query 快速验证样本。该接口不估算或替代正式全量周期。

## External Interfaces

`QueryRange(start, count)` 定义半开 query ID 区间。

`TraceSampleConfig(query_ranges, max_events, max_dependencies, scan_events, scan_backend)` 保存全部 quick-validation 选择和资源上限。

`dependency_closed_query_sample(trace, config, *, source_identity, progress)` 选择区间内的 consumer 和 gradient terminal，递归加入全部真实前置依赖，重排 event/dependency/payload offset，并返回通过结构校验的独立 `Trace`。超过显式事件或依赖上限、区间为空或 CUDA 后端不可用时抛出 `ValueError` 或 `RuntimeError`。

`QueryPacketSampleConfig(query_ranges, scan_events, scan_backend, query_lanes, ssim_radius)` 配置连续迭代的查询调度微基准。它不设置依赖闭包上限；`scan_events` 只控制源 trace 扫描块和展开块大小。

`real_query_packet_sample(trace, config, *, source_identity, progress)` 从源 trace 提取每个查询区间的真实 `RELATION`、候选 Gaussian、候选排序键和 consumer 标志，再将每个查询区间重定位到一个物理行并展开完整前向、缓存、归约、consumer、伴随和梯度链。同一迭代可选择多个物理行；输入必须覆盖连续迭代，且各轮的模板、行形状和相对查询位置完全一致。`gala-query-packet-sample-v2` 保留每轮源状态版本，因此可在 quick scope 内运行两个 Oracle、必要下界和 `0000,1000,1010,0100,0101,1100,1111` 七项配置；C 必须依赖 A，D 必须依赖 B。旧 v1 的版本零样本继续只允许查询调度策略。两种格式均固定为 `quick_cycle_validation`，不得用于更新、质量或正式性能实验。

## Internal Helpers

CPU 后端使用 NumPy 大块扫描；CUDA 后端将连续 raw event bytes 交给 v9 decoder 的通用双 primitive/query-range mask。依赖闭包阶段保留原始 ID 集合，最终使用排序索引把依赖映射到样本内 dense event ID。真实查询关系包路径只读取关系的单一候选依赖，并以源候选 event ID 合并 mask，不遍历 optimizer 的全迭代 fan-in。mmap 页在每个扫描块后回收。
