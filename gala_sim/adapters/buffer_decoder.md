# buffer_decoder.py

加载官方 PyTorch 环境中的只读 CUDA trace decoder，并提供 raster、voxel 与 raw trace 扫描接口。

## External Interfaces

`load_buffer_decoder()` 编译或加载版本化 CUDA 扩展。

`decode_raster_buffers()` 与 `decode_voxel_buffers()` 解码官方 work buffer。

`scan_trace_terminals_cuda()` 将连续结构化 event bytes、dtype stride/offset 和 query ranges 传入 CUDA，返回匹配 consumer/gradient terminal 的局部 event index。

## Internal Helpers

扩展缓存按模块版本隔离；编译器默认选择 GCC/G++ 11。raw scanner 不硬编码 event schema offset。
