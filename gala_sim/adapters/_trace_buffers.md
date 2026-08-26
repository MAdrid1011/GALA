# _trace_buffers.cpp

PyTorch C++ 扩展绑定，暴露官方 CUDA work buffer 解码和 raw trace terminal 扫描接口。所有输入必须是连续 CUDA tensor，布局和范围错误会抛出参数异常。
