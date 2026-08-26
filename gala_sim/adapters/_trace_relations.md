# _trace_relations.cu

实现 raster/voxel 有效关系压紧 kernel，以及按调用方提供的 event stride、字段 offset 和 query ranges 扫描 raw event bytes 的 CUDA terminal-mask kernel。
