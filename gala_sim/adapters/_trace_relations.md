# _trace_relations.cu

CUDA kernels derive raster and voxel relation masks from official work buffers
and scan raw event bytes for requested terminal kinds. Kernels do not change
the source buffers or define model policy.
