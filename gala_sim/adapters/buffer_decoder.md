# buffer_decoder.py

## External Interfaces

`load_buffer_decoder()` builds or loads the versioned PyTorch CUDA extension.
Raster and voxel decoder functions copy point lists, keys, ranges, and valid
masks from official work buffers. `scan_trace_terminals_cuda()` returns local
event indices for requested terminal kinds.

## Internal Helpers

The extension cache is isolated by module version. Available GCC 11 compilers
are selected for compatibility without changing an explicitly configured
compiler environment.
