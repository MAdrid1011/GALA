# _trace_buffers.cpp

This PyTorch extension exposes read-only decoding of CUDA raster and voxel work
buffers and a terminal-event scan interface. Inputs must be contiguous CUDA
tensors with schema-compatible strides and offsets; invalid layouts raise a
Python exception.
