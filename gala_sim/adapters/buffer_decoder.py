"""Optional decoder for the official CUDA rasterizer's returned work buffers."""

from __future__ import annotations

from functools import lru_cache
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def load_buffer_decoder() -> Any:
    """Compile/load the small read-only decoder in the active Torch environment."""

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as error:  # pragma: no cover - exercised in official env
        raise RuntimeError("trace capture requires the official PyTorch environment") from error
    source = Path(__file__).with_name("_trace_buffers.cpp")
    if not source.is_file():
        raise FileNotFoundError(f"trace buffer decoder source is missing: {source}")
    name = "gala_trace_buffers_v5"
    try:
        from torch.utils.cpp_extension import get_default_build_root
        cache_root = Path(get_default_build_root())
        cached = next(cache_root.rglob(f"{name}*.so"), None)
        if cached is not None:
            sys.path.insert(0, str(cached.parent))
            try:
                return importlib.import_module(name)
            finally:
                sys.path.pop(0)
    except (ImportError, OSError):
        pass
    injected_environment: list[str] = []
    for variable, executable in (("CC", "gcc-11"), ("CXX", "g++-11")):
        compiler = shutil.which(executable)
        if variable not in os.environ and compiler is not None:
            os.environ[variable] = compiler
            injected_environment.append(variable)
    try:
        return load(
            name=name,
            sources=[str(source), str(source.with_name("_trace_relations.cu"))],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=["-O2"],
            with_cuda=True,
            verbose=False,
        )
    finally:
        for variable in injected_environment:
            os.environ.pop(variable, None)


def decode_raster_buffers(
    decoder: Any, binning_buffer: Any, image_buffer: Any, rendered: int, image_elements: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy raster point IDs, sorted keys and tile ranges to host arrays."""

    point_ids = decoder.copy_raster_point_list(binning_buffer, int(rendered)).cpu().numpy()
    point_keys = decoder.copy_raster_point_keys(binning_buffer, int(rendered)).cpu().numpy()
    ranges = decoder.copy_ranges(image_buffer, int(image_elements)).cpu().numpy()
    return (
        np.asarray(point_ids, dtype=np.int64),
        np.asarray(point_keys, dtype=np.uint64),
        np.asarray(ranges, dtype=np.int64),
    )


def decode_voxel_buffers(
    decoder: Any, binning_buffer: Any, image_buffer: Any, rendered: int, voxel_elements: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy voxel point IDs, sorted keys and voxel-tile ranges to host arrays."""

    point_ids = decoder.copy_voxel_point_list(binning_buffer, int(rendered)).cpu().numpy()
    point_keys = decoder.copy_voxel_point_keys(binning_buffer, int(rendered)).cpu().numpy()
    ranges = decoder.copy_ranges(image_buffer, int(voxel_elements)).cpu().numpy()
    return (
        np.asarray(point_ids, dtype=np.int64),
        np.asarray(point_keys, dtype=np.uint64),
        np.asarray(ranges, dtype=np.int64),
    )
