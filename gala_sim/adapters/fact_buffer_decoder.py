"""Decode FaCT-GS split rasterizer tensors into GALA relation packets."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from gala_sim.trace import VirtualTracePacket


RASTER_BLOCK = (16, 16)
VOXEL_BLOCK = (8, 8, 8)
MASK_WORD_BITS = 32
_VIRTUAL_CANDIDATE_CHUNK = 2048


def _candidate_slice(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    start: int,
    count: int,
) -> tuple[Any, Any, Any]:
    import torch

    candidate_count = int(gaussian_ids_sorted.numel())
    if start < 0 or count < 0 or start + count > candidate_count:
        raise ValueError("FaCT candidate slice is outside the sorted point list")
    indexes = torch.arange(
        start, start + count, device=gaussian_ids_sorted.device, dtype=torch.int64,
    )
    if count:
        starts = tile_bins[:, 0].to(dtype=torch.int64)
        ends = tile_bins[:, 1].to(dtype=torch.int64)
        nonempty = ends > starts
        active_tiles = torch.nonzero(nonempty, as_tuple=False).reshape(-1)
        active_ends = ends[nonempty]
        active_indexes = torch.searchsorted(active_ends, indexes, right=True)
        if bool(torch.any(active_indexes >= int(active_tiles.numel())).item()):
            raise ValueError("FaCT tile bins do not cover the sorted point list")
        tile_ids = active_tiles[active_indexes]
        if bool(torch.any(indexes < starts[tile_ids]).item()):
            raise ValueError("FaCT tile bins contain a gap in the sorted point list")
    else:
        tile_ids = indexes
    point_ids = gaussian_ids_sorted[start:start + count].to(dtype=torch.int64)
    point_keys = tile_ids << 32
    return point_ids, point_keys, tile_ids


def _raster_validity(
    point_ids: Any,
    tile_ids: Any,
    pos2d: Any,
    conics_mu: Any,
    intensities: Any,
    image_shape: tuple[int, int],
) -> Any:
    import torch

    height, width = (int(value) for value in image_shape)
    blocks_x = (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
    local = torch.arange(
        RASTER_BLOCK[0] * RASTER_BLOCK[1], device=point_ids.device,
        dtype=torch.int64,
    )
    local_y = local // RASTER_BLOCK[1]
    local_x = local % RASTER_BLOCK[1]
    pixel_x = (tile_ids % blocks_x)[:, None] * RASTER_BLOCK[1] + local_x
    pixel_y = (tile_ids // blocks_x)[:, None] * RASTER_BLOCK[0] + local_y
    inside = (pixel_x < width) & (pixel_y < height)

    positions = pos2d[point_ids]
    conics = conics_mu[point_ids]
    values = intensities[point_ids]
    if values.ndim == 1:
        values = values[:, None]
    delta_x = positions[:, 0, None] - pixel_x.to(dtype=positions.dtype)
    delta_y = positions[:, 1, None] - pixel_y.to(dtype=positions.dtype)
    sigma = (
        0.5 * (
            conics[:, 0, None] * delta_x.square()
            + conics[:, 2, None] * delta_y.square()
        )
        + conics[:, 1, None] * delta_x * delta_y
    )
    alpha = torch.exp(-sigma) * conics[:, 3, None]
    return (
        inside
        & torch.isfinite(sigma)
        & (sigma >= 0)
        & (alpha * values.mean(dim=-1)[:, None] >= 1.0e-5)
    )


def _voxel_validity(
    point_ids: Any,
    tile_ids: Any,
    pos3d_radii: Any,
    conics: Any,
    intensities: Any,
    volume_shape: tuple[int, int, int],
    *,
    virtual_order: bool,
) -> Any:
    import torch

    voxel_x, voxel_y, voxel_z = (int(value) for value in volume_shape)
    blocks_x = (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
    blocks_y = (voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1]
    local = torch.arange(
        VOXEL_BLOCK[0] * VOXEL_BLOCK[1] * VOXEL_BLOCK[2],
        device=point_ids.device, dtype=torch.int64,
    )
    if virtual_order:
        local_x = local // (VOXEL_BLOCK[1] * VOXEL_BLOCK[2])
        local_y = (local // VOXEL_BLOCK[2]) % VOXEL_BLOCK[1]
        local_z = local % VOXEL_BLOCK[2]
    else:
        local_x = local % VOXEL_BLOCK[0]
        local_y = (local // VOXEL_BLOCK[0]) % VOXEL_BLOCK[1]
        local_z = local // (VOXEL_BLOCK[0] * VOXEL_BLOCK[1])

    tile_x = tile_ids % blocks_x
    tile_y = (tile_ids // blocks_x) % blocks_y
    tile_z = tile_ids // (blocks_x * blocks_y)
    query_x = tile_x[:, None] * VOXEL_BLOCK[0] + local_x
    query_y = tile_y[:, None] * VOXEL_BLOCK[1] + local_y
    query_z = tile_z[:, None] * VOXEL_BLOCK[2] + local_z
    inside = (query_x < voxel_x) & (query_y < voxel_y) & (query_z < voxel_z)

    positions = pos3d_radii[point_ids]
    gaussian_conics = conics[point_ids]
    values = intensities[point_ids]
    if values.ndim == 1:
        values = values[:, None]
    delta_x = positions[:, 0, None] - query_x.to(dtype=positions.dtype) - 0.5
    delta_y = positions[:, 1, None] - query_y.to(dtype=positions.dtype) - 0.5
    delta_z = positions[:, 2, None] - query_z.to(dtype=positions.dtype) - 0.5
    sigma = (
        -0.5 * (
            gaussian_conics[:, 0, None] * delta_x.square()
            + gaussian_conics[:, 3, None] * delta_y.square()
            + gaussian_conics[:, 5, None] * delta_z.square()
        )
        - gaussian_conics[:, 1, None] * delta_x * delta_y
        - gaussian_conics[:, 2, None] * delta_x * delta_z
        - gaussian_conics[:, 4, None] * delta_y * delta_z
    )
    alpha = torch.exp(sigma)
    return (
        inside
        & torch.isfinite(sigma)
        & (sigma <= 0)
        & (alpha * values.mean(dim=-1)[:, None] >= 1.0e-6)
    )


def _records(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    start: int,
    count: int,
    validity_fn: Callable[[Any, Any], Any],
) -> Any:
    import torch

    point_ids, point_keys, tile_ids = _candidate_slice(
        gaussian_ids_sorted, tile_bins, start, count,
    )
    valid = validity_fn(point_ids, tile_ids)
    relation_indexes = torch.nonzero(valid, as_tuple=False)
    records = torch.zeros(
        (count + int(relation_indexes.shape[0]), 4),
        device=gaussian_ids_sorted.device, dtype=torch.int64,
    )
    if count:
        records[:count, 1] = torch.arange(
            count, device=gaussian_ids_sorted.device, dtype=torch.int64,
        )
        records[:count, 2] = point_ids
        records[:count, 3] = point_keys
    if relation_indexes.numel():
        records[count:, 0] = 1
        records[count:, 1:3] = relation_indexes
    return records


def decode_fact_raster_records(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    pos2d: Any,
    conics_mu: Any,
    intensities: Any,
    image_shape: tuple[int, int],
    start: int,
    count: int,
) -> Any:
    return _records(
        gaussian_ids_sorted, tile_bins, start, count,
        lambda point_ids, tile_ids: _raster_validity(
            point_ids, tile_ids, pos2d, conics_mu, intensities, image_shape,
        ),
    )


def decode_fact_voxel_records(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    pos3d_radii: Any,
    conics: Any,
    intensities: Any,
    volume_shape: tuple[int, int, int],
    start: int,
    count: int,
) -> Any:
    return _records(
        gaussian_ids_sorted, tile_bins, start, count,
        lambda point_ids, tile_ids: _voxel_validity(
            point_ids, tile_ids, pos3d_radii, conics, intensities,
            volume_shape, virtual_order=False,
        ),
    )


def _pack_masks(valid: Any) -> np.ndarray:
    import torch

    if valid.shape[1] % MASK_WORD_BITS:
        raise ValueError("FaCT validity width is not a whole number of mask words")
    bits = torch.arange(MASK_WORD_BITS, device=valid.device, dtype=torch.int64)
    words = torch.sum(
        valid.reshape(valid.shape[0], -1, MASK_WORD_BITS).to(dtype=torch.int64)
        << bits,
        dim=-1,
    )
    return words.detach().cpu().numpy().astype(np.dtype("<u4"), copy=False)


def _virtual_packet(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    validity_fn: Callable[[Any, Any], Any],
    *,
    iteration_id: int,
    template_id: int,
    query_base: int,
    query_shape: tuple[int, ...],
    state_version: int,
    field_mask: int,
) -> VirtualTracePacket:
    point_id_parts: list[np.ndarray] = []
    point_key_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    candidate_count = int(gaussian_ids_sorted.numel())
    for start in range(0, candidate_count, _VIRTUAL_CANDIDATE_CHUNK):
        count = min(_VIRTUAL_CANDIDATE_CHUNK, candidate_count - start)
        point_ids, point_keys, tile_ids = _candidate_slice(
            gaussian_ids_sorted, tile_bins, start, count,
        )
        point_id_parts.append(point_ids.detach().cpu().numpy().astype(np.int64, copy=False))
        point_key_parts.append(point_keys.detach().cpu().numpy().astype(np.uint64, copy=False))
        mask_parts.append(_pack_masks(validity_fn(point_ids, tile_ids)))
    local_queries = RASTER_BLOCK[0] * RASTER_BLOCK[1]
    if template_id != 1:
        local_queries = VOXEL_BLOCK[0] * VOXEL_BLOCK[1] * VOXEL_BLOCK[2]
    return VirtualTracePacket.from_capture_buffers(
        iteration_id=iteration_id,
        template_id=template_id,
        query_base=query_base,
        query_shape=query_shape,
        point_ids=(np.concatenate(point_id_parts) if point_id_parts else np.empty(0, np.int64)),
        point_keys=(np.concatenate(point_key_parts) if point_key_parts else np.empty(0, np.uint64)),
        masks=(
            np.concatenate(mask_parts)
            if mask_parts
            else np.empty((0, local_queries // MASK_WORD_BITS), dtype=np.dtype("<u4"))
        ),
        state_version=state_version,
        field_mask=field_mask,
    )


def decode_fact_raster_virtual_packet(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    pos2d: Any,
    conics_mu: Any,
    intensities: Any,
    image_shape: tuple[int, int],
    *,
    iteration_id: int,
    template_id: int,
    query_base: int,
    state_version: int,
    field_mask: int,
) -> VirtualTracePacket:
    return _virtual_packet(
        gaussian_ids_sorted, tile_bins,
        lambda point_ids, tile_ids: _raster_validity(
            point_ids, tile_ids, pos2d, conics_mu, intensities, image_shape,
        ),
        iteration_id=iteration_id, template_id=template_id, query_base=query_base,
        query_shape=image_shape, state_version=state_version, field_mask=field_mask,
    )


def decode_fact_voxel_virtual_packet(
    gaussian_ids_sorted: Any,
    tile_bins: Any,
    pos3d_radii: Any,
    conics: Any,
    intensities: Any,
    volume_shape: tuple[int, int, int],
    *,
    iteration_id: int,
    template_id: int,
    query_base: int,
    state_version: int,
    field_mask: int,
) -> VirtualTracePacket:
    return _virtual_packet(
        gaussian_ids_sorted, tile_bins,
        lambda point_ids, tile_ids: _voxel_validity(
            point_ids, tile_ids, pos3d_radii, conics, intensities,
            volume_shape, virtual_order=True,
        ),
        iteration_id=iteration_id, template_id=template_id, query_base=query_base,
        query_shape=volume_shape, state_version=state_version, field_mask=field_mask,
    )
