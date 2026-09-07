"""Exact physical RelationPacket batches decoded directly from packed masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from gala_sim.trace.virtual import (
    RASTER_BLOCK,
    RASTER_TEMPLATE_ID,
    VOXEL_BLOCK,
    VOXEL_TEMPLATE_ID,
    VirtualTracePacket,
)


PACKED_RELATION_PACKET_DTYPE = np.dtype([
    ("packet_ordinal", "<u8"),
    ("query_pack", "<u8"),
    ("candidate_index", "<i8"),
    ("query_base", "<i8"),
    ("gaussian_id", "<i8"),
    ("point_key", "<u8"),
    ("consumer_release_pack", "<u8"),
    ("lane_mask", "u1"),
    ("active_lanes", "u1"),
])

_POPCOUNT8 = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
_FIRST_LANE8 = np.asarray([
    (value & -value).bit_length() - 1 if value else 0
    for value in range(256)
], dtype=np.uint8)


@dataclass(frozen=True)
class PackedRelationPacketBatch:
    records: np.ndarray

    def __post_init__(self) -> None:
        records = np.asarray(self.records)
        if records.dtype != PACKED_RELATION_PACKET_DTYPE or records.ndim != 1:
            raise ValueError("packed RelationPacket records have an invalid layout")

    @property
    def packet_count(self) -> int:
        return int(self.records.size)

    @property
    def logical_relation_count(self) -> int:
        return int(self.records["active_lanes"].sum(dtype=np.uint64))


@dataclass(frozen=True)
class PackedTileStatistics:
    tile_ids: np.ndarray
    candidate_counts: np.ndarray
    logical_relation_counts: np.ndarray
    physical_packet_counts: np.ndarray
    lane_histograms: np.ndarray

    def __post_init__(self) -> None:
        size = int(np.asarray(self.tile_ids).size)
        if any(
            np.asarray(value).shape != (size,)
            for value in (
                self.candidate_counts,
                self.logical_relation_counts,
                self.physical_packet_counts,
            )
        ) or np.asarray(self.lane_histograms).shape != (size, 9):
            raise ValueError("packed tile statistics have inconsistent shapes")


def packed_tile_statistics(source: VirtualTracePacket) -> PackedTileStatistics:
    """Aggregate exact candidate, relation, and physical-packet counts by tile."""

    tiles = np.right_shift(
        np.asarray(source.point_keys, dtype=np.uint64), 32,
    ).astype(np.int64, copy=False)
    tile_count = source._tile_count()
    if bool((tiles < 0).any()) or bool((tiles >= tile_count).any()):
        raise ValueError("packed packet contains an out-of-domain tile")
    byte_masks = np.asarray(source.masks, dtype=np.dtype("<u4")).view(np.uint8)
    # Empty CUDA queries are valid (for example, a sparse voxel tile).  NumPy
    # cannot infer ``-1`` when the first dimension is zero, so derive the
    # fixed mask width from the template's local query domain.
    mask_bytes_per_candidate = source.local_query_count // 8
    byte_masks = byte_masks.reshape(
        source.candidate_count, mask_bytes_per_candidate,
    )
    active_lanes = _POPCOUNT8[byte_masks]
    candidate_counts = np.bincount(tiles, minlength=tile_count).astype(
        np.uint64, copy=False,
    )
    relation_by_candidate = active_lanes.sum(axis=1, dtype=np.uint64)
    logical_relation_counts = np.zeros(tile_count, dtype=np.uint64)
    np.add.at(logical_relation_counts, tiles, relation_by_candidate)
    packet_by_candidate = np.count_nonzero(active_lanes, axis=1).astype(
        np.uint64, copy=False,
    )
    physical_packet_counts = np.zeros(tile_count, dtype=np.uint64)
    np.add.at(physical_packet_counts, tiles, packet_by_candidate)
    lane_histograms = np.zeros((tile_count, 9), dtype=np.uint64)
    for lane_count in range(1, 9):
        per_candidate = np.count_nonzero(
            active_lanes == lane_count, axis=1,
        ).astype(np.uint64, copy=False)
        np.add.at(lane_histograms[:, lane_count], tiles, per_candidate)
    return PackedTileStatistics(
        tile_ids=np.arange(tile_count, dtype=np.int64),
        candidate_counts=candidate_counts,
        logical_relation_counts=logical_relation_counts,
        physical_packet_counts=physical_packet_counts,
        lane_histograms=lane_histograms,
    )


def iter_packed_relation_packet_batches(
    source: VirtualTracePacket,
    *,
    query_lanes: int,
    max_packets: int,
) -> Iterator[PackedRelationPacketBatch]:
    """Decode exact physical packets without materializing logical relations."""

    if query_lanes != 8:
        raise ValueError("packed RelationPacket decoding currently requires eight lanes")
    if max_packets <= 0:
        raise ValueError("packed RelationPacket batch size must be positive")
    if source.local_query_count % query_lanes:
        raise ValueError("local query domain is not divisible by RelationPacket lanes")
    byte_masks = np.asarray(source.masks, dtype=np.dtype("<u4")).view(np.uint8)
    byte_masks = byte_masks.reshape(
        source.candidate_count, source.local_query_count // query_lanes,
    )
    tiles = np.right_shift(
        np.asarray(source.point_keys, dtype=np.uint64), 32,
    ).astype(np.int64, copy=False)
    order = np.argsort(tiles, kind="stable")
    sorted_tiles = tiles[order]
    tile_ids = np.arange(source._tile_count(), dtype=np.int64)
    starts = np.searchsorted(sorted_tiles, tile_ids, side="left")
    ends = np.searchsorted(sorted_tiles, tile_ids, side="right")
    release_by_query = source._consumer_ready_pack_indices(
        np.arange(source.query_count, dtype=np.int64),
        query_lanes=query_lanes,
    )
    buffered: list[np.ndarray] = []
    buffered_count = 0
    packet_ordinal = 0
    for query_pack, query_offset, candidates, masks in _iter_packet_masks(
        source, byte_masks, order, starts, ends,
    ):
        first_lanes = _FIRST_LANE8[masks]
        local_order = np.argsort(first_lanes, kind="stable")
        candidates = candidates[local_order].astype(np.int64, copy=False)
        masks = masks[local_order].astype(np.uint8, copy=False)
        release_packs = np.full(masks.size, -1, dtype=np.int64)
        for lane in range(query_lanes):
            target = query_offset + lane
            lane_release = (
                int(release_by_query[target])
                if target < source.query_count
                and target // int(source.query_shape[-1])
                == query_offset // int(source.query_shape[-1])
                else -1
            )
            if lane_release >= 0:
                release_packs = np.where(
                    masks & np.uint8(1 << lane),
                    np.maximum(release_packs, lane_release),
                    release_packs,
                )
        if bool((release_packs < 0).any()):
            raise ValueError("physical RelationPacket has no active lane")
        records = np.empty(candidates.size, dtype=PACKED_RELATION_PACKET_DTYPE)
        records["packet_ordinal"] = np.arange(
            packet_ordinal, packet_ordinal + candidates.size, dtype=np.uint64,
        )
        packet_ordinal += int(candidates.size)
        records["query_pack"] = query_pack
        records["candidate_index"] = candidates
        records["query_base"] = source.query_base + query_offset
        records["gaussian_id"] = source.point_ids[candidates]
        records["point_key"] = source.point_keys[candidates]
        records["consumer_release_pack"] = release_packs
        records["lane_mask"] = masks
        records["active_lanes"] = _POPCOUNT8[masks]
        record_offset = 0
        while record_offset < records.size:
            take = min(max_packets - buffered_count, records.size - record_offset)
            buffered.append(records[record_offset:record_offset + take])
            buffered_count += take
            record_offset += take
            if buffered_count == max_packets:
                yield PackedRelationPacketBatch(np.concatenate(buffered))
                buffered.clear()
                buffered_count = 0
    if buffered_count:
        yield PackedRelationPacketBatch(np.concatenate(buffered))


def _iter_packet_masks(
    source: VirtualTracePacket,
    byte_masks: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    lanes = 8
    if source.template_id == RASTER_TEMPLATE_ID:
        height, width = source.query_shape
        blocks_x = (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
        local_packs_per_row = RASTER_BLOCK[1] // lanes
        fast_packs = (width + lanes - 1) // lanes
        for tile_y in range((height + RASTER_BLOCK[0] - 1) // RASTER_BLOCK[0]):
            valid_rows = min(
                RASTER_BLOCK[0], height - tile_y * RASTER_BLOCK[0],
            )
            for local_y in range(valid_rows):
                query_y = tile_y * RASTER_BLOCK[0] + local_y
                for tile_x in range(blocks_x):
                    tile = tile_y * blocks_x + tile_x
                    tile_candidates = order[starts[tile]:ends[tile]]
                    if tile_candidates.size == 0:
                        continue
                    valid_columns = min(
                        RASTER_BLOCK[1], width - tile_x * RASTER_BLOCK[1],
                    )
                    for local_fast_pack in range(
                        (valid_columns + lanes - 1) // lanes
                    ):
                        local_pack = (
                            local_y * local_packs_per_row + local_fast_pack
                        )
                        masks = byte_masks[tile_candidates, local_pack]
                        active = masks != 0
                        if not bool(active.any()):
                            continue
                        query_x = (
                            tile_x * RASTER_BLOCK[1] + local_fast_pack * lanes
                        )
                        yield (
                            query_y * fast_packs + query_x // lanes,
                            query_y * width + query_x,
                            tile_candidates[active],
                            masks[active],
                        )
        return
    if source.template_id == VOXEL_TEMPLATE_ID:
        voxel_x, voxel_y, voxel_z = source.query_shape
        blocks_x = (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
        blocks_y = (voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1]
        blocks_z = (voxel_z + VOXEL_BLOCK[2] - 1) // VOXEL_BLOCK[2]
        fast_packs = (voxel_z + lanes - 1) // lanes
        for tile_x in range(blocks_x):
            valid_x = min(VOXEL_BLOCK[0], voxel_x - tile_x * VOXEL_BLOCK[0])
            for local_x in range(valid_x):
                query_x = tile_x * VOXEL_BLOCK[0] + local_x
                for tile_y in range(blocks_y):
                    valid_y = min(
                        VOXEL_BLOCK[1], voxel_y - tile_y * VOXEL_BLOCK[1],
                    )
                    for local_y in range(valid_y):
                        query_y = tile_y * VOXEL_BLOCK[1] + local_y
                        local_pack = local_x * VOXEL_BLOCK[1] + local_y
                        for tile_z in range(blocks_z):
                            tile = (
                                tile_z * blocks_x * blocks_y
                                + tile_y * blocks_x + tile_x
                            )
                            tile_candidates = order[starts[tile]:ends[tile]]
                            if tile_candidates.size == 0:
                                continue
                            masks = byte_masks[tile_candidates, local_pack]
                            active = masks != 0
                            if not bool(active.any()):
                                continue
                            query_z = tile_z * VOXEL_BLOCK[2]
                            yield (
                                (query_x * voxel_y + query_y) * fast_packs
                                + query_z // lanes,
                                query_x * voxel_y * voxel_z
                                + query_y * voxel_z + query_z,
                                tile_candidates[active],
                                masks[active],
                            )
        return
    raise ValueError(f"unsupported virtual trace template: {source.template_id}")
