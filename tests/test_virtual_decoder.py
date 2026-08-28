from __future__ import annotations

import numpy as np

from gala_sim.adapters.buffer_decoder import (
    decode_raster_virtual_packet,
    decode_voxel_virtual_packet,
)


class _Tensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value

    def detach(self) -> "_Tensor":
        return self


class _Decoder:
    def copy_raster_point_list(self, _buffer, count: int) -> _Tensor:
        return _Tensor(np.arange(count, dtype=np.int32))

    def copy_raster_point_keys(self, _buffer, count: int) -> _Tensor:
        return _Tensor(np.zeros(count, dtype=np.uint64))

    def copy_voxel_point_list(self, _buffer, count: int) -> _Tensor:
        return _Tensor(np.arange(count, dtype=np.int32))

    def copy_voxel_point_keys(self, _buffer, count: int) -> _Tensor:
        return _Tensor(np.zeros(count, dtype=np.uint64))

    def raster_valid_masks(self, *_args) -> _Tensor:
        masks = np.zeros((2, 8), dtype=np.int32)
        masks[0, 0] = 1
        return _Tensor(masks)

    def voxel_valid_masks(self, *_args) -> _Tensor:
        masks = np.zeros((1, 16), dtype=np.int32)
        masks[0, 0] = 1
        return _Tensor(masks)


class _VoxelKeyDecoder(_Decoder):
    def copy_voxel_point_keys(self, _buffer, count: int) -> _Tensor:
        return _Tensor(np.full(count, np.uint64(1) << np.uint64(32), dtype=np.uint64))


def test_raster_decoder_builds_packet_without_expanding_relations() -> None:
    packet = decode_raster_virtual_packet(
        _Decoder(), object(), object(), 2, 2, 1, 1,
        iteration_id=4, query_base=8,
    )
    assert packet.candidate_count == 2
    assert packet.logical_relation_count == 1
    assert packet.materialize_relations().tolist() == [[0, 8, 0, 0]]


def test_voxel_decoder_builds_packet_with_expected_mask_width() -> None:
    packet = decode_voxel_virtual_packet(
        _Decoder(), object(), object(), 1, 1, 1, 1, 1,
        iteration_id=4, query_base=8,
    )
    assert packet.masks.shape == (1, 16)
    assert list(packet.iter_relations()) == [(0, 8, 0, 0)]


def test_voxel_decoder_preserves_kernel_point_key_tile() -> None:
    packet = decode_voxel_virtual_packet(
        _VoxelKeyDecoder(), object(), object(), 1, 1, 9, 8, 8,
        iteration_id=4, query_base=0,
    )
    assert list(packet.iter_relations()) == [(0, 512, 0, 1 << 32)]
