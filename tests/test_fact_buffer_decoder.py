from __future__ import annotations

import numpy as np
import pytest

from gala_sim.adapters.fact_buffer_decoder import (
    decode_fact_raster_records,
    decode_fact_raster_virtual_packet,
    decode_fact_voxel_records,
    decode_fact_voxel_virtual_packet,
)
from gala_sim.adapters.trace_capture import (
    RASTER_TEMPLATE_ID,
    STATE_FIELD_MASK,
    VOXEL_TEMPLATE_ID,
)
from gala_sim.timing import CycleConfig, CycleEngine, ModuleTiming


torch = pytest.importorskip("torch")


class _Memory:
    def submit(
        self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int,
    ) -> int:
        return arrival_cycle + size_bytes // 64 + int(is_write)


def _cycle_config() -> CycleConfig:
    timing = ModuleTiming(
        latency=2, initiation_interval=1, queue_capacity=1024, ports=4, banks=8,
    )
    return CycleConfig(
        modules={
            name: timing for name in (
                "relation_constructor", "fusion_issue", "semantic_cache",
                "compute_pod", "bidirectional_query", "reconstruction_update",
                "shared_sram",
            )
        },
        memory=_Memory(), clock_frequency_hz=500_000_000,
        relation_seed_fifo_entries=1024, candidate_lanes=8,
    )


def test_fact_raster_decoder_preserves_nonempty_tile_ids_and_chunk_indexes() -> None:
    gaussian_ids = torch.tensor([1, 0], dtype=torch.int32)
    tile_bins = torch.tensor(
        [[0, 1], [1, 2], [0, 0], [0, 0]], dtype=torch.int32,
    )
    pos2d = torch.tensor([[16.0, 0.0], [0.0, 0.0]])
    conics = torch.tensor([[100.0, 0.0, 100.0, 1.0]] * 2)
    intensities = torch.ones((2, 1))

    records = decode_fact_raster_records(
        gaussian_ids, tile_bins, pos2d, conics, intensities, (17, 17), 1, 1,
    ).numpy()

    assert records.tolist() == [
        [0, 0, 0, 1 << 32],
        [1, 0, 0, 0],
    ]


def test_fact_raster_virtual_packet_matches_exact_relations() -> None:
    gaussian_ids = torch.tensor([1, 0], dtype=torch.int32)
    tile_bins = torch.tensor(
        [[0, 1], [1, 2], [0, 0], [0, 0]], dtype=torch.int32,
    )
    pos2d = torch.tensor([[16.0, 0.0], [0.0, 0.0]])
    conics = torch.tensor([[100.0, 0.0, 100.0, 1.0]] * 2)
    intensities = torch.ones((2, 1))

    packet = decode_fact_raster_virtual_packet(
        gaussian_ids, tile_bins, pos2d, conics, intensities, (17, 17),
        iteration_id=3, template_id=RASTER_TEMPLATE_ID, query_base=10,
        state_version=2, field_mask=STATE_FIELD_MASK,
    )

    assert packet.point_ids.tolist() == [1, 0]
    assert packet.point_keys.tolist() == [0, 1 << 32]
    assert packet.logical_relation_count == 2
    assert list(packet.iter_relations()) == [
        (0, 10, 1, 0),
        (1, 26, 0, 1 << 32),
    ]


def test_fact_raster_packet_replays_end_to_end_in_cycle_engine(
    tmp_path,
) -> None:
    packet = decode_fact_raster_virtual_packet(
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([[0, 1], [1, 2], [0, 0], [0, 0]], dtype=torch.int32),
        torch.tensor([[16.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[100.0, 0.0, 100.0, 1.0]] * 2),
        torch.ones((2, 1)),
        (17, 17),
        iteration_id=1,
        template_id=RASTER_TEMPLATE_ID,
        query_base=0,
        state_version=0,
        field_mask=STATE_FIELD_MASK,
    ).with_capture_fields(loss_flags=1, backward_confirmed=True)

    result = CycleEngine(_cycle_config()).run_virtual(
        [packet], trace_root=tmp_path / "trace",
        max_events=128, max_total_events=2048,
    )

    assert result.total_cycles > 0
    assert result.event_counts["RELATION"] == packet.logical_relation_count
    assert result.event_counts["FORWARD"] == packet.logical_relation_count
    assert result.event_counts["ADJOINT"] == packet.logical_relation_count
    assert result.event_counts["QUERY_CLOSE"] == packet.query_count


def test_fact_voxel_decoder_reorders_cuda_mask_bits_for_virtual_packets() -> None:
    gaussian_ids = torch.tensor([0], dtype=torch.int32)
    tile_bins = torch.tensor([[0, 1]], dtype=torch.int32)
    pos3d_radii = torch.tensor([[1.5, 0.5, 0.5, 1.0]])
    conics = torch.tensor([[100.0, 0.0, 0.0, 100.0, 0.0, 100.0]])
    intensities = torch.ones((1, 1))

    records = decode_fact_voxel_records(
        gaussian_ids, tile_bins, pos3d_radii, conics, intensities,
        (8, 1, 2), 0, 1,
    ).numpy()
    packet = decode_fact_voxel_virtual_packet(
        gaussian_ids, tile_bins, pos3d_radii, conics, intensities, (8, 1, 2),
        iteration_id=4, template_id=VOXEL_TEMPLATE_ID, query_base=20,
        state_version=1, field_mask=STATE_FIELD_MASK,
    )

    assert records[:, :3].tolist() == [[0, 0, 0], [1, 0, 1]]
    assert packet.masks[0, 2] == np.uint32(1)
    assert packet.logical_relation_count == 1
    assert list(packet.iter_relations()) == [(0, 22, 0, 0)]


def test_fact_decoder_rejects_gapped_tile_bins() -> None:
    with pytest.raises(ValueError, match="gap"):
        decode_fact_raster_records(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
            torch.ones((1, 1)),
            (16, 16), 0, 1,
        )
