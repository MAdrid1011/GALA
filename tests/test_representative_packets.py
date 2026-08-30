from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gala_sim.tools.representative_packets import plan_representative_packet_groups
from gala_sim.trace import (
    VirtualPacketArchiveReader, VirtualPacketArchiveWriter, VirtualTracePacket,
)


def _source(
    *, iteration: int, template_id: int, query_base: int,
) -> VirtualTracePacket:
    if template_id == 1:
        query_shape = (16, 32)
        words = 8
        loss_flags = 2
    else:
        query_shape = (8, 8, 16)
        words = 16
        loss_flags = 4
    masks = np.zeros((4, words), dtype=np.dtype("<u4"))
    masks[0, 0] = np.uint32(0b11111111)
    masks[1, 0] = np.uint32(0b00001111)
    masks[2, 0] = np.uint32(0b00000111)
    masks[3, 0] = np.uint32(0b00000001)
    return VirtualTracePacket(
        iteration_id=iteration,
        template_id=template_id,
        query_base=query_base,
        query_shape=query_shape,
        point_ids=np.arange(4, dtype=np.int64),
        point_keys=(
            np.asarray([0, 0, 1, 1], dtype=np.uint64) << np.uint64(32)
        ) | np.arange(4, dtype=np.uint64),
        masks=masks,
        loss_flags=loss_flags,
        backward_confirmed=True,
    )


def _archive(root: Path) -> None:
    writer = VirtualPacketArchiveWriter(root, max_chunk_bytes=1024)
    writer.initialize_gaussians(4)
    query_base = 0
    for iteration in range(1, 7):
        for template_id in (1, 2):
            source = _source(
                iteration=iteration,
                template_id=template_id,
                query_base=query_base,
            )
            writer.append_packet(source)
            query_base += source.query_count
        writer.close_iteration(iteration)
    writer.finish()


def test_representative_plan_uses_adjacent_phase_windows_and_common_tiles(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "representative_iterations:\n"
        "  - {iteration: 1, roles: [initial]}\n"
        "  - {iteration: 3, roles: [densification]}\n"
        "  - {iteration: 6, roles: [post_densification]}\n",
        encoding="utf-8",
    )

    report = plan_representative_packet_groups(
        archive, campaign, expected_group_count=4,
    )

    assert report["group_count"] == 4
    assert report["iteration_count"] == 4
    assert report["iterations"] == [2, 3, 5, 6]
    assert [group["iterations"] for group in report["groups"]] == [
        [2, 3], [2, 3], [5, 6], [5, 6],
    ]
    assert [group["template_id"] for group in report["groups"]] == [1, 2, 1, 2]
    assert [group["tile_id"] for group in report["groups"]] == [0, 0, 0, 0]
    assert all(
        sum(group["packets"][0]["lane_histogram"][1:])
        == group["packets"][0]["physical_packet_count"]
        for group in report["groups"]
    )


def test_archive_descriptors_filter_without_materializing_unselected_packets(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    reader = VirtualPacketArchiveReader(archive)
    descriptors = tuple(reader.packet_descriptors(iterations={3, 5}))

    assert [(item.iteration_id, item.template_id) for item in descriptors] == [
        (3, 1), (3, 2), (5, 1), (5, 2),
    ]
    assert reader.packet(descriptors[0]).iteration_id == 3


def test_representative_plan_enforces_requested_group_count(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "representative_iterations:\n"
        "  - {iteration: 3, roles: [densification]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="produced 2 groups, expected 3"):
        plan_representative_packet_groups(
            archive, campaign, expected_group_count=3,
        )
