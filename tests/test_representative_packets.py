from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.tools.representative_packets import (
    build_representative_packet_trace, plan_representative_packet_groups,
)
from gala_sim.trace import (
    VirtualPacketArchiveReader, VirtualPacketArchiveWriter, VirtualTracePacket,
    snapshot_live_archive_prefix, validate_trace,
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


def test_archive_descriptor_filter_opens_only_selected_chunks(
    monkeypatch, tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    reader = VirtualPacketArchiveReader(archive)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    selected_chunks = {
        int(item["chunk"])
        for item in map(
            json.loads,
            (archive / "stream.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if item.get("type") == "packet"
    }
    opened: list[Path] = []
    original_load = np.load

    def tracked_load(path, *args, **kwargs):
        opened.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    descriptors = tuple(reader.packet_descriptors(iterations={3}))

    assert [(item.iteration_id, item.template_id) for item in descriptors] == [
        (3, 1), (3, 2),
    ]
    opened_chunks = {
        int(path.stem.removeprefix("chunk-")) for path in opened
    }
    assert opened_chunks < selected_chunks


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


def test_representative_plan_selects_one_closed_iteration_without_campaign(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)

    report = plan_representative_packet_groups(
        archive, None, expected_group_count=2, single_iteration=1,
    )

    assert report["single_iteration"] is True
    assert report["campaign_complete"] is False
    assert report["iterations"] == [1]
    assert [group["iterations"] for group in report["groups"]] == [[1], [1]]
    assert [group["template_id"] for group in report["groups"]] == [1, 2]
    assert all(
        group["selection"] == "nonempty_tile_nearest_median_physical_packets"
        for group in report["groups"]
    )


def test_representative_plan_can_select_only_closed_live_prefix_windows(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    snapshot = tmp_path / "snapshot"
    snapshot_live_archive_prefix(
        archive,
        snapshot,
        initial_gaussian_count=4,
        through_iteration=3,
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "representative_iterations:\n"
        "  - {iteration: 3, roles: [densification]}\n"
        "  - {iteration: 6, roles: [post_densification]}\n",
        encoding="utf-8",
    )

    report = plan_representative_packet_groups(
        snapshot,
        campaign,
        expected_group_count=2,
        live_prefix=True,
    )

    assert report["groups"][0]["iterations"] == [2, 3]
    assert report["live_prefix"] is True
    assert report["campaign_complete"] is False


def test_representative_packet_trace_expands_one_complete_window(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "representative_iterations:\n"
        "  - {iteration: 3, roles: [densification]}\n"
        "  - {iteration: 6, roles: [post_densification]}\n",
        encoding="utf-8",
    )
    plan = plan_representative_packet_groups(
        archive, campaign, expected_group_count=4,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    trace = build_representative_packet_trace(
        archive,
        plan_path,
        window_index=1,
        max_events=128,
        query_lanes=8,
        model_id="fixture_model",
        dataset_id="fixture_dataset",
    )

    validate_trace(trace)
    assert trace.metadata["trace_sample"]["iterations"] == [5, 6]
    assert len(trace.metadata["trace_sample"]["packets"]) == 4
    assert [
        packet["query_count"]
        for packet in trace.metadata["trace_sample"]["packets"]
    ] == [256, 512, 256, 512]
    assert set(trace.events["iteration_id"].tolist()) == {5, 6}
    update_end = np.flatnonzero(
        trace.events["primitive_kind"] == int(PrimitiveKind.UPDATE_END)
    )
    assert update_end.size == 1
    next_candidates = trace.events[
        (trace.events["iteration_id"] == 6)
        & (trace.events["primitive_kind"] == int(PrimitiveKind.RELATION_CANDIDATE))
    ]
    assert all(
        trace.dependency_ids(row).tolist() == update_end.tolist()
        for row in next_candidates
    )


def test_representative_packet_trace_preserves_single_iteration_update(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    _archive(archive)
    plan = plan_representative_packet_groups(
        archive, None, expected_group_count=2, single_iteration=1,
    )
    plan_path = tmp_path / "single-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    trace = build_representative_packet_trace(
        archive, plan_path, window_index=0, max_events=128, query_lanes=8,
        model_id="fixture_model", dataset_id="fixture_dataset",
    )

    validate_trace(trace)
    assert trace.metadata["trace_sample"]["iterations"] == [1]
    assert trace.metadata["model_id"] == "fixture_model"
    assert trace.metadata["dataset_id"] == "fixture_dataset"
    assert (
        trace.metadata["trace_sample"]["selection"]
        == "single_iteration_median_physical_tile"
    )
    update_begin = np.flatnonzero(
        trace.events["primitive_kind"] == int(PrimitiveKind.UPDATE_BEGIN)
    )
    update_end = np.flatnonzero(
        trace.events["primitive_kind"] == int(PrimitiveKind.UPDATE_END)
    )
    assert update_begin.size == update_end.size == 1
    assert trace.dependency_ids(trace.events[update_end[0]]).tolist() == [
        int(trace.events[update_begin[0]]["event_id"]),
    ]
