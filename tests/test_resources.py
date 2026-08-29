from __future__ import annotations

from pathlib import Path

import pytest

from gala_sim.timing import CycleConfig, ModuleTiming, ResourceEnvelope, ResourceUsage
from gala_sim.config import load_config


ROOT = Path(__file__).parents[1]


def test_resource_envelope_accepts_only_frozen_topology() -> None:
    envelope = ResourceEnvelope(
        shared_sram_bytes=2_883_584, pods=4, clusters_per_pod=5,
        fma_lanes=320, transcendental_lanes=40, external_channels=8,
    )
    usage = ResourceUsage(
        shared_sram_bytes=2_800_000, pods=4, clusters=20, fma_lanes=320,
        transcendental_lanes=40, external_channels=8, regions={"cache": 1},
    )
    envelope.check(usage)
    with pytest.raises(ValueError, match="SRAM"):
        envelope.check(ResourceUsage(
            shared_sram_bytes=2_883_585, pods=4, clusters=20, fma_lanes=320,
            transcendental_lanes=40, external_channels=8, regions={},
        ))


def test_resource_envelope_is_derived_from_registered_architecture_values() -> None:
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    envelope = ResourceEnvelope.from_gala(config)
    assert envelope.shared_sram_bytes == 2_883_584
    assert envelope.pods == 4
    assert envelope.clusters_per_pod == 5
    assert envelope.fma_lanes == 320
    assert envelope.transcendental_lanes == 40
    assert envelope.external_channels == 8


def test_production_resource_usage_closes_all_six_shared_sram_regions() -> None:
    config = CycleConfig.from_gala(
        load_config(ROOT / "configs/architecture/gala.yaml"), _Memory()
    )

    assert config.resource_usage is not None
    assert config.resource_usage.shared_sram_bytes == 2_883_584
    assert config.resource_usage.regions == {
        "active_gaussian": 512 * 1024,
        "relation_window": 640 * 1024,
        "query_volume": 512 * 1024,
        "gradient_update": 640 * 1024,
        "index_graph": 256 * 1024,
        "control_metadata": 256 * 1024,
    }


class _Memory:
    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        return arrival_cycle


def test_cycle_config_rejects_usage_outside_resource_envelope() -> None:
    timing = ModuleTiming(latency=1, initiation_interval=1, queue_capacity=1, ports=1, banks=1)
    modules = {name: timing for name in (
        "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
        "bidirectional_query", "reconstruction_update", "shared_sram",
    )}
    envelope = ResourceEnvelope(
        shared_sram_bytes=1024, pods=1, clusters_per_pod=1,
        fma_lanes=1, transcendental_lanes=1, external_channels=1,
    )
    usage = ResourceUsage(
        shared_sram_bytes=1025, pods=1, clusters=1, fma_lanes=1,
        transcendental_lanes=1, external_channels=1, regions={},
    )
    with pytest.raises(ValueError, match="shared SRAM"):
        CycleConfig(
            modules=modules, memory=_Memory(), clock_frequency_hz=1,
            relation_seed_fifo_entries=1, candidate_lanes=1,
            resource_envelope=envelope, resource_usage=usage,
        )
