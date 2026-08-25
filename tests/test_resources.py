from __future__ import annotations

import pytest

from gala_sim.timing import ResourceEnvelope, ResourceUsage


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
