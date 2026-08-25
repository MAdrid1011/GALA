"""Shared SRAM and execution-resource envelope checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceUsage:
    shared_sram_bytes: int
    pods: int
    clusters: int
    fma_lanes: int
    transcendental_lanes: int
    external_channels: int
    regions: dict[str, int]

    def __post_init__(self) -> None:
        if min(self.shared_sram_bytes, self.pods, self.clusters, self.fma_lanes,
               self.transcendental_lanes, self.external_channels) <= 0:
            raise ValueError("resource usage values must be positive")


@dataclass(frozen=True)
class ResourceEnvelope:
    shared_sram_bytes: int
    pods: int
    clusters_per_pod: int
    fma_lanes: int
    transcendental_lanes: int
    external_channels: int

    def check(self, usage: ResourceUsage) -> None:
        if usage.shared_sram_bytes > self.shared_sram_bytes:
            raise ValueError("shared SRAM resource envelope exceeded")
        if usage.pods != self.pods or usage.clusters != self.pods * self.clusters_per_pod:
            raise ValueError("compute Pod resource envelope changed")
        if usage.fma_lanes != self.fma_lanes:
            raise ValueError("FMA resource envelope changed")
        if usage.transcendental_lanes != self.transcendental_lanes:
            raise ValueError("transcendental resource envelope changed")
        if usage.external_channels != self.external_channels:
            raise ValueError("external memory channel envelope changed")
        if any(value < 0 for value in usage.regions.values()):
            raise ValueError("resource region has negative bytes")
