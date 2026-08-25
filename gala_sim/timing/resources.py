"""Shared SRAM and execution-resource envelope checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gala_sim.config import GalaConfig


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

    @classmethod
    def from_gala(cls, config: "GalaConfig") -> "ResourceEnvelope":
        """Build the immutable top-level envelope from registered parameters."""

        try:
            pods = int(config.value("top.num_pods"))
            clusters_per_pod = int(config.value("compute.clusters_per_pod"))
            fma_lanes_per_cluster = int(config.value("compute.fma_lanes_per_cluster"))
            transcendental_lanes_per_cluster = int(
                config.value("compute.transcendental_lanes_per_cluster")
            )
            shared_sram_bytes = int(config.value("top.shared_sram_bytes"))
            external_channels = int(config.value("memory.channels"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("registered resource envelope is incomplete") from error
        return cls(
            shared_sram_bytes=shared_sram_bytes,
            pods=pods,
            clusters_per_pod=clusters_per_pod,
            fma_lanes=pods * clusters_per_pod * fma_lanes_per_cluster,
            transcendental_lanes=pods * clusters_per_pod * transcendental_lanes_per_cluster,
            external_channels=external_channels,
        )

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
