"""Explicit timing inputs; no hardware latency is hidden in module code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gala_sim.config import GalaConfig

from .resources import ResourceEnvelope, ResourceUsage


class MemoryBackend(Protocol):
    def submit(self, *, address: int, size_bytes: int, is_write: bool, arrival_cycle: int) -> int:
        """Return the backend-provided completion cycle for one request."""


class AsyncMemoryBackend(Protocol):
    def submit_async(self, *, address: int, size_bytes: int, is_write: bool,
                     arrival_cycle: int) -> int: ...

    def advance(self, cycle: int) -> None: ...

    def next_wakeup(self) -> int | None: ...

    def pop_completions(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class ModuleTiming:
    latency: int
    initiation_interval: int
    queue_capacity: int
    ports: int
    banks: int

    def __post_init__(self) -> None:
        if min(self.latency, self.initiation_interval, self.queue_capacity, self.ports, self.banks) <= 0:
            raise ValueError("module timing values must be positive")


@dataclass(frozen=True)
class CycleConfig:
    modules: dict[str, ModuleTiming]
    memory: MemoryBackend | AsyncMemoryBackend
    clock_frequency_hz: int
    relation_seed_fifo_entries: int
    candidate_lanes: int
    cache_instances: int | None = None
    cache_capacity_per_instance: int | None = None
    cache_directory_banks: int | None = None
    cache_sector_bytes: int | None = None
    cache_multicast_destinations: int | None = None
    fusion_forward_ports: int | None = None
    fusion_consumer_ports: int | None = None
    fusion_adjoint_ports: int | None = None
    resource_envelope: ResourceEnvelope | None = None
    resource_usage: ResourceUsage | None = None

    def __post_init__(self) -> None:
        if min(self.clock_frequency_hz, self.relation_seed_fifo_entries, self.candidate_lanes) <= 0:
            raise ValueError("cycle clock, seed FIFO, and candidate lanes must be positive")
        optional_cache_values = (
            self.cache_instances, self.cache_capacity_per_instance,
            self.cache_directory_banks, self.cache_sector_bytes,
            self.cache_multicast_destinations,
        )
        if any(value is not None and value <= 0 for value in optional_cache_values):
            raise ValueError("optional cache timing values must be positive")
        optional_fusion_values = (
            self.fusion_forward_ports, self.fusion_consumer_ports, self.fusion_adjoint_ports,
        )
        if any(value is not None and value <= 0 for value in optional_fusion_values):
            raise ValueError("optional fusion port values must be positive")
        if (self.resource_envelope is None) != (self.resource_usage is None):
            raise ValueError("resource envelope and usage must be provided together")
        if self.resource_envelope is not None and self.resource_usage is not None:
            self.resource_envelope.check(self.resource_usage)
        required = {
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        }
        missing = required.difference(self.modules)
        if missing:
            raise ValueError(f"cycle configuration lacks modules: {sorted(missing)}")

    @classmethod
    def from_gala(cls, config: GalaConfig,
                  memory: MemoryBackend | AsyncMemoryBackend,
                  resource_usage: ResourceUsage | None = None) -> "CycleConfig":
        """Build timing inputs only when every latency is present in GalaConfig."""

        module_names = (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )
        modules: dict[str, ModuleTiming] = {}
        missing: list[str] = []
        for name in module_names:
            try:
                modules[name] = ModuleTiming(
                    latency=int(config.value(f"latency.{name}.latency")),
                    initiation_interval=int(config.value(f"latency.{name}.initiation_interval")),
                    queue_capacity=int(config.value(f"latency.{name}.queue_capacity")),
                    ports=int(config.value(f"latency.{name}.ports")),
                    banks=int(config.value(f"latency.{name}.banks")),
                )
            except (KeyError, TypeError, ValueError):
                missing.append(name)
        try:
            frequency = int(config.value("clock.frequency"))
            seed_fifo = int(config.value("relation.seed_fifo_entries"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cycle configuration lacks clock or seed FIFO") from error
        if missing:
            raise ValueError("cycle latency configuration is incomplete: " + ", ".join(missing))
        if not config.ready:
            config.require_ready()
        return cls(modules=modules, memory=memory, clock_frequency_hz=frequency,
                   relation_seed_fifo_entries=seed_fifo,
                   candidate_lanes=int(config.value("issue.candidate_lanes")),
                   cache_instances=int(config.value("cache.instances")),
                   cache_capacity_per_instance=int(config.value("cache.active_records_per_instance")),
                   cache_directory_banks=int(config.value("cache.directory_banks_per_instance")),
                   cache_sector_bytes=int(config.value("cache.sector_bytes")),
                   cache_multicast_destinations=int(config.value("cache.multicast_destinations")),
                   fusion_forward_ports=int(config.value("issue.forward_ports")),
                   fusion_consumer_ports=int(config.value("issue.consumer_ports")),
                   fusion_adjoint_ports=int(config.value("issue.adjoint_ports")),
                   resource_envelope=ResourceEnvelope.from_gala(config),
                   resource_usage=resource_usage or _resource_usage_from_gala(config))


def _resource_usage_from_gala(config: GalaConfig) -> ResourceUsage:
    """Derive fixed execution resources and registered SRAM regions."""

    try:
        pods = int(config.value("top.num_pods"))
        clusters_per_pod = int(config.value("compute.clusters_per_pod"))
        clusters = pods * clusters_per_pod
        active_sram = pods * int(config.value("cache.active_sram_bytes_per_instance"))
        microcontexts = clusters * int(config.value("compute.microcontext_bytes_per_cluster"))
        regions = {
            "semantic_cache_active": active_sram,
            "compute_microcontexts": microcontexts,
        }
        return ResourceUsage(
            shared_sram_bytes=sum(regions.values()),
            pods=pods,
            clusters=clusters,
            fma_lanes=clusters * int(config.value("compute.fma_lanes_per_cluster")),
            transcendental_lanes=(
                clusters * int(config.value("compute.transcendental_lanes_per_cluster"))
            ),
            external_channels=int(config.value("memory.channels")),
            regions=regions,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("registered resource usage is incomplete") from error
