"""Explicit timing inputs; no hardware latency is hidden in module code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

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
class ComputeStage:
    """One serialized stage in a ComputePod execution template.

    Resource demands are expressed per cluster.  ``latency`` is the stage's
    result latency, while the issue interval remains the enclosing module's
    configured initiation interval.  Keeping the sequence in configuration
    makes template-specific work auditable and prevents a hidden average
    compute latency in the module implementation.
    """

    name: str
    latency: int
    fma_groups: int = 0
    transcendental_lanes: int = 0
    reduction_trees: int = 0
    register_reads: int = 0
    register_writes: int = 0
    feedback_lanes: int = 0

    def __post_init__(self) -> None:
        if self.name not in {"TRANSFORM", "EVALUATE", "COMBINE"}:
            raise ValueError(f"unsupported ComputePod stage: {self.name}")
        if self.latency <= 0:
            raise ValueError("ComputePod stage latency must be positive")
        if min(self.fma_groups, self.transcendental_lanes, self.reduction_trees,
               self.register_reads, self.register_writes, self.feedback_lanes) < 0:
            raise ValueError("ComputePod resource demands cannot be negative")


@dataclass(frozen=True)
class ComputePathProfile:
    """One template path and its relation-level cluster admission contract."""

    stages: tuple[ComputeStage, ...]
    cluster_issue_slots: int = 1
    cluster_issue_cycles: int = 1
    packet_first_result_latency: int | None = None
    packet_last_result_offset: int | None = None
    packet_lane_issue_interval: int = 1
    packet_lanes_per_issue: int = 1
    packet_completion_at_last: bool = False

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("ComputePod path has no stages")
        if not all(isinstance(stage, ComputeStage) for stage in self.stages):
            raise ValueError("ComputePod path contains an invalid stage")
        if min(self.cluster_issue_slots, self.cluster_issue_cycles) <= 0:
            raise ValueError("ComputePod cluster issue values must be positive")
        packet_values = (
            self.packet_first_result_latency,
            self.packet_last_result_offset,
        )
        if any(value is not None and value <= 0 for value in packet_values):
            raise ValueError("ComputePod packet result offsets must be positive")
        if min(self.packet_lane_issue_interval, self.packet_lanes_per_issue) <= 0:
            raise ValueError("ComputePod packet lane timing must be positive")
        if (
            self.packet_first_result_latency is not None
            and self.packet_last_result_offset is not None
            and self.packet_last_result_offset < self.packet_first_result_latency
        ):
            raise ValueError("ComputePod packet last result precedes its first result")

    @property
    def latency(self) -> int:
        return sum(stage.latency for stage in self.stages)

    def packet_completion_offset(self, lane: int) -> int:
        """Return one logical lane's completion offset in a physical pack."""

        if lane < 0:
            raise ValueError("ComputePod packet lane must be non-negative")
        first = self.packet_first_result_latency or self.latency
        last = self.packet_last_result_offset or first
        if self.packet_completion_at_last:
            return last
        offset = first + (
            lane // self.packet_lanes_per_issue
        ) * self.packet_lane_issue_interval
        if offset > last:
            raise ValueError("ComputePod lane completion exceeds packet last result")
        return offset

    def packet_issue_cycles(self, active_lanes: int) -> int:
        """Return cluster admission occupancy for one real physical packet."""

        if active_lanes <= 0:
            raise ValueError("ComputePod packet must contain an active lane")
        issue_groups = (
            active_lanes + self.packet_lanes_per_issue - 1
        ) // self.packet_lanes_per_issue
        cycles = issue_groups * self.packet_lane_issue_interval
        if cycles > self.cluster_issue_cycles:
            raise ValueError("ComputePod packet exceeds the configured full-pack issue window")
        return cycles


@dataclass(frozen=True)
class ComputeTemplateProfile:
    """Audited per-template resource sequences for one or more task paths."""

    template_id: int
    paths: Mapping[str, ComputePathProfile]

    def __post_init__(self) -> None:
        if self.template_id < 0:
            raise ValueError("ComputePod template ID cannot be negative")
        if not self.paths:
            raise ValueError("ComputePod template profile needs at least one path")
        for path, profile in self.paths.items():
            if not isinstance(path, str) or not path:
                raise ValueError("ComputePod path name must be non-empty")
            if not isinstance(profile, ComputePathProfile):
                raise ValueError(f"ComputePod path {path} contains an invalid stage")

    def path_for(self, path: str) -> ComputePathProfile:
        try:
            return self.paths[path]
        except KeyError as error:
            raise KeyError(
                f"ComputePod template {self.template_id} has no {path} path"
            ) from error

    def stages_for(self, path: str) -> tuple[ComputeStage, ...]:
        return self.path_for(path).stages

    def latency_for(self, path: str) -> int:
        return self.path_for(path).latency


@dataclass(frozen=True)
class CycleConfig:
    modules: dict[str, ModuleTiming]
    memory: MemoryBackend | AsyncMemoryBackend
    clock_frequency_hz: int
    relation_seed_fifo_entries: int
    candidate_lanes: int
    relation_support_lanes: int | None = None
    relation_query_lanes: int = 1
    shared_sram_read_ports_per_bank: int = 1
    shared_sram_write_ports_per_bank: int = 1
    query_state_entries: int | None = None
    fusion_query_state_banks: int | None = None
    candidate_fifo_entries: int | None = None
    fusion_bank_head_lookahead: bool | None = None
    fusion_bank_head_index_bytes: int | None = None
    fusion_semantic_bundle_index_bytes: int | None = None
    cache_instances: int | None = None
    cache_capacity_per_instance: int | None = None
    cache_directory_banks: int | None = None
    cache_sector_bytes: int | None = None
    cache_multicast_destinations: int | None = None
    cache_ready_head_index_bytes: int | None = None
    fusion_forward_ports: int | None = None
    fusion_consumer_ports: int | None = None
    fusion_adjoint_ports: int | None = None
    resource_envelope: ResourceEnvelope | None = None
    resource_usage: ResourceUsage | None = None
    config_sha256: str | None = None
    compute_templates: Mapping[int, ComputeTemplateProfile] | None = None
    compute_resource_capacities: Mapping[str, int] | None = None
    owner_gradient_slots_per_cluster: int | None = None
    compute_ready_head_index_bytes: int | None = None
    query_reduction_banks: int | None = None
    query_partial_sum_groups_per_bank: int | None = None
    query_loss_fma_lanes: int | None = None
    query_loss_queries_per_cycle: int | None = None
    query_adjoint_replay_lanes: int | None = None
    query_replay_queue_entries: int | None = None
    query_relation_window_entries: int | None = None
    query_relation_window_entry_bytes: int | None = None
    query_relation_store_banks: int | None = None
    query_relation_store_records: int | None = None
    query_relation_store_record_bytes: int | None = None
    query_relation_candidate_ordinal_bits: int | None = None
    trace_continuation_query_packs: int | None = None
    query_volume_banks: int | None = None
    query_volume_word_bytes: int | None = None
    query_volume_bank_mapping: str = "linear"
    query_volume_bank_xor_shift: int | None = None
    memory_peak_bandwidth_bytes_per_second: int | None = None

    def __post_init__(self) -> None:
        if min(
            self.clock_frequency_hz,
            self.relation_seed_fifo_entries,
            self.candidate_lanes,
            self.relation_query_lanes,
        ) <= 0:
            raise ValueError(
                "cycle clock, seed FIFO, candidate lanes, and relation query lanes "
                "must be positive"
            )
        if min(
            self.shared_sram_read_ports_per_bank,
            self.shared_sram_write_ports_per_bank,
        ) <= 0:
            raise ValueError("Shared SRAM read/write ports per bank must be positive")
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
        if any(
            value is not None and value <= 0
            for value in (
                self.query_state_entries, self.fusion_query_state_banks,
            )
        ):
            raise ValueError("query-state table capacity and banks must be positive")
        if self.relation_support_lanes is not None and self.relation_support_lanes <= 0:
            raise ValueError("relation support lane count must be positive")
        if self.candidate_fifo_entries is not None and self.candidate_fifo_entries <= 0:
            raise ValueError("candidate FIFO capacity must be positive")
        if (
            self.fusion_bank_head_index_bytes is not None
            and self.fusion_bank_head_index_bytes <= 0
        ):
            raise ValueError("Fusion Bank head index metadata must be positive")
        if (
            self.fusion_semantic_bundle_index_bytes is not None
            and self.fusion_semantic_bundle_index_bytes <= 0
        ):
            raise ValueError("Fusion semantic bundle index metadata must be positive")
        if (
            self.fusion_bank_head_lookahead is not None
            and not isinstance(self.fusion_bank_head_lookahead, bool)
        ):
            raise ValueError("Fusion Bank head lookahead must be boolean")
        if (self.fusion_bank_head_lookahead is True) != (
            self.fusion_bank_head_index_bytes is not None
        ):
            raise ValueError(
                "Fusion Bank head lookahead and metadata budget must be configured together"
            )
        if (
            self.cache_ready_head_index_bytes is not None
            and self.cache_ready_head_index_bytes <= 0
        ):
            raise ValueError("cache ready-head index metadata must be positive")
        if (self.resource_envelope is None) != (self.resource_usage is None):
            raise ValueError("resource envelope and usage must be provided together")
        if self.resource_envelope is not None and self.resource_usage is not None:
            self.resource_envelope.check(self.resource_usage)
            indexed_control_bytes = sum(
                value or 0 for value in (
                    self.fusion_bank_head_index_bytes,
                    self.fusion_semantic_bundle_index_bytes,
                    self.cache_ready_head_index_bytes,
                    self.compute_ready_head_index_bytes,
                )
            )
            if indexed_control_bytes > self.resource_usage.regions.get(
                "control_metadata", 0
            ):
                raise ValueError(
                    "ready-head indexes exceed control metadata region"
                )
        if self.compute_templates is not None:
            if not self.compute_templates:
                raise ValueError("compute template profiles cannot be empty")
            if any(int(key) != profile.template_id
                   for key, profile in self.compute_templates.items()):
                raise ValueError("compute template profile keys do not match IDs")
        if self.compute_resource_capacities is not None:
            if any(not isinstance(name, str) or not name or int(value) <= 0
                   for name, value in self.compute_resource_capacities.items()):
                raise ValueError("compute resource capacities must be positive")
        if (
            self.owner_gradient_slots_per_cluster is not None
            and self.owner_gradient_slots_per_cluster <= 0
        ):
            raise ValueError("owner-gradient slot capacity must be positive")
        if (
            self.compute_ready_head_index_bytes is not None
            and self.compute_ready_head_index_bytes <= 0
        ):
            raise ValueError("ComputePod ready-head index metadata must be positive")
        query_values = (
            self.query_reduction_banks,
            self.query_partial_sum_groups_per_bank,
            self.query_loss_fma_lanes,
            self.query_loss_queries_per_cycle,
            self.query_adjoint_replay_lanes,
            self.query_replay_queue_entries,
            self.query_relation_window_entries,
            self.query_relation_window_entry_bytes,
            self.query_relation_store_banks,
            self.query_relation_store_records,
            self.query_relation_store_record_bytes,
            self.query_relation_candidate_ordinal_bits,
            self.query_volume_banks,
            self.query_volume_word_bytes,
        )
        if any(value is not None and value <= 0 for value in query_values):
            raise ValueError("optional query resource values must be positive")
        if any(value is None for value in query_values) and any(
            value is not None for value in query_values
        ):
            raise ValueError("query execution resources must be configured together")
        if (
            self.query_relation_store_banks is not None
            and self.query_relation_store_records is not None
            and self.query_relation_store_record_bytes is not None
            and self.resource_usage is not None
        ):
            relation_bytes = self.resource_usage.regions.get("relation_window")
            if relation_bytes is None:
                raise ValueError("relation-store resource region is missing")
            bytes_per_bank = relation_bytes // self.query_relation_store_banks
            derived_records = (
                self.query_relation_store_banks
                * (bytes_per_bank // self.query_relation_store_record_bytes)
            )
            if derived_records != self.query_relation_store_records:
                raise ValueError(
                    "relation-store records disagree with bytes, banks, and record width"
                )
        if (
            self.trace_continuation_query_packs is not None
            and self.trace_continuation_query_packs <= 0
        ):
            raise ValueError("trace continuation query-pack count must be positive")
        if (
            self.query_reduction_banks is not None
            and self.query_reduction_banks & (self.query_reduction_banks - 1)
        ):
            raise ValueError("query reduction bank count must be a power of two")
        if (
            self.query_volume_banks is not None
            and self.query_volume_banks & (self.query_volume_banks - 1)
        ):
            raise ValueError("query-volume bank count must be a power of two")
        if self.query_volume_bank_mapping not in {
            "linear", "xor_folded", "xor_shift",
        }:
            raise ValueError("query-volume Bank mapping is unsupported")
        if (self.query_volume_bank_mapping == "xor_shift") != (
            self.query_volume_bank_xor_shift is not None
        ):
            raise ValueError(
                "query-volume XOR-shift mapping and shift must be configured together"
            )
        if (
            self.query_volume_bank_xor_shift is not None
            and self.query_volume_bank_xor_shift <= 0
        ):
            raise ValueError("query-volume XOR shift must be positive")
        if (
            self.query_partial_sum_groups_per_bank is not None
            and self.query_partial_sum_groups_per_bank
            < self.modules["bidirectional_query"].latency
        ):
            raise ValueError(
                "query partial-sum groups cannot expose an FP32 feedback hazard"
            )
        if (
            self.query_loss_fma_lanes is not None
            and self.query_loss_queries_per_cycle is not None
            and self.query_loss_fma_lanes % self.query_loss_queries_per_cycle
        ):
            raise ValueError("query loss FMA lanes must divide into query issue lanes")
        if (
            self.memory_peak_bandwidth_bytes_per_second is not None
            and self.memory_peak_bandwidth_bytes_per_second <= 0
        ):
            raise ValueError("memory peak bandwidth must be positive")
        required = {
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        }
        missing = required.difference(self.modules)
        if missing:
            raise ValueError(f"cycle configuration lacks modules: {sorted(missing)}")

    def query_volume_bank(self, query_id: int) -> int:
        """Map one query SRAM address to its configured physical Bank."""

        if self.query_volume_banks is None:
            raise ValueError("query-volume Bank mapping is unavailable")
        query_id = int(query_id)
        if query_id < 0:
            raise ValueError("query-volume address must be non-negative")
        mask = self.query_volume_banks - 1
        if self.query_volume_bank_mapping == "linear":
            return query_id & mask
        if self.query_volume_bank_mapping == "xor_shift":
            assert self.query_volume_bank_xor_shift is not None
            return (
                query_id ^ (query_id >> self.query_volume_bank_xor_shift)
            ) & mask
        shift = mask.bit_length()
        folded = 0
        while query_id:
            folded ^= query_id & mask
            query_id >>= shift
        return folded

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
        compute_templates = _compute_templates_from_gala(config)
        clusters = int(config.value("top.num_pods")) * int(
            config.value("compute.clusters_per_pod")
        )
        return cls(modules=modules, memory=memory, clock_frequency_hz=frequency,
                   relation_seed_fifo_entries=seed_fifo,
                   candidate_lanes=int(config.value("issue.candidate_lanes")),
                   relation_support_lanes=int(
                       config.value("relation.support_lanes")
                   ),
                   relation_query_lanes=int(
                       config.value("compute.relations_per_microcontext")
                   ),
                   shared_sram_read_ports_per_bank=int(
                       config.value("shared_sram.read_ports_per_bank")
                   ),
                   shared_sram_write_ports_per_bank=int(
                       config.value("shared_sram.write_ports_per_bank")
                   ),
                   query_state_entries=int(config.value("issue.query_state_entries")),
                   fusion_query_state_banks=int(
                       config.value("issue.query_state_banks")
                   ),
                   candidate_fifo_entries=int(
                       config.value("issue.candidate_fifo_entries")
                   ),
                   fusion_bank_head_lookahead=config.value(
                       "issue.bank_head_lookahead"
                   ),
                   fusion_bank_head_index_bytes=int(
                       config.value("issue.bank_head_index_bytes")
                   ),
                   fusion_semantic_bundle_index_bytes=int(
                       config.value("issue.semantic_bundle_index_bytes")
                   ),
                   cache_instances=int(config.value("cache.instances")),
                   cache_capacity_per_instance=int(config.value("cache.active_records_per_instance")),
                   cache_directory_banks=int(config.value("cache.directory_banks_per_instance")),
                   cache_sector_bytes=int(config.value("cache.sector_bytes")),
                   cache_multicast_destinations=int(config.value("cache.multicast_destinations")),
                   cache_ready_head_index_bytes=int(
                       config.value("cache.ready_head_index_bytes")
                   ),
                   fusion_forward_ports=int(config.value("issue.forward_ports")),
                   fusion_consumer_ports=int(config.value("issue.consumer_ports")),
                   fusion_adjoint_ports=int(config.value("issue.adjoint_ports")),
                   resource_envelope=ResourceEnvelope.from_gala(config),
                   resource_usage=resource_usage or _resource_usage_from_gala(config),
                   config_sha256=config.sha256,
                   compute_templates=compute_templates,
                   compute_resource_capacities={
                       "pods": int(config.value("top.num_pods")),
                       "clusters_per_pod": int(config.value("compute.clusters_per_pod")),
                       "clusters": clusters,
                       "cluster_issue": clusters * int(
                           config.value("compute.cluster_issue_slots_per_cluster")
                       ),
                       "fma_groups": clusters * int(config.value("compute.fma_groups_per_cluster")),
                       "transcendental_lanes": clusters * int(config.value("compute.transcendental_lanes_per_cluster")),
                       "reduction_trees": clusters,
                       "microcontext_slots": (
                           clusters
                           * int(config.value("compute.microcontexts_per_cluster"))
                       ),
                       "feedback_lanes": clusters * int(config.value("compute.boundary_selectors_per_cluster")),
                   },
                   owner_gradient_slots_per_cluster=int(
                       config.value("compute.owner_gradient_slots_per_cluster")
                   ),
                   compute_ready_head_index_bytes=int(
                       config.value("compute.ready_head_index_bytes")
                   ),
                   query_reduction_banks=int(config.value("query.reduction_banks")),
                   query_partial_sum_groups_per_bank=int(
                       config.value("query.partial_sum_groups_per_bank")
                   ),
                   query_loss_fma_lanes=int(config.value("query.loss_fma_lanes")),
                   query_loss_queries_per_cycle=int(
                       config.value("query.loss_queries_per_cycle")
                   ),
                   query_adjoint_replay_lanes=int(
                       config.value("query.adjoint_replay_lanes")
                   ),
                   query_replay_queue_entries=int(
                       config.value("query.replay_queue_entries")
                   ),
                   query_relation_window_entries=int(
                       config.value("query.relation_window_entries")
                   ),
                   query_relation_window_entry_bytes=int(
                       config.value("query.relation_window_entry_bytes")
                   ),
                   query_relation_store_banks=int(
                       config.value("query.relation_store_banks")
                   ),
                   query_relation_store_records=int(
                       config.value("query.relation_store_records")
                   ),
                   query_relation_store_record_bytes=int(
                       config.value("query.relation_store_record_bytes")
                   ),
                   query_relation_candidate_ordinal_bits=int(
                       config.value("query.relation_candidate_ordinal_bits")
                   ),
                   trace_continuation_query_packs=int(
                       config.value("trace.continuation_query_packs")
                   ),
                   query_volume_banks=int(config.value("query.query_volume_banks")),
                   query_volume_word_bytes=int(
                       config.value("query.query_volume_word_bytes")
                   ),
                   query_volume_bank_mapping=str(
                       config.value("query.query_volume_bank_mapping")
                   ),
                   query_volume_bank_xor_shift=int(
                       config.value("query.query_volume_bank_xor_shift")
                   ),
                   memory_peak_bandwidth_bytes_per_second=int(
                       config.value("memory.peak_bandwidth_bytes_per_second")
                   ))


def _compute_templates_from_gala(config: GalaConfig) -> dict[int, ComputeTemplateProfile] | None:
    """Parse the registered template sequences, if this config declares them."""

    try:
        raw_profiles = config.value("compute.template_profiles")
    except KeyError:
        return None
    if not isinstance(raw_profiles, (list, tuple)):
        raise ValueError("compute.template_profiles must be a list")
    profiles: dict[int, ComputeTemplateProfile] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("compute template profile must be a mapping")
        try:
            template_id = int(raw_profile["template_id"])
            raw_paths = raw_profile["paths"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("compute template profile metadata is incomplete") from error
        if not isinstance(raw_paths, Mapping):
            raise ValueError("compute template profile paths must be a mapping")
        paths: dict[str, ComputePathProfile] = {}
        for path, raw_stages in raw_paths.items():
            if not isinstance(path, str) or not isinstance(raw_stages, Mapping):
                raise ValueError("compute template path is malformed")
            stage_sequence = raw_stages.get("stages")
            if not isinstance(stage_sequence, (list, tuple)):
                raise ValueError("compute template path stages must be a list")
            stages: list[ComputeStage] = []
            for raw_stage in stage_sequence:
                if not isinstance(raw_stage, Mapping):
                    raise ValueError("compute template stage must be a mapping")
                try:
                    raw_latency = raw_stage.get("latency")
                    if raw_latency is None:
                        fma_passes = int(raw_stage.get("fma_passes", 0))
                        transcendental_ops = int(raw_stage.get("transcendental_ops", 0))
                        reduction_passes = int(raw_stage.get("reduction_passes", 0))
                        if fma_passes + transcendental_ops + reduction_passes <= 0:
                            raise ValueError(
                                "stage needs latency or primitive operation counts"
                            )
                        raw_latency = (
                            fma_passes * int(config.value("compute.fma_latency"))
                            + transcendental_ops * int(config.value("compute.transcendental_latency"))
                            + reduction_passes * int(config.value("compute.reduction_latency"))
                        )
                    stage = ComputeStage(
                        name=str(raw_stage["name"]),
                        latency=int(raw_latency),
                        fma_groups=int(raw_stage.get("fma_groups", 0)),
                        transcendental_lanes=int(raw_stage.get("transcendental_lanes", 0)),
                        reduction_trees=int(raw_stage.get("reduction_trees", 0)),
                        register_reads=int(raw_stage.get("register_reads", 0)),
                        register_writes=int(raw_stage.get("register_writes", 0)),
                        feedback_lanes=int(raw_stage.get("feedback_lanes", 0)),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("compute template stage metadata is malformed") from error
                stages.append(stage)
            paths[path] = ComputePathProfile(
                tuple(stages),
                cluster_issue_slots=int(raw_stages.get("cluster_issue_slots", 1)),
                cluster_issue_cycles=int(raw_stages.get("cluster_issue_cycles", 1)),
                packet_first_result_latency=(
                    int(raw_stages["packet_first_result_latency"])
                    if "packet_first_result_latency" in raw_stages else None
                ),
                packet_last_result_offset=(
                    int(raw_stages["packet_last_result_offset"])
                    if "packet_last_result_offset" in raw_stages else None
                ),
                packet_lane_issue_interval=int(
                    raw_stages.get("packet_lane_issue_interval", 1)
                ),
                packet_lanes_per_issue=int(
                    raw_stages.get("packet_lanes_per_issue", 1)
                ),
                packet_completion_at_last=bool(
                    raw_stages.get("packet_completion_at_last", False)
                ),
            )
        if template_id in profiles:
            raise ValueError(f"duplicate ComputePod template ID: {template_id}")
        profiles[template_id] = ComputeTemplateProfile(template_id, paths)
    return profiles


def _resource_usage_from_gala(config: GalaConfig) -> ResourceUsage:
    """Derive fixed execution resources and registered SRAM regions."""

    try:
        pods = int(config.value("top.num_pods"))
        clusters_per_pod = int(config.value("compute.clusters_per_pod"))
        clusters = pods * clusters_per_pod
        regions = {
            "active_gaussian": int(config.value("shared_sram.active_gaussian_bytes")),
            "relation_window": int(config.value("shared_sram.relation_window_bytes")),
            "query_volume": int(config.value("shared_sram.query_volume_bytes")),
            "gradient_update": int(config.value("shared_sram.gradient_update_bytes")),
            "index_graph": int(config.value("shared_sram.index_graph_bytes")),
            "control_metadata": int(config.value("shared_sram.control_metadata_bytes")),
        }
        registered_shared_sram = int(config.value("top.shared_sram_bytes"))
        if sum(regions.values()) != registered_shared_sram:
            raise ValueError(
                "registered shared SRAM regions do not exactly close the top-level budget"
            )
        return ResourceUsage(
            shared_sram_bytes=registered_shared_sram,
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
