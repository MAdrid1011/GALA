"""Auditable necessary cycle bounds for one complete trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.trace.model import Trace
from gala_sim.timing.packets import (
    PhysicalPacketStage,
    RelationPacketPlan,
    RelationWindowPlan,
)


@dataclass(frozen=True)
class CycleBoundComponent:
    name: str
    category: str
    cycles: int
    evidence: Mapping[str, Any]
    limitation: str


@dataclass(frozen=True)
class ScenarioCycleBound:
    scenario: str
    minimum_cycles: int
    limiting_component: str
    memory_request_lower_bound: int
    memory_byte_lower_bound: int
    components: tuple[CycleBoundComponent, ...]


@dataclass(frozen=True)
class TargetReachability:
    scenario: str
    target_speedup_vs_base_asic: float
    target_cycle_budget: int
    lower_bound_cycles: int
    maximum_possible_speedup_vs_base_asic: float
    maximum_coverable_cycles: int
    minimum_uncovered_cycles: int
    status: str


@dataclass(frozen=True)
class CycleLowerBoundReport:
    schema_version: str
    trace_event_count: int
    trace_dependency_count: int
    base_asic_cycles: int
    scenarios: tuple[ScenarioCycleBound, ...]
    reachability: tuple[TargetReachability, ...]
    capacity_diagnostics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _issue_completion_bound(
    count: int, *, ports: int, initiation_interval: int, minimum_service: int,
) -> int:
    if count <= 0:
        return 0
    issue_batches = math.ceil(count / ports)
    return (issue_batches - 1) * initiation_interval + minimum_service


def _event_service_cycles(
    engine: Any, trace: Trace, packet_plan: RelationPacketPlan,
) -> np.ndarray:
    service = np.zeros(trace.event_count, dtype=np.uint64)
    kinds = trace.events["primitive_kind"]
    templates = trace.events["template_id"]
    for kind in PrimitiveKind:
        kind_ids = np.flatnonzero(kinds == int(kind))
        if not kind_ids.size:
            continue
        stages = engine._stages_for(kind)
        generic_cycles = sum(
            engine.modules[name].timing.latency
            for name in stages if name != "compute_pod"
        )
        if "compute_pod" not in stages:
            service[kind_ids] = generic_cycles
            continue
        for event_id in kind_ids:
            row = trace.events[int(event_id)]
            physical_stage = packet_plan.stage_for_event(int(event_id))
            if physical_stage is None:
                compute_cycles = engine._service_cycles("compute_pod", row, kind)
            elif kind is PrimitiveKind.FORWARD:
                compute = engine.modules["compute_pod"]
                path = compute.path_for(int(row["template_id"]), kind)
                lane = physical_stage.lanes[
                    physical_stage.event_ids.index(int(event_id))
                ]
                compute_cycles = path.packet_completion_offset(lane)
            else:
                compute_cycles = engine._physical_service_cycles(
                    "compute_pod", row, kind, physical_stage,
                )
            service[int(event_id)] = generic_cycles + compute_cycles
    return service


def _dependency_bound(
    trace: Trace, service: np.ndarray, packet_plan: RelationPacketPlan,
) -> int:
    """Collapse each physical stage into one atomic-ready DAG node."""

    node_members: list[tuple[int, ...]] = [stage.event_ids for stage in packet_plan.stages]
    event_to_node = np.full(trace.event_count, -1, dtype=np.int64)
    for stage in packet_plan.stages:
        event_to_node[list(stage.event_ids)] = stage.stage_id
    for event_id in range(trace.event_count):
        if event_to_node[event_id] < 0:
            event_to_node[event_id] = len(node_members)
            node_members.append((event_id,))
    predecessor_nodes: list[set[int]] = [set() for _ in node_members]
    successor_nodes: list[set[int]] = [set() for _ in node_members]
    for node_id, members in enumerate(node_members):
        for event_id in members:
            row = trace.events[event_id]
            for dependency in trace.dependency_ids(row):
                predecessor = int(event_to_node[int(dependency)])
                if predecessor == node_id:
                    raise ValueError("physical packet stage has an internal dependency")
                predecessor_nodes[node_id].add(predecessor)
                successor_nodes[predecessor].add(node_id)
    remaining = np.asarray(
        [len(predecessors) for predecessors in predecessor_nodes], dtype=np.int64,
    )
    ready = [int(node_id) for node_id in np.flatnonzero(remaining == 0)]
    earliest = np.zeros(trace.event_count, dtype=np.uint64)
    processed = 0
    while ready:
        node_id = ready.pop()
        members = node_members[node_id]
        start = 0
        for event_id in members:
            dependencies = trace.dependency_ids(trace.events[event_id])
            if dependencies.size:
                start = max(start, int(np.max(earliest[dependencies])))
        for event_id in members:
            earliest[event_id] = start + int(service[event_id])
        processed += 1
        for successor in successor_nodes[node_id]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
    if processed != len(node_members):
        raise ValueError("physical packet dependency graph contains a cycle")
    return int(np.max(earliest)) if earliest.size else 0


def _module_event_ids(engine: Any, trace: Trace, module_name: str) -> np.ndarray:
    kinds = trace.events["primitive_kind"]
    accepted = [
        int(kind) for kind in PrimitiveKind
        if module_name in engine._stages_for(kind)
    ]
    return np.flatnonzero(np.isin(kinds, accepted))


def _module_work_items(
    engine: Any, trace: Trace, module_name: str, packet_plan: RelationPacketPlan,
) -> list[tuple[int, PhysicalPacketStage | None]]:
    items: list[tuple[int, PhysicalPacketStage | None]] = []
    seen_stages: set[int] = set()
    for event_id, row in enumerate(trace.events):
        kind = PrimitiveKind(int(row["primitive_kind"]))
        if module_name not in engine._stages_for(kind):
            continue
        physical_stage = packet_plan.stage_for_event(event_id)
        if physical_stage is not None:
            if physical_stage.stage_id in seen_stages:
                continue
            seen_stages.add(physical_stage.stage_id)
            event_id = physical_stage.head_event_id
        items.append((event_id, physical_stage))
    return items


def _module_components(
    engine: Any,
    trace: Trace,
    packet_plan: RelationPacketPlan,
    *,
    compulsory_cache_writes: bool = False,
) -> list[CycleBoundComponent]:
    components: list[CycleBoundComponent] = []
    compulsory_request_ids = (
        set(_cache_request_ids(trace, packet_plan, compulsory_only=True))
        if compulsory_cache_writes else None
    )
    for module_name, module in engine.modules.items():
        if module_name == "bidirectional_query" and engine._has_query_resources():
            continue
        items = _module_work_items(engine, trace, module_name, packet_plan)
        if compulsory_request_ids is not None and module_name == "shared_sram":
            items = [
                item for item in items
                if PrimitiveKind(int(trace.events[item[0]]["primitive_kind"]))
                is not PrimitiveKind.CACHE_REQUEST
                or item[0] in compulsory_request_ids
            ]
        if not items:
            continue
        ids = np.asarray([item[0] for item in items], dtype=np.int64)
        services = np.asarray([
            engine._physical_service_cycles(
                module_name,
                trace.events[event_id],
                PrimitiveKind(int(trace.events[event_id]["primitive_kind"])),
                physical_stage,
            )
            for event_id, physical_stage in items
        ], dtype=np.uint64)
        count = int(ids.size)
        partitions = np.asarray([
            engine._module_partition(module_name, trace.events[int(event_id)])
            for event_id in ids
        ], dtype=np.int64)
        partition_counts = np.bincount(
            partitions, minlength=engine._module_partition_count(module_name)
        )
        ports_per_partition = np.asarray([
            engine._module_partition_issue_limit(module_name, partition)
            for partition in range(partition_counts.size)
        ], dtype=np.int64)
        port_cycles = max(
            _issue_completion_bound(
                int(partition_count),
                ports=int(ports_per_partition[partition]),
                initiation_interval=module.timing.initiation_interval,
                minimum_service=int(np.min(services[partitions == partition])),
            )
            for partition, partition_count in enumerate(partition_counts)
            if partition_count
        )
        components.append(CycleBoundComponent(
            name=f"{module_name}.issue_ports",
            category="module_issue",
            cycles=port_cycles,
            evidence={
                "physical_stage_count": count,
                "logical_lane_event_count": sum(
                    len(stage.event_ids) if stage is not None else 1
                    for _event_id, stage in items
                ),
                "partitions": int(partition_counts.size),
                "event_stage_count_per_partition": partition_counts.tolist(),
                "ports_per_partition": ports_per_partition.tolist(),
                "aggregate_ports": int(engine._module_issue_ports(module_name)),
                "initiation_interval_cycles": module.timing.initiation_interval,
                "minimum_service_cycles": int(np.min(services)),
            },
            limitation="Assumes every input is ready at cycle zero and chooses the shortest tail.",
        ))
        if engine._uses_generic_module_limits(module_name):
            occupancy_cycles = max(
                math.ceil(
                    int(np.sum(services[partitions == partition], dtype=np.uint64))
                    / module.timing.queue_capacity
                )
                for partition in range(partition_counts.size)
                if partition_counts[partition]
            )
            components.append(CycleBoundComponent(
                name=f"{module_name}.queue_occupancy",
                category="queue_capacity",
                cycles=occupancy_cycles,
                evidence={
                    "service_cycle_sum_per_partition": [
                        int(np.sum(services[partitions == partition], dtype=np.uint64))
                        for partition in range(partition_counts.size)
                    ],
                    "queue_capacity_per_partition": module.timing.queue_capacity,
                },
                limitation="Uses ideal continuous occupancy with no dependency or port bubbles.",
            ))
            if engine._uses_generic_module_bank_limits(module_name):
                if module_name == "shared_sram":
                    bank_count = module.timing.banks
                    read_counts = np.zeros((partition_counts.size, bank_count), dtype=np.int64)
                    write_counts = np.zeros((partition_counts.size, bank_count), dtype=np.int64)
                    for index, event_id in enumerate(ids):
                        row = trace.events[int(event_id)]
                        bank = engine._shared_sram_bank(row)
                        kind = PrimitiveKind(int(row["primitive_kind"]))
                        if engine._shared_sram_direction(kind) == "read":
                            read_counts[partitions[index], bank] += 1
                        else:
                            write_counts[partitions[index], bank] += 1
                    read_cycles = max(
                        _issue_completion_bound(
                            int(count),
                            ports=engine.config.shared_sram_read_ports_per_bank,
                            initiation_interval=module.timing.initiation_interval,
                            minimum_service=int(np.min(services)),
                        )
                        for count in read_counts.ravel()
                        if count
                    ) if np.any(read_counts) else 0
                    write_cycles = max(
                        _issue_completion_bound(
                            int(count),
                            ports=engine.config.shared_sram_write_ports_per_bank,
                            initiation_interval=module.timing.initiation_interval,
                            minimum_service=int(np.min(services)),
                        )
                        for count in write_counts.ravel()
                        if count
                    ) if np.any(write_counts) else 0
                    components.append(CycleBoundComponent(
                        name=f"{module_name}.banks",
                        category="bank_issue",
                        cycles=max(read_cycles, write_cycles),
                        evidence={
                            "partitions": int(partition_counts.size),
                            "banks_per_partition": bank_count,
                            "read_ports_per_bank": engine.config.shared_sram_read_ports_per_bank,
                            "write_ports_per_bank": engine.config.shared_sram_write_ports_per_bank,
                            "read_accesses_per_bank": read_counts.tolist(),
                            "write_accesses_per_bank": write_counts.tolist(),
                            "read_cycles": read_cycles,
                            "write_cycles": write_cycles,
                        },
                        limitation="Each physical Bank has independent 1R1W ports; same-address read/write replay and dependency bubbles are outside this necessary bound.",
                    ))
                    continue
                elif (
                    module_name == "fusion_issue"
                    and engine.config.fusion_query_state_banks is not None
                ):
                    bank_count = engine.config.fusion_query_state_banks
                    lines = np.asarray(
                        trace.events["query_id"][ids], dtype=np.int64,
                    ) // engine.config.relation_query_lanes
                else:
                    bank_count = module.timing.banks
                    tokens = np.asarray(
                        trace.events["address_token"][ids], dtype=np.uint64,
                    )
                    lines = np.where(
                        (tokens >= 64) & (tokens % 64 == 0),
                        tokens // 64,
                        tokens,
                    )
                bank_counts = np.bincount(
                    np.asarray(
                        partitions * bank_count + (lines % bank_count),
                        dtype=np.int64,
                    ),
                    minlength=partition_counts.size * bank_count,
                )
                busiest = int(np.max(bank_counts))
                bank_cycles = busiest - 1 + int(np.min(services))
                components.append(CycleBoundComponent(
                    name=f"{module_name}.banks",
                    category="bank_issue",
                    cycles=bank_cycles,
                    evidence={
                        "partitions": int(partition_counts.size),
                        "banks_per_partition": bank_count,
                        "busiest_bank_event_count": busiest,
                        "minimum_service_cycles": int(np.min(services)),
                    },
                    limitation="Assumes one issue per bank per cycle and the shortest possible final service.",
                ))
    return components


def _query_components(
    engine: Any, trace: Trace, packet_plan: RelationPacketPlan,
) -> list[CycleBoundComponent]:
    if not engine._has_query_resources():
        return []
    banks = engine.config.query_reduction_banks
    loss_lanes = engine.config.query_loss_fma_lanes
    loss_slots = engine.config.query_loss_queries_per_cycle
    replay_lanes = engine.config.query_adjoint_replay_lanes
    groups = engine.config.query_partial_sum_groups_per_bank
    assert banks is not None and loss_lanes is not None and loss_slots is not None
    assert replay_lanes is not None and groups is not None
    timing = engine.modules["bidirectional_query"].timing
    kinds = trace.events["primitive_kind"]

    reduction_ids = np.flatnonzero(np.isin(kinds, [
        int(PrimitiveKind.FORWARD), int(PrimitiveKind.QUERY_REDUCTION),
    ]))
    query_ids = np.asarray(trace.events["query_id"][reduction_ids], dtype=np.int64)
    reduction_slots = (query_ids % banks) * groups + (query_ids // banks) % groups
    slot_counts = np.bincount(reduction_slots, minlength=banks * groups)
    busiest_slot = int(np.max(slot_counts, initial=0))
    components = [CycleBoundComponent(
        name="bidirectional_query.reduction_banks",
        category="query_reduction",
        cycles=_issue_completion_bound(
            busiest_slot,
            ports=1,
            initiation_interval=timing.initiation_interval,
            minimum_service=timing.latency,
        ),
        evidence={
            "logical_forward_contributions": int(np.count_nonzero(
                kinds == int(PrimitiveKind.FORWARD)
            )),
            "query_completion_events": int(np.count_nonzero(
                kinds == int(PrimitiveKind.QUERY_REDUCTION)
            )),
            "reduction_banks": banks,
            "partial_sum_groups_per_bank": groups,
            "reduction_slots": banks * groups,
            "busiest_bank_group_events": busiest_slot,
        },
        limitation="Assumes all interleaved bank groups are independently fed and every contribution is ready.",
    )]

    consumers = int(np.count_nonzero(kinds == int(PrimitiveKind.CONSUMER)))
    components.append(CycleBoundComponent(
        name="bidirectional_query.loss_fma_lanes",
        category="query_loss",
        cycles=_issue_completion_bound(
            consumers,
            ports=loss_slots,
            initiation_interval=timing.initiation_interval,
            minimum_service=timing.latency,
        ),
        evidence={
            "consumer_events": consumers,
            "fma_lanes_per_consumer": loss_lanes,
            "queries_per_cycle": loss_slots,
            "fma_lanes_per_query_issue": loss_lanes // loss_slots,
        },
        limitation="Uses the configured query issue width and assumes every consumer is ready; dependencies and volume-bank conflicts are free.",
    ))

    volume_banks = engine.config.query_volume_banks
    assert volume_banks is not None
    volume_reads = np.zeros(volume_banks, dtype=np.int64)
    volume_writes = np.zeros(volume_banks, dtype=np.int64)
    seen_volume_stages: set[int] = set()
    for raw_event_id, row in enumerate(trace.events):
        kind = PrimitiveKind(int(row["primitive_kind"]))
        if kind not in {
            PrimitiveKind.QUERY_REDUCTION,
            PrimitiveKind.CONSUMER,
            PrimitiveKind.ADJOINT,
        }:
            continue
        physical_stage = packet_plan.stage_for_event(raw_event_id)
        if physical_stage is not None:
            if physical_stage.stage_id in seen_volume_stages:
                continue
            seen_volume_stages.add(physical_stage.stage_id)
            query_ids = tuple(
                int(trace.events[event_id]["query_id"])
                for event_id in physical_stage.event_ids
            )
        else:
            query_ids = (int(row["query_id"]),)
        if kind in {PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT}:
            for query_id in query_ids:
                volume_reads[query_id % volume_banks] += 1
        if kind in {PrimitiveKind.CONSUMER, PrimitiveKind.QUERY_REDUCTION}:
            for query_id in query_ids:
                volume_writes[query_id % volume_banks] += 1

    for direction, counts in (("read", volume_reads), ("write", volume_writes)):
        busiest = int(np.max(counts, initial=0))
        if not busiest:
            continue
        components.append(CycleBoundComponent(
            name=f"bidirectional_query.volume_{direction}_banks",
            category="query_volume_bank",
            cycles=_issue_completion_bound(
                busiest,
                ports=1,
                initiation_interval=timing.initiation_interval,
                minimum_service=timing.latency,
            ),
            evidence={
                "volume_banks": volume_banks,
                "accesses_per_bank": counts.tolist(),
                "busiest_bank_accesses": busiest,
            },
            limitation="One access per SRAM bank per cycle; assumes all accesses are ready and ignores read/write pipeline coupling.",
        ))

    replay_counts = np.zeros(replay_lanes, dtype=np.int64)
    seen_stages: set[int] = set()
    adjoint_ids = np.flatnonzero(kinds == int(PrimitiveKind.ADJOINT))
    physical_packets = 0
    for raw_event_id in adjoint_ids:
        event_id = int(raw_event_id)
        physical_stage = packet_plan.stage_for_event(event_id)
        if physical_stage is None:
            replay_counts[int(trace.events[event_id]["query_id"]) % replay_lanes] += 1
            physical_packets += 1
            continue
        if physical_stage.stage_id in seen_stages:
            continue
        seen_stages.add(physical_stage.stage_id)
        replay_counts[list(physical_stage.lanes)] += 1
        physical_packets += 1
    total_replay_lane_work = int(np.sum(replay_counts, dtype=np.int64))
    # Replay dispatch is work-conserving: a ready packet can use any free
    # physical replay lane.  Fixed packet lane positions are useful telemetry
    # but are not a necessary lower bound on the critical path.
    replay_work_cycles = _issue_completion_bound(
        total_replay_lane_work,
        ports=replay_lanes,
        initiation_interval=timing.initiation_interval,
        minimum_service=timing.latency,
    )
    busiest_replay = int(np.max(replay_counts, initial=0))
    components.append(CycleBoundComponent(
        name="bidirectional_query.adjoint_replay_lanes",
        category="query_adjoint_replay",
        cycles=replay_work_cycles,
        evidence={
            "physical_adjoint_packets": physical_packets,
            "logical_adjoint_lanes": int(adjoint_ids.size),
            "replay_lanes": replay_lanes,
            "events_per_replay_lane": replay_counts.tolist(),
            "busiest_replay_lane_events": busiest_replay,
            "total_replay_lane_work": total_replay_lane_work,
            "work_conserving_lanes": replay_lanes,
        },
        limitation="Uses total physical replay lane work divided across eight work-conserving lanes; dependencies and packet launch bubbles are free.",
    ))
    return components


def _fusion_components(
    engine: Any, trace: Trace, packet_plan: RelationPacketPlan,
) -> list[CycleBoundComponent]:
    components: list[CycleBoundComponent] = []
    kinds = trace.events["primitive_kind"]
    fusion_kinds = (
        PrimitiveKind.FORWARD, PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT,
    )
    for kind in fusion_kinds:
        items = [
            item for item in _module_work_items(
                engine, trace, "fusion_issue", packet_plan,
            )
            if PrimitiveKind(int(trace.events[item[0]]["primitive_kind"])) is kind
        ]
        count = len(items)
        if not count:
            continue
        port_name, ports = engine._fusion_port_limit(kind)
        service = engine.modules["fusion_issue"].timing.latency
        cycles = _issue_completion_bound(
            count,
            ports=ports,
            initiation_interval=engine.modules["fusion_issue"].timing.initiation_interval,
            minimum_service=service,
        )
        components.append(CycleBoundComponent(
            name=f"fusion_issue.{port_name}_ports",
            category="fusion_issue_class",
            cycles=cycles,
            evidence={
                "physical_task_count": count,
                "logical_event_count": sum(
                    len(stage.event_ids) if stage is not None else 1
                    for _event_id, stage in items
                ),
                "ports": ports,
            },
            limitation="Assumes all other fusion classes, banks, and dependencies are free.",
        ))
    conflict_work: dict[tuple[str, int], int] = {}
    for event_id, physical_stage in _module_work_items(
        engine, trace, "fusion_issue", packet_plan,
    ):
        row = trace.events[event_id]
        kind = PrimitiveKind(int(row["primitive_kind"]))
        if kind is PrimitiveKind.ADJOINT:
            keys = (("gaussian", int(row["gaussian_id"])),)
        else:
            keys = tuple(
                ("query", int(trace.events[member]["query_id"]))
                for member in physical_stage.event_ids
            ) if physical_stage is not None else ((
                "query",
                int(row["reduction_key"])
                if int(row["reduction_key"]) >= 0 else int(row["query_id"]),
            ),)
        for key in keys:
            conflict_work[key] = conflict_work.get(key, 0) + (
                engine.modules["fusion_issue"].timing.latency
            )
    if conflict_work:
        key, cycles = max(conflict_work.items(), key=lambda item: item[1])
        components.append(CycleBoundComponent(
            name="fusion_issue.reduction_conflicts",
            category="semantic_conflict",
            cycles=cycles,
            evidence={"busiest_domain": key[0], "busiest_key": key[1]},
            limitation="Only serializes events sharing the same frozen reduction key.",
        ))
    return components


def _compute_components(
    engine: Any, trace: Trace, packet_plan: RelationPacketPlan,
) -> list[CycleBoundComponent]:
    capacities = engine.config.compute_resource_capacities
    templates = engine.config.compute_templates
    if not capacities or not templates:
        return []
    demand: dict[str, int] = {}
    demand_by_pod: dict[int, dict[str, int]] = {}
    count_by_path: dict[str, int] = {}
    pods = int(capacities.get("pods", 1))
    clusters = int(capacities.get("clusters", 1))
    if clusters % pods:
        raise ValueError("ComputePod clusters are not divisible by Pod count")
    capacity_per_pod = {
        resource: int(value) // pods
        for resource, value in capacities.items()
        if resource not in {"pods", "clusters_per_pod"}
    }
    compute_items = _module_work_items(engine, trace, "compute_pod", packet_plan)
    for event_id, physical_stage in compute_items:
        row = trace.events[event_id]
        kind = PrimitiveKind(int(row["primitive_kind"]))
        profile = engine.modules["compute_pod"].path_for(
            int(row["template_id"]), kind
        )
        path_name = engine.modules["compute_pod"]._path_for(kind)
        count_by_path[path_name] = count_by_path.get(path_name, 0) + 1
        event_demand: dict[str, int] = {
            "cluster_issue": profile.cluster_issue_slots * profile.cluster_issue_cycles,
            "microcontext_slots": profile.cluster_issue_cycles,
        }
        for stage in profile.stages:
            for resource, value in engine.modules["compute_pod"]._demands(stage):
                if value and resource in capacities:
                    event_demand[resource] = event_demand.get(resource, 0) + value
        pod = int(row["gaussian_id"]) % pods if int(row["gaussian_id"]) >= 0 else 0
        pod_demand = demand_by_pod.setdefault(pod, {})
        for resource, value in event_demand.items():
            demand[resource] = demand.get(resource, 0) + value
            pod_demand[resource] = pod_demand.get(resource, 0) + value
    components: list[CycleBoundComponent] = []
    for resource, total in sorted(demand.items()):
        capacity = int(capacities[resource])
        pod_cycles = [
            math.ceil(pod_values.get(resource, 0) / capacity_per_pod[resource])
            for pod_values in demand_by_pod.values()
            if pod_values.get(resource, 0)
        ]
        components.append(CycleBoundComponent(
            name=f"compute_pod.{resource}",
            category="compute_reservation",
            cycles=max(math.ceil(total / capacity), max(pod_cycles, default=0)),
            evidence={
                "total_slot_cycles_or_demands": total,
                "global_capacity_per_cycle": capacity,
                "pod_demand": {
                    str(pod): values.get(resource, 0)
                    for pod, values in sorted(demand_by_pod.items())
                },
                "pod_capacity_per_cycle": capacity_per_pod[resource],
                "event_count_by_path": dict(sorted(count_by_path.items())),
                "logical_lane_events": sum(
                    len(stage.event_ids) if stage is not None else 1
                    for _event_id, stage in compute_items
                ),
            },
            limitation="Aggregates flexible cluster assignment within each Pod; it is still optimistic because it ignores stage overlap and exact per-cluster packing.",
        ))
    return components


def _memory_metadata(engine: Any) -> Mapping[str, Any]:
    metadata = getattr(engine.config.memory, "metadata", None)
    if not callable(metadata):
        return {}
    value = metadata()
    return value if isinstance(value, Mapping) else {}


def _memory_components(
    engine: Any,
    *,
    request_count: int,
    byte_count: int,
    scenario: str,
) -> list[CycleBoundComponent]:
    components: list[CycleBoundComponent] = []
    peak = engine.config.memory_peak_bandwidth_bytes_per_second
    if peak is not None and byte_count:
        cycles = math.ceil(byte_count * engine.config.clock_frequency_hz / peak)
        components.append(CycleBoundComponent(
            name=f"{scenario}.memory_peak_bandwidth",
            category="memory_bandwidth",
            cycles=cycles,
            evidence={
                "minimum_bytes": byte_count,
                "peak_bandwidth_bytes_per_second": peak,
                "clock_frequency_hz": engine.config.clock_frequency_hz,
            },
            limitation="Physical peak-bandwidth lower bound; excludes every command, bank, and refresh penalty.",
        ))
    metadata = _memory_metadata(engine)
    read_latency = metadata.get("read_latency_cycles")
    if request_count and isinstance(read_latency, int) and read_latency > 0:
        components.append(CycleBoundComponent(
            name=f"{scenario}.ramulator_read_return",
            category="memory_command_timing",
            cycles=read_latency,
            evidence={
                "minimum_logical_requests": request_count,
                "read_latency_cycles": read_latency,
                "ramulator_clock_ratio": 1,
                "ramulator_config_sha256": metadata.get("config_sha256"),
            },
            limitation="Minimum RD-to-return latency only; ACT, PRE, arbitration, and refresh can only add cycles.",
        ))
    return components


def _cache_request_floor(
    trace: Trace, packet_plan: RelationPacketPlan, *, compulsory_only: bool,
) -> tuple[int, int]:
    request_ids = _cache_request_ids(
        trace, packet_plan, compulsory_only=compulsory_only,
    )
    requests = trace.events[np.asarray(request_ids, dtype=np.int64)]
    return int(requests.size), int(np.sum(requests["data_bytes"], dtype=np.uint64))


def _cache_request_ids(
    trace: Trace, packet_plan: RelationPacketPlan, *, compulsory_only: bool,
) -> tuple[int, ...]:
    """Return cache-request stage heads, optionally limited to miss leaders."""

    mask = trace.events["primitive_kind"] == int(PrimitiveKind.CACHE_REQUEST)
    request_ids = np.flatnonzero(mask)
    request_ids = np.asarray([
        int(event_id) for event_id in request_ids
        if packet_plan.is_stage_head(int(event_id))
    ], dtype=np.int64)
    if not compulsory_only:
        return tuple(int(event_id) for event_id in request_ids)
    first_by_key: dict[tuple[int, int], int] = {}
    for event_id in request_ids:
        row = trace.events[int(event_id)]
        key = (int(row["gaussian_id"]), int(row["state_version"]))
        first_by_key.setdefault(key, int(event_id))
    return tuple(first_by_key.values())


def _capacity_diagnostics(
    engine: Any, trace: Trace, packet_plan: RelationPacketPlan,
) -> dict[str, Any]:
    instances = engine.config.cache_instances
    capacity = engine.config.cache_capacity_per_instance
    diagnostics: dict[str, Any] = {
        "relation_packets": _packet_diagnostics(trace, packet_plan),
    }
    if not instances or not capacity:
        diagnostics["semantic_cache"] = {"status": "unavailable_configuration"}
    else:
        mask = trace.events["primitive_kind"] == int(PrimitiveKind.CACHE_REQUEST)
        unique_by_instance: dict[int, set[tuple[int, int]]] = {
            instance: set() for instance in range(instances)
        }
        for row in trace.events[mask]:
            gaussian_id = int(row["gaussian_id"])
            unique_by_instance[gaussian_id % instances].add(
                (gaussian_id, int(row["state_version"]))
            )
        counts = {str(key): len(value) for key, value in unique_by_instance.items()}
        diagnostics["semantic_cache"] = {
            "instances": instances,
            "active_records_per_instance": capacity,
            "unique_keys_per_instance": counts,
            "all_unique_keys_fit_without_eviction": max(counts.values(), default=0) <= capacity,
            "interpretation": "Unique-key capacity check, not a timed liveness proof.",
        }

    relation_windows = _relation_window_diagnostics(engine, trace, packet_plan)
    diagnostics["relation_windows"] = relation_windows

    replay_capacity = engine.config.query_replay_queue_entries
    replay_keys: dict[tuple[int, int], int] = {}
    consumers: set[tuple[int, int]] = set()
    for row in trace.events:
        kind = PrimitiveKind(int(row["primitive_kind"]))
        key = (int(row["iteration_id"]), int(row["query_id"]))
        if kind is PrimitiveKind.ADJOINT:
            replay_keys[key] = replay_keys.get(key, 0) + 1
        elif kind is PrimitiveKind.CONSUMER:
            consumers.add(key)
    diagnostics["query_replay"] = {
        "capacity_entries": replay_capacity,
        "queries_with_adjoint_replay": len(replay_keys),
        "consumer_queries_with_replay": len(consumers & replay_keys.keys()),
        "max_adjoint_lanes_per_query": max(replay_keys.values(), default=0),
        "interpretation": "Identity and multiplicity counts only; event order is not a liveness proof.",
    }

    capacities = engine.config.compute_resource_capacities or {}
    pods = int(capacities.get("pods", 1))
    clusters_per_pod = int(capacities.get("clusters_per_pod", capacities.get("clusters", 1)))
    owner_keys: dict[int, set[tuple[int, int, int]]] = {}
    for row in trace.events:
        if PrimitiveKind(int(row["primitive_kind"])) is not PrimitiveKind.ADJOINT:
            continue
        gaussian_id = int(row["gaussian_id"])
        if gaussian_id < 0:
            continue
        pod = gaussian_id % pods
        cluster = pod * clusters_per_pod + gaussian_id % clusters_per_pod
        owner_keys.setdefault(cluster, set()).add(
            (int(row["iteration_id"]), gaussian_id, int(row["state_version"]))
        )
    diagnostics["owner_gradient"] = {
        "slots_per_cluster": engine.config.owner_gradient_slots_per_cluster,
        "unique_epoch_keys_per_cluster": {
            str(cluster): len(keys) for cluster, keys in sorted(owner_keys.items())
        },
        "max_unique_epoch_keys_per_cluster": max(
            (len(keys) for keys in owner_keys.values()), default=0,
        ),
        "interpretation": "Unique-key capacity check; exact in-flight lifetime still depends on dependencies.",
    }

    envelope = engine.config.resource_envelope
    usage = engine.config.resource_usage
    diagnostics["shared_sram"] = {
        "envelope_bytes": envelope.shared_sram_bytes if envelope is not None else None,
        "usage_bytes": usage.shared_sram_bytes if usage is not None else None,
        "usage_regions": dict(usage.regions) if usage is not None else {},
        "status": "checked" if envelope is not None and usage is not None else "unavailable_configuration",
    }
    diagnostics["module_queues"] = {
        name: module.timing.queue_capacity
        for name, module in engine.modules.items()
    }
    return diagnostics


def _relation_window_diagnostics(
    engine: Any, trace: Trace, packet_plan: RelationPacketPlan,
) -> dict[str, Any]:
    """Report declared window/store occupancy without guessing a schedule."""

    window_plan = RelationWindowPlan.from_trace(trace, packet_plan)
    if window_plan is None:
        return {"status": "unavailable_trace_sidecar"}
    descriptors = window_plan.descriptors
    records = [descriptor.relation_records for descriptor in descriptors]
    capacity = engine.config.query_relation_window_entries
    store_capacity = engine.config.query_relation_store_records
    return {
        "status": "declared",
        "declared_windows": len(descriptors),
        "relation_records_per_window": records,
        "max_relation_records_per_window": max(records, default=0),
        "window_capacity": capacity,
        "relation_store_capacity_records": store_capacity,
        "every_window_fits_store": (
            store_capacity is None or max(records, default=0) <= store_capacity
        ),
        "declared_windows_fit_table": (
            capacity is None or len(descriptors) <= capacity
        ),
        "interpretation": "Declared packet occupancy only; simultaneous liveness follows dependency order.",
    }


def _packet_diagnostics(
    trace: Trace, packet_plan: RelationPacketPlan,
) -> dict[str, Any]:
    logical = int(np.count_nonzero(
        trace.events["primitive_kind"] == int(PrimitiveKind.RELATION)
    ))
    physical = packet_plan.relation_packet_count
    return {
        "logical_relation_lane_events": logical,
        "physical_relation_packets": physical,
        "query_lanes_per_packet": packet_plan.query_lanes,
        "mean_active_lanes": logical / physical if physical else 0.0,
        "lane_utilization": (
            logical / (physical * packet_plan.query_lanes) if physical else 0.0
        ),
    }


def analyze_cycle_lower_bounds(
    engine: Any,
    trace: Trace,
    *,
    base_asic_cycles: int,
    targets: Mapping[str, float],
) -> CycleLowerBoundReport:
    """Compute necessary, optimistic bounds without changing trace or resources."""

    if base_asic_cycles <= 0:
        raise ValueError("Base ASIC cycles must be positive")
    required_targets = {"query", "residency", "full"}
    if set(targets) != required_targets or any(value <= 1.0 for value in targets.values()):
        raise ValueError("targets must define positive query, residency, and full speedups")
    packet_plan = RelationPacketPlan.from_trace(
        trace, query_lanes=engine.config.relation_query_lanes,
    )
    event_service = _event_service_cycles(engine, trace, packet_plan)
    dependency_cycles = _dependency_bound(trace, event_service, packet_plan)
    relation_lane_events = int(np.count_nonzero(
        trace.events["primitive_kind"] == int(PrimitiveKind.RELATION)
    ))
    lane_utilization = (
        relation_lane_events
        / (packet_plan.relation_packet_count * packet_plan.query_lanes)
        if packet_plan.relation_packet_count else 0.0
    )
    common_components = [CycleBoundComponent(
        name="trace.dependency_critical_path",
        category="dependency_dag",
        cycles=dependency_cycles,
        evidence={
            "event_count": trace.event_count,
            "dependency_count": int(trace.dependencies.size),
            "logical_relation_lane_events": relation_lane_events,
            "physical_relation_packets": packet_plan.relation_packet_count,
            "relation_lane_utilization": lane_utilization,
        },
        limitation="Unlimited hardware resources; preserves every real dependency and event service path.",
    )]
    module_components = _module_components(engine, trace, packet_plan)
    common_components.extend(_fusion_components(engine, trace, packet_plan))
    common_components.extend(_compute_components(engine, trace, packet_plan))
    common_components.extend(_query_components(engine, trace, packet_plan))

    scenarios: list[ScenarioCycleBound] = []
    for scenario in ("query", "residency", "full"):
        compulsory_only = scenario != "query"
        request_count, byte_count = _cache_request_floor(
            trace, packet_plan, compulsory_only=compulsory_only
        )
        scenario_module_components = (
            module_components
            if not compulsory_only
            else _module_components(
                engine, trace, packet_plan, compulsory_cache_writes=True,
            )
        )
        components = tuple([
            *common_components,
            *scenario_module_components,
            *_memory_components(
                engine,
                request_count=request_count,
                byte_count=byte_count,
                scenario=scenario,
            ),
        ])
        limiting = max(components, key=lambda item: item.cycles)
        scenarios.append(ScenarioCycleBound(
            scenario=scenario,
            minimum_cycles=limiting.cycles,
            limiting_component=limiting.name,
            memory_request_lower_bound=request_count,
            memory_byte_lower_bound=byte_count,
            components=components,
        ))

    reachability: list[TargetReachability] = []
    for scenario in scenarios:
        target = float(targets[scenario.scenario])
        budget = math.floor(base_asic_cycles / target)
        lower = scenario.minimum_cycles
        reachability.append(TargetReachability(
            scenario=scenario.scenario,
            target_speedup_vs_base_asic=target,
            target_cycle_budget=budget,
            lower_bound_cycles=lower,
            maximum_possible_speedup_vs_base_asic=base_asic_cycles / lower,
            maximum_coverable_cycles=max(base_asic_cycles - lower, 0),
            minimum_uncovered_cycles=lower,
            status=(
                "ruled_out_by_necessary_lower_bound"
                if lower > budget else "not_ruled_out_by_necessary_lower_bound"
            ),
        ))
    return CycleLowerBoundReport(
        schema_version="gala-cycle-lower-bound-v1",
        trace_event_count=trace.event_count,
        trace_dependency_count=int(trace.dependencies.size),
        base_asic_cycles=base_asic_cycles,
        scenarios=tuple(scenarios),
        reachability=tuple(reachability),
        capacity_diagnostics=_capacity_diagnostics(engine, trace, packet_plan),
    )
