"""Compiler-generated semantic placement for resident Gaussian worksets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
from types import MappingProxyType
from typing import Iterable, Mapping

from gala_sim.clamp.events import PrimitiveKind
from gala_sim.trace.model import Trace
from gala_sim.trace.virtual import VirtualTracePacket

from .config import CycleConfig
from .packets import RelationPacketPlan


PlacementKey = tuple[int, int]

# Gaussian IDs in the captured Chest workload are correlated in their low
# bits.  Fold two higher address bits into the public routing key so every
# configuration sees the same low-cost, deterministic distribution.
GAUSSIAN_ROUTE_FOLD_SHIFTS = (6, 13)


def folded_gaussian_index(gaussian_id: int, modulo: int) -> int:
    """Map one Gaussian ID through the frozen XOR-fold address hash."""

    gaussian_id = int(gaussian_id)
    modulo = int(modulo)
    if gaussian_id < 0:
        raise ValueError("Gaussian route IDs must be non-negative")
    if modulo <= 0:
        raise ValueError("Gaussian route modulus must be positive")
    folded = gaussian_id
    for shift in GAUSSIAN_ROUTE_FOLD_SHIFTS:
        folded ^= gaussian_id >> shift
    return folded % modulo


@dataclass(frozen=True)
class SemanticPlacement:
    """Deterministic LPT assignment emitted with one semantic workset."""

    cluster_by_key: Mapping[PlacementKey, int]
    demand_by_key: Mapping[PlacementKey, int]
    cluster_loads_by_iteration: Mapping[int, tuple[int, ...]]
    cluster_count: int

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        packet_plan: RelationPacketPlan,
        config: CycleConfig,
    ) -> "SemanticPlacement":
        """Compile exact physical packet demand from an expanded trace."""

        if config.compute_templates is None:
            return cls._from_demands({}, 1)
        path_names = {
            PrimitiveKind.FORWARD: "forward",
            PrimitiveKind.ADJOINT: "adjoint",
            PrimitiveKind.GRADIENT_REDUCTION: "gradient_reduction",
        }
        demands: dict[PlacementKey, int] = defaultdict(int)
        for stage in packet_plan.stages:
            path_name = path_names.get(stage.kind)
            if path_name is None:
                continue
            event_index = stage.head_event_id - packet_plan.event_id_base
            if not 0 <= event_index < trace.event_count:
                raise ValueError("semantic placement stage is outside its trace")
            row = trace.events[event_index]
            gaussian_id = int(row["gaussian_id"])
            if gaussian_id < 0:
                raise ValueError("semantic placement packet has no Gaussian identity")
            try:
                path = config.compute_templates[int(row["template_id"])].path_for(
                    path_name
                )
            except KeyError as error:
                raise ValueError(str(error)) from error
            issue_cycles = (
                path.packet_issue_cycles(stage.active_lanes)
                if stage.kind in {PrimitiveKind.FORWARD, PrimitiveKind.ADJOINT}
                else path.cluster_issue_cycles
            )
            key = (int(row["iteration_id"]), gaussian_id)
            demands[key] += issue_cycles * path.cluster_issue_slots
        return cls._from_demands(demands, cls._cluster_count(config))

    @classmethod
    def from_virtual_packets(
        cls,
        packets: Iterable[VirtualTracePacket],
        config: CycleConfig,
        *,
        max_relations: int,
    ) -> "SemanticPlacement":
        """Compile the same demand directly from compact point/mask packets."""

        if max_relations <= 0:
            raise ValueError("semantic placement relation bound must be positive")
        if config.compute_templates is None:
            return cls._from_demands({}, 1)
        demands: dict[PlacementKey, int] = defaultdict(int)
        for packet in packets:
            try:
                profile = config.compute_templates[packet.template_id]
                forward = profile.path_for("forward")
                adjoint = profile.path_for("adjoint")
                gradient = profile.path_for("gradient_reduction")
            except KeyError as error:
                raise ValueError(str(error)) from error
            histograms = packet.relation_packet_lane_histograms_by_candidate(
                query_lanes=config.relation_query_lanes,
                max_relations=max_relations,
            )
            for candidate, gaussian_id in enumerate(packet.point_ids):
                demand = 0
                for active_lanes in range(1, config.relation_query_lanes + 1):
                    packet_count = int(histograms[candidate, active_lanes])
                    if not packet_count:
                        continue
                    demand += packet_count * (
                        forward.packet_issue_cycles(active_lanes)
                        * forward.cluster_issue_slots
                    )
                    if packet.backward_confirmed:
                        demand += packet_count * (
                            adjoint.packet_issue_cycles(active_lanes)
                            * adjoint.cluster_issue_slots
                            + gradient.cluster_issue_cycles
                            * gradient.cluster_issue_slots
                        )
                if demand:
                    demands[(packet.iteration_id, int(gaussian_id))] += demand
        return cls._from_demands(demands, cls._cluster_count(config))

    @staticmethod
    def _cluster_count(config: CycleConfig) -> int:
        capacities = config.compute_resource_capacities
        if capacities is None or "clusters" not in capacities:
            raise ValueError("semantic placement requires ComputePod topology")
        cluster_count = int(capacities["clusters"])
        if cluster_count <= 0:
            raise ValueError("semantic placement cluster count must be positive")
        return cluster_count

    @classmethod
    def _from_demands(
        cls,
        raw_demands: Mapping[PlacementKey, int],
        cluster_count: int,
    ) -> "SemanticPlacement":
        demands = {
            (int(key[0]), int(key[1])): int(value)
            for key, value in raw_demands.items()
            if int(value) > 0
        }
        if any(min(key) < 0 for key in demands):
            raise ValueError("semantic placement keys must be non-negative")
        by_iteration: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (iteration_id, gaussian_id), demand in demands.items():
            by_iteration[iteration_id].append((gaussian_id, demand))
        cluster_by_key: dict[PlacementKey, int] = {}
        loads_by_iteration: dict[int, tuple[int, ...]] = {}
        for iteration_id, work in sorted(by_iteration.items()):
            heap = [(0, cluster) for cluster in range(cluster_count)]
            heapq.heapify(heap)
            loads = [0] * cluster_count
            for gaussian_id, demand in sorted(
                work, key=lambda item: (-item[1], item[0]),
            ):
                load, cluster = heapq.heappop(heap)
                cluster_by_key[(iteration_id, gaussian_id)] = cluster
                load += demand
                loads[cluster] = load
                heapq.heappush(heap, (load, cluster))
            loads_by_iteration[iteration_id] = tuple(loads)
        return cls(
            cluster_by_key=MappingProxyType(cluster_by_key),
            demand_by_key=MappingProxyType(demands),
            cluster_loads_by_iteration=MappingProxyType(loads_by_iteration),
            cluster_count=cluster_count,
        )

    def cluster_for(self, iteration_id: int, gaussian_id: int) -> int | None:
        return self.cluster_by_key.get((int(iteration_id), int(gaussian_id)))
