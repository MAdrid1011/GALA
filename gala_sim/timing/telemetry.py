"""Optional ComputePod timing telemetry for cycle-path diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from gala_sim.clamp.events import PrimitiveKind


_COMPUTE_KINDS = {
    PrimitiveKind.FORWARD,
    PrimitiveKind.ADJOINT,
    PrimitiveKind.GRADIENT_REDUCTION,
}


@dataclass(frozen=True)
class ComputeEventTiming:
    event_id: int
    primitive_kind: str
    pod: int | None
    cluster: int | None
    dependency_ready_cycle: int
    fusion_issue_cycle: int | None
    query_issue_cycle: int | None
    compute_issue_cycle: int
    finish_cycle: int


@dataclass(frozen=True)
class ComputeClusterOccupancyRun:
    """Half-open cycle range with an exact active-microcontext vector."""

    start_cycle: int
    end_cycle: int
    active_microcontexts: tuple[int, ...]


@dataclass(frozen=True)
class ComputeTelemetry:
    cluster_count: int
    occupancy_metric: str
    event_timings: tuple[ComputeEventTiming, ...]
    cluster_occupancy_runs: tuple[ComputeClusterOccupancyRun, ...]


class ComputeTelemetryCollector:
    """Collect exact stage times and run-length encoded cluster occupancy."""

    def __init__(self, *, cluster_count: int, clusters_per_pod: int) -> None:
        if cluster_count <= 0 or clusters_per_pod <= 0:
            raise ValueError("ComputePod telemetry topology must be positive")
        if cluster_count % clusters_per_pod:
            raise ValueError("ComputePod telemetry topology does not close")
        self.cluster_count = cluster_count
        self.clusters_per_pod = clusters_per_pod
        self._events: dict[int, dict[str, int | str | None]] = {}
        self._occupancy_deltas: dict[int, dict[int, int]] = {}

    def mark_dependency_ready(
        self, event_id: int, kind: PrimitiveKind, cycle: int,
    ) -> None:
        if kind not in _COMPUTE_KINDS:
            return
        record = self._events.setdefault(event_id, {
            "event_id": event_id,
            "primitive_kind": kind.name,
            "pod": None,
            "cluster": None,
            "dependency_ready_cycle": cycle,
            "fusion_issue_cycle": None,
            "query_issue_cycle": None,
            "compute_issue_cycle": None,
            "finish_cycle": None,
        })
        record["dependency_ready_cycle"] = min(
            int(record["dependency_ready_cycle"]), cycle
        )

    def mark_issue(
        self,
        event_ids: Iterable[int],
        kind: PrimitiveKind,
        module_name: str,
        cycle: int,
        *,
        compute_plan: tuple[tuple[str, int, int], ...] = (),
    ) -> None:
        if kind not in _COMPUTE_KINDS:
            return
        cluster = self._cluster_from_plan(compute_plan)
        for event_id in event_ids:
            record = self._events.get(event_id)
            if record is None:
                raise ValueError(
                    f"compute event {event_id} issued before dependency readiness"
                )
            if module_name == "fusion_issue":
                record["fusion_issue_cycle"] = cycle
            elif module_name == "bidirectional_query":
                record["query_issue_cycle"] = cycle
            elif module_name == "compute_pod":
                record["compute_issue_cycle"] = cycle
                record["cluster"] = cluster
                record["pod"] = (
                    cluster // self.clusters_per_pod
                    if cluster is not None else None
                )
        if module_name == "compute_pod":
            self._record_occupancy(compute_plan)

    def mark_finish(self, event_id: int, cycle: int) -> None:
        record = self._events.get(event_id)
        if record is not None:
            record["finish_cycle"] = cycle

    def finish(self, total_cycles: int) -> ComputeTelemetry:
        timings: list[ComputeEventTiming] = []
        for event_id in sorted(self._events):
            record = self._events[event_id]
            if record["compute_issue_cycle"] is None or record["finish_cycle"] is None:
                raise ValueError(f"compute event {event_id} has incomplete telemetry")
            timings.append(ComputeEventTiming(
                event_id=event_id,
                primitive_kind=str(record["primitive_kind"]),
                pod=self._optional_int(record["pod"]),
                cluster=self._optional_int(record["cluster"]),
                dependency_ready_cycle=int(record["dependency_ready_cycle"]),
                fusion_issue_cycle=self._optional_int(record["fusion_issue_cycle"]),
                query_issue_cycle=self._optional_int(record["query_issue_cycle"]),
                compute_issue_cycle=int(record["compute_issue_cycle"]),
                finish_cycle=int(record["finish_cycle"]),
            ))
        return ComputeTelemetry(
            cluster_count=self.cluster_count,
            occupancy_metric="active_microcontexts",
            event_timings=tuple(timings),
            cluster_occupancy_runs=self._occupancy_runs(total_cycles),
        )

    @staticmethod
    def _optional_int(value: int | str | None) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _cluster_from_plan(
        plan: tuple[tuple[str, int, int], ...],
    ) -> int | None:
        for resource, _cycle, _demand in plan:
            if resource.startswith("cluster_issue:"):
                return int(resource.partition(":")[2])
        return None

    def _record_occupancy(
        self, plan: tuple[tuple[str, int, int], ...],
    ) -> None:
        points: dict[int, list[int]] = {}
        for resource, cycle, demand in plan:
            if not resource.startswith("microcontext_slots:"):
                continue
            cluster = int(resource.partition(":")[2])
            points.setdefault(cluster, []).extend([cycle] * demand)
        for cluster, cycles in points.items():
            start = min(cycles)
            end = max(cycles) + 1
            if len(cycles) != end - start:
                raise ValueError("ComputePod microcontext reservation is not contiguous")
            self._add_delta(start, cluster, 1)
            self._add_delta(end, cluster, -1)

    def _add_delta(self, cycle: int, cluster: int, delta: int) -> None:
        changes = self._occupancy_deltas.setdefault(cycle, {})
        changes[cluster] = changes.get(cluster, 0) + delta

    def _occupancy_runs(
        self, total_cycles: int,
    ) -> tuple[ComputeClusterOccupancyRun, ...]:
        if total_cycles < 0:
            raise ValueError("ComputePod telemetry total cycles cannot be negative")
        occupancy = [0] * self.cluster_count
        runs: list[ComputeClusterOccupancyRun] = []
        cursor = 0
        for cycle in sorted(self._occupancy_deltas):
            clipped = min(cycle, total_cycles)
            if cursor < clipped:
                runs.append(ComputeClusterOccupancyRun(
                    start_cycle=cursor,
                    end_cycle=clipped,
                    active_microcontexts=tuple(occupancy),
                ))
                cursor = clipped
            if cycle > total_cycles:
                break
            for cluster, delta in self._occupancy_deltas[cycle].items():
                occupancy[cluster] += delta
                if occupancy[cluster] < 0:
                    raise ValueError("ComputePod telemetry occupancy underflows")
        if cursor < total_cycles:
            runs.append(ComputeClusterOccupancyRun(
                start_cycle=cursor,
                end_cycle=total_cycles,
                active_microcontexts=tuple(occupancy),
            ))
        return tuple(runs)
